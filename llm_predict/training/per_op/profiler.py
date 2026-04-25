"""profiler.py — torch CUDA-event profiler for per-op training.

Adapted from experiment/profile_phase1.py. Measures median per-sub-module
latency (self_attn, mlp, input_layernorm, post_attention_layernorm) over
a (bs, seq) grid using only the first 2 transformer layers, so it fits
in small-VRAM GPUs (RTX 2080 Ti, 22 GB).

Phase 2 extension (2026-04-25): adds a decode grid that profiles single-
token decode steps at various kv_cache_len values. For each (bs, kv_len),
a prefill pass populates the KV cache, then the decode step is measured
with use_cache=True. This gives the serving_e2e predictor real per-op
decode training data instead of the Phase 1 bandwidth approximation.

Output: {out_root}/{gpu}/{model_dir}/tp1/phase1_bsseq_profiles.csv
  columns: model_name, layer_name, bs, seq, n_tokens, kv_cache_len,
           tp_size, latency_ns

  Prefill rows: kv_cache_len=0, seq=1..512.
  Decode rows: kv_cache_len>0, seq=1.

  — matches the legacy schema; labeler.py accepts it via column
  normalisation (latency_ns → duration_us).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


# v4-dense grid: bs=1, every seq 1..512 (matches existing A100 Llama-3.3-70B coverage)
DEFAULT_GRID = [(1, s) for s in range(1, 513)]

# Phase 2 decode grid: single-token decode at various KV cache depths.
DECODE_GRID = [
    (bs, kv_len)
    for bs in [1, 2, 4, 8]
    for kv_len in [128, 256, 512, 1024, 2048, 4096, 8192]
]

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _measure_layer(layer, layer_args, layer_kwargs, sub_names, warmup, repeat):
    """Measure total-block and per-sub-module latencies. Returns list of (name, median_ns)."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(*layer_args, **layer_kwargs)
        torch.cuda.synchronize()

    lats = []
    with torch.no_grad():
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            _ = layer(*layer_args, **layer_kwargs)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e) * 1e6)
    lats.sort()
    total_ns = int(lats[len(lats) // 2])
    out = [("total_block", total_ns)]

    for sub_name in sub_names:
        sub_mod = getattr(layer, sub_name)
        lats = []
        with torch.no_grad():
            for _ in range(repeat):
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                def pre_h(_m, _args, _se=se):
                    _se.record()
                def post_h(_m, _args, _out, _ee=ee):
                    _ee.record()
                h1 = sub_mod.register_forward_pre_hook(pre_h)
                h2 = sub_mod.register_forward_hook(post_h)
                torch.cuda.synchronize()
                _ = layer(*layer_args, **layer_kwargs)
                torch.cuda.synchronize()
                h1.remove(); h2.remove()
                lats.append(se.elapsed_time(ee) * 1e6)
        lats.sort()
        out.append((sub_name, int(lats[len(lats) // 2])))

    return out


def run(model_path: Path, gpu: str, device: str, dtype: torch.dtype,
        out_root: Path, warmup: int = 5, repeat: int = 15,
        n_layers: int = 2,
        skip_prefill: bool = False, skip_decode: bool = False) -> Path:
    model_name = model_path.name
    out_dir = out_root / gpu / model_name / "tp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase1_bsseq_profiles.csv"

    print(f"[*] Loading {model_name} ({n_layers} layers, dtype={dtype})")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.num_hidden_layers = n_layers
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=dtype,
        device_map=device, ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[0]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[0]
    else:
        raise RuntimeError("cannot find transformer layers in model")

    sub_names = [n for n, _ in layer.named_children()]
    print(f"[*] sub-modules: {sub_names}")

    results: list[dict] = []

    def make_hook():
        captured: dict = {}
        def hook_fn(_m, args, kwargs):
            captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
            captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
        return captured, hook_fn

    # ── Prefill grid ──────────────────────────────────────────────────────
    if not skip_prefill:
        print(f"[*] Prefill grid: {len(DEFAULT_GRID)} cells")
        for bs, seq in DEFAULT_GRID:
            n_tok = bs * seq
            print(f"  [prefill] bs={bs:<3} seq={seq:<4} (ntok={n_tok})...", end="", flush=True)
            try:
                input_ids = torch.randint(0, 1000, (bs, seq), device=device)
                captured, hook_fn = make_hook()
                h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
                with torch.no_grad():
                    _ = model(input_ids=input_ids)
                h.remove()
                layer_args = captured["args"]
                layer_kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
                layer_kwargs["use_cache"] = False

                measurements = _measure_layer(layer, layer_args, layer_kwargs, sub_names, warmup, repeat)
                for name, ns in measurements:
                    results.append({"model_name": model_name, "layer_name": name,
                                    "bs": bs, "seq": seq, "n_tokens": n_tok,
                                    "kv_cache_len": 0, "tp_size": 1, "latency_ns": ns})
                print(f" {measurements[0][1]/1000:.0f}us")
            except torch.cuda.OutOfMemoryError:
                print(" OOM")
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f" ERROR: {ex}")

    # ── Decode grid ───────────────────────────────────────────────────────
    if not skip_decode:
        print(f"[*] Decode grid: {len(DECODE_GRID)} cells")
        for bs, kv_len in DECODE_GRID:
            print(f"  [decode] bs={bs:<3} kv_len={kv_len:<5}...", end="", flush=True)
            try:
                # Step 1: prefill to populate KV cache.
                prefill_ids = torch.randint(0, 1000, (bs, kv_len), device=device)
                with torch.no_grad():
                    prefill_out = model(input_ids=prefill_ids, use_cache=True)
                past_kv = prefill_out.past_key_values

                # Step 2: single-token decode — hook captures layer inputs
                # including the per-layer past_key_values slice.
                decode_ids = torch.randint(0, 1000, (bs, 1), device=device)
                captured, hook_fn = make_hook()
                h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
                with torch.no_grad():
                    _ = model(input_ids=decode_ids, past_key_values=past_kv, use_cache=True)
                h.remove()

                layer_args = captured["args"]
                layer_kwargs = dict(captured["kwargs"])

                measurements = _measure_layer(layer, layer_args, layer_kwargs, sub_names, warmup, repeat)
                for name, ns in measurements:
                    results.append({"model_name": model_name, "layer_name": name,
                                    "bs": bs, "seq": 1, "n_tokens": bs,
                                    "kv_cache_len": kv_len, "tp_size": 1, "latency_ns": ns})
                print(f" {measurements[0][1]/1000:.0f}us")

                del prefill_out, past_kv
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(" OOM")
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f" ERROR: {ex}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model_name", "layer_name", "bs", "seq", "n_tokens",
            "kv_cache_len", "tp_size", "latency_ns"])
        w.writeheader()
        w.writerows(results)
    print(f"[+] saved {len(results)} rows → {csv_path}")
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--gpu", required=True, help="GPU short name, e.g. RTX2080Ti")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--out-root", type=Path, default=Path.home() / "per_op_traces")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--skip-prefill", action="store_true",
                    help="Skip the prefill grid (only run decode grid)")
    ap.add_argument("--skip-decode", action="store_true",
                    help="Skip the decode grid (only run prefill grid)")
    args = ap.parse_args()

    run(args.model_path, args.gpu, args.device, DTYPE_MAP[args.dtype],
        args.out_root, args.warmup, args.repeat, args.n_layers,
        skip_prefill=args.skip_prefill, skip_decode=args.skip_decode)


if __name__ == "__main__":
    main()
