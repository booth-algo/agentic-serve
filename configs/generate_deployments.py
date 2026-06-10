#!/usr/bin/env python3
"""Generate deployment JSONs for every (GPU, model, TP, engine) ground-truth run in the central
bench store, so the simulator page can cover the full matrix as labelled first-cuts.

For each ``/mnt/100g/.../synthetic_distributional/<gpu>_<model>_tp<N>_<engine>`` dir we:
  - map gpu/model -> the configs/gpus + configs/models JSONs (must already exist),
  - read ``gpu_memory_utilization`` + ``tensor_parallel_size`` from a sample bench file,
  - compute the KV pool analytically (configs/kv_pool.py),
  - write a deployment JSON whose ``data`` manifest honestly marks every kernel artifact as
    inherited/placeholder/missing (no measured kernels for these configs -> analytic decode
    roofline + H100-anchored ceiling/prefill) and the ground truth as measured.

The three CALIBRATED Llama-3.1-8B configs (H100 tp1, H100 tp2, A100 tp1) are hand-authored and
SKIPPED here so we never duplicate / overwrite them. The redundant ``h100-2`` host (only Llama-3.1-8B
tp1, identical to the H100 config) is skipped too.

gpu_key follows the dashboard's SERVING_GPU_ORDER convention: physical GPU + TP via ``x`` notation
(H100x2, A100x4, RTX3090x2, ...), SGLang gets a `` (sglang)`` suffix, vLLM none. The model stays a
ROW dimension (the dashboard model dropdown disambiguates), so multiple models share one gpu_key.

    python3 -m configs.generate_deployments            # write configs/deployments/*.json
    python3 -m configs.generate_deployments --dry-run  # print what would be written
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.kv_pool import available_kv_blocks, RESERVE_BYTES  # noqa: E402

BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
DEPLOYMENTS_DIR = REPO_ROOT / "configs" / "deployments"
SAMPLE_BENCH = "chat-multiturn-synth_conc1.json"

# dir gpu token -> (gpu JSON name, dashboard display prefix, roofline_peak provenance)
GPU_MAP = {
    "h100":   ("H100", "H100", "measured"),
    "a100":   ("A100", "A100", "measured"),
    "3090":   ("RTX3090", "RTX3090", "placeholder"),
    "2080ti": ("RTX2080Ti", "RTX2080Ti", "placeholder"),
}
TP_SUFFIX = {1: "", 2: "x2", 4: "x4", 8: "x8"}
ENGINE_SUFFIX = {"vllm": "", "sglang": " (sglang)"}

# bench_dirs owned by the hand-authored calibrated configs (never regenerate these)
SKIP_BENCH_DIRS = {
    "h100_Llama-3.1-8B_tp1_vllm",   # -> H100 (kernel-calibrated headline)
    "h100_Llama-3.1-8B_tp2_vllm",   # -> H100x2 (tp2 measured decode grid)
    "a100_Llama-3.1-8B_tp1_vllm",   # -> A100 (measured decode grid + ceiling)
}
SKIP_GPU_TOKENS = {"h100-2"}  # redundant 2nd-h100 host (only Llama-3.1-8B tp1; == H100)


def vllm_max_num_batched_tokens(total_memory_bytes: float, gpu_name: str) -> int:
    """The vLLM OpenAI-server default chunked-prefill budget (ENGINE CONFIG, not a model
    choice): 8192 for >=70GiB non-A100 devices, else 2048 — vllm/engine/arg_utils.py
    get_batch_defaults (the A100 carve-out cites vLLM PR #17885). Benchmarks ran the
    OpenAI API server with max_num_batched_tokens unset (server metadata: null), so the
    resolved default IS the engine value. Emitted into every generated vLLM deployment so
    regeneration preserves the key (audit-v2 item S14: the 2048 pins were originally
    hand-added and a regeneration without this function silently reverted them to the
    loader's 8192 default, quadrupling the overflow budget on small GPUs)."""
    return 8192 if (total_memory_bytes >= 70 * 1024**3
                    and "a100" not in gpu_name.lower()) else 2048


def sglang_chunked_prefill_size(total_memory_bytes: float) -> int:
    """The sglang resolved ``chunked_prefill_size`` default (ENGINE CONFIG, not a model
    choice) — a PER-DEVICE memory-tier rule, NOT vLLM's >=70GiB/non-A100 device rule
    (audit-v2 item G8: the sglang configs used to inherit the loader's vLLM 8192,
    a 4x error on 24GiB devices). Source: sglang
    ``python/sglang/srt/server_args.py`` ``ServerArgs._handle_gpu_memory_settings``
    (upstream main @ 255843d45462, fetched 2026-06-10; gpu_mem compared in MiB):

        gpu_mem <  20 GiB -> 2048   (T4, 4080; covers RTX2080Ti 11GiB)
        gpu_mem <  35 GiB -> 2048   (A10, 4090, 5090; covers RTX3090 24GiB)
        gpu_mem <  60 GiB -> 4096   (A100 40GB, L40)
        gpu_mem <  90 GiB -> 8192   (H100, A100 80GB)
        gpu_mem < 160 GiB -> 8192   (H20, H200)
        else              -> 16384  (B200, MI300)

    Benchmarks ran the sglang server with --chunked-prefill-size unset, so the
    resolved default IS the engine value. ``total_memory_bytes`` is the per-device
    gpu JSON value (matching sglang's get_device_memory_capacity per-GPU read; TP
    does not change the tier). Emitted as ``max_num_batched_tokens`` — the loader
    key the simulator consumes as the per-step chunked-prefill token budget."""
    gib = total_memory_bytes / 1024**3
    if gib < 20:
        return 2048
    if gib < 35:
        return 2048
    if gib < 60:
        return 4096
    if gib < 90:
        return 8192
    if gib < 160:
        return 8192
    return 16384


def parse_dir(name: str) -> tuple[str, str, int, str] | None:
    """``<gpu>_<model...>_tp<N>_<engine>`` -> (gpu_token, model, tp, engine)."""
    parts = name.split("_")
    if len(parts) < 4 or not parts[-2].startswith("tp"):
        return None
    gpu_token, engine, tp_tok = parts[0], parts[-1], parts[-2]
    model = "_".join(parts[1:-2])
    try:
        tp = int(tp_tok[2:])
    except ValueError:
        return None
    return gpu_token, model, tp, engine


def util_provenance(gpu_name: str) -> dict:
    """util_flops/util_bw/scheduler_overhead provenance for a GPU (all H100-anchored)."""
    if gpu_name == "H100":
        return {"status": "inherited", "from": "H100-Llama-3.1-8B",
                "note": "H100 util scalars reused for a different model on the same GPU"}
    return {"status": "placeholder", "from": "H100",
            "note": "H100 util scalars; re-anchor from a measured grid for this GPU"}


def build_manifest(gpu_name: str, peak_prov: str, bench_dir: str, kv_value: int,
                   util_note: dict, kv_note: str, kv_status: str) -> dict:
    return {
        "decode_grid":         {"status": "missing",
                                "note": "no measured kernel grid for this GPU/model -> analytic decode roofline (configs scale by weight bytes / bandwidth / KV)"},
        "kv_pool":             {"status": kv_status, "value": kv_value,
                                "source": "configs/kv_pool.py", "note": kv_note},
        "saturated_ceiling":   {"status": "inherited", "from": "H100",
                                "note": "TPOT plateau ceiling is H100/Llama-3.1-8B-anchored; under-caps large-model plateaus -> first-cut"},
        "roofline_peak":       {"status": peak_prov,
                                "note": "datasheet peak FLOPS/BW" + ("" if peak_prov == "measured" else " (FP16-accumulate ambiguity on consumer GPUs)")},
        "util_flops":          util_note,
        "util_bw":             util_note,
        "scheduler_overhead":  util_note,
        "cached_prefill_grid": {"status": "inherited", "from": "H100", "note": "TTFT prefill law H100-measured -> first-cut"},
        "fa3_grid":            {"status": "inherited", "from": "H100", "note": "FA3 prefill grid H100-measured -> first-cut"},
        "ground_truth":        {"status": "measured", "path": bench_dir, "source": "/mnt/100g"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BENCH_BASE.exists():
        raise SystemExit(f"bench store not mounted: {BENCH_BASE}")
    models_avail = {p.stem for p in (REPO_ROOT / "configs" / "models").glob("*.json")}
    gpus_avail = {p.stem for p in (REPO_ROOT / "configs" / "gpus").glob("*.json")}

    written, skipped = 0, []
    for d in sorted(BENCH_BASE.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_dir(d.name)
        if parsed is None:
            continue
        gpu_token, model, tp, engine = parsed
        if gpu_token in SKIP_GPU_TOKENS or d.name in SKIP_BENCH_DIRS:
            skipped.append((d.name, "calibrated/redundant"))
            continue
        if gpu_token not in GPU_MAP:
            skipped.append((d.name, f"unknown gpu '{gpu_token}'"))
            continue
        gpu_name, disp, peak_prov = GPU_MAP[gpu_token]
        if model not in models_avail:
            skipped.append((d.name, f"no model JSON '{model}'"))
            continue
        if gpu_name not in gpus_avail:
            skipped.append((d.name, f"no gpu JSON '{gpu_name}'"))
            continue
        if engine not in ENGINE_SUFFIX or tp not in TP_SUFFIX:
            skipped.append((d.name, f"unsupported tp{tp}/{engine}"))
            continue

        sample = d / SAMPLE_BENCH
        gpu_mem_util = 0.9
        if sample.exists():
            cfg = json.loads(sample.read_text()).get("config", {})
            gpu_mem_util = float(cfg.get("gpu_memory_utilization") or 0.9)

        gpu_json = json.loads((REPO_ROOT / "configs" / "gpus" / f"{gpu_name}.json").read_text())
        mdl_json = json.loads((REPO_ROOT / "configs" / "models" / f"{model}.json").read_text())
        weight_bytes = float(mdl_json["n_params"]) * float(mdl_json["bytes_per_param"])
        total_mem = float(gpu_json["total_memory_bytes"])
        kv_blocks = available_kv_blocks(
            total_memory_bytes=total_mem,
            gpu_mem_util=gpu_mem_util, weight_bytes=weight_bytes, tp=tp,
            kv_bytes_per_token=float(mdl_json["kv_bytes_per_token"]),
            kv_heads=int(mdl_json["kv_heads"]),
        )
        # If bf16 weights/tp don't fit the budget, the actual run must have quantized (we can't know the
        # effective weight size) -> the analytic pool is unreliable. Flag kv_pool MISSING so the coverage
        # report surfaces it; predictions for these near-OOM configs saturate (poor but honest first-cut).
        budget = total_mem * gpu_mem_util - weight_bytes / tp - RESERVE_BYTES
        kv_status = "derived" if budget > 0 else "missing"
        kv_note = (f"{total_mem / 1024**3:.0f}GiB util {gpu_mem_util}, "
                   f"weights {weight_bytes / 1e9:.1f}GB/tp{tp}, reserve {RESERVE_BYTES / 1e9:.1f}GB"
                   + ("" if budget > 0 else " -> bf16 weights exceed budget (run likely quantized); pool UNRELIABLE"))

        gpu_key = f"{disp}{TP_SUFFIX[tp]}{ENGINE_SUFFIX[engine]}"
        dep = {
            "gpu_key": gpu_key,
            "gpu": gpu_name,
            "model": model,
            "tp": tp,
            "engine": engine,
            "available_kv_blocks": kv_blocks,
            # Per-engine chunked-prefill budget rule (both ENGINE CONFIG): vLLM's device
            # rule vs sglang's per-device memory-tier rule (audit-v2 G8, resolved
            # 2026-06-10 — sglang configs no longer inherit the vLLM 8192 default).
            "max_num_batched_tokens": (
                vllm_max_num_batched_tokens(total_mem, gpu_name) if engine == "vllm"
                else sglang_chunked_prefill_size(total_mem)),
            "bench_dir": d.name,
            "backend": f"{disp.lower()}-tp{tp}-{engine}-analytic-roofline",
            "calibration_status": f"{gpu_token}_tp{tp}_{engine}_analytic_roofline_firstcut",
            "ground_truth": True,
            "data": build_manifest(gpu_name, peak_prov, d.name, kv_blocks,
                                   util_provenance(gpu_name), kv_note, kv_status),
        }
        out = DEPLOYMENTS_DIR / f"{gpu_token}_{model}_tp{tp}_{engine}.json"
        if args.dry_run:
            print(f"WOULD WRITE {out.name:48s} key={gpu_key!r:22s} kv={kv_blocks}")
        else:
            out.write_text(json.dumps(dep, indent=2) + "\n")
            written += 1

    print(f"\n{'(dry-run) ' if args.dry_run else ''}deployments written: {written}; skipped: {len(skipped)}")
    for name, why in skipped:
        print(f"  skip {name}: {why}")


if __name__ == "__main__":
    main()
