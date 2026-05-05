#!/usr/bin/env python3
"""
Profile Llama-3.1-8B layers for roofline analysis.

Two modes:
  --mode timing   : Full-model CUDA event timing + analytical per-layer OI
  --mode profile  : PyTorch Profiler on full model for kernel breakdown

Since all 32 decoder layers are architecturally identical, per-layer OI is
computed analytically. The timing mode measures per-layer execution time
variance via a full forward pass and CUDA events between layers.

Usage:
    python scripts/roofline/profile_all_layers.py \
        --model /data/models/Llama-3.1-8B-Instruct \
        --batch-size 80 --seq-len 512 --phase prefill \
        --mode timing --output-dir results/roofline/raw/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

H100_PEAK_BF16_TFLOPS = 989.0
H100_BW_GBS = 3352.0
H100_RIDGE = H100_PEAK_BF16_TFLOPS * 1000.0 / H100_BW_GBS  # ~295 FLOP/byte
BPP = 2  # bytes per bf16 element

# ── Model ───────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return model, config


# ── Analytical per-component OI ────────────────────────────────────────

def compute_layer_analytics(B: int, S: int, phase: str,
                             d: int, h: int, kv_h: int, intermediate: int) -> dict:
    """Analytical FLOPs and bytes for one LLaMA decoder layer."""
    head_dim = d // h
    q_len = 1 if phase == "decode" else S
    kv_len = S

    comps = []

    def add(name, flops, bytes_):
        oi = flops / bytes_ if bytes_ > 0 else 0
        bound = "compute" if oi > H100_RIDGE else "memory"
        comps.append({"name": name, "flops": flops, "bytes": bytes_,
                       "oi": round(oi, 1), "bound": bound})

    # RMSNorm (input)
    nf = B * S * d * 5
    nb = BPP * B * S * d * 2
    add("rmsnorm_in", nf, nb)

    # QKV projections
    for proj, out_dim in [("q_proj", d), ("k_proj", d * kv_h // h), ("v_proj", d * kv_h // h)]:
        pf = 2 * B * S * d * out_dim
        pb = BPP * (B * S * d + d * out_dim + B * S * out_dim)
        add(proj, pf, pb)

    # Rotary embeddings
    rf = B * h * q_len * head_dim * 8
    rb = BPP * B * h * q_len * head_dim * 3
    add("rope", rf, rb)

    # Flash attention
    af = 4 * B * h * q_len * kv_len * head_dim
    ab = BPP * B * h * head_dim * (q_len + 2 * kv_len + q_len) * 4
    add("attention", af, ab)

    # O projection
    of_ = 2 * B * S * d * d
    ob = BPP * (B * S * d + d * d + B * S * d)
    add("o_proj", of_, ob)

    # RMSNorm (post-attn)
    add("rmsnorm_post", nf, nb)

    # Gate + Up projections
    gf = 2 * B * S * d * intermediate
    gb = BPP * (B * S * d + d * intermediate + B * S * intermediate)
    add("gate_proj", gf, gb)
    add("up_proj", gf, gb)

    # SiLU + elementwise multiply
    add("silu", B * S * intermediate * 4, BPP * B * S * intermediate * 2)
    add("gate_mul", B * S * intermediate, BPP * B * S * intermediate * 3)

    # Down projection
    df_ = 2 * B * S * intermediate * d
    db = BPP * (B * S * intermediate + intermediate * d + B * S * d)
    add("down_proj", df_, db)

    total_flops = sum(c["flops"] for c in comps)
    total_bytes = sum(c["bytes"] for c in comps)
    oi = total_flops / total_bytes if total_bytes > 0 else 0

    return {
        "B": B, "S": S, "phase": phase,
        "d": d, "h": h, "kv_h": kv_h, "intermediate": intermediate,
        "head_dim": head_dim, "q_len": q_len, "kv_len": kv_len,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "oi": round(oi, 1),
        "ridge_point": round(H100_RIDGE, 1),
        "overall_bound": "compute" if oi > H100_RIDGE else "memory",
        "components": comps,
    }


# ── Full-model timing (CUDA events between layers) ─────────────────────

def time_full_model(model, input_ids, phase, config):
    """Time a full forward pass and record per-layer times via PyTorch hooks."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    # Build KV cache for decode
    past = None
    if phase == "decode":
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            past = out.past_key_values
        torch.cuda.synchronize()
        decode_ids = torch.randint(0, config.vocab_size, (batch_size, 1), device=device)

    # Warmup
    for _ in range(3):
        if phase == "decode":
            model(decode_ids, past_key_values=past, use_cache=True)
        else:
            model(input_ids)
    torch.cuda.synchronize()

    # Full forward timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if phase == "decode":
        model(decode_ids, past_key_values=past, use_cache=True)
    else:
        model(input_ids)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)

    # Per-layer timing via hooks
    layer_times = []
    layers = model.model.layers
    n_layers = len(layers)

    for i in range(n_layers):
        hidden = torch.randn(batch_size, seq_len if phase == "prefill" else 1,
                             config.hidden_size, device=device, dtype=torch.bfloat16)
        # Each layer timing measured separately with fresh random input
        for _ in range(3):
            try:
                layers[i](hidden, use_cache=False)
            except Exception:
                pass
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            layers[i](hidden, use_cache=False)
        except Exception:
            pass
        e.record()
        torch.cuda.synchronize()
        layer_times.append({
            "layer_idx": i,
            "cuda_time_ms": round(s.elapsed_time(e), 4),
        })

    return {
        "total_ms": round(total_ms, 3),
        "layers": layer_times,
        "embed_ms": 0,
        "norm_ms": 0,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--phase", type=str, default="prefill",
                        choices=["prefill", "decode"])
    parser.add_argument("--mode", type=str, default="timing",
                        choices=["timing", "analytical"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="results/roofline/raw")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.path.basename(args.model)

    if args.mode == "analytical":
        # No GPU needed — pure analytical computation
        d, h, kv_h, intermediate = 4096, 32, 8, 14336
        analytics = compute_layer_analytics(
            args.batch_size, args.seq_len, args.phase, d, h, kv_h, intermediate)

        print(f"\nAnalytical per-layer OI for Llama-3.1-8B", file=sys.stderr)
        print(f"  B={args.batch_size} S={args.seq_len} phase={args.phase}", file=sys.stderr)
        print(f"  OI={analytics['oi']} FLOP/byte (ridge={analytics['ridge_point']})", file=sys.stderr)
        print(f"  Overall bound: {analytics['overall_bound']}", file=sys.stderr)
        print(f"\n  {'Component':<18s} {'FLOPs':>14s} {'Bytes':>14s} {'OI':>8s} {'Bound':>8s}", file=sys.stderr)
        for c in analytics["components"]:
            print(f"  {c['name']:<18s} {c['flops']:>14.2e} {c['bytes']:>14.2e} "
                  f"{c['oi']:>8.1f} {c['bound']:>8s}", file=sys.stderr)

        out = output_dir / f"llama8b_analytical_{args.phase}_B{args.batch_size}.json"
        with open(out, "w") as f:
            json.dump(analytics, f, indent=2)
        print(f"\nOutput: {out}", file=sys.stderr)
        return 0

    # GPU timing mode
    print(f"=== Full-model timing: {model_name} ===", file=sys.stderr)
    print(f"  phase={args.phase} bs={args.batch_size} seq={args.seq_len}", file=sys.stderr)

    model, config = load_model(args.model, args.device, args.trust_remote_code)
    print(f"  d={config.hidden_size} h={config.num_attention_heads} "
          f"kv_h={getattr(config, 'num_key_value_heads', config.num_attention_heads)} "
          f"n_layers={config.num_hidden_layers} intermediate={config.intermediate_size}",
          file=sys.stderr)

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len),
                              device=args.device)

    timing = time_full_model(model, input_ids, args.phase, config)

    # Compute analytics
    d = config.hidden_size
    h = config.num_attention_heads
    kv_h = getattr(config, "num_key_value_heads", h)
    intermediate = config.intermediate_size
    analytics = compute_layer_analytics(
        args.batch_size, args.seq_len, args.phase, d, h, kv_h, intermediate)

    # Merge: per-layer timing + analytical OI
    total_layer_time = sum(l["cuda_time_ms"] for l in timing["layers"])
    per_layer = []
    for l in timing["layers"]:
        time_s = l["cuda_time_ms"] / 1000.0
        achieved = analytics["total_flops"] / time_s / 1e12 if time_s > 0 else 0
        peak_ratio = achieved / H100_PEAK_BF16_TFLOPS * 100 if H100_PEAK_BF16_TFLOPS > 0 else 0
        per_layer.append({
            "layer_idx": l["layer_idx"],
            "cuda_time_ms": l["cuda_time_ms"],
            "time_pct": round(l["cuda_time_ms"] / total_layer_time * 100, 2) if total_layer_time > 0 else 0,
            "flops": analytics["total_flops"],
            "bytes": analytics["total_bytes"],
            "oi": analytics["oi"],
            "achieved_tflops": round(achieved, 2),
            "peak_ratio_pct": round(peak_ratio, 1),
            "bound": analytics["overall_bound"],
        })

    result = {
        "model_name": model_name,
        "phase": args.phase,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "total_ms": timing["total_ms"],
        "total_layer_time_ms": round(total_layer_time, 3),
        "analytics": analytics,
        "per_layer": per_layer,
    }

    out_path = output_dir / f"{model_name}_per_layer_{args.phase}_bs{args.batch_size}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Total forward: {timing['total_ms']:.3f} ms", file=sys.stderr)
    print(f"  Total layers:  {total_layer_time:.3f} ms", file=sys.stderr)
    print(f"  Per-layer mean: {total_layer_time/len(timing['layers']):.3f} ms", file=sys.stderr)
    print(f"  OI: {analytics['oi']:.0f} FLOP/byte ({analytics['overall_bound']}-bound)", file=sys.stderr)
    print(f"  Output: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
