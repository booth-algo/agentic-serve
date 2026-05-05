#!/usr/bin/env python3
"""
Parse per-layer timing data and compute analytical OI, FLOPs, bytes per layer.

Combines measured per-layer CUDA times from profile_all_layers.py with
analytical FLOP/byte counts from model architecture to determine whether
each layer is memory-bound or compute-bound.

Output: per_layer_analysis.json with per-layer OI classification.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ── H100 Specs ───────────────────────────────────────────────────────────

P_PEAK = 989.0         # TFLOP/s (bf16 tensor core)
BW = 3352.0            # GB/s = 3.352 TB/s
RIDGE_POINT = P_PEAK * 1000.0 / BW  # ~295 FLOP/byte
BPP = 2                # bytes per param (bf16)

# ── Analytical model ────────────────────────────────────────────────────

@dataclass
class LayerComponent:
    name: str
    flops: float       # in FLOPs
    bytes_: float      # in bytes
    fraction: float    # fraction of total layer time (estimated)


def compute_layer_analytics(
    B: int, S: int, phase: str,
    d: int, h: int, kv_h: int, intermediate: int,
) -> dict:
    """Compute per-component FLOPs and bytes for one LLaMA decoder layer.

    Args:
        B: batch size
        S: sequence length (prefill) or kv_len (decode)
        phase: "prefill" or "decode"
        d: hidden dimension
        h: num attention heads
        kv_h: num key/value heads (GQA)
        intermediate: FFN intermediate size
    """
    head_dim = d // h

    # Decode: query length = 1. Prefill: query length = S.
    q_len = 1 if phase == "decode" else S
    kv_len = S  # KV cache length

    components = []

    # 1. Input RMSNorm
    norm_flops = B * S * d * 5  # ~5 FLOPs per element
    norm_bytes = BPP * B * S * d * 2  # read input + write output
    components.append(LayerComponent("rmsnorm_in", norm_flops, norm_bytes, 0.02))

    # 2. Q projection: [B, S, d] @ [d, d] -> [B, S, d]
    q_flops = 2 * B * S * d * d
    q_bytes = BPP * (B * S * d + d * d + B * S * d)  # read input, read weight, write output
    components.append(LayerComponent("q_proj", q_flops, q_bytes, 0.10))

    # 3. K projection: [B, S, d] @ [d, d*kv_h/h] -> [B, S, d*kv_h/h]
    k_d = d * kv_h // h
    k_flops = 2 * B * S * d * k_d
    k_bytes = BPP * (B * S * d + d * k_d + B * S * k_d)
    components.append(LayerComponent("k_proj", k_flops, k_bytes, 0.08))

    # 4. V projection: same as K
    v_flops = k_flops
    v_bytes = k_bytes
    components.append(LayerComponent("v_proj", v_flops, v_bytes, 0.08))

    # 5. Rotary embeddings (elementwise on Q and K)
    rope_flops = B * h * q_len * head_dim * 8  # cos + sin + mul + add
    rope_bytes = BPP * B * h * q_len * head_dim * 3  # read Q/K, write rotated
    components.append(LayerComponent("rope", rope_flops, rope_bytes, 0.03))

    # 6. Flash Attention
    # FLOPs = 4 * B * h * q_len * kv_len * head_dim  (fwd flash attention)
    attn_flops = 4 * B * h * q_len * kv_len * head_dim
    # Memory: Q, K, V read + O write. Flash attention minimizes DRAM.
    # Typical: ~4 passes through data in SRAM, DRAM bytes ≈ 4 * (Q + K + V + O)
    attn_bytes = BPP * B * h * head_dim * (q_len + 2 * kv_len + q_len) * 4
    components.append(LayerComponent("attention", attn_flops, attn_bytes, 0.15))

    # 7. O projection: [B, S, d] @ [d, d] -> [B, S, d]
    o_flops = 2 * B * S * d * d
    o_bytes = BPP * (B * S * d + d * d + B * S * d)
    components.append(LayerComponent("o_proj", o_flops, o_bytes, 0.10))

    # 8. Post-attention RMSNorm
    components.append(LayerComponent("rmsnorm_post", norm_flops, norm_bytes, 0.02))

    # 9. Gate projection: [B, S, d] @ [d, intermediate] -> [B, S, intermediate]
    gate_flops = 2 * B * S * d * intermediate
    gate_bytes = BPP * (B * S * d + d * intermediate + B * S * intermediate)
    components.append(LayerComponent("gate_proj", gate_flops, gate_bytes, 0.12))

    # 10. Up projection: same as gate
    up_flops = gate_flops
    up_bytes = gate_bytes
    components.append(LayerComponent("up_proj", up_flops, up_bytes, 0.12))

    # 11. SiLU activation
    silu_flops = B * S * intermediate * 4
    silu_bytes = BPP * B * S * intermediate * 2  # read + write
    components.append(LayerComponent("silu", silu_flops, silu_bytes, 0.03))

    # 12. Elementwise multiply (gate * up)
    mul_flops = B * S * intermediate
    mul_bytes = BPP * B * S * intermediate * 3  # read gate, read up, write
    components.append(LayerComponent("gate_mul", mul_flops, mul_bytes, 0.02))

    # 13. Down projection: [B, S, intermediate] @ [intermediate, d] -> [B, S, d]
    down_flops = 2 * B * S * intermediate * d
    down_bytes = BPP * (B * S * intermediate + intermediate * d + B * S * d)
    components.append(LayerComponent("down_proj", down_flops, down_bytes, 0.13))

    total_flops = sum(c.flops for c in components)
    total_bytes = sum(c.bytes_ for c in components)
    oi = total_flops / total_bytes if total_bytes > 0 else 0

    # Classify each component
    for c in components:
        c_oi = c.flops / c.bytes_ if c.bytes_ > 0 else 0
        c.bound = "compute" if c_oi > RIDGE_POINT else "memory"
        c.oi = c_oi

    # Per-layer peak compute ceiling
    gemm_components = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    gemm_flops = sum(c.flops for c in components if c.name in gemm_components)
    gemm_bytes = sum(c.bytes_ for c in components if c.name in gemm_components)
    gemm_oi = gemm_flops / gemm_bytes if gemm_bytes > 0 else 0

    # Effective peak for this layer
    peak_tflops = min(P_PEAK, BW * gemm_oi / 1000.0)

    return {
        "B": B, "S": S, "phase": phase,
        "d": d, "h": h, "kv_h": kv_h, "intermediate": intermediate,
        "head_dim": head_dim, "q_len": q_len, "kv_len": kv_len,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "oi": round(oi, 2),
        "gemm_oi": round(gemm_oi, 2),
        "ridge_point": round(RIDGE_POINT, 2),
        "peak_tflops": round(peak_tflops, 2),
        "overall_bound": "compute" if oi > RIDGE_POINT else "memory",
        "gemm_bound": "compute" if gemm_oi > RIDGE_POINT else "memory",
        "components": [
            {
                "name": c.name,
                "flops": c.flops,
                "bytes": c.bytes_,
                "oi": round(c.flops / c.bytes_, 2) if c.bytes_ > 0 else 0,
                "bound": "compute" if (c.flops / c.bytes_ if c.bytes_ > 0 else 0) > RIDGE_POINT else "memory",
                "tflops_at_peak": round(min(P_PEAK, BW * (c.flops / c.bytes_ if c.bytes_ > 0 else 0) / 1000.0), 2),
            }
            for c in components
        ],
    }


# ── Per-layer analysis from timing data ─────────────────────────────────

def analyze_per_layer(timing_json: Path, analytics: dict) -> dict:
    """Combine measured per-layer times with analytical FLOPs/bytes."""
    with open(timing_json) as f:
        data = json.load(f)

    total_layer_time = sum(l["cuda_time_ms"] for l in data["layers"])
    B = data["batch_size"]
    S = data["seq_len"]

    per_layer = []
    for l in data["layers"]:
        layer_idx = l["layer_idx"]
        time_s = l["cuda_time_ms"] / 1000.0
        flops = analytics["total_flops"]
        bytes_ = analytics["total_bytes"]
        achieved_tflops = (flops / time_s) / 1e12 if time_s > 0 else 0
        peak_ratio = achieved_tflops / analytics["peak_tflops"] if analytics["peak_tflops"] > 0 else 0

        per_layer.append({
            "layer_idx": layer_idx,
            "cuda_time_ms": l["cuda_time_ms"],
            "time_pct": round(l["cuda_time_ms"] / total_layer_time * 100, 2) if total_layer_time > 0 else 0,
            "flops": flops,
            "bytes": bytes_,
            "oi": analytics["oi"],
            "gemm_oi": analytics["gemm_oi"],
            "achieved_tflops": round(achieved_tflops, 2),
            "peak_tflops": analytics["peak_tflops"],
            "peak_ratio_pct": round(peak_ratio * 100, 1),
            "bound": analytics["overall_bound"],
            "gemm_bound": analytics["gemm_bound"],
        })

    return {
        "model_name": data.get("model_name", "unknown"),
        "phase": data.get("phase", "unknown"),
        "batch_size": B,
        "seq_len": S,
        "total_layer_time_ms": round(total_layer_time, 3),
        "embed_ms": data.get("embed_ms", 0),
        "norm_ms": data.get("norm_ms", 0),
        "analytics": analytics,
        "per_layer": per_layer,
    }


# ── Print / export ──────────────────────────────────────────────────────

def print_table(result: dict):
    """Print a formatted per-layer table."""
    analytics = result["analytics"]
    print(f"\n{'='*95}")
    print(f"Model: {result['model_name']}  phase={result['phase']}  "
          f"B={result['batch_size']}  S={result['seq_len']}")
    print(f"H100 ridge: {RIDGE_POINT:.0f} FLOP/byte  "
          f"Peak: {P_PEAK:.0f} TFLOP/s  BW: {BW:.0f} GB/s")
    print(f"Layer OI: {analytics['oi']:.1f} FLOP/byte  "
          f"GEMM OI: {analytics['gemm_oi']:.1f} FLOP/byte  "
          f"Overall bound: {analytics['overall_bound'].upper()}  "
          f"GEMM bound: {analytics['gemm_bound'].upper()}")
    print(f"{'='*95}")
    print(f"{'Layer':>6s} {'Time ms':>9s} {'Time %':>7s} "
          f"{'TFLOPS':>8s} {'%Peak':>6s} {'Bound':>8s} {'Bar'}")
    print(f"{'-'*6} {'-'*9} {'-'*7} {'-'*8} {'-'*6} {'-'*8} {'-'*40}")

    for l in result["per_layer"]:
        bar = "#" * int(l["peak_ratio_pct"] / 2.5)
        print(f"{l['layer_idx']:>6d} {l['cuda_time_ms']:>9.3f} "
              f"{l['time_pct']:>6.1f}% {l['achieved_tflops']:>8.1f} "
              f"{l['peak_ratio_pct']:>5.1f}% {l['bound']:>8s} {bar}")

    print(f"\n--- Per-component OI ---")
    print(f"{'Component':<20s} {'FLOPs':>15s} {'Bytes':>15s} {'OI':>10s} {'Bound':>8s}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*10} {'-'*8}")
    for c in analytics["components"]:
        print(f"{c['name']:<20s} {c['flops']:>15.2e} {c['bytes']:>15.2e} "
              f"{c['oi']:>10.1f} {c['bound']:>8s}")


def main():
    parser = argparse.ArgumentParser(description="Parse per-layer timing into roofline analysis")
    parser.add_argument("--timing-json", type=str, required=True,
                        help="Per-layer timing JSON from profile_all_layers.py --mode timing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--print-table", action="store_true", default=True,
                        help="Print formatted table")
    args = parser.parse_args()

    timing_path = Path(args.timing_json)
    with open(timing_path) as f:
        timing_data = json.load(f)

    B = timing_data["batch_size"]
    S = timing_data["seq_len"]
    phase = timing_data["phase"]
    d = timing_data["hidden_size"]
    h = timing_data["num_attention_heads"]
    kv_h = timing_data["num_key_value_heads"]
    intermediate = timing_data["intermediate_size"]

    # Compute analytical model
    analytics = compute_layer_analytics(B, S, phase, d, h, kv_h, intermediate)

    # Combine with timing
    result = analyze_per_layer(timing_path, analytics)

    if args.print_table:
        print_table(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
