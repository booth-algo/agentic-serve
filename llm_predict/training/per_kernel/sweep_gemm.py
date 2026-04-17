"""Dense-model GEMM roofline sweep for per-kernel predictor training.

Covers every distinct (N, K) shape that dense Llama-3.1-8B / Llama-3.1-70B /
Llama-3.3-70B / Qwen2.5-72B hit during prefill (QKV fused, O, FFN gate/up,
FFN down, LM head), swept over a dense M grid from M=1 to M=16384. Intended
to close the shape-extrapolation gap that leaves held-out 70B-class aggregate
MAPE at ~65% (see reports/training_report.md 2026-04-17).

Replaces `/root/per-kernel-rebuild/sweep_gemm.py`. Differences:
- 21 M values (vs 14) — adds non-power-of-2 intermediates + M=16384 for
  long-prefill coverage.
- 17 (N, K) templates covering all dense models we care about (vs 7 in v1).
- Writes a `shape_manifest.json` alongside the ncu-rep so the labeler can
  assign exact (M, N, K) to each kernel row (no back-annotation).

Run with:
    ncu --set default --nvtx --target-processes all \\
        -o /tmp/ncu_gemm_sweep --force-overwrite \\
        python sweep_gemm.py --out-manifest /tmp/ncu_gemm_sweep.manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.cuda import nvtx


# Dense M grid — 21 values. Power-of-2 + fills (384, 768, 1536, 3072, 6144,
# 12288) so the predictor sees non-aligned shapes during training.
M_VALUES = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 12288, 16384,
]

# Per-model canonical shapes (N, K, tag). Deduped across models.
SHAPE_TEMPLATES: list[tuple[int, int, str]] = [
    # Llama-3.1-8B (d=4096, ffn=14336, vocab=128256)
    (4096,  4096,   "llama8b_q_or_o"),
    (6144,  4096,   "llama8b_qkv_fused"),    # (32+16)*128 = 6144
    (1024,  4096,   "llama8b_kv"),           # 8*128
    (14336, 4096,   "llama8b_ffn_gate_up"),
    (4096,  14336,  "llama8b_ffn_down"),
    (128256, 4096,  "llama_lm_head_d4k"),

    # Llama-70B / Llama-3.3-70B (d=8192, ffn=28672, vocab=128256)
    (8192,  8192,   "llama70b_q_or_o"),
    (10240, 8192,   "llama70b_qkv_fused"),   # (64+16)*128 = 10240
    (1024,  8192,   "llama70b_kv"),
    (28672, 8192,   "llama70b_ffn_gate_up"),
    (8192,  28672,  "llama70b_ffn_down"),
    (128256, 8192,  "llama_lm_head_d8k"),

    # Qwen-72B extras (d=8192, ffn=29568, vocab=152064)
    # shared with Llama-70B for q_or_o, qkv_fused, kv
    (29568, 8192,   "qwen72b_ffn_gate_up"),
    (8192,  29568,  "qwen72b_ffn_down"),
    (152064, 8192,  "qwen72b_lm_head"),

    # Small-shape fillers — close gaps between model shapes for XGBoost
    (2048,  2048,   "filler_sq2k"),
    (5120,  5120,   "filler_sq5k"),
]

REPS = 3


def run_sweep(dtype: torch.dtype, manifest_out: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run on a GPU host.")
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    total = len(M_VALUES) * len(SHAPE_TEMPLATES) * REPS
    print(f"[*] GPU: {name} sm_{cc[0]}{cc[1]}")
    print(f"[*] {len(M_VALUES)} M × {len(SHAPE_TEMPLATES)} templates × {REPS} reps "
          f"= {total} matmuls, dtype={dtype}")

    manifest: list[dict] = []
    idx = 0
    for M in M_VALUES:
        for (N, K, tag) in SHAPE_TEMPLATES:
            shape_tag = f"gemm_M{M}_N{N}_K{K}_{tag}"
            nvtx.range_push(shape_tag)
            a = torch.randn(M, K, dtype=dtype, device="cuda")
            b = torch.randn(K, N, dtype=dtype, device="cuda")
            torch.cuda.synchronize()
            for _ in range(REPS):
                c = a @ b
                manifest.append({
                    "idx": idx, "M": M, "N": N, "K": K,
                    "dtype": str(dtype).replace("torch.", ""),
                    "tag": tag, "nvtx_range": shape_tag,
                })
                idx += 1
            torch.cuda.synchronize()
            nvtx.range_pop()
            del a, b, c
    torch.cuda.empty_cache()

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump({
            "gpu_name": name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "dtype": str(dtype),
            "n_matmuls": len(manifest),
            "m_values": M_VALUES,
            "shape_templates": [
                {"N": N, "K": K, "tag": t} for (N, K, t) in SHAPE_TEMPLATES
            ],
            "matmuls": manifest,
        }, f, indent=2)
    print(f"[+] manifest -> {manifest_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    run_sweep(dtype_map[args.dtype], Path(args.out_manifest))


if __name__ == "__main__":
    main()
