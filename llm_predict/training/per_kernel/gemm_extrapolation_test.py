"""GEMM shape extrapolation test.

Generates random (M, N, K) shapes in the model-relevant range,
profiles them with ncu on the target GPU, then compares predicted
vs measured kernel time to prove the XGBoost predictor generalizes
beyond its training shapes.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def generate_shapes(n: int = 50, seed: int = 42) -> list[tuple[int, int, int]]:
    """Generate random GEMM shapes in the model-relevant range."""
    rng = random.Random(seed)
    shapes = []
    # Model-relevant ranges:
    # M: 1-512 (seq lengths for prefill)
    # N: 1024-32768 (hidden/ffn sizes)
    # K: 1024-16384 (hidden sizes)
    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
    n_values = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
                10240, 12288, 14336, 16384, 20480, 28672, 32768]
    k_values = [1024, 2048, 2880, 3584, 4096, 5120, 7168, 8192,
                10240, 14336, 16384]
    for _ in range(n):
        m = rng.choice(m_values)
        n_val = rng.choice(n_values)
        k = rng.choice(k_values)
        shapes.append((m, n_val, k))
    return shapes


def profile_shapes(shapes: list[tuple[int, int, int]], dtype=torch.bfloat16,
                   warmup: int = 5, repeat: int = 10) -> list[dict]:
    """Profile each shape using CUDA events (no ncu needed)."""
    results = []
    device = "cuda"
    for i, (m, n, k) in enumerate(shapes):
        linear = nn.Linear(k, n, bias=False, dtype=dtype, device=device)
        x = torch.randn(m, k, dtype=dtype, device=device)
        # Warmup
        for _ in range(warmup):
            _ = linear(x)
        torch.cuda.synchronize()
        # Measure
        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            _ = linear(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[len(times) // 2]
        results.append({
            "M": m, "N": n, "K": k,
            "measured_ms": median_ms,
        })
        if (i + 1) % 10 == 0:
            print("  profiled %d/%d shapes" % (i + 1, len(shapes)))
        del linear, x
    torch.cuda.empty_cache()
    return results


def predict_shapes(shapes: list[tuple[int, int, int]], gpu: str) -> list[float]:
    """Predict GEMM latency using the per-kernel XGBoost predictor."""
    from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor
    pred = PerKernelPredictor(gpu=gpu)
    pred.load()
    predictions = []
    for m, n, k in shapes:
        t = pred.predict_gemm(M=m, N=n, K=k)
        predictions.append(max(0.0, t))
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--n-shapes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print("[*] Generating %d random GEMM shapes..." % args.n_shapes)
    shapes = generate_shapes(args.n_shapes, args.seed)

    print("[*] Profiling on GPU (CUDA events, nn.Linear)...")
    measured = profile_shapes(shapes)

    print("[*] Predicting with per-kernel XGBoost...")
    predicted = predict_shapes(shapes, args.gpu)

    # Compute errors
    print()
    print("%5s %6s %6s %10s %10s %8s" % ("M", "N", "K", "pred_ms", "meas_ms", "err"))
    print("-" * 50)
    errors = []
    for i, (m, n, k) in enumerate(shapes):
        pred_ms = predicted[i]
        meas_ms = measured[i]["measured_ms"]
        if meas_ms > 0.001:
            err = abs(pred_ms - meas_ms) / meas_ms * 100
        else:
            err = 0
        errors.append(err)
        print("%5d %6d %6d %10.4f %10.4f %7.1f%%" % (m, n, k, pred_ms, meas_ms, err))

    print()
    print("MAPE: %.1f%%" % np.mean(errors))
    print("Median error: %.1f%%" % np.median(errors))
    p90 = np.percentile(errors, 90)
    print("P90 error: %.1f%%" % p90)
    within_20 = sum(1 for e in errors if e < 20) / len(errors) * 100
    print("Within 20%%: %.0f%%" % within_20)

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["M", "N", "K", "pred_ms", "meas_ms", "err_pct"])
            w.writeheader()
            for i, (m, n, k) in enumerate(shapes):
                w.writerow({"M": m, "N": n, "K": k,
                           "pred_ms": "%.4f" % predicted[i],
                           "meas_ms": "%.4f" % measured[i]["measured_ms"],
                           "err_pct": "%.1f" % errors[i]})
        print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
