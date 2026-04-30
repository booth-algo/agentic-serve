"""Calibrate elementwise kernel parameters from ncu measurements.

Fits floor_us and effective_bw per kernel type per GPU from measured
(tensor_bytes, latency_us) pairs using least-squares on:
    latency_us = floor_us + tensor_bytes / (eff_bw_gb_s * 1e3)
"""

import csv
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "elementwise"

KERNEL_TYPES = ["rmsnorm", "silu_mul", "rotary_emb", "residual_add"]


def calibrate_kernel(measurements: list[tuple[float, float]]) -> dict:
    if len(measurements) < 2:
        if measurements:
            bm, lat = measurements[0]
            return {"floor_us": 0.0, "eff_bw_gb_s": bm / (lat * 1e3) if lat > 0 else 1000.0}
        return {"floor_us": 3.0, "eff_bw_gb_s": 1000.0}

    bytes_arr = np.array([m[0] for m in measurements])
    lat_arr = np.array([m[1] for m in measurements])

    A = np.column_stack([np.ones_like(bytes_arr), bytes_arr])
    result = np.linalg.lstsq(A, lat_arr, rcond=None)
    floor_us = max(float(result[0][0]), 0.0)
    slope = float(result[0][1])
    eff_bw_gb_s = 1.0 / (slope * 1e3) if slope > 0 else 2000.0

    return {"floor_us": round(floor_us, 3), "eff_bw_gb_s": round(eff_bw_gb_s, 1)}


def calibrate_gpu(gpu: str) -> dict:
    result = {}
    for kt in KERNEL_TYPES:
        path = DATA_DIR / f"{gpu}_{kt}.csv"
        if not path.exists():
            continue
        measurements = []
        with open(path) as f:
            for row in csv.DictReader(f):
                measurements.append((float(row["bytes_moved"]), float(row["latency_us"])))
        result[kt] = calibrate_kernel(measurements)

    out_path = DATA_DIR / f"{gpu}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else "A100"
    result = calibrate_gpu(gpu)
    print(f"Calibrated elementwise for {gpu}:")
    for kt, params in result.items():
        print(f"  {kt}: {params}")
