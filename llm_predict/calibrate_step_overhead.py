"""Calibrate step_overhead_{base,per_req}_us from benchmark data.

For each row with measured median_tpot_ms and known active decode count D:
  measured_step_wall = median_tpot_ms × D   (wall-clock decode step time)
  composer_step = predict_decode_step_us(bs=D, kv_len=ctx_est, tp=tp)
  overhead(D) = measured_step_wall - composer_step

Fit: overhead = base_us/1000 + per_req_us/1000 × D  via linear regression.

Usage:
  python3 -m llm_predict.calibrate_step_overhead
  python3 -m llm_predict.calibrate_step_overhead --write  # update gpu_specs.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add repo root so composer/configs imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from llm_predict.composer import Composer
from llm_predict.configs.model_configs import MODEL_CONFIGS, get_model

BENCH_DATA = _REPO_ROOT / "inference-benchmark" / "dashboard" / "public" / "data.json"
GPU_SPECS_PATH = _REPO_ROOT / "llm_predict" / "configs" / "gpu_specs.py"
_MAX_SAMPLE_PER_GPU = 2000  # avoid 62K composer calls
_HW_MAP = {
    "A100-40GB": "A100",
    "H100": "H100",
    "3090": "RTX3090",
    "2080Ti": "RTX2080Ti",
}
_MOE_MODELS = {"gpt-oss-20b", "gpt-oss-120b", "Mixtral-8x7B"}


def _serving_gpu_key(raw_hw: str) -> str:
    match = re.fullmatch(r"(.+?)x(\d+)", raw_hw)
    if match:
        base, tp = match.groups()
        return f"{_HW_MAP.get(base, base)}x{tp}"
    return _HW_MAP.get(raw_hw, raw_hw)


def _predictor_gpu_key(serving_gpu: str) -> str:
    match = re.fullmatch(r"(.+?)x\d+", serving_gpu)
    return match.group(1) if match else serving_gpu


def _resolve_model(raw: str) -> str | None:
    if not raw:
        return None
    for key in MODEL_CONFIGS:
        if key.lower() in raw.lower():
            return key
    return None


def _ctx_isl_osl(entry: dict, per_turn: list[dict] | None) -> tuple[int, int, int]:
    """Return estimated (context, isl, osl) for the data point."""
    if per_turn:
        # per-turn: ctx is median per-turn input, osl is median output
        ctx = 0
        osl_avg = 0
        n = 0
        for t in per_turn:
            if int(t.get("successful", 0)) > 0:
                c = t.get("median_input_tokens", 0) or t.get("avg_input_tokens", 0) or 0
                o = t.get("median_output_tokens", 0) or t.get("avg_output_tokens", 0) or 0
                if c > 0:
                    ctx = max(ctx, int(c))
                    osl_avg += o
                    n += 1
        osl = int(osl_avg / n) if n > 0 else 1
        isl = ctx  # approximate
    else:
        summary = entry.get("summary", {})
        succ = max(1, summary.get("successful_requests", 1))
        isl = int(summary.get("total_input_tokens", 0) / succ)
        osl = int(summary.get("total_output_tokens", 0) / succ)
        ctx = isl
    return max(1, ctx), max(1, isl), max(1, osl)


def collect_data_points(data: list[dict]) -> list[dict[str, Any]]:
    """Extract (GPU, D, tpot_meas, ctx, osl, tp) tuples from data.json."""
    points = []
    for entry in data:
        cfg = entry.get("config", {})
        raw_model = cfg.get("model", "")
        model_key = _resolve_model(raw_model)
        if model_key is None or model_key in _MOE_MODELS:
            continue

        hardware = entry.get("hardware", "")
        serving_gpu = _serving_gpu_key(hardware)
        predictor_gpu = _predictor_gpu_key(serving_gpu)
        tp_match = re.fullmatch(r".+?x(\d+)", hardware)
        tp = int(tp_match.group(1)) if tp_match else 1

        backend = cfg.get("backend", "")
        per_turn = entry.get("perTurn")

        if per_turn:
            for t in per_turn:
                D = int(t.get("successful", 0))
                tpot = t.get("median_tpot_ms")
                if D <= 0 or not tpot or tpot <= 0:
                    continue
                ctx = int(t.get("median_input_tokens", 0) or t.get("avg_input_tokens", 0) or 0)
                osl = int(t.get("median_output_tokens", 0) or t.get("avg_output_tokens", 0) or 0)
                if ctx <= 0:
                    continue
                points.append({
                    "gpu": predictor_gpu,
                    "serving_gpu": serving_gpu,
                    "D": D,
                    "tpot_meas": float(tpot),
                    "ctx": ctx,
                    "osl": max(1, osl),
                    "tp": tp,
                    "model": model_key,
                    "backend": backend,
                    "source": "per_turn",
                })
        else:
            D = int(cfg.get("concurrency", 1))
            summary = entry.get("summary", {})
            tpot = summary.get("median_tpot_ms")
            if D <= 0 or not tpot or tpot <= 0:
                continue
            succ = max(1, summary.get("successful_requests", 1))
            isl = int(summary.get("total_input_tokens", 0) / succ)
            osl = int(summary.get("total_output_tokens", 0) / succ)
            ctx = isl
            if ctx <= 0:
                continue
            points.append({
                "gpu": predictor_gpu,
                "serving_gpu": serving_gpu,
                "D": D,
                "tpot_meas": float(tpot),
                "ctx": ctx,
                "osl": max(1, osl),
                "tp": tp,
                "model": model_key,
                "backend": backend,
                "source": "single_turn",
            })
    return points


def calibrate(points: list[dict[str, Any]],
              verbose: bool = True) -> dict[str, dict[str, float]]:
    """Fit overhead = base_us + per_req_us × D per GPU."""

    # Sample to avoid excessive composer calls
    import random
    random.seed(42)
    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        by_gpu[p["gpu"]].append(p)
    sampled: list[dict] = []
    for gpu, gpts in by_gpu.items():
        if len(gpts) > _MAX_SAMPLE_PER_GPU:
            sampled.extend(random.sample(gpts, _MAX_SAMPLE_PER_GPU))
        else:
            sampled.extend(gpts)
    if verbose:
        print(f"Sampled {len(sampled)} points (max {_MAX_SAMPLE_PER_GPU} per GPU)", flush=True)

    composers: dict[str, Composer] = {}

    # Group by (gpu, tp, model)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in sampled:
        key = (p["gpu"], p["tp"])
        groups[key].append(p)

    results: dict[str, dict[str, float]] = {}

    for (gpu, tp), g_points in sorted(groups.items()):
        if len(g_points) < 5:
            continue

        # Compute overhead per point
        X: list[float] = []  # D
        Y: list[float] = []  # overhead (us)
        composer = composers.setdefault((gpu, tp), Composer(gpu))

        for i, p in enumerate(g_points):
            if i % 500 == 0 and verbose:
                print(f"  {gpu} tp={tp}: {i}/{len(g_points)}...", flush=True)

            cfg = get_model(p["model"])
            try:
                pred_step_us = composer.predict_decode_step_us(
                    cfg, kv_len=p["ctx"], bs=p["D"],
                    tensor_parallel_size=tp,
                )
            except Exception:
                continue

            measured_step_us = p["tpot_meas"] * p["D"] * 1000.0
            overhead_us = measured_step_us - pred_step_us

            max_overhead = 100.0 * p["D"] * 1000  # 100ms per request
            if 0 < overhead_us < max_overhead:
                X.append(float(p["D"]))
                Y.append(overhead_us)

        if len(X) < 5:
            continue

        X_arr = np.array(X, dtype=np.float64)
        Y_arr = np.array(Y, dtype=np.float64)

        # Weight by 1/sqrt(D) to reduce influence of high-C (VRAM-constrained) rows
        weights = 1.0 / np.sqrt(np.maximum(X_arr, 1.0))

        A = np.column_stack([np.ones_like(X_arr), X_arr])
        W = np.diag(weights)
        try:
            coeffs = np.linalg.lstsq(W @ A, W @ Y_arr, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        base_us = max(0.0, coeffs[0])
        per_req_us = max(0.0, coeffs[1])

        pred = A @ coeffs
        ss_res = np.sum((Y_arr - pred) ** 2)
        ss_tot = np.sum((Y_arr - np.mean(Y_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        key = f"{gpu}_tp{tp}"
        results[key] = {
            "base_us": round(base_us, 1),
            "per_req_us": round(per_req_us, 1),
            "n_points": len(X),
            "r2": round(float(r2), 3),
            "D_range": f"{int(min(X))}-{int(max(X))}",
            "mean_overhead_ms": round(float(np.mean(Y)) / 1000.0, 2),
        }

        if verbose:
            print(f"  → base={base_us:.0f}us  per_req={per_req_us:.0f}us  "
                  f"n={len(X)}  r²={r2:.3f}  D∈[{min(X):.0f},{max(X):.0f}]  "
                  f"mean_overhead={np.mean(Y)/1000:.1f}ms", flush=True)

    return results


def write_gpu_specs(results: dict[str, dict[str, float]],
                    dry_run: bool = True) -> None:
    """Update gpu_specs.py with calibrated values."""
    if dry_run:
        print("\nDry run — would update gpu_specs.py with:")
        for key, vals in sorted(results.items()):
            print(f"  {key}: base={vals['base_us']:.0f}us per_req={vals['per_req_us']:.0f}us")
        return

    content = GPU_SPECS_PATH.read_text()
    for key, vals in results.items():
        gpu = key.rsplit("_tp", 1)[0]
        base = f"{vals['base_us']:.0f}"
        per_req = f"{vals['per_req_us']:.0f}"
        # Find and replace the overhead lines for this GPU
        # Pattern: step_overhead_base_us=0.0,  # ... or similar
        base_pat = re.compile(
            rf'("{gpu}".*?step_overhead_base_us=)[\d.]+',
            re.DOTALL,
        )
        per_req_pat = re.compile(
            rf'("{gpu}".*?step_overhead_per_req_us=)[\d.]+',
            re.DOTALL,
        )
        # Simpler: do line-by-line replacement
        lines = content.split("\n")
        in_gpu_block = False
        out_lines = []
        for line in lines:
            if f'"{gpu}"' in line and "GpuSpec(" in line:
                in_gpu_block = True
            elif in_gpu_block and ")" in line and not "(" in line and not "}" in line:
                in_gpu_block = False

            if in_gpu_block and "step_overhead_base_us=" in line:
                line = re.sub(
                    r"step_overhead_base_us=[\d.]+",
                    f"step_overhead_base_us={base}.0",
                    line,
                )
            if in_gpu_block and "step_overhead_per_req_us=" in line:
                line = re.sub(
                    r"step_overhead_per_req_us=[\d.]+",
                    f"step_overhead_per_req_us={per_req}.0",
                    line,
                )
            out_lines.append(line)
        content = "\n".join(out_lines)

    GPU_SPECS_PATH.write_text(content)
    print(f"Updated {GPU_SPECS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate step_overhead_{base,per_req}_us from benchmark data."
    )
    parser.add_argument("--write", action="store_true",
                        help="Write calibrated values to gpu_specs.py")
    parser.add_argument("--data", type=Path, default=BENCH_DATA,
                        help="Path to data.json")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        sys.exit(1)

    with open(args.data) as f:
        data = json.load(f)

    points = collect_data_points(data)
    print(f"Collected {len(points)} data points from {args.data}")

    results = calibrate(points, verbose=True)

    if results:
        write_gpu_specs(results, dry_run=not args.write)
    else:
        print("\nNo calibration results produced — check data coverage.")


if __name__ == "__main__":
    main()
