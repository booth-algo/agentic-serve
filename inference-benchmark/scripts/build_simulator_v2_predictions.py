#!/usr/bin/env python3
"""Build simulator-v2sim-predictions.json for the dashboard "Simulator v2" tab.

Runs the simulator_v2 (kernel-composition) backtest over the measured H100 /
Llama-3.1-8B cells and emits the same schema the existing "Simulator" tab consumes
(`{gpu_key: ServingRow[]}`, each row carrying `multiturn_turn_predictions`). Scope
matches what simulator_v2 supports today: H100, Llama-3.1-8B.

Usage:  python3 inference-benchmark/scripts/build_simulator_v2_predictions.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # make simulator_v2 (at repo root) importable
os.chdir(REPO)  # simulator_v2 resolves kernel artifacts via a CWD-relative path

from simulator_v2.engine.predict import predict  # noqa: E402
from simulator_v2.getters.hardware import load_kernel_composition_hardware  # noqa: E402
from simulator_v2.getters.workload import load_benchmark  # noqa: E402

GPU_KEY = "H100"
GPU_YAML = REPO / "simulator_v2/configs/gpu_configs/h100.yaml"
MODEL_YAML = REPO / "simulator_v2/configs/model_configs/llama3.1-8b.yaml"
BENCH_DIR = Path("/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm")
OUT = REPO / "inference-benchmark/dashboard/public/simulator-v2sim-predictions.json"
# Some benchmark files carry a "<model>_tp<N>_<backend>_" filename prefix that the
# cell-name parser sweeps into the profile; strip it for clean dashboard names.
CONFIG_PREFIX = BENCH_DIR.name.split("_", 1)[1] + "_"  # e.g. "Llama-3.1-8B_tp1_vllm_"

# Scope: match v1's emitter (profiling/process/build_simulator_rows.py) -- the four
# multi-turn profiles only. chat-singleturn was dropped after v1 and is not on the
# dashboard; including it also pulls in its c500 cell. Keeps v1/v2 tabs comparable.
PROFILES = frozenset({
    "chat-multiturn-synth",
    "osworld-multiturn-synth",
    "swebench-multiturn-synth",
    "terminalbench-multiturn-synth",
})


def _clean_profile(raw: str) -> str:
    return raw[len(CONFIG_PREFIX):] if raw.startswith(CONFIG_PREFIX) else raw


def _ape(pred: float, meas: float) -> float | None:
    return round(abs(pred - meas) / meas * 100.0, 2) if meas and meas > 0 else None


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.mean(xs), 4) if xs else 0.0


def _cell_mape(per_turn: list[dict], err_key: str) -> float | None:
    """Mean of the per-turn APEs -- matches v1's build_simulator_rows._cell_mape
    (every turn weighted equally), NOT a ratio of the cell-mean pred/meas."""
    apes = [pt[err_key] for pt in per_turn if isinstance(pt.get(err_key), (int, float))]
    return round(st.mean(apes), 2) if apes else None


def _row(cell, preds) -> dict:
    turns, gt = cell.turns, cell.ground_truth or []
    profile = _clean_profile(cell.profile)
    per_turn = []
    for ti, (t, g, p) in enumerate(zip(turns, gt, preds)):
        total = float(t.isl_tokens)
        per_turn.append({
            "turn_index": ti,
            "successful": int(t.scheduled_requests) or 1,
            "total_context_tokens": total,
            "new_prefill_tokens": float(t.new_prefill_tokens),
            "cached_context_tokens": float(t.cache_hit_tokens),
            "cache_hit_rate": (t.cache_hit_tokens / total) if total > 0 else 0.0,
            "output_tokens": float(t.osl_tokens),
            "scheduled_requests": float(t.scheduled_requests),
            "ttft_meas": g.ttft_ms, "ttft_pred": round(p.ttft_ms, 4), "ttft_err": _ape(p.ttft_ms, g.ttft_ms),
            "tpot_meas": g.tpot_ms, "tpot_pred": round(p.tpot_ms, 4), "tpot_err": _ape(p.tpot_ms, g.tpot_ms),
            "e2el_meas": g.e2el_ms, "e2el_pred": round(p.e2el_ms, 4), "e2el_err": _ape(p.e2el_ms, g.e2el_ms),
        })

    tpot_p, tpot_m = _mean([x["tpot_pred"] for x in per_turn]), _mean([x["tpot_meas"] for x in per_turn])
    ttft_p, ttft_m = _mean([x["ttft_pred"] for x in per_turn]), _mean([x["ttft_meas"] for x in per_turn])
    e2el_p, e2el_m = _mean([x["e2el_pred"] for x in per_turn]), _mean([x["e2el_meas"] for x in per_turn])
    return {
        "model": "Llama-3.1-8B",
        "backend": "h100-tp1-vllm-kernel-v2",
        "profile": profile,
        "data_scope": "synthetic_distributional",
        "mode": "single-turn" if len(turns) <= 1 else "multi-turn",
        "concurrency": cell.concurrency,
        "isl": float(turns[0].isl_tokens) if turns else 0.0,
        "osl": float(turns[0].osl_tokens) if turns else 0.0,
        "tensor_parallel_size": 1,
        "predicted_turn_count": len(turns),
        "multiturn_prediction_mode": "kernel_composition_v2",
        "calibration_status": "simulator_v2_kernel_composition",
        "multiturn_turn_predictions": per_turn,
        "tpot_pred": tpot_p, "tpot_meas": tpot_m, "tpot_err": _cell_mape(per_turn, "tpot_err"),
        "ttft_pred": ttft_p, "ttft_meas": ttft_m, "ttft_err": _cell_mape(per_turn, "ttft_err"),
        "e2el_pred": e2el_p, "e2el_meas": e2el_m, "e2el_err": _cell_mape(per_turn, "e2el_err"),
    }


def main() -> None:
    if not BENCH_DIR.exists():
        raise SystemExit(f"benchmark dir missing: {BENCH_DIR}")
    hw = load_kernel_composition_hardware(GPU_YAML, MODEL_YAML, tp=1)
    files = sorted(f for f in glob.glob(str(BENCH_DIR / "*_conc*.json")) if "_per_turn" not in f)

    rows = []
    for f in files:
        cell = load_benchmark(Path(f))
        if _clean_profile(cell.profile) not in PROFILES:
            continue  # out of v1's scope (e.g. chat-singleturn)
        if not cell.turns or not cell.ground_truth:
            continue
        preds = predict(
            hw, cell.turns, cell.concurrency,
            trajectories=cell.trajectories, shared_prefix_tokens=cell.shared_prefix_tokens,
        )
        rows.append(_row(cell, preds))
        print(f"  {cell.profile} conc{cell.concurrency}: {len(cell.turns)} turns")

    OUT.write_text(json.dumps({GPU_KEY: rows}, indent=2) + "\n")
    print(f"wrote {len(rows)} cells -> {OUT}")


if __name__ == "__main__":
    main()
