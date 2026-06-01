#!/usr/bin/env python3
"""Inject a FIRST-CUT tp2 (2xH100 tensor-parallel) prediction config into the dashboard
``simulator-predictions.json`` under the ``H100x2`` GPU key.

INFRA-FIRST / PHYSICS-NEXT (deliberate, labelled): the prediction reuses the tp1-calibrated
kernels (decode/prefill grids) with only the KV pool resized for tp2 (config-derived: weights
split across 2 ranks frees HBM, and the KV per token is split across ranks, so the pooled token
capacity is ~2.35x tp1). The decode/prefill STEP TIMES are therefore still tp1-magnitude (tp2 is
~1.6x faster in reality) — so expect the tp2 MAPEs to be poor until tp2 kernels are profiled.
The GROUND TRUTH (ttft_meas / tpot_meas / e2el_meas) is the real h100_..._tp2_vllm benchmark, so
the dashboard shows measured-vs-(first-cut)-prediction honestly. Replace the kernels (profile tp2)
to upgrade the prediction without touching the dashboard.

Usage:
    python3 -m profiling.process.emitters.build_tp2_simulator_rows
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot  # noqa: E402
from simulator.ttft_queue_sim import predict_cell_ttft_qsim  # noqa: E402

DASHBOARD_JSON = Path("inference-benchmark/dashboard/public/simulator-predictions.json")
TP2_BENCH_ROOT = Path(
    "/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp2_vllm"
)
GPU_KEY = "H100x2"
MODEL = "Llama-3.1-8B"
PROFILES = [
    "chat-multiturn-synth",
    "osworld-multiturn-synth",
    "swebench-multiturn-synth",
    "terminalbench-multiturn-synth",
]
CONCURRENCIES = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]

# Config-derived tp2 KV pool (FIRST CUT). tp1 = 27250 blocks ~= 54.5 GiB KV (72 GiB - 16 GiB
# bf16 weights - overhead, KV = 128 KiB/token). tp2: each rank holds half the weights (8 GiB) and
# half the KV heads (64 KiB/token), so per-GPU KV budget ~= 72 - 8 - 1.5 ~= 62.5 GiB -> 62.5 GiB /
# 64 KiB ~= 1.02M tokens of pooled capacity -> ~64000 blocks (~2.35x tp1). Measured tp2 engine
# trace would refine this; it is the dominant TTFT lever and the one thing we resize for tp2.
TP2_AVAILABLE_KV_BLOCKS = 64_000


def _ape(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None or meas <= 0 or pred <= 0:
        return None
    return abs(pred - meas) / meas * 100.0


def _cell_mape(turns: list[dict[str, Any]], pred_key: str, meas_key: str) -> float | None:
    apes = [_ape(t.get(pred_key), t.get(meas_key)) for t in turns]
    apes = [a for a in apes if a is not None]
    return round(st.mean(apes), 4) if apes else None


def build_turns(bench_file: Path) -> list[dict[str, Any]]:
    """Per-turn median ground truth (cached/new/output + ttft/tpot/e2el) from the tp2 benchmark."""
    data = json.loads(bench_file.read_text())
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in data.get("per_request") or []:
        if not r.get("success"):
            continue
        by_turn[int(r.get("turn_index") or 0)].append(r)
    turns: list[dict[str, Any]] = []
    for ti in sorted(by_turn):
        reqs = by_turn[ti]

        def med(key: str) -> float:
            vals = [float(r[key]) for r in reqs if isinstance(r.get(key), (int, float))]
            return st.median(vals) if vals else 0.0

        turns.append({
            "turn_index": ti,
            "successful": len(reqs),
            "scheduled_requests": len(reqs),
            "total_context_tokens": med("total_context_tokens"),
            "cached_context_tokens": med("cached_context_tokens"),
            "new_prefill_tokens": med("new_prefill_tokens"),
            "cache_hit_rate": med("cache_hit_rate"),
            "output_tokens": max(1.0, med("output_tokens")),
            "ttft_meas": round(med("ttft_ms"), 4),
            "tpot_meas": round(med("tpot_ms"), 4),
            "e2el_meas": round(med("e2el_ms"), 4),
        })
    return turns


def build_row(profile: str, conc: int, params: RooflineParams) -> dict[str, Any] | None:
    bench = TP2_BENCH_ROOT / f"{profile}_conc{conc}.json"
    if not bench.exists():
        return None
    turns = build_turns(bench)
    if not turns:
        return None

    # FIRST-CUT predictions: kernel TPOT + queue-sim TTFT with the tp2 KV pool (tp1 kernels).
    kin = [KernelTurnInput(t["cached_context_tokens"], t["new_prefill_tokens"],
                           t["output_tokens"], t["scheduled_requests"]) for t in turns]
    tpot_pred = predict_cell_tpot(kin, params)
    ttft_pred = predict_cell_ttft_qsim(turns, profile, float(conc), params)
    for t, tp, tf in zip(turns, tpot_pred, ttft_pred):
        out = float(t["output_tokens"])
        t["tpot_pred"] = round(float(tp), 4)
        t["ttft_pred"] = round(float(tf), 4)
        t["e2el_pred"] = round(float(tf) + out * float(tp), 4)
        for m in ("tpot", "ttft", "e2el"):
            pred, meas = t[f"{m}_pred"], t[f"{m}_meas"]
            t[f"{m}_err"] = round(_ape(pred, meas), 4) if _ape(pred, meas) is not None else None
            t[f"{m}_signed_err_ms"] = round(pred - meas, 4) if meas else None
            t[f"{m}_abs_err_ms"] = round(abs(pred - meas), 4) if meas else None

    ctxs = [t["total_context_tokens"] for t in turns]
    outs = [t["output_tokens"] for t in turns]
    return {
        "model": MODEL,
        "backend": "kernel-tp1-first-cut-tp2-kv",  # labelled: tp1 kernels + tp2 KV pool
        "profile": profile,
        "data_scope": "synthetic_distributional",
        "mode": "multi-turn",
        "concurrency": conc,
        "isl": round(st.median(ctxs)) if ctxs else 0,
        "osl": round(st.median(outs)) if outs else 1,
        "calibration_status": "tp2_first_cut_tp1_kernels_config_kv",
        "tensor_parallel_size": 2,
        "available_kv_blocks": params.available_kv_blocks,
        "predicted_turn_count": len(turns),
        "multiturn_prediction_mode": "tp2_first_cut",
        "multiturn_turn_predictions": turns,
        "tpot_pred": round(st.mean([t["tpot_pred"] for t in turns]), 4),
        "tpot_meas": round(st.mean([t["tpot_meas"] for t in turns]), 4),
        "tpot_err": _cell_mape(turns, "tpot_pred", "tpot_meas"),
        "ttft_meas": round(st.mean([t["ttft_meas"] for t in turns]), 4),
        "ttft_pred": round(st.mean([t["ttft_pred"] for t in turns]), 4),
        "ttft_err": _cell_mape(turns, "ttft_pred", "ttft_meas"),
        "e2el_meas": round(st.mean([t["e2el_meas"] for t in turns]), 4),
        "e2el_pred": round(st.mean([t["e2el_pred"] for t in turns]), 4),
        "e2el_err": _cell_mape(turns, "e2el_pred", "e2el_meas"),
    }


def main() -> None:
    if not TP2_BENCH_ROOT.exists():
        raise SystemExit(f"tp2 bench root not found: {TP2_BENCH_ROOT}")
    params = RooflineParams(available_kv_blocks=TP2_AVAILABLE_KV_BLOCKS)
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        for conc in CONCURRENCIES:
            row = build_row(profile, conc, params)
            if row:
                rows.append(row)
                print(f"  {GPU_KEY} {profile} c{conc}: ttft_err={row['ttft_err']} "
                      f"tpot_err={row['tpot_err']} e2el_err={row['e2el_err']}")
    if not rows:
        raise SystemExit("no tp2 rows built")
    payload = json.loads(DASHBOARD_JSON.read_text())
    payload[GPU_KEY] = rows
    DASHBOARD_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(rows)} {GPU_KEY} rows -> {DASHBOARD_JSON} (kv_blocks={params.available_kv_blocks})")


if __name__ == "__main__":
    main()
