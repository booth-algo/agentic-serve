#!/usr/bin/env python3
"""Inject a FIRST-CUT tp2 (2xH100 tensor-parallel) prediction config into the dashboard
``simulator-predictions.json`` under the ``H100x2`` GPU key.

PHYSICS (2026-06-01, profiled on 2xH100 GPU 4,5): the prediction now uses the MEASURED tp2
**decode grid** (decode_steps.py --tensor-parallel-size 2, ~0.70x tp1) and the MEASURED tp2 **KV
pool** (62416 blocks, "GPU KV cache size: 998,656 tokens") — so TPOT and E2EL (decode-dominated)
use tp2 timings, and the eviction/preemption cliff sits at the real tp2 pool. STILL tp1: the
serving **prefill** law (FLOOR/NEW/HOST in ttft_queue_sim, anchored on tp1 c1) — tp2 prefill is
~0.4-0.8x tp1 (compute splits), so per-turn TTFT prefill stays over-predicted until the prefill
law is re-anchored on tp2 c1 (offline follow-up). GROUND TRUTH (ttft/tpot/e2el_meas) is the real
h100_..._tp2_vllm benchmark. tp1 modules are untouched — the decode-grid swap is local here.

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

import simulator.kernel_step_cost as kernel_step_cost  # noqa: E402
from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.kernel_step_cost import load_grid  # noqa: E402
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

# MEASURED tp2 KV pool (2026-06-01, GPU 4,5 engine trace: "GPU KV cache size: 998,656 tokens"
# -> 998656/16 = 62416 blocks; reconfirmed 999,184 on a second launch). ~2.29x tp1's 27250. The
# config-derived first cut (64000) was within 2.5%. This is the dominant TTFT/preemption lever.
TP2_AVAILABLE_KV_BLOCKS = 62_416
# MEASURED tp2 decode-step grid (decode_steps.py --tensor-parallel-size 2 on 2xH100, ~0.70x tp1
# across all shapes). Loaded into the decode kernel so TPOT/E2EL use tp2 timings, not tp1.
TP2_DECODE_GRID = Path("profiling/results/decode_profile_H100x2_2026-06-01.csv")


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
        "backend": "kernel-tp2-decode-measured-kv",  # tp2 decode grid + measured KV; prefill still tp1
        "profile": profile,
        "data_scope": "synthetic_distributional",
        "mode": "multi-turn",
        "concurrency": conc,
        "isl": round(st.median(ctxs)) if ctxs else 0,
        "osl": round(st.median(outs)) if outs else 1,
        "calibration_status": "tp2_decode_grid_measured_kv_tp1_prefill",
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
    # Swap the decode kernel grid to the MEASURED tp2 grid (decode_step_ms reads the module-global
    # _default_grid()). This makes predict_cell_tpot (TPOT) and the qsim's decode step use tp2
    # timings. tp1 modules are untouched — the swap is local to this generator process.
    if TP2_DECODE_GRID.exists():
        _tp2_grid = load_grid(TP2_DECODE_GRID)
        kernel_step_cost._default_grid = lambda: _tp2_grid
        print(f"using MEASURED tp2 decode grid: {TP2_DECODE_GRID.name} ({len(_tp2_grid.cells)} cells)")
    else:
        print(f"WARNING: tp2 decode grid {TP2_DECODE_GRID} missing -> tp1 decode kernel (worse TPOT)")
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
