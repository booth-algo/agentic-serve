#!/usr/bin/env python3
"""Build the dashboard per-turn TTFT / TPOT / E2EL predictions (``simulator-predictions.json``)
for EVERY (GPU, model, TP) config from ONE config-driven, headline-only generator.

Replaces the old split pipeline (the heavyweight engine-step base producer + the kernel/ttft
augmenters for tp1, plus the bespoke ``build_tp2_simulator_rows.py``). For each config x (profile,
concurrency): per-turn medians from the benchmark ``per_request`` -> headline kernel-composed TPOT
(``predict_cell_tpot``) + queue-sim TTFT (``predict_cell_ttft_qsim``) + E2EL (= ttft + output*tpot)
+ ground truth, with errors. Same predictors -> the headline MAPEs are preserved (TPOT 16.48% /
swebench-plateau 9.20%, TTFT 33.12% / E2EL 19.67%).

Field convention is identical across GPUs: ``tpot_pred`` (kernel comp), ``ttft_pred`` (queue sim),
``e2el_pred``, and ``{tpot,ttft,e2el}_{meas,err,signed_err_ms,abs_err_ms}``. To add a GPU/TP config,
profile its decode grid + KV pool (see profile_data/README.md) and add a ``Config`` row below.

Usage:
    python3 -m profiling.process.build_simulator_rows
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulator.kernel_step_cost as kernel_step_cost  # noqa: E402
from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.kernel_step_cost import load_grid  # noqa: E402
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot  # noqa: E402
from simulator.ttft_queue_sim import predict_cell_ttft_qsim  # noqa: E402

DASHBOARD_JSON = Path("inference-benchmark/dashboard/public/simulator-predictions.json")
BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
MODEL = "Llama-3.1-8B"
PROFILES = [
    "chat-multiturn-synth",
    "osworld-multiturn-synth",
    "swebench-multiturn-synth",
    "terminalbench-multiturn-synth",
]
CONCURRENCIES = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]


@dataclass(frozen=True)
class Config:
    gpu_key: str            # dashboard GPU key (e.g. "H100", "H100x2")
    tp: int                 # tensor-parallel degree
    bench_dir: str          # under BENCH_BASE; the measured ground-truth run
    available_kv_blocks: int
    decode_grid: Path | None  # measured decode grid; None -> default (tp1 H100) grid
    backend: str
    calibration_status: str


# To add a GPU/TP config: profile its decode grid + KV pool, then add a Config row.
CONFIGS = [
    Config("H100", 1, "h100_Llama-3.1-8B_tp1_vllm", 27_250, None,
           "kernel-headline", "kernel_tpot_qsim_ttft_headline"),
    Config("H100x2", 2, "h100_Llama-3.1-8B_tp2_vllm", 62_416,
           Path("profile_data/results/decode_profile_H100x2_2026-06-01.csv"),
           "kernel-tp2-decode-measured-kv", "tp2_decode_grid_measured_kv_tp1_prefill"),
]


def _ape(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None or meas <= 0 or pred <= 0:
        return None
    return abs(pred - meas) / meas * 100.0


def _cell_mape(turns: list[dict[str, Any]], pred_key: str, meas_key: str) -> float | None:
    apes = [_ape(t.get(pred_key), t.get(meas_key)) for t in turns]
    apes = [a for a in apes if a is not None]
    return round(st.mean(apes), 4) if apes else None


def build_turns(bench_file: Path) -> list[dict[str, Any]]:
    """Per-turn median ground truth (cached/new/output + ttft/tpot/e2el) from a benchmark run."""
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


def build_row(profile: str, conc: int, params: RooflineParams, cfg: Config,
              bench_root: Path) -> dict[str, Any] | None:
    bench = bench_root / f"{profile}_conc{conc}.json"
    if not bench.exists():
        return None
    turns = build_turns(bench)
    if not turns:
        return None

    # Headline predictions: kernel-composed TPOT + queue-sim TTFT (the active decode grid + KV pool
    # are set per-config in main()). E2EL composes on the kernel TPOT.
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
        "backend": cfg.backend,
        "profile": profile,
        "data_scope": "synthetic_distributional",
        "mode": "multi-turn",
        "concurrency": conc,
        "isl": round(st.median(ctxs)) if ctxs else 0,
        "osl": round(st.median(outs)) if outs else 1,
        "calibration_status": cfg.calibration_status,
        "tensor_parallel_size": cfg.tp,
        "available_kv_blocks": cfg.available_kv_blocks,
        "predicted_turn_count": len(turns),
        "multiturn_prediction_mode": "kernel_tpot_qsim_ttft_headline",
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
    orig_default_grid = kernel_step_cost._default_grid  # the cached tp1 H100 decode grid
    payload: dict[str, list[dict[str, Any]]] = {}
    for cfg in CONFIGS:
        bench_root = BENCH_BASE / cfg.bench_dir
        if not bench_root.exists():
            print(f"SKIP {cfg.gpu_key}: bench root not found ({bench_root})")
            continue
        # Swap the decode kernel grid for this config (decode_step_ms reads the module-global
        # _default_grid()). tp1 modules are untouched — the swap is local to this generator.
        if cfg.decode_grid is not None and cfg.decode_grid.exists():
            grid = load_grid(cfg.decode_grid)
            kernel_step_cost._default_grid = lambda grid=grid: grid
            print(f"{cfg.gpu_key}: measured decode grid {cfg.decode_grid.name} ({len(grid.cells)} cells)")
        else:
            kernel_step_cost._default_grid = orig_default_grid
            print(f"{cfg.gpu_key}: default decode grid")
        params = RooflineParams(available_kv_blocks=cfg.available_kv_blocks)
        rows: list[dict[str, Any]] = []
        for profile in PROFILES:
            for conc in CONCURRENCIES:
                row = build_row(profile, conc, params, cfg, bench_root)
                if row:
                    rows.append(row)
        payload[cfg.gpu_key] = rows
        print(f"  {cfg.gpu_key}: {len(rows)} rows  "
              f"(tpot {_overall(rows,'tpot_err')} / ttft {_overall(rows,'ttft_err')} "
              f"/ e2el {_overall(rows,'e2el_err')} cell-MAPE)")
    kernel_step_cost._default_grid = orig_default_grid
    if not payload:
        raise SystemExit("no configs produced rows")
    DASHBOARD_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {sum(len(v) for v in payload.values())} rows across {len(payload)} GPU configs "
          f"-> {DASHBOARD_JSON}")


def _overall(rows: list[dict[str, Any]], key: str) -> str:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return f"{st.mean(vals):.1f}%" if vals else "n/a"


if __name__ == "__main__":
    main()
