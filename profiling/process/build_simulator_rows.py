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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulator.kernel_step_cost as kernel_step_cost  # noqa: E402
import simulator.kernel_tpot as kernel_tpot  # noqa: E402
from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.cohort_scale import cohort_scale_mean  # noqa: E402
from simulator.kernel_step_cost import load_grid  # noqa: E402
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot  # noqa: E402
from simulator.ttft_queue_sim import (  # noqa: E402
    QSimSchedConfig, _prefill_floor_for, predict_cell_ttft_qsim,
)
from configs.loader import Deployment, all_deployments  # noqa: E402

DASHBOARD_JSON = Path("inference-benchmark/dashboard/public/simulator-predictions.json")
BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
PROFILES = [
    "chat-multiturn-synth",
    "osworld-multiturn-synth",
    "swebench-multiturn-synth",
    "terminalbench-multiturn-synth",
]
CONCURRENCIES = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]


# Deployments live in configs/deployments/*.json, composed by configs/loader.py (model + GPU + deployment
# + a per-input `data` manifest). To add a GPU/TP config, add a deployment JSON — no code change here.
# Each Deployment exposes: gpu_key, tp, bench_dir, available_kv_blocks, decode_grid, saturated_ceiling,
# backend, calibration_status, ground_truth, roofline (composed RooflineParams), data (manifest).
CONFIGS = all_deployments()


def _ape(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None or meas <= 0 or pred <= 0:
        return None
    return abs(pred - meas) / meas * 100.0


def _cell_mape(turns: list[dict[str, Any]], pred_key: str, meas_key: str) -> float | None:
    apes = [_ape(t.get(pred_key), t.get(meas_key)) for t in turns]
    apes = [a for a in apes if a is not None]
    return round(st.mean(apes), 4) if apes else None


def _shared_prefix_tokens(reqs: list[dict[str, Any]]) -> float:
    """Median ``request_metadata.shared_prefix_actual_tokens`` over a turn's successful requests.

    The ``prefix_aware_synthetic`` workloads inject a profile-constant cross-session APC prefix
    (swebench/osworld 1024, terminalbench 976, chat 48) that vLLM dedups across sessions but the
    per-session cache estimate records as cached=0. Threading it lets the queue sim credit it once
    instead of re-prefilling it for every concurrent session. 0.0 when absent (non-prefix-aware)."""
    vals = [
        float((r.get("request_metadata") or {}).get("shared_prefix_actual_tokens"))
        for r in reqs
        if isinstance((r.get("request_metadata") or {}).get("shared_prefix_actual_tokens"), (int, float))
    ]
    vals = [v for v in vals if v > 0.0]
    return st.median(vals) if vals else 0.0


def build_turns(bench_file: Path) -> tuple[list[dict[str, Any]], float]:
    """Per-turn median ground truth (cached/new/output + ttft/tpot/e2el) from a benchmark run,
    plus the cell's shared cross-session APC prefix size (from turn-0 request_metadata; 0 if none)."""
    data = json.loads(bench_file.read_text())
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in data.get("per_request") or []:
        if not r.get("success"):
            continue
        by_turn[int(r.get("turn_index") or 0)].append(r)
    shared_prefix_tokens = _shared_prefix_tokens(by_turn[min(by_turn)]) if by_turn else 0.0
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
    return turns, shared_prefix_tokens


def build_row(profile: str, conc: int, params: RooflineParams, cfg: Deployment,
              bench_root: Path) -> dict[str, Any] | None:
    bench = bench_root / f"{profile}_conc{conc}.json"
    if not bench.exists():
        return None
    turns, shared_prefix_tokens = build_turns(bench)
    if not turns:
        return None

    # Headline predictions: kernel-composed TPOT + queue-sim TTFT (the active decode grid + KV pool
    # are set per-config in main()). E2EL composes on the kernel TPOT. ``shared_prefix_tokens`` lets
    # the queue sim dedup the cross-session APC prefix instead of re-prefilling it per session.
    # ``cohort_scale_mean`` (qbar) = trapezoid mean of the cell's MEASURED context-scale quantiles
    # (one value per cell; profiles without a spec -> 1.0): qbar sizes the distribution-integrated
    # overflow mass z = pressure*qbar once the pool is physically full (pressure >= 1) — the round-2
    # chunk-quantized eviction-drain weight (kernel_tpot._overflow_weight; predict_cell_tpot also
    # tracks the per-turn eviction-development state). The quantile pools are profile-level workload
    # spreads (Llama-derived realized artifacts) — resolved with the same Llama-only gpu_key gate as
    # the replay cohort below.
    qbar_gpu_key = cfg.gpu_key if cfg.model == "Llama-3.1-8B" else None
    qbar = cohort_scale_mean(profile, float(conc), qbar_gpu_key)
    kin = [KernelTurnInput(t["cached_context_tokens"], t["new_prefill_tokens"],
                           t["output_tokens"], t["scheduled_requests"],
                           cohort_scale_mean=qbar) for t in turns]
    tpot_pred = predict_cell_tpot(kin, params)
    # Two INDEPENDENT per-config inputs for the TTFT sim:
    #  - cohort_gpu_key: the trajectory-REPLAY cohort. Llama-only (the pools are Llama-derived). Kept
    #    ON for all TP: replay HELPS some tp>=2 configs (A100x2 25.9->25.6) and only under-helps where
    #    the DECODE amplifier over-prices (H100x2) — a per-config replay on/off would be metric
    #    cherry-picking, so the amplifier is fixed instead (the floor below is the first, fit-free part).
    #  - floor_gpu_key: the measured per-config PREFILL FLOOR — correct for ALL Llama configs (tp2/tp4
    #    inherited the wrong tp1 floor of 26 ms; their real floor is lower, e.g. H100x2=14), applied
    #    independently of the cohort via the explicit ``prefill_floor_ms`` (so it survives any future
    #    cohort gating). Monotonic: every config's TTFT <= the pre-floor value, tp1 byte-identical.
    cohort_gpu_key = cfg.gpu_key if cfg.model == "Llama-3.1-8B" else None
    # Per-config vLLM scheduler truth for the queue sim's ADMISSION arithmetic — built ONLY
    # when the deployment manifest pins it (verified GT server metadata + resolved engine
    # defaults; see the manifest's scheduler note + QSimSchedConfig). Unpinned configs pass
    # None -> the sim's module-level H100 constants (byte-identical). The token budget is
    # the SAME per-deployment ``max_num_batched_tokens`` the kernel-TPOT side already
    # prices with (engine-truth parity between the two consumers of one engine config);
    # the chunk cap keeps the sim's established int(max_model_len*0.04) rule with the
    # config's OWN GT-recorded max_model_len (L10-tp1sub20 round 2, 2026-06-11).
    sched = None
    if cfg.max_model_len is not None or cfg.max_num_seqs is not None:
        sched = QSimSchedConfig(
            max_num_batched_tokens=int(params.max_num_batched_tokens),
            long_prefill_token_threshold=(
                int(cfg.max_model_len * 0.04) if cfg.max_model_len is not None else None),
            max_num_seqs=cfg.max_num_seqs,
        )
    ttft_pred = predict_cell_ttft_qsim(
        turns, profile, float(conc), params, shared_prefix_tokens=shared_prefix_tokens,
        gpu_key=cohort_gpu_key, prefill_floor_ms=_prefill_floor_for(cohort_gpu_key),
        sched=sched)
    for t, tp, tf in zip(turns, tpot_pred, ttft_pred):
        out = float(t["output_tokens"])
        t["tpot_pred"] = round(float(tp), 4)
        t["ttft_pred"] = round(float(tf), 4)
        t["e2el_pred"] = round(float(tf) + out * float(tp), 4)
        if not cfg.ground_truth:  # predictions-only: bench_dir gave only the workload structure
            for m in ("tpot", "ttft", "e2el"):
                t[f"{m}_meas"] = None
        for m in ("tpot", "ttft", "e2el"):
            pred, meas = t[f"{m}_pred"], t[f"{m}_meas"]
            t[f"{m}_err"] = round(_ape(pred, meas), 4) if _ape(pred, meas) is not None else None
            t[f"{m}_signed_err_ms"] = round(pred - meas, 4) if meas else None
            t[f"{m}_abs_err_ms"] = round(abs(pred - meas), 4) if meas else None

    ctxs = [t["total_context_tokens"] for t in turns]
    outs = [t["output_tokens"] for t in turns]

    def cell_meas(key: str) -> float | None:
        if not cfg.ground_truth:
            return None
        vals = [t[key] for t in turns if isinstance(t.get(key), (int, float))]
        return round(st.mean(vals), 4) if vals else None

    return {
        "model": cfg.model,
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
        "tpot_meas": cell_meas("tpot_meas"),
        "tpot_err": _cell_mape(turns, "tpot_pred", "tpot_meas"),
        "ttft_meas": cell_meas("ttft_meas"),
        "ttft_pred": round(st.mean([t["ttft_pred"] for t in turns]), 4),
        "ttft_err": _cell_mape(turns, "ttft_pred", "ttft_meas"),
        "e2el_meas": cell_meas("e2el_meas"),
        "e2el_pred": round(st.mean([t["e2el_pred"] for t in turns]), 4),
        "e2el_err": _cell_mape(turns, "e2el_pred", "e2el_meas"),
    }


def main() -> None:
    orig_default_grid = kernel_step_cost._default_grid  # the cached tp1 H100 decode grid
    orig_ceiling = kernel_tpot._active_ceiling_json     # the default (H100) saturated-ITL ceiling
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
            # No measured grid for this GPU/model -> use the full decode ROOFLINE (scales with the
            # config's own weight bytes / bandwidth / KV), NOT the H100 8B grid. First-cut but physical.
            agrid = kernel_step_cost.analytic_grid()
            kernel_step_cost._default_grid = lambda agrid=agrid: agrid
            print(f"{cfg.gpu_key}: analytic decode roofline (no measured grid; launch_floor "
                  f"{kernel_step_cost.default_launch_floor_ms():.2f} ms)")
        # Swap the saturated-ITL ceiling for this config (mirror the grid swap). None -> H100 anchors.
        if cfg.saturated_ceiling is not None and cfg.saturated_ceiling.exists():
            kernel_tpot._active_ceiling_json = cfg.saturated_ceiling
            print(f"{cfg.gpu_key}: saturated ceiling {cfg.saturated_ceiling.name}")
        else:
            kernel_tpot._active_ceiling_json = orig_ceiling
        params = cfg.roofline  # composed by configs.loader from the gpu + model + deployment JSONs
        print(f"{cfg.gpu_key}: roofline flops={params.peak_flops_per_s:.3g} bw={params.peak_bw_bytes_per_s:.3g} "
              f"kv={params.available_kv_blocks} tp={params.tensor_parallel}"
              + ("" if cfg.ground_truth else f"  (PREDICTIONS-ONLY: structure from {cfg.bench_dir})"))
        rows: list[dict[str, Any]] = []
        for profile in PROFILES:
            for conc in CONCURRENCIES:
                row = build_row(profile, conc, params, cfg, bench_root)
                if row:
                    rows.append(row)
        # Multiple deployments can share one gpu_key (e.g. different models on the same GPU/TP/engine,
        # disambiguated by the row `model` field + the dashboard model dropdown) -> accumulate, never
        # overwrite. The gpu_key gate (validate_*) filters to model=Llama-3.1-8B so this stays gate-safe.
        payload.setdefault(cfg.gpu_key, []).extend(rows)
        print(f"  {cfg.gpu_key} += {len(rows)} rows [{cfg.model}]  "
              f"(tpot {_overall(rows,'tpot_err')} / ttft {_overall(rows,'ttft_err')} "
              f"/ e2el {_overall(rows,'e2el_err')} cell-MAPE)")
    kernel_step_cost._default_grid = orig_default_grid
    kernel_tpot._active_ceiling_json = orig_ceiling
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
