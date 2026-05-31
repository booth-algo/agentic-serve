"""Inject measured + predicted TTFT / E2EL into the dashboard
``simulator-predictions.json`` per turn.

The base dashboard JSON carries per-turn ``tpot_meas`` but **no** ``ttft_meas`` /
``e2el_meas`` (0/1043 populated). The measured ground truth lives in the raw
benchmark files at
``<bench_root>/<profile>_conc<C>.json`` → ``per_request[]`` with ``ttft_ms`` /
``e2el_ms`` / ``tpot_ms`` per (session, turn). This augmenter lifts them in,
aggregated **median over successful requests grouped by turn_index** — the same
rule the base JSON uses for ``tpot_meas`` (verified: JSON matches the median, not
the mean).

It also injects forward (workload-only) predictions when available:
  * ``ttft_pred`` / ``e2el_pred``  — **HEADLINE = the forward closed-loop queue sim**
    (``simulator/ttft_queue_sim.py``: barrier round-robin + chunked-prefill budget +
    block-level prefix-cache w/ herd-protected rotation). Beats the static M0 (60.78%
    vs 61.95% TTFT; 29.71% vs 30.01% E2EL), so it is now the primary prediction.
  * ``ttft_pred_static`` / ``e2el_pred_static``  — the static M0 comparison (prefill
    baseline × Little's-law queue amplifier, ``simulator/ttft_predict.py``).
  * ``e2el_*`` compose on the headline kernel TPOT (``tpot_pred_kernel``), no re-fit.

This augmenter never repoints any TPOT field (kernel headline stays byte-identical).

Usage:
    python3 -m profiling.process.emitters.augment_simulator_predictions_with_ttft
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DASHBOARD_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)
DEFAULT_BENCH_ROOT = Path(
    "/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm"
)


def measured_by_turn(
    profile: str, concurrency: int, bench_root: Path
) -> dict[int, dict[str, float]]:
    """Median measured ``ttft_ms`` / ``e2el_ms`` per ``turn_index`` for a cell.

    Reads ``<bench_root>/<profile>_conc<C>.json`` (the non-``_per_turn`` variant
    carries ``per_request``), success-filters, groups by ``turn_index``, and
    returns ``{turn_index: {"ttft_meas": med, "e2el_meas": med}}``. Returns an
    empty dict when the raw file is absent.
    """
    path = bench_root / f"{profile}_conc{concurrency}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    per_request = data.get("per_request") or []
    buckets: dict[int, dict[str, list[float]]] = {}
    for row in per_request:
        if not row.get("success"):
            continue
        t = row.get("turn_index")
        if t is None:
            continue
        ttft = row.get("ttft_ms")
        e2el = row.get("e2el_ms")
        bucket = buckets.setdefault(int(t), {"ttft": [], "e2el": []})
        if isinstance(ttft, (int, float)):
            bucket["ttft"].append(float(ttft))
        if isinstance(e2el, (int, float)):
            bucket["e2el"].append(float(e2el))
    out: dict[int, dict[str, float]] = {}
    for t, b in buckets.items():
        rec: dict[str, float] = {}
        if b["ttft"]:
            rec["ttft_meas"] = round(statistics.median(b["ttft"]), 4)
        if b["e2el"]:
            rec["e2el_meas"] = round(statistics.median(b["e2el"]), 4)
        if rec:
            out[t] = rec
    return out


def _cell_mean(turns: list[dict[str, Any]], key: str) -> float | None:
    vals = [
        float(t[key])
        for t in turns
        if isinstance(t.get(key), (int, float))
    ]
    return round(sum(vals) / len(vals), 6) if vals else None


def augment(dashboard_json: Path, bench_root: Path) -> dict[str, int]:
    """Inject measured ``ttft_meas`` / ``e2el_meas`` (and predictions when the
    forward predictor is importable) into every turn. Additive only."""
    payload = json.loads(dashboard_json.read_text())
    rows = payload.get("H100", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"no H100 rows found in {dashboard_json}; nothing to augment")

    # The forward predictor is optional at this stage (built in a later step);
    # measured extraction must work standalone.
    try:
        from simulator.ttft_predict import PROFILE_DIST, predict_cell_ttft  # noqa: E402

        have_predictor = True
    except Exception:  # pragma: no cover - predictor lands in a later step
        PROFILE_DIST = {}  # type: ignore[assignment]
        predict_cell_ttft = None  # type: ignore[assignment]
        have_predictor = False

    # The forward closed-loop queue sim (ttft_pred_qsim) is additive and optional —
    # it never repoints ttft_pred / tpot_* (M0 + kernel headline stay byte-identical).
    try:
        from simulator.ttft_queue_sim import (  # noqa: E402
            predict_cell_e2el_qsim,
            predict_cell_ttft_qsim,
        )

        have_qsim = True
    except Exception:  # pragma: no cover - sim is the headline addition here
        predict_cell_ttft_qsim = None  # type: ignore[assignment]
        predict_cell_e2el_qsim = None  # type: ignore[assignment]
        have_qsim = False

    counts = {
        "ttft_meas": 0,
        "e2el_meas": 0,
        "ttft_pred": 0,          # HEADLINE = queue sim
        "e2el_pred": 0,
        "ttft_pred_static": 0,   # comparison = static M0
        "e2el_pred_static": 0,
        "ttft_err": 0,
        "e2el_err": 0,
        "ttft_signed_err_ms": 0,
        "e2el_signed_err_ms": 0,
        "ttft_abs_err_ms": 0,
        "e2el_abs_err_ms": 0,
        "ttft_err_static": 0,
        "e2el_err_static": 0,
        "cells": 0,
    }
    missing_raw: list[str] = []
    for row in rows:
        c = row.get("concurrency")
        profile = row.get("profile")
        turns = row.get("multiturn_turn_predictions") or []
        if c is None or profile is None or not turns:
            continue
        counts["cells"] += 1

        # ---- measured (always) ----
        meas = measured_by_turn(str(profile), int(c), bench_root)
        if not meas:
            missing_raw.append(f"{profile}_conc{c}")
        for turn in turns:
            rec = meas.get(int(turn.get("turn_index", -1)))
            if not rec:
                continue
            if "ttft_meas" in rec:
                turn["ttft_meas"] = rec["ttft_meas"]
                counts["ttft_meas"] += 1
            if "e2el_meas" in rec:
                turn["e2el_meas"] = rec["e2el_meas"]
                counts["e2el_meas"] += 1

        # ---- static M0 prediction (forward, workload-only) — COMPARISON column ----
        # The prefill-baseline × Little's-law queue amplifier. Once the headline; now the
        # static comparison line (ttft_pred_static), since the queue sim beats it.
        if have_predictor and profile in PROFILE_DIST:
            ttft_preds = predict_cell_ttft(turns, str(profile), float(c))
            for turn, ttft_pred in zip(turns, ttft_preds):
                if ttft_pred is None:
                    continue
                turn["ttft_pred_static"] = round(float(ttft_pred), 4)
                counts["ttft_pred_static"] += 1
                # E2EL composition: ttft + output·tpot_pred_kernel (headline TPOT).
                tpot_k = turn.get("tpot_pred_kernel")
                out = turn.get("output_tokens")
                if isinstance(tpot_k, (int, float)) and isinstance(out, (int, float)):
                    turn["e2el_pred_static"] = round(
                        float(ttft_pred) + float(out) * float(tpot_k), 4
                    )
                    counts["e2el_pred_static"] += 1

        # ---- forward closed-loop queue-sim TTFT — HEADLINE (ttft_pred / e2el_pred) ----
        # The emergent per-turn TTFT from the event-driven multi-turn queue sim (barrier
        # round-robin + chunked-prefill budget + block-level prefix-cache w/ herd-protected
        # rotation). Beats the static M0 on TTFT (60.78% vs 61.95%) and E2EL (29.71% vs
        # 30.01%) so it is now the headline ttft_pred / e2el_pred. e2el composes it with the
        # EXISTING kernel TPOT column (tpot_pred_kernel) — byte-identical composition, no
        # re-fit. The kernel headline (tpot_pred_kernel) is untouched.
        if have_qsim and profile in PROFILE_DIST:
            ttft_qsim = predict_cell_ttft_qsim(turns, str(profile), float(c))
            tpot_kernels = [turn.get("tpot_pred_kernel") for turn in turns]
            e2el_qsim = predict_cell_e2el_qsim(
                turns,
                str(profile),
                float(c),
                ttft_qsim,
                tpot_preds=[
                    float(tk) if isinstance(tk, (int, float)) else 0.0
                    for tk in tpot_kernels
                ],
            )
            for turn, ttft_q, e2el_q, tpot_k in zip(
                turns, ttft_qsim, e2el_qsim, tpot_kernels
            ):
                if ttft_q is None:
                    continue
                turn["ttft_pred"] = round(float(ttft_q), 4)
                counts["ttft_pred"] += 1
                # Only emit e2el_pred when the kernel TPOT it composes on is present.
                if isinstance(tpot_k, (int, float)):
                    turn["e2el_pred"] = round(float(e2el_q), 4)
                    counts["e2el_pred"] += 1

        # ---- per-turn error fields (ADDITIVE; mirror the tpot_* convention) ----
        # For TTFT/E2EL the per-turn dict carried meas+pred (and qsim pred) but no
        # per-turn error columns, so the dashboard per-turn UI rendered N/A. These
        # mirror tpot_err / tpot_signed_err_ms / tpot_abs_err_ms exactly (round 4),
        # written only when both pred and meas are numeric and meas>0. Never repoint.
        for pred_key, meas_key, err_key, signed_key, abs_key in (
            ("ttft_pred", "ttft_meas", "ttft_err", "ttft_signed_err_ms", "ttft_abs_err_ms"),
            ("e2el_pred", "e2el_meas", "e2el_err", "e2el_signed_err_ms", "e2el_abs_err_ms"),
        ):
            for turn in turns:
                pred = turn.get(pred_key)
                meas = turn.get(meas_key)
                if not (isinstance(pred, (int, float)) and isinstance(meas, (int, float))):
                    continue
                signed = float(pred) - float(meas)
                turn[signed_key] = round(signed, 4)
                counts[signed_key] += 1
                turn[abs_key] = round(abs(signed), 4)
                counts[abs_key] += 1
                if float(meas) > 0:
                    turn[err_key] = round(abs(signed) / float(meas) * 100.0, 4)
                    counts[err_key] += 1
        # per-turn static-M0 MAPE (additive) — lets the static comparison line carry a tone.
        for qsim_key, meas_key, err_key in (
            ("ttft_pred_static", "ttft_meas", "ttft_err_static"),
            ("e2el_pred_static", "e2el_meas", "e2el_err_static"),
        ):
            for turn in turns:
                pred = turn.get(qsim_key)
                meas = turn.get(meas_key)
                if (
                    isinstance(pred, (int, float))
                    and isinstance(meas, (int, float))
                    and float(meas) > 0
                ):
                    turn[err_key] = round(abs(float(pred) - float(meas)) / float(meas) * 100.0, 4)
                    counts[err_key] += 1

        # ---- cell-level summaries (mean of per-turn, matching tpot_meas) ----
        for key in (
            "ttft_meas",
            "e2el_meas",
            "ttft_pred",
            "e2el_pred",
            "ttft_pred_static",
            "e2el_pred_static",
        ):
            v = _cell_mean(turns, key)
            if v is not None:
                row[key] = v
        # cell-level APE (mean of per-turn APE), additive ttft_err / e2el_err.
        for pred_key, meas_key, err_key in (
            ("ttft_pred", "ttft_meas", "ttft_err"),
            ("e2el_pred", "e2el_meas", "e2el_err"),
            ("ttft_pred_static", "ttft_meas", "ttft_err_static"),
            ("e2el_pred_static", "e2el_meas", "e2el_err_static"),
        ):
            apes = [
                abs(float(t[pred_key]) - float(t[meas_key])) / float(t[meas_key]) * 100.0
                for t in turns
                if isinstance(t.get(pred_key), (int, float))
                and isinstance(t.get(meas_key), (int, float))
                and float(t[meas_key]) > 0
            ]
            if apes:
                row[err_key] = round(sum(apes) / len(apes), 4)
        # cell-level signed / abs latency error (mean over per-turn), mirroring how
        # tpot_signed_err_ms / tpot_abs_err_ms are the cell aggregates. Enables the
        # ServingMiniMetric tooltip + per-turn breakdown headers to read the explicit
        # field instead of the (pred-meas) fallback. Additive only.
        for signed_turn_key, cell_key in (
            ("ttft_signed_err_ms", "ttft_signed_err_ms"),
            ("e2el_signed_err_ms", "e2el_signed_err_ms"),
            ("ttft_abs_err_ms", "ttft_abs_err_ms"),
            ("e2el_abs_err_ms", "e2el_abs_err_ms"),
        ):
            v = _cell_mean(turns, signed_turn_key)
            if v is not None:
                row[cell_key] = v

    dashboard_json.write_text(json.dumps(payload, indent=2) + "\n")
    if missing_raw:
        print(f"WARNING: {len(missing_raw)} cells had no raw bench file: {missing_raw[:8]}")
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dashboard-json", type=Path, default=DEFAULT_DASHBOARD_JSON)
    p.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    counts = augment(args.dashboard_json, args.bench_root)
    print(
        "injected ttft_meas={ttft_meas} e2el_meas={e2el_meas} "
        "ttft_pred(qsim)={ttft_pred} e2el_pred(qsim)={e2el_pred} "
        "ttft_pred_static={ttft_pred_static} e2el_pred_static={e2el_pred_static} "
        "ttft_err={ttft_err} e2el_err={e2el_err} "
        "ttft_signed_err_ms={ttft_signed_err_ms} e2el_signed_err_ms={e2el_signed_err_ms} "
        "ttft_abs_err_ms={ttft_abs_err_ms} e2el_abs_err_ms={e2el_abs_err_ms} "
        "ttft_err_static={ttft_err_static} e2el_err_static={e2el_err_static} "
        "across {cells} cells".format(**counts)
    )
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
