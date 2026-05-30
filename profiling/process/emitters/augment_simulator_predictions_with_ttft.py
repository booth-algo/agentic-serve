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
  * ``ttft_pred``  — forward prefill-baseline × Little's-law queue amplifier
    (``simulator/ttft_predict.py``).
  * ``e2el_pred``  — composition ``ttft_pred + output·tpot_pred_kernel`` (uses the
    headline kernel TPOT, *not* the stale roofline ``tpot_pred``).

This augmenter is **additive only** — it never repoints any existing TPOT field.

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
        "ttft_pred": 0,
        "e2el_pred": 0,
        "ttft_pred_qsim": 0,
        "e2el_pred_qsim": 0,
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

        # ---- predicted (forward, workload-only) ----
        if have_predictor and profile in PROFILE_DIST:
            ttft_preds = predict_cell_ttft(turns, str(profile), float(c))
            for turn, ttft_pred in zip(turns, ttft_preds):
                if ttft_pred is None:
                    continue
                turn["ttft_pred"] = round(float(ttft_pred), 4)
                counts["ttft_pred"] += 1
                # E2EL composition: ttft + output·tpot_pred_kernel (headline TPOT).
                tpot_k = turn.get("tpot_pred_kernel")
                out = turn.get("output_tokens")
                if isinstance(tpot_k, (int, float)) and isinstance(out, (int, float)):
                    turn["e2el_pred"] = round(
                        float(ttft_pred) + float(out) * float(tpot_k), 4
                    )
                    counts["e2el_pred"] += 1

        # ---- forward closed-loop queue-sim TTFT (ADDITIVE; never repoints) ----
        # ttft_pred_qsim is the emergent per-turn TTFT from the event-driven multi-turn
        # queue simulation. e2el_pred_qsim composes it with the EXISTING kernel TPOT
        # column (tpot_pred_kernel) — byte-identical composition, no re-fit. The static
        # M0 (ttft_pred) and the kernel headline (tpot_pred_kernel) are untouched.
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
                turn["ttft_pred_qsim"] = round(float(ttft_q), 4)
                counts["ttft_pred_qsim"] += 1
                # Only emit e2el_qsim when the kernel TPOT it composes on is present.
                if isinstance(tpot_k, (int, float)):
                    turn["e2el_pred_qsim"] = round(float(e2el_q), 4)
                    counts["e2el_pred_qsim"] += 1

        # ---- cell-level summaries (mean of per-turn, matching tpot_meas) ----
        for key in (
            "ttft_meas",
            "e2el_meas",
            "ttft_pred",
            "e2el_pred",
            "ttft_pred_qsim",
            "e2el_pred_qsim",
        ):
            v = _cell_mean(turns, key)
            if v is not None:
                row[key] = v
        # cell-level APE (mean of per-turn APE), additive ttft_err / e2el_err.
        for pred_key, meas_key, err_key in (
            ("ttft_pred", "ttft_meas", "ttft_err"),
            ("e2el_pred", "e2el_meas", "e2el_err"),
            ("ttft_pred_qsim", "ttft_meas", "ttft_err_qsim"),
            ("e2el_pred_qsim", "e2el_meas", "e2el_err_qsim"),
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
        "ttft_pred={ttft_pred} e2el_pred={e2el_pred} "
        "ttft_pred_qsim={ttft_pred_qsim} e2el_pred_qsim={e2el_pred_qsim} "
        "across {cells} cells".format(**counts)
    )
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
