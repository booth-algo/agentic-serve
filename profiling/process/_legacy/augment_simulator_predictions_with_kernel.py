"""Inject kernel-composition TPOT predictions into the dashboard
``simulator-predictions.json`` as ``tpot_pred_kernel`` per turn.

The kernel predictor (``simulator/kernel_tpot.py``) is fully workload-only — no
engine telemetry — so this augmenter reads only the per-turn aggregates already
present in each row: ``cached_context_tokens``, ``new_prefill_tokens``,
``output_tokens``, and ``scheduled_requests`` (falling back to the cell
concurrency when a turn lacks a scheduled count).

Also injects two side-by-side comparison columns (never repointed into the MAPE
matrix / KPIs — the headline ``tpot_err`` always tracks the production
``tpot_pred_kernel``, byte-identical):
  * ``tpot_pred_kernel_hint`` — classifier stepping-ramp soft hint
    (``simulator/kernel_tpot_hint.py``).
  * ``tpot_pred_ramp`` — forward 3D-roofline eviction-deficit ramp predictor
    (``simulator/ramp_tpot.py``); forecasts the cohort drain from the profile's
    session-length distribution (only for profiles with a known distribution).

Usage:
    python3 -m profiling.process._legacy.augment_simulator_predictions_with_kernel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot  # noqa: E402
from simulator._legacy.kernel_tpot_hint import predict_cell_tpot_hinted  # noqa: E402  (retired 2026-06-10, audit-v2 D9)
from simulator.ramp_tpot import PROFILE_DIST, predict_cell_tpot_ramp  # noqa: E402


DEFAULT_DASHBOARD_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)


def _scheduled(turn: dict[str, Any], fallback_c: float) -> float:
    for key in ("scheduled_requests", "successful"):
        v = turn.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return fallback_c


def _aggregate_kernel_error(
    turns: list[dict[str, Any]], preds: list[float]
) -> dict[str, float | None]:
    """Cell-level error metrics for kernel vs observed ``tpot_meas``.

    These overwrite the row-level ``tpot_err`` / ``tpot_signed_err_ms`` /
    ``tpot_abs_err_ms`` so the dashboard's matrix MAPE columns + KPI cards show
    kernel-predictor accuracy (the "repoint" — kernel is now the headline).
    """
    signed_errs: list[float] = []
    abs_errs: list[float] = []
    apes: list[float] = []
    for turn, pred in zip(turns, preds):
        meas = turn.get("tpot_meas")
        if not isinstance(meas, (int, float)) or meas <= 0:
            continue
        meas_f = float(meas)
        signed_errs.append(pred - meas_f)
        abs_errs.append(abs(pred - meas_f))
        apes.append(abs(pred - meas_f) / meas_f * 100.0)
    if not apes:
        return {
            "tpot_signed_err_ms": None,
            "tpot_abs_err_ms": None,
            "tpot_err": None,
        }
    return {
        "tpot_signed_err_ms": round(sum(signed_errs) / len(signed_errs), 4),
        "tpot_abs_err_ms": round(sum(abs_errs) / len(abs_errs), 4),
        "tpot_err": round(sum(apes) / len(apes), 4),
    }


def augment(dashboard_json: Path, *, repoint: bool = True) -> int:
    """Inject ``tpot_pred_kernel`` per turn.

    When ``repoint`` (default), also overwrite the per-turn and cell-level TPOT
    error fields (``tpot_err`` etc.) with kernel-vs-measured errors so the
    dashboard matrix MAPE + KPIs report the kernel predictor. The base
    ``tpot_pred`` (roofline) line is left untouched so the chart still shows it
    alongside kernel / kernel+hint / fwd-ramp.
    """
    payload = json.loads(dashboard_json.read_text())
    rows = payload.get("H100", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"no H100 rows found in {dashboard_json}; nothing to augment")

    params = RooflineParams()
    injected = 0
    for row in rows:
        c = row.get("concurrency")
        profile = row.get("profile")
        turns = row.get("multiturn_turn_predictions") or []
        if c is None or not turns:
            continue
        inputs = [
            KernelTurnInput(
                cached_context_tokens=float(turn.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(turn.get("new_prefill_tokens") or 0.0),
                output_tokens=float(turn.get("output_tokens") or 0.0),
                scheduled_requests=_scheduled(turn, float(c)),
            )
            for turn in turns
        ]
        # Cell path applies the per-cell median-output ceiling (Step-2 de-swing).
        preds = predict_cell_tpot(inputs, params)
        # Classifier stepping-ramp soft-hint variant (comparison line; never
        # repointed). Needs turn_index (in stored order) so its turn-space ramp
        # lines up with the classifier window.
        turn_indices = [int(turn.get("turn_index", i)) for i, turn in enumerate(turns)]
        preds_hint = predict_cell_tpot_hinted(inputs, turn_indices, params)
        # Forward 3D-roofline eviction-deficit ramp (comparison line; never repointed).
        # Fully forward: forecasts the cohort from the profile survival curve, so it
        # only runs for profiles with a known session-length distribution.
        if profile in PROFILE_DIST:
            preds_ramp = predict_cell_tpot_ramp(turns, profile, float(c), params)
        else:
            preds_ramp = [None] * len(turns)
        for turn, pred, pred_hint, pred_ramp in zip(turns, preds, preds_hint, preds_ramp):
            turn["tpot_pred_kernel"] = round(pred, 4)
            turn["tpot_pred_kernel_hint"] = round(pred_hint, 4)
            if pred_ramp is not None:
                turn["tpot_pred_ramp"] = round(pred_ramp, 4)
            if repoint:
                meas = turn.get("tpot_meas")
                if isinstance(meas, (int, float)) and meas > 0:
                    meas_f = float(meas)
                    turn["tpot_signed_err_ms"] = round(pred - meas_f, 4)
                    turn["tpot_abs_err_ms"] = round(abs(pred - meas_f), 4)
                    turn["tpot_err"] = round(abs(pred - meas_f) / meas_f * 100.0, 4)
            injected += 1
        if repoint:
            row.update(_aggregate_kernel_error(turns, preds))
    dashboard_json.write_text(json.dumps(payload, indent=2) + "\n")
    return injected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dashboard-json", type=Path, default=DEFAULT_DASHBOARD_JSON)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = augment(args.dashboard_json)
    print(f"injected tpot_pred_kernel (+ tpot_pred_kernel_hint, tpot_pred_ramp) into {n} turn records")
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
