"""Inject three-regime TPOT predictions into the dashboard
``simulator-predictions.json`` as ``tpot_pred_three_regime`` and the
classification label ``tpot_regime`` per turn.

Architecture (see plan):
- T_min[t]    = ``tpot_pred[t]``                  ← closed-form roofline (already injected)
- T_max       = min(physical_T_upper ≈ 205 ms,
                    ``tpot_pred_llm_d`` for this cell)
- regime      = classify_cell(workload trajectory)  ← FLAT / PERTURBING / SATURATING
- per-turn TPOT follows the regime-specific shape (see three_regime_tpot.py)

No fitted constants. Workload-only inputs:
``scheduled_requests`` (cohort active count from the bench output), plus the
existing per-turn ``cached_context_tokens``, ``new_prefill_tokens``,
``output_tokens``.

Usage:
    python3 -m profiling.process.emitters.augment_simulator_predictions_with_three_regime
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
from simulator.three_regime_tpot import (  # noqa: E402
    TurnObservation,
    classify_cell,
    compute_t_max,
    predict_cell_tpot,
)


DEFAULT_DASHBOARD_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)


def _request_count(turn: dict[str, Any], fallback_c: int) -> int:
    """Workload-derived active session count for this turn.

    Prefers ``scheduled_requests`` (cohort intent — what the bench submitted
    for turn t). This signal correctly distinguishes osworld (cohort
    completion drops scheduled_requests 160 → 66+) from swe (cohort stays at
    80 throughout, only engine throttles its admission). Engine throttling
    (visible in ``engine_max_decode_batch``) is NOT the same as cohort
    completion and shouldn't trigger PERTURBING reclassification.
    """
    for key in ("scheduled_requests", "successful"):
        v = turn.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return fallback_c


def predict_cell(
    turns: list[dict[str, Any]],
    cohort_c: int,
    params: RooflineParams,
    *,
    use_kernel_lookup: bool = True,
) -> tuple[str, list[float], dict[str, Any]]:
    """Classify the cell and emit per-turn three-regime TPOT predictions.

    ``use_kernel_lookup=True`` (default) uses ``cached_prefill_step_ms(U, P) +
    scheduler_overhead`` as the per-turn T_max ceiling, capped by the
    regime-dependent ``t_max_ms`` from ``compute_t_max``. Set False for
    legacy constant-T_max behaviour.

    Returns (regime_name, per_turn_predictions_aligned_with_turns, classification_info).
    """
    observations = [
        TurnObservation(
            turn_index=int(turn.get("turn_index", 0)),
            cached_context_tokens=float(turn.get("cached_context_tokens") or 0.0),
            new_prefill_tokens=float(turn.get("new_prefill_tokens") or 0.0),
            output_tokens=float(turn.get("output_tokens") or 0.0),
            request_count=_request_count(turn, cohort_c),
        )
        for turn in turns
    ]

    classification = classify_cell(observations, params, cohort_c=cohort_c)

    # Coverage = engine_total_decode_slots / (engine_steps × engine_max_decode_batch).
    # When telemetry is present, this lifts PERTURBING T_max from engine step
    # time toward client-measured ITL (engine_step_ms / coverage). When telemetry
    # is missing or running batch is zero, coverage stays 1.0 (no inflation).
    coverage_by_turn: dict[int, float] = {}
    for turn in turns:
        slots = turn.get("engine_total_decode_slots")
        steps = turn.get("engine_steps")
        running = turn.get("engine_max_decode_batch")
        if (
            isinstance(slots, (int, float)) and slots > 0
            and isinstance(steps, (int, float)) and steps > 0
            and isinstance(running, (int, float)) and running > 0
        ):
            cov = float(slots) / (float(steps) * float(running))
            coverage_by_turn[int(turn.get("turn_index", 0))] = max(0.05, min(1.0, cov))

    # T_max anchor: regime-dependent.
    # SATURATING → physical_T_upper (true asymptote, no llm-d cap)
    # PERTURBING → min(physical, llm-d per-cell mean) — bounded by observable envelope
    llmd_value: float | None = None
    for turn in turns:
        v = turn.get("tpot_pred_llm_d")
        if isinstance(v, (int, float)) and v > 0:
            llmd_value = float(v)
            break
    t_max_ms = compute_t_max(llmd_value, params, regime=classification.regime)

    # T_min[t] = existing roofline pred per turn
    sorted_turns = sorted(turns, key=lambda t: int(t.get("turn_index", 0)))
    sorted_observations = sorted(observations, key=lambda o: o.turn_index)
    t_min_per_turn: list[float] = []
    for turn in sorted_turns:
        v = turn.get("tpot_pred")
        if isinstance(v, (int, float)) and v > 0:
            t_min_per_turn.append(float(v))
        else:
            # Fallback: if no roofline value, leave a moderate guess
            t_min_per_turn.append(10.0)

    sorted_preds = predict_cell_tpot(
        classification, sorted_observations, t_min_per_turn, t_max_ms,
        params=params, use_kernel_lookup=use_kernel_lookup,
        coverage_by_turn=coverage_by_turn or None,
    )

    # Re-align predictions back to the input `turns` order
    pred_by_idx = {
        obs.turn_index: pred
        for obs, pred in zip(sorted_observations, sorted_preds)
    }
    aligned = [pred_by_idx[int(t.get("turn_index", 0))] for t in turns]

    info = {
        "regime": classification.regime.value,
        "jump_turn": classification.jump_turn,
        "peak_pressure": round(classification.peak_pressure, 4),
        "late_pressure": round(classification.late_pressure, 4),
        "saturated_turns": classification.saturated_turns,
        "t_max_ms": round(t_max_ms, 4),
        "perturbation_turns": list(classification.perturbation_turns),
    }
    return classification.regime.value, aligned, info


def _aggregate_three_regime_error(
    turns: list[dict[str, Any]],
    preds: list[float],
) -> dict[str, float | None]:
    """Cell-level error metrics for three-regime vs observed `tpot_meas`.

    These overwrite the row-level `tpot_err`, `tpot_signed_err_ms`,
    `tpot_abs_err_ms` so the dashboard's matrix MAPE columns show
    three-regime accuracy instead of the closed-form roofline.
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


def augment(dashboard_json: Path) -> int:
    payload = json.loads(dashboard_json.read_text())
    rows = payload.get("H100", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            f"no H100 rows found in {dashboard_json}; nothing to augment"
        )

    params = RooflineParams()
    injected = 0
    for row in rows:
        c = row.get("concurrency")
        turns = row.get("multiturn_turn_predictions") or []
        if c is None or not turns:
            continue
        regime, preds, info = predict_cell(turns, int(c), params)
        for turn, pred_ms in zip(turns, preds):
            turn["tpot_pred_three_regime"] = round(pred_ms, 4)
            turn["tpot_regime"] = regime
            # Per-turn errors also reflect the three-regime prediction so the
            # per-turn breakdown shows the right deltas.
            meas = turn.get("tpot_meas")
            if isinstance(meas, (int, float)) and meas > 0:
                meas_f = float(meas)
                turn["tpot_signed_err_ms"] = round(pred_ms - meas_f, 4)
                turn["tpot_abs_err_ms"] = round(abs(pred_ms - meas_f), 4)
                turn["tpot_err"] = round(abs(pred_ms - meas_f) / meas_f * 100.0, 4)
            injected += 1
        # Cell-level diagnostic — useful for the dashboard's cell header chip
        row["tpot_three_regime_info"] = info
        # Overwrite cell-level TPOT error fields with three-regime aggregates
        # so the matrix MAPE columns + KPI cards display three-regime accuracy.
        row.update(_aggregate_three_regime_error(turns, preds))

    dashboard_json.write_text(json.dumps(payload, indent=2) + "\n")
    return injected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dashboard-json", type=Path, default=DEFAULT_DASHBOARD_JSON)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = augment(args.dashboard_json)
    print(f"injected tpot_pred_three_regime into {n} turn records")
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
