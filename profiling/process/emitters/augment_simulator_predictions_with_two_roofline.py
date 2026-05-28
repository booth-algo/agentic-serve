"""Inject two-roofline TPOT predictions into the dashboard ``simulator-predictions.json``
as ``tpot_pred_two_roofline`` per turn.

The two-roofline model decomposes each turn's scheduler steps into decode-only
(``T_lower``) and mixed prefill+decode (``T_upper``). See
``simulator/two_roofline_tpot.py`` for the derivation; no fitted constants.

Turn-history rule (see plan): for each (profile, c) cell we iterate turns in
order and track a `consecutive_saturated` counter that ticks up on any turn
where ``active × per_session_blocks / available_kv_blocks >= 1`` and resets
on any turn below capacity. Once the counter hits ``K_SUSTAIN`` (= 2 turns
at capacity), the cell is declared "sustained-saturated" and the predictor
uses the cohort `c` as effective concurrency (cohort dominates) instead of
the observed active count. This catches sustained-saturation regimes like
swebench c=80 from turn ~12 onward, where active and capacity_batch shrink
together and a per-turn snapshot looks barely-pressured.

Re-run this whenever the bench's per-turn data changes.

Usage:
    python3 -m profiling.process.emitters.augment_simulator_predictions_with_two_roofline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.two_roofline_tpot import TurnWorkload, predict_two_roofline  # noqa: E402


DEFAULT_DASHBOARD_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)

# Two consecutive turns at pool capacity is the physical minimum to
# distinguish "first-turn boundary blip" from "sustained admission cycle".
# Not a fit — see plan.
K_SUSTAIN = 2

# A single-turn drop in active sessions exceeding 15% of the cohort signals
# "bulk completion" rather than steady-state cycling. Steady-state cycling
# completes roughly (1 / planned_turns_per_session) × cohort_c per turn —
# for 10–30-turn cohorts that's 3–10%. A 15%+ drop is well above that floor,
# so we treat it as a permanent regime shift: from this point on, the
# remaining sessions don't dominate KV — use observed active for pressure,
# not the original cohort size.
BURST_COMPLETION_FRACTION = 0.15


def _per_session_blocks(cached: float, new_prefill: float, output: float,
                        block_size: int) -> int:
    ctx_mid = cached + new_prefill + 0.5 * max(1.0, output)
    return max(1, math.ceil(ctx_mid / max(1, block_size)))


def predict_cell_turns(
    turns: list[dict[str, Any]],
    cohort_c: int,
    params: RooflineParams,
) -> list[float]:
    """Return per-turn ``tpot_pred_two_roofline`` for one (profile, c) cell.

    Turns must contain ``turn_index``; sorted internally before iteration.
    Maintains the ``consecutive_saturated`` counter across turns in this cell.
    """
    sorted_turns = sorted(turns, key=lambda t: int(t.get("turn_index", 0)))
    consecutive_saturated = 0
    burst_completion_detected = False
    prev_active: int | None = None
    burst_threshold = BURST_COMPLETION_FRACTION * cohort_c
    predictions_by_turn_index: dict[int, float] = {}

    for turn in sorted_turns:
        cached = float(turn.get("cached_context_tokens") or 0.0)
        new_prefill = float(turn.get("new_prefill_tokens") or 0.0)
        output = float(turn.get("output_tokens") or 0.0)
        edb = turn.get("engine_max_decode_batch")
        active = int(edb) if isinstance(edb, (int, float)) and edb > 0 else None

        # Detect bulk completion: a single-turn drop in active sessions
        # exceeding BURST_COMPLETION_FRACTION × cohort_c. Once detected, the
        # cohort has lost too many sessions for "cohort dominates" to apply —
        # use observed active for the rest of the cell.
        if prev_active is not None and active is not None:
            if (prev_active - active) > burst_threshold:
                burst_completion_detected = True
        prev_active = active if active is not None else prev_active

        # Update sustained-saturation state using observed active vs capacity.
        if active is not None and not burst_completion_detected:
            psb = _per_session_blocks(
                cached, new_prefill, output, params.cache_block_size,
            )
            capacity_batch = max(1, params.available_kv_blocks // psb)
            pressure_active = active / capacity_batch
            if pressure_active >= 1.0:
                consecutive_saturated += 1
            else:
                consecutive_saturated = 0
        else:
            # No telemetry or already in burst-completion mode → reset state.
            consecutive_saturated = 0

        # Effective concurrency: cohort dominates once sustained AND no burst
        # completion has fired; otherwise use the observed active.
        if consecutive_saturated >= K_SUSTAIN and not burst_completion_detected:
            effective_c = cohort_c
        else:
            effective_c = active if active is not None else cohort_c

        workload = TurnWorkload(
            cached_context_tokens=cached,
            new_prefill_tokens=new_prefill,
            output_tokens=output,
        )
        pred = predict_two_roofline(
            workload, cohort_c, p=params, active_sessions=effective_c,
        )
        predictions_by_turn_index[int(turn.get("turn_index", 0))] = pred.predicted_tpot_ms

    return [
        predictions_by_turn_index[int(t.get("turn_index", 0))]
        for t in turns
    ]


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
        if c is None:
            continue
        turns = row.get("multiturn_turn_predictions") or []
        if not turns:
            continue
        per_turn_preds = predict_cell_turns(turns, int(c), params)
        for turn, pred_ms in zip(turns, per_turn_preds):
            turn["tpot_pred_two_roofline"] = round(pred_ms, 4)
            injected += 1

    dashboard_json.write_text(json.dumps(payload, indent=2) + "\n")
    return injected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dashboard-json", type=Path, default=DEFAULT_DASHBOARD_JSON)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = augment(args.dashboard_json)
    print(f"injected tpot_pred_two_roofline into {n} turn records")
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
