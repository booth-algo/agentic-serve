"""Inject two-roofline TPOT predictions into the dashboard ``simulator-predictions.json``
as ``tpot_pred_two_roofline`` per turn.

The two-roofline model decomposes each turn's scheduler steps into decode-only
(``T_lower``) and mixed prefill+decode (``T_upper``). See
``simulator/two_roofline_tpot.py`` for the derivation; no fitted constants.

Re-run this whenever the bench's per-turn data changes.

Usage:
    python3 -m profiling.process.emitters.augment_simulator_predictions_with_two_roofline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402
from simulator.two_roofline_tpot import TurnWorkload, predict_two_roofline  # noqa: E402


DEFAULT_DASHBOARD_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)


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
        for turn in row.get("multiturn_turn_predictions") or []:
            cached = float(turn.get("cached_context_tokens") or 0.0)
            new_prefill = float(turn.get("new_prefill_tokens") or 0.0)
            output = float(turn.get("output_tokens") or 0.0)
            workload = TurnWorkload(
                cached_context_tokens=cached,
                new_prefill_tokens=new_prefill,
                output_tokens=output,
            )
            # Use observed active count if present — captures session-completion
            # dynamics the analytical model can't predict from workload alone.
            edb = turn.get("engine_max_decode_batch")
            active = int(edb) if isinstance(edb, (int, float)) and edb > 0 else None
            pred = predict_two_roofline(
                workload, int(c), p=params, active_sessions=active,
            )
            turn["tpot_pred_two_roofline"] = round(pred.predicted_tpot_ms, 4)
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
