"""Inject llm-d-inference-sim parametric TPOT predictions into the dashboard
``simulator-predictions.json`` as ``tpot_pred_llm_d`` per turn.

The added field is a **constant per (profile, concurrency)** by design — it's
the llm-d parametric model's output:

    base_itl = mean(tpot_meas at min-c for the profile)
    tf       = mean(tpot_meas at max-c for the profile) / base_itl
    tpot     = base_itl × (1 + (tf − 1) × (c − 1) / (c_max − 1))

llm-d's ITL formula has no notion of turn / context length (see
``pkg/llm-d-inference-sim/latencies.go: GetInterTokenLatency``), so the
prediction is the same value for every turn within a cell.  The dashboard
chart plots it as a flat dotted amber line — that flatness is the point: it
exposes that llm-d's formula can't model the KV-exhaustion regime shift
visible on swebench / terminal at high c.

Re-run this whenever the bench's per-turn ``tpot_meas`` data changes.

Usage:
    python3 -m profiling.process.emitters.augment_simulator_predictions_with_llm_d
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


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

    # Mean of tpot_meas per (profile, concurrency)
    mean_by_pc: dict[tuple[str, int], float] = {}
    for row in rows:
        p, c = row.get("profile"), row.get("concurrency")
        vals = [
            t.get("tpot_meas")
            for t in (row.get("multiturn_turn_predictions") or [])
            if isinstance(t.get("tpot_meas"), (int, float))
        ]
        if p is not None and c is not None and vals:
            mean_by_pc[(p, c)] = statistics.mean(vals)

    # Fit (base_itl, tf, c_max) per profile from min-c and max-c anchors
    profiles = sorted({p for (p, _) in mean_by_pc})
    params: dict[str, tuple[float, float, int]] = {}
    for prof in profiles:
        cs = sorted({c for (p, c) in mean_by_pc if p == prof})
        if not cs:
            continue
        c_min, c_max = cs[0], cs[-1]
        base = mean_by_pc[(prof, c_min)]
        peak = mean_by_pc[(prof, c_max)]
        tf = peak / base if base > 0 else 1.0
        params[prof] = (base, tf, c_max)
        print(
            f"  {prof}: base_itl={base:.2f}ms  tf={tf:.2f}  c_max={c_max}"
        )

    def llmd(prof: str, c: int) -> float:
        base, tf, c_max = params[prof]
        if c_max <= 1:
            return base
        return base * (1 + (tf - 1) * (c - 1) / (c_max - 1))

    injected = 0
    for row in rows:
        p, c = row.get("profile"), row.get("concurrency")
        if p not in params or c is None:
            continue
        pred = llmd(p, int(c))
        for turn in row.get("multiturn_turn_predictions") or []:
            turn["tpot_pred_llm_d"] = round(pred, 4)
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
    print(f"\ninjected tpot_pred_llm_d into {n} turn records")
    print(f"wrote {args.dashboard_json}")


if __name__ == "__main__":
    main()
