"""Measurement gate for the kernel-composition TPOT predictor.

Reports, for every predictor in ``simulator-predictions.json`` (roofline, llm-d,
two-roofline, three-regime, kernel), the per-profile MAPE / median / p90 / max APE
across all turns, PLUS a **plateau-only** slice (turns with ``tpot_meas > 100`` ms)
where the capacity-pressure jump lives. This is the gate used after each fix in the
jump-tracking plan — the headline numbers to watch are overall MAPE and the swebench
plateau MAPE.

Console-only by default (no doc artifact). Use after editing simulator/kernel_tpot.py
and re-running augment_simulator_predictions_with_kernel.

Usage:
    python3 -m profiling.process.validate_tpot
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SIM_PREDICTIONS = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)
PLATEAU_MS = 100.0  # turns above this are "on the plateau" (the capacity jump)

PREDICTORS = [
    # Headline = the kernel-composed TPOT, now written directly to tpot_pred by
    # build_simulator_rows.py (the old roofline / kernel+hint / fwd-ramp comparison lines
    # were retired with the engine-step pipeline).
    ("tpot_pred", "tpot"),
]


def collect_rows(sim_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(sim_json.read_text())
    out: list[dict[str, Any]] = []
    for cell in payload.get("H100", []):
        # The "H100" payload key now also holds other-model rows (gpt-oss, Qwen) run on the same
        # H100 tp1 vllm; the gate is the kernel-calibrated Llama-3.1-8B config only.
        if cell.get("model") not in (None, "Llama-3.1-8B"):
            continue
        prof = cell.get("profile")
        c = cell.get("concurrency")
        if prof is None or c is None:
            continue
        for turn in cell.get("multiturn_turn_predictions") or []:
            meas = turn.get("tpot_meas")
            if not isinstance(meas, (int, float)) or meas <= 0:
                continue
            rec: dict[str, Any] = {
                "profile": prof,
                "concurrency": int(c),
                "turn_index": int(turn.get("turn_index", -1)),
                "tpot_meas_ms": float(meas),
            }
            for key, _ in PREDICTORS:
                v = turn.get(key)
                rec[key] = float(v) if isinstance(v, (int, float)) else None
            out.append(rec)
    return out


def ape(pred: float | None, obs: float) -> float | None:
    if pred is None or pred <= 0 or obs <= 0:
        return None
    return abs(pred - obs) / obs * 100.0


def _stats(apes: list[float]) -> dict[str, float]:
    apes_f = [a for a in apes if a is not None and not math.isnan(a)]
    if not apes_f:
        return {"mape": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan"), "n": 0}
    s = sorted(apes_f)
    return {
        "mape": sum(s) / len(s),
        "median": statistics.median(s),
        "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
        "max": max(s),
        "n": len(s),
    }


def _summary(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """profile -> predictor_name -> stats. Profile key '__overall__' aggregates all."""
    by_prof: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prof[r["profile"]].append(r)
        by_prof["__overall__"].append(r)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for prof, prof_rows in by_prof.items():
        out[prof] = {
            name: _stats([ape(r.get(key), r["tpot_meas_ms"]) for r in prof_rows])
            for key, name in PREDICTORS
        }
    return out


def _print_table(title: str, summary: dict[str, dict[str, dict[str, float]]], metric: str) -> None:
    print(f"\n=== {title} — {metric.upper()} ===")
    names = [name for _, name in PREDICTORS]
    header = f"  {'profile':<32}{'n':>6}" + "".join(f"{n:>10}" for n in names)
    print(header)
    order = sorted(k for k in summary if k != "__overall__") + ["__overall__"]
    for prof in order:
        per = summary[prof]
        n = per[names[0]]["n"]
        label = "OVERALL" if prof == "__overall__" else prof
        cells = "".join(
            f"{per[n][metric]:>9.1f}%" if not math.isnan(per[n][metric]) else f"{'n/a':>10}"
            for n in names
        )
        print(f"  {label:<32}{n:>6}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator-predictions-json", type=Path, default=DEFAULT_SIM_PREDICTIONS)
    args = ap.parse_args()

    rows = collect_rows(args.simulator_predictions_json)
    plateau = [r for r in rows if r["tpot_meas_ms"] > PLATEAU_MS]
    print(f"loaded {len(rows)} turns ({len(plateau)} on plateau, tpot_meas>{PLATEAU_MS:.0f}ms)")

    all_summary = _summary(rows)
    plateau_summary = _summary(plateau)

    _print_table("ALL turns", all_summary, "mape")
    _print_table("ALL turns", all_summary, "median")
    _print_table(f"PLATEAU only (tpot_meas>{PLATEAU_MS:.0f}ms)", plateau_summary, "mape")

    # The two headline gate numbers.
    ov = all_summary["__overall__"]["tpot"]["mape"]
    swe_plateau = plateau_summary.get("swebench-multiturn-synth", {}).get("tpot", {}).get("mape", float("nan"))
    print(f"\nGATE  overall TPOT MAPE = {ov:.2f}%   swebench-plateau TPOT MAPE = {swe_plateau:.2f}%")


if __name__ == "__main__":
    main()
