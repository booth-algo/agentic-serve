"""Update ``llama31-8b-h100-tpot-fit.json`` aggregates from kernel-composition
predictions (``tpot_pred_kernel``) in ``simulator-predictions.json``.

This repoints the dashboard's "TPOT MAPE" KPI badge + comparison rows to the
kernel predictor (see ``simulator/kernel_tpot.py``), replacing the previous
three-regime aggregates. The per-turn prediction lines for every predictor are
left untouched — only the headline summary numbers change.

Usage:
    python3 -m profiling.process.emitters.update_tpot_fit_with_kernel
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SIM_PREDICTIONS = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)
DEFAULT_TPOT_FIT = Path(
    "inference-benchmark/dashboard/public/llama31-8b-h100-tpot-fit.json"
)


def collect_kernel_errors(sim_predictions: Path) -> dict[str, float | int]:
    """Aggregate kernel APE / signed / abs error across all turns."""
    payload = json.loads(sim_predictions.read_text())
    rows = payload.get("H100") or []
    apes: list[float] = []
    signed_errs: list[float] = []
    abs_errs: list[float] = []
    for row in rows:
        for turn in row.get("multiturn_turn_predictions") or []:
            pred = turn.get("tpot_pred_kernel")
            meas = turn.get("tpot_meas")
            if not isinstance(pred, (int, float)) or pred <= 0:
                continue
            if not isinstance(meas, (int, float)) or meas <= 0:
                continue
            meas_f = float(meas)
            err_abs = abs(pred - meas_f)
            apes.append(err_abs / meas_f * 100.0)
            signed_errs.append(pred - meas_f)
            abs_errs.append(err_abs)
    if not apes:
        raise SystemExit("no kernel predictions found in simulator-predictions.json")
    return {
        "rows": len(apes),
        "mape": sum(apes) / len(apes),
        "median_ape": statistics.median(apes),
        "max_ape": max(apes),
        "mean_signed_error_ms": sum(signed_errs) / len(signed_errs),
        "mean_abs_error_ms": sum(abs_errs) / len(abs_errs),
        "max_abs_error_ms": max(abs_errs),
    }


def patch_fit_json(fit_json: Path, agg: dict[str, float | int]) -> None:
    payload = json.loads(fit_json.read_text())
    fit = payload.setdefault("fit_summary", {})
    fit["rows"] = agg["rows"]
    fit["kernel_composed_mape"] = round(agg["mape"], 4)
    fit["kernel_composed_median_ape"] = round(agg["median_ape"], 4)
    fit["kernel_composed_max_ape"] = round(agg["max_ape"], 4)
    fit["kernel_composed_mean_signed_error_ms"] = round(agg["mean_signed_error_ms"], 4)
    fit["kernel_composed_mean_abs_error_ms"] = round(agg["mean_abs_error_ms"], 4)
    fit["kernel_composed_max_abs_error_ms"] = round(agg["max_abs_error_ms"], 4)
    fit["physics_loo_mape"] = fit["kernel_composed_mape"]
    fit["physics_loo_median_ape"] = fit["kernel_composed_median_ape"]
    fit["physics_loo_max_ape"] = fit["kernel_composed_max_ape"]
    fit["engine_step_model"] = "kernel_composition"
    fit["wave_policy"] = "pressure-amplifier (kernel)"

    for comp_list in (payload.get("dashboard_comparison") or [], *(
        payload.get("page_comparisons", {}).get(k, []) for k in ("serving", "simulator")
    )):
        for comp in comp_list:
            if comp.get("backend") and comp.get("tpot_mape") is not None:
                comp["backend"] = "kernel"
                comp["label"] = "kernel-composition predictor"
                comp["rows"] = agg["rows"]
                comp["tpot_mape"] = fit["kernel_composed_mape"]
                comp["tpot_median_ape"] = fit["kernel_composed_median_ape"]
                comp["tpot_max_ape"] = fit["kernel_composed_max_ape"]

    fit_json.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulator-predictions-json", type=Path, default=DEFAULT_SIM_PREDICTIONS)
    p.add_argument("--tpot-fit-json", type=Path, default=DEFAULT_TPOT_FIT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg = collect_kernel_errors(args.simulator_predictions_json)
    patch_fit_json(args.tpot_fit_json, agg)
    print(
        f"updated {args.tpot_fit_json} with kernel aggregates: "
        f"MAPE={agg['mape']:.2f}% median={agg['median_ape']:.2f}% max={agg['max_ape']:.2f}% "
        f"(n={agg['rows']})"
    )


if __name__ == "__main__":
    main()
