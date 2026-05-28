"""Validate the three-regime TPOT predictor against measured ``tpot_meas``.

Reports:
- Per-profile median APE for: roofline, llm-d, two-roofline, three-regime
- Per-(profile, c tier) APE breakdown
- Regime **confusion matrix**: predicted regime (from workload) vs
  detected regime (from observed tpot_meas trajectory)
- Spot checks on the three motivating cells

Outputs:
  profiling/results/three_regime_validation.csv         per-(profile, c, turn)
  profiling/docs/three-regime-tpot-2026-05-28.md        verdict markdown
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT = Path("profiling/results/three_regime_validation.csv")
DEFAULT_REPORT = Path("profiling/docs/three-regime-tpot-2026-05-28.md")


PREDICTORS = [
    ("tpot_pred", "roofline"),
    ("tpot_pred_llm_d", "llm_d"),
    ("tpot_pred_two_roofline", "two_roofline"),
    ("tpot_pred_three_regime", "three_regime"),
]


def collect_rows(sim_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(sim_json.read_text())
    out: list[dict[str, Any]] = []
    for cell in payload.get("H100", []):
        prof = cell.get("profile")
        c = cell.get("concurrency")
        if prof is None or c is None:
            continue
        predicted_regime = (cell.get("tpot_three_regime_info") or {}).get("regime")
        for turn in cell.get("multiturn_turn_predictions") or []:
            tpot_meas = turn.get("tpot_meas")
            if not isinstance(tpot_meas, (int, float)) or tpot_meas <= 0:
                continue
            rec = {
                "profile": prof,
                "concurrency": int(c),
                "turn_index": int(turn.get("turn_index", -1)),
                "tpot_meas_ms": float(tpot_meas),
                "predicted_regime": predicted_regime,
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


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile", "concurrency", "turn_index", "tpot_meas_ms",
        "predicted_regime",
        *(key for key, _ in PREDICTORS),
        *(f"{name}_ape_pct" for _, name in PREDICTORS),
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = dict(r)
            for key, name in PREDICTORS:
                row_out[f"{name}_ape_pct"] = ape(r.get(key), r["tpot_meas_ms"])
            for k, v in list(row_out.items()):
                if isinstance(v, float) and not math.isnan(v):
                    row_out[k] = round(v, 3)
            w.writerow(row_out)


def detect_regime_from_meas(meas_series: list[float]) -> str:
    """Detect observed regime from the tpot_meas trajectory of one cell.

    FLAT       — peak / baseline < 1.5
    SATURATING — peak / baseline ≥ 1.5 AND last / peak > 0.6 (still elevated)
    PERTURBING — peak / baseline ≥ 1.5 AND last / peak ≤ 0.6 (recovered)
    """
    if not meas_series or len(meas_series) < 3:
        return "FLAT"
    baseline = statistics.median(meas_series[:3])
    peak = max(meas_series)
    if baseline <= 0:
        return "FLAT"
    if peak / baseline < 1.5:
        return "FLAT"
    last = meas_series[-1]
    if last / peak > 0.6:
        return "SATURATING"
    return "PERTURBING"


def per_profile_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_prof: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prof[r["profile"]].append(r)
    summary: dict[str, dict[str, float]] = {}
    for prof, prof_rows in sorted(by_prof.items()):
        prof_summary: dict[str, float] = {"n_turns": float(len(prof_rows))}
        for key, name in PREDICTORS:
            apes = [ape(r.get(key), r["tpot_meas_ms"]) for r in prof_rows]
            apes_f = [a for a in apes if a is not None and not math.isnan(a)]
            prof_summary[f"{name}_mape_pct"] = (
                sum(apes_f) / len(apes_f) if apes_f else float("nan")
            )
            prof_summary[f"{name}_median_ape_pct"] = median(apes_f)
            if apes_f:
                apes_sorted = sorted(apes_f)
                prof_summary[f"{name}_p90_ape_pct"] = apes_sorted[
                    min(len(apes_sorted) - 1, int(len(apes_sorted) * 0.9))
                ]
                prof_summary[f"{name}_max_ape_pct"] = max(apes_f)
            else:
                prof_summary[f"{name}_p90_ape_pct"] = float("nan")
                prof_summary[f"{name}_max_ape_pct"] = float("nan")
        summary[prof] = prof_summary
    return summary


def tier(c: int) -> str:
    if c <= 20:
        return "low (≤20)"
    if c <= 80:
        return "mid (21–80)"
    return "high (>80)"


def per_tier_summary(rows: list[dict]) -> list[tuple[str, str, int, dict[str, float]]]:
    by_pt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_pt[(r["profile"], tier(r["concurrency"]))].append(r)

    def tier_key(pt: tuple[str, str]) -> tuple[str, int]:
        order = ["low (≤20)", "mid (21–80)", "high (>80)"]
        return (pt[0], order.index(pt[1]))

    summary: list[tuple[str, str, int, dict[str, float]]] = []
    for pt in sorted(by_pt.keys(), key=tier_key):
        prof, t = pt
        rs = by_pt[pt]
        per: dict[str, float] = {}
        for key, name in PREDICTORS:
            apes = [ape(r.get(key), r["tpot_meas_ms"]) for r in rs]
            apes_f = [a for a in apes if a is not None]
            per[name] = median(apes_f) if apes_f else float("nan")
        summary.append((prof, t, len(rs), per))
    return summary


def regime_confusion_matrix(rows: list[dict]) -> dict[tuple[str, str], int]:
    """Predicted regime vs detected (from tpot_meas) per cell.

    One vote per (profile, c) cell — collapse all turns to the cell's
    detected regime.
    """
    # Group by (profile, c)
    cells: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["profile"], r["concurrency"])
        cells.setdefault(key, {"meas_series": [], "predicted_regime": r.get("predicted_regime")})
        cells[key]["meas_series"].append((r["turn_index"], r["tpot_meas_ms"]))

    matrix: dict[tuple[str, str], int] = defaultdict(int)
    for key, info in cells.items():
        series = [m for _, m in sorted(info["meas_series"])]
        detected = detect_regime_from_meas(series)
        predicted = info["predicted_regime"] or "?"
        matrix[(predicted, detected)] += 1
    return matrix


def write_report(rows: list[dict], summary: dict[str, dict[str, float]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Three-regime TPOT predictor — validation verdict")
    lines.append("")
    lines.append(
        "Per-profile MAPE (mean) and median APE across all turns × concurrencies. "
        "Three-regime uses no fitted constants — every input is `RooflineParams`, "
        "the empirically-confirmed `max_num_batched_tokens=8192`, per-turn "
        "workload, or the existing `tpot_pred` / `tpot_pred_llm_d` anchors. "
        "**MAPE is what the dashboard's TPOT MAPE badge displays**; median APE is "
        "robust to outliers (heavy tail on extreme cells)."
    )
    lines.append("")
    lines.append("## Per-profile MAPE (mean) — dashboard metric")
    lines.append("")
    lines.append("| profile | turns | roofline | llm-d | two-roofline | **three-regime** |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for prof, s in sorted(summary.items()):
        lines.append(
            f"| {prof} | {int(s['n_turns'])} | "
            f"{s['roofline_mape_pct']:.1f}% | "
            f"{s['llm_d_mape_pct']:.1f}% | "
            f"{s['two_roofline_mape_pct']:.1f}% | "
            f"**{s['three_regime_mape_pct']:.1f}%** |"
        )
    lines.append("")
    lines.append("## Per-profile median APE — tail-robust view")
    lines.append("")
    lines.append("| profile | turns | roofline | llm-d | two-roofline | **three-regime** |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for prof, s in sorted(summary.items()):
        lines.append(
            f"| {prof} | {int(s['n_turns'])} | "
            f"{s['roofline_median_ape_pct']:.1f}% | "
            f"{s['llm_d_median_ape_pct']:.1f}% | "
            f"{s['two_roofline_median_ape_pct']:.1f}% | "
            f"**{s['three_regime_median_ape_pct']:.1f}%** |"
        )
    lines.append("")
    lines.append("## Per-profile tail of the three-regime error distribution")
    lines.append("")
    lines.append("| profile | turns | median | MAPE | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for prof, s in sorted(summary.items()):
        lines.append(
            f"| {prof} | {int(s['n_turns'])} | "
            f"{s['three_regime_median_ape_pct']:.1f}% | "
            f"{s['three_regime_mape_pct']:.1f}% | "
            f"{s['three_regime_p90_ape_pct']:.1f}% | "
            f"{s['three_regime_max_ape_pct']:.1f}% |"
        )

    # Overall
    overall_apes: dict[str, list[float]] = {name: [] for _, name in PREDICTORS}
    for r in rows:
        for key, name in PREDICTORS:
            a = ape(r.get(key), r["tpot_meas_ms"])
            if a is not None:
                overall_apes[name].append(a)
    lines.append("")
    lines.append("**Overall (median across all 4 profiles):**")
    lines.append("")
    for _, name in PREDICTORS:
        lines.append(f"- `{name}`: median APE = {median(overall_apes[name]):.1f}%")
    lines.append("")

    # Per-tier
    lines.append("## Per-profile breakdown by concurrency tier")
    lines.append("")
    lines.append("| profile | c tier | turns | roofline | llm-d | two-roofline | **three-regime** |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for prof, t, n, per in per_tier_summary(rows):
        lines.append(
            f"| {prof} | {t} | {n} | "
            f"{per['roofline']:.1f}% | "
            f"{per['llm_d']:.1f}% | "
            f"{per['two_roofline']:.1f}% | "
            f"**{per['three_regime']:.1f}%** |"
        )

    # Regime confusion matrix
    lines.append("")
    lines.append("## Regime classification: predicted vs detected")
    lines.append("")
    lines.append(
        "Predicted regime (from physics + workload) vs detected regime "
        "(from `tpot_meas` trajectory: peak/baseline ≥ 1.5 and last/peak > 0.6 ⇒ saturating; "
        "≤ 0.6 ⇒ perturbing; otherwise flat). One vote per (profile, c) cell."
    )
    lines.append("")
    matrix = regime_confusion_matrix(rows)
    regimes = ["FLAT", "PERTURBING", "SATURATING"]
    lines.append("| predicted ↓ / detected → | FLAT | PERTURBING | SATURATING |")
    lines.append("|---|---:|---:|---:|")
    for pred in regimes:
        cells = " | ".join(str(matrix.get((pred, det), 0)) for det in regimes)
        lines.append(f"| **{pred}** | {cells} |")
    correct = sum(matrix.get((r, r), 0) for r in regimes)
    total = sum(matrix.values())
    lines.append("")
    lines.append(
        f"Diagonal (correct): {correct} / {total} = {100 * correct / max(1, total):.0f}%"
    )

    # Spot checks on motivating cells
    lines.append("")
    lines.append("## Spot checks (motivating cells)")
    lines.append("")
    lines.append("| profile | c | regime (predicted) | turn-0 obs / pred | turn-mid obs / pred | turn-last obs / pred |")
    lines.append("|---|---:|---|---|---|---|")
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["profile"], r["concurrency"])].append(r)
    spot = [
        ("chat-multiturn-synth", 5),
        ("chat-multiturn-synth", 320),
        ("osworld-multiturn-synth", 160),
        ("swebench-multiturn-synth", 80),
        ("terminalbench-multiturn-synth", 80),
    ]
    for cell_key in spot:
        rs = sorted(by_cell.get(cell_key, []), key=lambda r: r["turn_index"])
        if not rs:
            continue
        regime = rs[0].get("predicted_regime")
        mid = rs[len(rs) // 2]
        last = rs[-1]
        first = rs[0]
        lines.append(
            f"| {cell_key[0]} | {cell_key[1]} | {regime} | "
            f"{first['tpot_meas_ms']:.1f} / {first.get('tpot_pred_three_regime') or 0:.1f} | "
            f"{mid['tpot_meas_ms']:.1f} / {mid.get('tpot_pred_three_regime') or 0:.1f} | "
            f"{last['tpot_meas_ms']:.1f} / {last.get('tpot_pred_three_regime') or 0:.1f} |"
        )

    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "**Where three-regime wins**: cells where the regime classification "
        "matches reality. SATURATING shape (swe/terminal mid-high c) is "
        "captured by the linear T_min → T_max ramp from the predicted jump "
        "turn; FLAT cells reduce to the existing roofline (so identical to "
        "roofline APE in low-c regimes)."
    )
    lines.append("")
    lines.append(
        "**Documented limitations**:"
    )
    lines.append("")
    lines.append(
        "1. **osworld c≥160 misclassified as SATURATING**: the workload's "
        "`scheduled_requests` stays high enough that pressure ≥ 1 sustains in "
        "the model, but vLLM's chunked-prefill scheduler actually throttles "
        "admission, so observed `tpot_meas` recovers by turn 14+. Pure-forward "
        "prediction (workload only, no engine telemetry) can't capture vLLM "
        "throttle dynamics. This is the documented trade-off vs. the "
        "two-roofline predictor which uses `engine_max_decode_batch`."
    )
    lines.append("")
    lines.append(
        "2. **Perturbing-regime magnitude**: regime-2 cells (chat c=256/320, "
        "osworld c=80/120) spike to `T_max` at perturbation turns. Observed "
        "magnitudes are often smaller than `T_max` (e.g. chat c=320 turn 8 "
        "observed 50 ms vs predicted T_max ~28 ms — predicted ceiling capped "
        "by llm-d mean). Modeling spike magnitude < T_max needs an additional "
        "term; deferred."
    )
    lines.append("")
    lines.append(
        "3. **SATURATING ramp shape**: linear T_min → T_max across "
        "(jump_turn .. last_turn) was the simplest choice. Real curves often "
        "climb faster early, then plateau. A sigmoid ramp would fit better "
        "but adds tunable steepness."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator-predictions-json", type=Path, default=DEFAULT_SIM_PREDICTIONS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    rows = collect_rows(args.simulator_predictions_json)
    write_csv(rows, args.output)
    print(f"wrote {args.output}  ({len(rows)} rows)")
    summary = per_profile_summary(rows)
    write_report(rows, summary, args.report_output)
    print(f"wrote {args.report_output}")
    print()
    print(f"  {'profile':<35}  {'metric':<7}  {'roofline':>9}  {'llm-d':>9}  {'two-rfl':>9}  {'three-rg':>9}")
    for prof, s in sorted(summary.items()):
        for metric in ("mape", "median"):
            key = "mape_pct" if metric == "mape" else "median_ape_pct"
            print(
                f"  {prof:<35}  {metric:<7}  "
                f"{s[f'roofline_{key}']:>8.1f}%  "
                f"{s[f'llm_d_{key}']:>8.1f}%  "
                f"{s[f'two_roofline_{key}']:>8.1f}%  "
                f"{s[f'three_regime_{key}']:>8.1f}%"
            )


if __name__ == "__main__":
    main()
