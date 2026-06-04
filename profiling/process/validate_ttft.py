"""Measurement gate for the forward TTFT predictor + E2EL composition.

Reports, from ``simulator-predictions.json``, the per-profile MAPE / median / p90 /
max APE for ``ttft_pred`` vs ``ttft_meas`` and ``e2el_pred`` vs ``e2el_meas``, across
all turns PLUS a **high-TTFT** slice (``ttft_meas > 200`` ms) — the saturated,
queue-dominated regime that is the TTFT analog of the TPOT plateau.

Console-only. Run after editing simulator/ttft_predict.py and re-running
augment_simulator_predictions_with_ttft.

Usage:
    python3 -m profiling.process.validate_ttft
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
HIGH_TTFT_MS = 200.0  # queue-dominated slice (TTFT analog of the TPOT plateau)

# (pred_key, meas_key, label) per metric. ``ttft_pred`` / ``e2el_pred`` are the HEADLINE forward
# closed-loop queue sim (the static M0 comparison was retired with the old augmenter pipeline).
METRICS = [
    ("ttft_pred", "ttft_meas", "ttft"),
    ("e2el_pred", "e2el_meas", "e2el"),
]


def collect_rows(sim_json: Path, gpu_key: str = "H100") -> list[dict[str, Any]]:
    payload = json.loads(sim_json.read_text())
    out: list[dict[str, Any]] = []
    for cell in payload.get(gpu_key, []):
        # The payload key may also hold other-model rows (gpt-oss, Qwen) run on the same
        # GPU/tp/engine; the gate is the kernel-calibrated Llama-3.1-8B config only.
        if cell.get("model") not in (None, "Llama-3.1-8B"):
            continue
        prof = cell.get("profile")
        c = cell.get("concurrency")
        if prof is None or c is None:
            continue
        for turn in cell.get("multiturn_turn_predictions") or []:
            ttft_meas = turn.get("ttft_meas")
            if not isinstance(ttft_meas, (int, float)) or ttft_meas <= 0:
                continue
            rec: dict[str, Any] = {
                "profile": prof,
                "concurrency": int(c),
                "turn_index": int(turn.get("turn_index", -1)),
                "ttft_meas": float(ttft_meas),
            }
            for pred_key, meas_key, _ in METRICS:
                pv = turn.get(pred_key)
                mv = turn.get(meas_key)
                rec[pred_key] = float(pv) if isinstance(pv, (int, float)) else None
                rec[meas_key] = float(mv) if isinstance(mv, (int, float)) else None
            out.append(rec)
    return out


def ape(pred: float | None, obs: float | None) -> float | None:
    if pred is None or obs is None or pred <= 0 or obs <= 0:
        return None
    return abs(pred - obs) / obs * 100.0


def _stats(apes: list[float | None]) -> dict[str, float]:
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


def _summary(rows: list[dict], pred_key: str, meas_key: str) -> dict[str, dict[str, float]]:
    by_prof: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prof[r["profile"]].append(r)
        by_prof["__overall__"].append(r)
    return {
        prof: _stats([ape(r.get(pred_key), r.get(meas_key)) for r in prof_rows])
        for prof, prof_rows in by_prof.items()
    }


def _print_table(title: str, summary: dict[str, dict[str, float]], metric_stat: str) -> None:
    print(f"\n=== {title} ({metric_stat}) ===")
    print(f"  {'profile':<32}{'n':>7}{'value':>10}")
    order = sorted(k for k in summary if k != "__overall__") + ["__overall__"]
    for prof in order:
        s = summary[prof]
        label = "OVERALL" if prof == "__overall__" else prof
        val = f"{s[metric_stat]:>9.1f}%" if not math.isnan(s[metric_stat]) else f"{'n/a':>10}"
        print(f"  {label:<32}{s['n']:>7}{val}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator-predictions-json", type=Path, default=DEFAULT_SIM_PREDICTIONS)
    ap.add_argument("--gpu-key", default="H100",
                    help="payload GPU key to gate (e.g. H100 [default, headline], A100, 3090, 2080ti)")
    args = ap.parse_args()

    rows = collect_rows(args.simulator_predictions_json, args.gpu_key)
    high = [r for r in rows if r["ttft_meas"] > HIGH_TTFT_MS]
    print(f"[{args.gpu_key}] loaded {len(rows)} turns ({len(high)} high-TTFT, ttft_meas>{HIGH_TTFT_MS:.0f}ms)")

    for pred_key, meas_key, label in METRICS:
        _print_table(f"{label.upper()} — ALL turns", _summary(rows, pred_key, meas_key), "mape")
        _print_table(f"{label.upper()} — ALL turns", _summary(rows, pred_key, meas_key), "median")
        _print_table(
            f"{label.upper()} — high-TTFT (ttft_meas>{HIGH_TTFT_MS:.0f}ms)",
            _summary(high, pred_key, meas_key),
            "mape",
        )

    ttft_ov = _summary(rows, "ttft_pred", "ttft_meas")["__overall__"]["mape"]
    e2el_ov = _summary(rows, "e2el_pred", "e2el_meas")["__overall__"]["mape"]
    print(f"\nGATE [{args.gpu_key}]  overall TTFT MAPE = {ttft_ov:.2f}%   overall E2EL MAPE = {e2el_ov:.2f}%  (headline = queue sim)")


if __name__ == "__main__":
    main()
