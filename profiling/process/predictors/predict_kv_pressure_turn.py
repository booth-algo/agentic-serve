"""Diagnose KV-capacity pressure by comparing the physics-predicted jump turn
against the data-detected jump turn.

The KV-pressure wave-factor in the closed-form predictor says:

    blocks_per_req(N) = ceil(ctx_mid(N) / cache_block_size)
    total_demand(N)   = concurrency × blocks_per_req(N)
    jump fires when total_demand(N) > available_kv_blocks

This script asks one question per (profile, concurrency) cell: at the turn where
the physics says KV exhausts, does the measured TPOT actually jump?

Success criterion: not aggregate MAPE, but whether the predicted jump turn
matches the data-detected jump turn within ±2 turns. If yes, the physics
explains the trend. If no, we know exactly which (profile, c) reveals the gap.

Inputs:
  - profiling/results/benchmark_per_request_llama31_8b_h100_vllm.json
    (per-turn workload distributions; ``per_turn`` buckets)
  - inference-benchmark/dashboard/public/simulator-predictions.json
    (ground-truth ``tpot_meas`` per turn under ``H100[*].multiturn_turn_predictions``)

Outputs:
  - profiling/results/kv_pressure_jump_analysis.csv
  - profiling/results/kv_pressure_jump_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402


DEFAULT_PER_REQUEST_JSON = Path(
    "profiling/results/benchmark_per_request_llama31_8b_h100_vllm.json"
)
DEFAULT_SIM_PREDICTIONS_JSON = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)
DEFAULT_ROOFLINE_PARAMS_JSON = Path(
    "profiling/data/roofline_params_H100_llama31_8b.json"
)
DEFAULT_OUTPUT_CSV = Path("profiling/results/kv_pressure_jump_analysis.csv")
DEFAULT_OUTPUT_MD = Path("profiling/results/kv_pressure_jump_analysis.md")


# ---------------------------------------------------------- input data types


@dataclass(frozen=True)
class TurnWorkload:
    turn_index: int
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float
    request_count: int


def load_per_request_workload(
    path: Path,
) -> dict[tuple[str, int], list[TurnWorkload]]:
    """Load per-(profile, concurrency) ordered turn workload from the sidecar.

    Returns a dict ``(profile, concurrency) -> [TurnWorkload sorted by turn_index]``.
    Means across the ``request_count`` per-request samples in each bucket.
    """
    payload = json.loads(Path(path).read_text())
    entries = payload.get("per_turn") if isinstance(payload, dict) else payload
    if not isinstance(entries, dict):
        return {}

    def _mean(field: str, bucket: dict) -> float:
        vals = bucket.get(field) or []
        if not isinstance(vals, list) or not vals:
            return 0.0
        return fmean(float(v) for v in vals)

    out: dict[tuple[str, int], list[TurnWorkload]] = {}
    for bucket in entries.values():
        try:
            profile = str(bucket["profile"])
            c = int(bucket["concurrency"])
            turn_index = int(bucket["turn_index"])
        except (KeyError, TypeError, ValueError):
            continue
        turn = TurnWorkload(
            turn_index=turn_index,
            cached_context_tokens=_mean("cached_context_tokens", bucket),
            new_prefill_tokens=_mean("new_prefill_tokens", bucket),
            output_tokens=_mean("output_tokens", bucket),
            request_count=int(bucket.get("request_count", 0) or 0),
        )
        out.setdefault((profile, c), []).append(turn)
    for k in out:
        out[k].sort(key=lambda t: t.turn_index)
    return out


def load_measured_tpot_series(
    path: Path,
) -> dict[tuple[str, int], list[tuple[int, float | None]]]:
    """Load per-(profile, concurrency) ordered tpot_meas series from the
    dashboard's simulator-predictions.json.

    Returns ``(profile, concurrency) -> [(turn_index, tpot_meas_ms_or_None)]``.
    """
    payload = json.loads(Path(path).read_text())
    rows: Iterable[dict] = []
    if isinstance(payload, dict):
        # The dashboard JSON nests rows under a hardware key (e.g. "H100").
        for value in payload.values():
            if isinstance(value, list):
                rows = value
                break

    out: dict[tuple[str, int], list[tuple[int, float | None]]] = {}
    for row in rows:
        try:
            profile = str(row["profile"])
            c = int(row["concurrency"])
        except (KeyError, TypeError, ValueError):
            continue
        mtp = row.get("multiturn_turn_predictions") or []
        if not isinstance(mtp, list):
            continue
        series: list[tuple[int, float | None]] = []
        for turn in mtp:
            if not isinstance(turn, dict):
                continue
            try:
                ti = int(turn["turn_index"])
            except (KeyError, TypeError, ValueError):
                continue
            meas = turn.get("tpot_meas")
            try:
                meas_v = float(meas) if meas is not None else None
            except (TypeError, ValueError):
                meas_v = None
            if meas_v is not None and not math.isfinite(meas_v):
                meas_v = None
            series.append((ti, meas_v))
        series.sort(key=lambda x: x[0])
        out[(profile, c)] = series
    return out


# ----------------------------------------------------- stage 1: prediction


@dataclass(frozen=True)
class JumpPrediction:
    predicted_jump_turn: int | None
    demand_at_jump: int | None
    available_blocks: int


def predict_jump_turn(
    workload: list[TurnWorkload],
    concurrency: int,
    params: RooflineParams,
) -> JumpPrediction:
    """Smallest turn N where ``c × per_session_blocks(N) > available_kv_blocks``.

    ctx_mid(N) = cached_context(N) + new_prefill(N) + 0.5 × output_tokens(N).
    Returns ``None`` for the turn if demand never exceeds capacity.
    """
    available = int(params.available_kv_blocks)
    block_size = max(1, int(params.cache_block_size))
    for turn in workload:
        ctx_mid = (
            turn.cached_context_tokens
            + turn.new_prefill_tokens
            + 0.5 * turn.output_tokens
        )
        if ctx_mid <= 0:
            continue
        blocks_per_req = max(1, math.ceil(ctx_mid / block_size))
        demand = concurrency * blocks_per_req
        if demand > available:
            return JumpPrediction(
                predicted_jump_turn=turn.turn_index,
                demand_at_jump=demand,
                available_blocks=available,
            )
    return JumpPrediction(
        predicted_jump_turn=None,
        demand_at_jump=None,
        available_blocks=available,
    )


# ------------------------------------------------------ stage 2: detection


@dataclass(frozen=True)
class JumpDetection:
    detected_jump_turn: int | None
    baseline_ms: float | None
    tpot_at_jump_ms: float | None
    tentative: bool  # True if last-turn jump with no persistence neighbor


def detect_jump_turn(
    series: list[tuple[int, float | None]],
    *,
    jump_threshold_ratio: float = 3.0,
    persistence_ratio: float = 2.0,
    baseline_n: int = 3,
) -> JumpDetection:
    """Find the smallest turn where TPOT jumps and stays jumped.

    baseline = median of the first ``baseline_n`` non-None values (by turn_index).
    Detection rule (for turn N, only N >= baseline_n considered):

        tpot[N]   >= jump_threshold_ratio × baseline   AND
        tpot[N+1] >= persistence_ratio    × baseline

    The persistence check protects against one-off spikes (e.g. GC stalls).
    If the last turn fires the threshold but has no neighbor, mark it
    ``tentative=True`` and report it anyway.
    """
    vals = [(ti, v) for ti, v in series if v is not None]
    if len(vals) < baseline_n + 1:
        return JumpDetection(None, None, None, False)
    head = [v for _, v in vals[:baseline_n]]
    baseline = float(median(head))
    if baseline <= 0:
        return JumpDetection(None, None, None, False)
    jump_floor = jump_threshold_ratio * baseline
    persist_floor = persistence_ratio * baseline
    # Iterate from index baseline_n (skip the baseline turns).
    for i in range(baseline_n, len(vals)):
        ti, v = vals[i]
        if v < jump_floor:
            continue
        # Need a neighbor to confirm persistence.
        if i + 1 < len(vals):
            _, nxt = vals[i + 1]
            if nxt >= persist_floor:
                return JumpDetection(
                    detected_jump_turn=ti,
                    baseline_ms=baseline,
                    tpot_at_jump_ms=v,
                    tentative=False,
                )
            # Doesn't persist — keep scanning for a later jump.
            continue
        # Last-turn jump with no neighbor — tentative.
        return JumpDetection(
            detected_jump_turn=ti,
            baseline_ms=baseline,
            tpot_at_jump_ms=v,
            tentative=True,
        )
    return JumpDetection(None, None, None, False)


# ----------------------------------------------------- stage 3: comparison


@dataclass(frozen=True)
class JumpComparison:
    profile: str
    concurrency: int
    predicted: JumpPrediction
    detected: JumpDetection

    @property
    def turn_delta(self) -> int | None:
        p, d = self.predicted.predicted_jump_turn, self.detected.detected_jump_turn
        if p is None or d is None:
            return None
        return p - d

    @property
    def category(self) -> str:
        p = self.predicted.predicted_jump_turn
        d = self.detected.detected_jump_turn
        if p is None and d is None:
            return "neither"
        if p is not None and d is None:
            return "false_positive"
        if p is None and d is not None:
            return "missed"
        return "matched"


def run_analysis(
    workloads: dict[tuple[str, int], list[TurnWorkload]],
    series_by_key: dict[tuple[str, int], list[tuple[int, float | None]]],
    params: RooflineParams,
    *,
    jump_threshold_ratio: float,
    persistence_ratio: float,
) -> list[JumpComparison]:
    out: list[JumpComparison] = []
    keys = sorted(set(workloads) & set(series_by_key))
    for key in keys:
        profile, c = key
        prediction = predict_jump_turn(workloads[key], c, params)
        detection = detect_jump_turn(
            series_by_key[key],
            jump_threshold_ratio=jump_threshold_ratio,
            persistence_ratio=persistence_ratio,
        )
        out.append(JumpComparison(profile, c, prediction, detection))
    return out


# ---------------------------------------------------------- output writers


def write_csv(path: Path, comparisons: list[JumpComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow([
            "profile",
            "concurrency",
            "predicted_jump_turn",
            "detected_jump_turn",
            "turn_delta",
            "category",
            "demand_at_jump_blocks",
            "available_blocks",
            "tpot_baseline_ms",
            "tpot_at_jump_ms",
            "tentative",
        ])
        for cmp in comparisons:
            w.writerow([
                cmp.profile,
                cmp.concurrency,
                cmp.predicted.predicted_jump_turn if cmp.predicted.predicted_jump_turn is not None else "",
                cmp.detected.detected_jump_turn if cmp.detected.detected_jump_turn is not None else "",
                cmp.turn_delta if cmp.turn_delta is not None else "",
                cmp.category,
                cmp.predicted.demand_at_jump if cmp.predicted.demand_at_jump is not None else "",
                cmp.predicted.available_blocks,
                f"{cmp.detected.baseline_ms:.2f}" if cmp.detected.baseline_ms is not None else "",
                f"{cmp.detected.tpot_at_jump_ms:.2f}" if cmp.detected.tpot_at_jump_ms is not None else "",
                "1" if cmp.detected.tentative else "0",
            ])


def write_report(path: Path, comparisons: list[JumpComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[JumpComparison]] = {}
    for cmp in comparisons:
        by_cat.setdefault(cmp.category, []).append(cmp)

    matched = by_cat.get("matched", [])
    within_2 = [c for c in matched if c.turn_delta is not None and abs(c.turn_delta) <= 2]
    n_total = len(comparisons)
    n_predictable = len(matched)
    n_neither = len(by_cat.get("neither", []))
    n_fp = len(by_cat.get("false_positive", []))
    n_missed = len(by_cat.get("missed", []))
    n_within_2 = len(within_2)

    lines: list[str] = []
    lines.append("# KV-Pressure Jump-Turn Diagnostic")
    lines.append("")
    lines.append(
        f"Compared physics-predicted vs data-detected KV-pressure jump turn "
        f"across **{n_total}** (profile, concurrency) cells."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if n_predictable > 0:
        lines.append(
            f"- Physics + data **both fire a jump** for **{n_predictable}** cells."
        )
        lines.append(
            f"- Of those, **{n_within_2}/{n_predictable}** match within ±2 turns "
            f"(success bar)."
        )
    else:
        lines.append("- No cells with both physics-predicted AND data-detected jump.")
    lines.append(f"- Neither fires (correctly identified low-pressure): **{n_neither}**.")
    lines.append(f"- Physics says jump, data doesn't (false positive): **{n_fp}**.")
    lines.append(f"- Data says jump, physics doesn't (missed): **{n_missed}**.")
    lines.append("")

    # Per-profile breakdown
    by_profile: dict[str, list[JumpComparison]] = {}
    for cmp in comparisons:
        by_profile.setdefault(cmp.profile, []).append(cmp)
    lines.append("## Per-profile breakdown")
    lines.append("")
    lines.append("| profile | total | matched | within ±2 | false_pos | missed | neither |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for prof in sorted(by_profile):
        rows = by_profile[prof]
        m = [r for r in rows if r.category == "matched"]
        w2 = [r for r in m if r.turn_delta is not None and abs(r.turn_delta) <= 2]
        fp = [r for r in rows if r.category == "false_positive"]
        ms = [r for r in rows if r.category == "missed"]
        ne = [r for r in rows if r.category == "neither"]
        lines.append(
            f"| {prof} | {len(rows)} | {len(m)} | {len(w2)} | "
            f"{len(fp)} | {len(ms)} | {len(ne)} |"
        )
    lines.append("")

    # Per-cell detail
    lines.append("## Per-cell detail")
    lines.append("")
    lines.append("| profile | c | predicted | detected | Δ | tpot_base→jump (ms) | demand vs cap (blocks) | category |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for cmp in sorted(comparisons, key=lambda c: (c.profile, c.concurrency)):
        p = cmp.predicted.predicted_jump_turn
        d = cmp.detected.detected_jump_turn
        delta = cmp.turn_delta
        base = cmp.detected.baseline_ms
        atj = cmp.detected.tpot_at_jump_ms
        demand = cmp.predicted.demand_at_jump
        avail = cmp.predicted.available_blocks
        cell_p = f"{p}" + (" (tentative)" if cmp.detected.tentative else "")
        cell_d = f"{d}" if d is not None else "—"
        cell_delta = f"{delta:+d}" if delta is not None else "—"
        cell_base = (
            f"{base:.1f} → {atj:.1f}"
            if base is not None and atj is not None
            else "—"
        )
        cell_demand = (
            f"{demand} / {avail}"
            if demand is not None
            else f"≤ {avail}"
        )
        lines.append(
            f"| {cmp.profile} | {cmp.concurrency} | {cell_p if p is not None else '—'} | "
            f"{cell_d} | {cell_delta} | {cell_base} | {cell_demand} | {cmp.category} |"
        )

    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-request-json",
        type=Path,
        default=DEFAULT_PER_REQUEST_JSON,
        help="Per-turn workload sidecar (cached/new_prefill/output_tokens per request).",
    )
    p.add_argument(
        "--simulator-predictions-json",
        type=Path,
        default=DEFAULT_SIM_PREDICTIONS_JSON,
        help="Dashboard predictions JSON containing multiturn_turn_predictions.tpot_meas.",
    )
    p.add_argument(
        "--roofline-params-json",
        type=Path,
        default=DEFAULT_ROOFLINE_PARAMS_JSON,
        help="Roofline params JSON (for available_kv_blocks + cache_block_size).",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--report-output", type=Path, default=DEFAULT_OUTPUT_MD)
    p.add_argument(
        "--jump-threshold-ratio",
        type=float,
        default=3.0,
        help=(
            "Detection: tpot_meas[N] >= ratio × baseline counts as a jump. "
            "Defaults to 3.0 to match the visually-dramatic KV-exhaustion jumps "
            "(wave_factor c/B_eff is typically ≥ 2.5× at full pressure)."
        ),
    )
    p.add_argument(
        "--persistence-ratio",
        type=float,
        default=2.0,
        help=(
            "Detection: tpot_meas[N+1] >= ratio × baseline required to confirm. "
            "Guards against single-turn spikes; defaults below jump-threshold-ratio."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workloads = load_per_request_workload(args.per_request_json)
    if not workloads:
        raise SystemExit(f"no per-turn workload found in {args.per_request_json}")
    series_by_key = load_measured_tpot_series(args.simulator_predictions_json)
    if not series_by_key:
        raise SystemExit(
            f"no measured TPOT series found in {args.simulator_predictions_json}"
        )
    if args.roofline_params_json.exists():
        params = RooflineParams.from_json(args.roofline_params_json)
    else:
        params = RooflineParams()
    comparisons = run_analysis(
        workloads,
        series_by_key,
        params,
        jump_threshold_ratio=args.jump_threshold_ratio,
        persistence_ratio=args.persistence_ratio,
    )
    if not comparisons:
        raise SystemExit("no overlapping (profile, concurrency) cells found")
    write_csv(args.output, comparisons)
    write_report(args.report_output, comparisons)
    print(f"matched {len(comparisons)} (profile, concurrency) cells")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
