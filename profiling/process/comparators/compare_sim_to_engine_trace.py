"""Compare simulator step counts vs real vLLM engine traces.

Reads ``profiling/results/vllm_engine_step_trace_*_summary.csv`` ground-truth
rows (one per (profile, concurrency, turn_index) triple) and joins them
against the simulator's per-row predictions in
``profiling/results/llama31_8b_h100_engine_step_tpot_predictions.csv``.

Emits a per-row diff CSV with absolute and percent deltas, plus a markdown
summary of the worst gaps.  This is the load-bearing yardstick for the
scheduler shape correction — every simulator change should reduce the diff
on the same set of trace-backed rows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TRACE_GLOB = "profiling/results/vllm_engine_step_trace_*_summary.csv"
DEFAULT_PREDICTIONS = Path(
    "profiling/results/llama31_8b_h100_engine_step_tpot_predictions.csv"
)
DEFAULT_OUTPUT = Path("profiling/results/sim_vs_engine_trace_diff.csv")
DEFAULT_REPORT = Path("profiling/results/sim_vs_engine_trace_diff_report.md")

# (trace column, sim column) — the metrics we compare.
_METRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("steps", "sim_steps"),
    ("decode_only_steps", "sim_decode_only_steps"),
    ("prefill_only_steps", "sim_prefill_only_steps"),
    ("mixed_decode_prefill_steps", "sim_mixed_decode_prefill_steps"),
    ("total_decode_slots", "sim_total_decode_slots"),
    ("total_prefill_tokens", "sim_total_prefill_tokens"),
    ("max_decode_batch", "sim_max_decode_batch"),
    ("mean_decode_batch", "sim_mean_decode_batch"),
    ("min_free_kv_blocks", "sim_min_free_kv_blocks"),
    ("total_preemptions", "sim_total_preemptions"),
)


@dataclass(frozen=True)
class RowDiff:
    profile: str
    concurrency: int
    turn_index: int
    trace_source: str
    metric: str
    real: float
    sim: float
    abs_delta: float
    pct_delta: float | None  # None when real is 0


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_truth_rows(glob_pattern: str) -> dict[tuple[str, int, int], dict]:
    """Index trace summary rows by (profile, concurrency, turn_index).

    When multiple traces cover the same triple, prefer the longer filename
    (typically the more specific / newer variant) — but we record the source
    on the diff row so a human can sanity-check.
    """
    out: dict[tuple[str, int, int], dict] = {}
    for path in sorted(glob.glob(glob_pattern)):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (
                        row["profile"],
                        int(row["concurrency"]),
                        int(row["turn_index"]),
                    )
                except (KeyError, ValueError):
                    continue
                existing = out.get(key)
                # Keep the row from the longest matching filename so
                # later-captured / more-specific traces win.
                if existing is None or len(path) > len(existing.get("_source", "")):
                    row["_source"] = path
                    out[key] = row
    return out


def load_sim_rows(path: Path) -> dict[tuple[str, int, int], dict]:
    out: dict[tuple[str, int, int], dict] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                key = (
                    row["profile"],
                    int(row["concurrency"]),
                    int(row["turn_index"]),
                )
            except (KeyError, ValueError):
                continue
            out[key] = row
    return out


def compute_diffs(
    truth: dict[tuple[str, int, int], dict],
    sim: dict[tuple[str, int, int], dict],
) -> list[RowDiff]:
    out: list[RowDiff] = []
    for key in sorted(truth.keys() & sim.keys()):
        truth_row = truth[key]
        sim_row = sim[key]
        for trace_col, sim_col in _METRIC_PAIRS:
            real_v = _to_float(truth_row.get(trace_col))
            sim_v = _to_float(sim_row.get(sim_col))
            if real_v is None or sim_v is None:
                continue
            abs_delta = sim_v - real_v
            pct = (abs_delta / real_v * 100.0) if real_v != 0 else None
            out.append(
                RowDiff(
                    profile=key[0],
                    concurrency=key[1],
                    turn_index=key[2],
                    trace_source=truth_row.get("_source", ""),
                    metric=trace_col,
                    real=real_v,
                    sim=sim_v,
                    abs_delta=abs_delta,
                    pct_delta=pct,
                )
            )
    return out


def write_diff_csv(path: Path, diffs: list[RowDiff]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile", "concurrency", "turn_index", "trace_source",
        "metric", "real", "sim", "abs_delta", "pct_delta",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for d in diffs:
            writer.writerow({
                "profile": d.profile,
                "concurrency": d.concurrency,
                "turn_index": d.turn_index,
                "trace_source": Path(d.trace_source).name if d.trace_source else "",
                "metric": d.metric,
                "real": f"{d.real:.6g}",
                "sim": f"{d.sim:.6g}",
                "abs_delta": f"{d.abs_delta:+.6g}",
                "pct_delta": f"{d.pct_delta:+.2f}" if d.pct_delta is not None else "",
            })


def _per_metric_stats(diffs: list[RowDiff]) -> dict[str, dict[str, float]]:
    by_metric: dict[str, list[RowDiff]] = {}
    for d in diffs:
        by_metric.setdefault(d.metric, []).append(d)
    out: dict[str, dict[str, float]] = {}
    for metric, rows in by_metric.items():
        abs_deltas = [r.abs_delta for r in rows]
        pcts = [r.pct_delta for r in rows if r.pct_delta is not None]
        out[metric] = {
            "n": len(rows),
            "mean_abs_delta": statistics.fmean(abs_deltas) if abs_deltas else 0.0,
            "median_abs_delta": statistics.median(abs_deltas) if abs_deltas else 0.0,
            "max_abs_delta": max(abs_deltas, key=abs, default=0.0),
            "mean_pct": statistics.fmean(pcts) if pcts else 0.0,
            "median_abs_pct": statistics.median(abs(p) for p in pcts) if pcts else 0.0,
            "max_abs_pct": max((abs(p) for p in pcts), default=0.0),
        }
    return out


def write_report(path: Path, diffs: list[RowDiff]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = _per_metric_stats(diffs)
    rows_by_key: dict[tuple[str, int, int], list[RowDiff]] = {}
    for d in diffs:
        rows_by_key.setdefault((d.profile, d.concurrency, d.turn_index), []).append(d)

    lines = [
        "# Simulator vs vLLM Engine Trace Diff",
        "",
        f"Compared **{len(rows_by_key)}** trace-backed (profile, concurrency, turn) "
        f"triples covering **{len(diffs)}** metric values.",
        "",
        "## Per-metric error",
        "",
        "| metric | n | mean abs Δ | median abs Δ | max abs Δ | mean Δ% | median abs Δ% | max abs Δ% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trace_col, _sim_col in _METRIC_PAIRS:
        s = stats.get(trace_col)
        if not s:
            continue
        lines.append(
            f"| {trace_col} | {int(s['n'])} | {s['mean_abs_delta']:+.2f} | "
            f"{s['median_abs_delta']:+.2f} | {s['max_abs_delta']:+.2f} | "
            f"{s['mean_pct']:+.2f}% | {s['median_abs_pct']:.2f}% | "
            f"{s['max_abs_pct']:.2f}% |"
        )

    lines.extend(["", "## Per-row breakdown", ""])
    for (profile, c, t), row_diffs in sorted(rows_by_key.items()):
        lines.append(f"### {profile} c={c} turn={t}")
        lines.append("")
        lines.append("| metric | real | sim | Δ | Δ% |")
        lines.append("|---|---:|---:|---:|---:|")
        for d in row_diffs:
            pct = f"{d.pct_delta:+.2f}%" if d.pct_delta is not None else "—"
            lines.append(
                f"| {d.metric} | {d.real:g} | {d.sim:g} | {d.abs_delta:+g} | {pct} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-glob", default=DEFAULT_TRACE_GLOB)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_truth_rows(args.trace_glob)
    sim = load_sim_rows(args.predictions)
    if not truth:
        raise SystemExit(f"no trace summary rows found via glob {args.trace_glob}")
    if not sim:
        raise SystemExit(f"no simulator predictions found at {args.predictions}")
    diffs = compute_diffs(truth, sim)
    if not diffs:
        raise SystemExit(
            "no overlapping (profile, concurrency, turn_index) rows between traces "
            "and simulator predictions"
        )
    write_diff_csv(args.output, diffs)
    write_report(args.report_output, diffs)
    print(
        f"matched {len({(d.profile, d.concurrency, d.turn_index) for d in diffs})} "
        f"trace-backed turns, {len(diffs)} metric values"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
