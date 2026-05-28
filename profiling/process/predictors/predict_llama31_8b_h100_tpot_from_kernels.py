"""Predict Llama-3.1-8B H100 vLLM TPOT from profiled kernel components.

This is Experiment 3's first predictor shape. It composes kernel-profile inputs
and uses measured decode TPOT only as a validation target.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


DEFAULT_DECODE_PROFILE = Path(
    "profiling/results/decode_profile_H100_large_2026-05-17.csv"
)
DEFAULT_ATTENTION_PROFILE = Path(
    "profiling/results/ncu_flash_attention_H100_full_2026-05-18/"
    "flash_attention_ncu_summary.csv"
)
DEFAULT_GEMM_SUMMARY = Path(
    "profiling/results/ncu_decode_kernels_H100_gemm_full_2026-05-18/"
    "decode_gemm_ncu_summary.csv"
)
DEFAULT_SMALL_KERNEL_SUMMARY = Path(
    "profiling/results/ncu_decode_kernels_H100_fused_full_2026-05-18/"
    "decode_fused_kernels_ncu_summary.csv"
)
DEFAULT_TRACE_SUMMARY = Path(
    "profiling/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "profiling/results/llama31_8b_h100_kernel_composed_tpot_predictions.csv"
)
DEFAULT_REPORT = Path(
    "profiling/results/llama31_8b_h100_kernel_composed_tpot_report.md"
)


@dataclass(frozen=True)
class DecodeTarget:
    batch_size: int
    context_len: int
    measured_tpot_ms: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.batch_size, self.context_len)


@dataclass(frozen=True)
class LookupValue:
    value: float
    status: str


@dataclass(frozen=True)
class TraceComponents:
    decode_step_ms: float
    attention_ms: float
    gemm_linear_ms: float
    small_kernel_ms: float


@dataclass(frozen=True)
class PredictionRow:
    batch_size: int
    context_len: int
    measured_tpot_ms: float
    pred_tpot_ms: float
    pct_error: float
    attention_ms: float
    gemm_linear_ms: float
    small_kernel_ms: float
    runtime_residual_ms: float
    attention_status: str
    gemm_status: str
    small_kernel_status: str
    trace_decode_step_ms: float | None
    trace_attention_ms: float | None
    trace_gemm_linear_ms: float | None
    trace_small_kernel_ms: float | None
    diagnostic_reason: str


@dataclass(frozen=True)
class ErrorSummary:
    rows: int
    mape: float
    median_ape: float
    max_ape: float


class LogSpaceBatchTable:
    def __init__(self, values: dict[int, float]):
        if not values:
            raise ValueError("batch table requires at least one value")
        self._values = dict(values)
        self._batches = sorted(values)

    def lookup(self, batch_size: int) -> LookupValue:
        exact = self._values.get(batch_size)
        if exact is not None:
            return LookupValue(exact, "exact")

        lower = [batch for batch in self._batches if batch <= batch_size]
        upper = [batch for batch in self._batches if batch >= batch_size]
        if lower and upper:
            low = max(lower)
            high = min(upper)
            return LookupValue(
                _lerp_log(batch_size, low, high, self._values[low], self._values[high]),
                "log_interpolated",
            )

        nearest = min(
            self._batches,
            key=lambda batch: abs(math.log2(batch) - math.log2(batch_size)),
        )
        return LookupValue(self._values[nearest], f"nearest_batch_{nearest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-profile", type=Path, default=DEFAULT_DECODE_PROFILE)
    parser.add_argument("--attention-profile", type=Path, default=DEFAULT_ATTENTION_PROFILE)
    parser.add_argument("--gemm-summary", type=Path, default=DEFAULT_GEMM_SUMMARY)
    parser.add_argument(
        "--small-kernel-summary",
        type=Path,
        default=DEFAULT_SMALL_KERNEL_SUMMARY,
    )
    parser.add_argument(
        "--include-diagnostic-small-kernels",
        action="store_true",
        help=(
            "Include rows marked diagnostic/unknown in the small-kernel "
            "summary. By default only source_of_truth rows are composed."
        ),
    )
    parser.add_argument("--trace-summary", type=Path, default=DEFAULT_TRACE_SUMMARY)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--runtime-residual-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_decode_targets(path: Path, gpu: str = "H100") -> list[DecodeTarget]:
    rows: list[DecodeTarget] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_gpu = (row.get("gpu") or gpu).strip()
            if row_gpu and row_gpu != gpu:
                continue
            rows.append(
                DecodeTarget(
                    batch_size=int(row["batch_size"]),
                    context_len=int(row["context_len"]),
                    measured_tpot_ms=float(row["decode_step_ms"]),
                )
            )
    return sorted(rows, key=lambda row: row.key)


def load_attention_profile(path: Path, gpu: str = "H100") -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_gpu = (row.get("gpu") or gpu).strip()
            if row_gpu and row_gpu != gpu:
                continue
            key = (int(row["batch_size"]), int(row["context_len"]))
            values[key] = _kernel_row_ms(
                row,
                direct_columns=(
                    "ncu_flash_full_model_ms_sum",
                    "attention_ms",
                    "flash_full_model_ms_median",
                    "flash_full_model_ms",
                ),
            )
    return values


def load_gemm_summary(path: Path) -> dict[int, float]:
    totals: dict[int, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            batch_size = int(row["batch_size"])
            totals[batch_size] = totals.get(batch_size, 0.0) + _kernel_row_ms(row)
    return totals


def load_small_kernel_summary(
    path: Path | None,
    *,
    include_diagnostic: bool = False,
) -> dict[tuple[int, int], float]:
    if path is None or not path.exists():
        return {}

    totals: dict[tuple[int, int], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                not include_diagnostic
                and "source_status" in row
                and row["source_status"] != "source_of_truth"
            ):
                continue
            key = (int(row["batch_size"]), int(row["context_len"]))
            totals[key] = totals.get(key, 0.0) + _kernel_row_ms(row)
    return totals


def load_trace_summary(path: Path | None) -> dict[tuple[int, int], TraceComponents]:
    if path is None or not path.exists():
        return {}

    traces: dict[tuple[int, int], TraceComponents] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(row["batch_size"]), int(row["context_len"]))
            decode_step_ms = float(row["decode_step_ms"])
            attention_ms = decode_step_ms * float(row["attention_pct"]) / 100.0
            gemm_ms = decode_step_ms * float(row["gemm_linear_pct"]) / 100.0
            small_ms = decode_step_ms * (
                float(row["norm_residual_pct"])
                + float(row["ffn_activation_pct"])
                + float(row["kv_rope_cache_pct"])
                + float(row["sampling_logits_pct"])
                + float(row["other_pct"])
            ) / 100.0
            traces[key] = TraceComponents(
                decode_step_ms=decode_step_ms,
                attention_ms=attention_ms,
                gemm_linear_ms=gemm_ms,
                small_kernel_ms=small_ms,
            )
    return traces


def build_prediction_rows(
    targets: Sequence[DecodeTarget],
    *,
    attention_ms_by_key: dict[tuple[int, int], float],
    gemm_ms_by_batch: dict[int, float],
    small_ms_by_key: dict[tuple[int, int], float] | None = None,
    trace_by_key: dict[tuple[int, int], TraceComponents] | None = None,
    runtime_residual_ms: float = 0.0,
) -> list[PredictionRow]:
    gemm_table = LogSpaceBatchTable(gemm_ms_by_batch)
    small_ms_by_key = small_ms_by_key or {}
    trace_by_key = trace_by_key or {}

    rows = []
    for target in targets:
        diagnostics = []
        attention_ms = attention_ms_by_key.get(target.key)
        attention_status = "exact"
        if attention_ms is None:
            attention_ms = 0.0
            attention_status = "missing_zero"
            diagnostics.append("missing_attention")

        gemm_lookup = gemm_table.lookup(target.batch_size)
        small_ms = small_ms_by_key.get(target.key)
        small_status = "exact"
        if small_ms is None:
            small_ms = 0.0
            small_status = "missing_zero"
            diagnostics.append("missing_small_kernel")

        pred = attention_ms + gemm_lookup.value + small_ms + runtime_residual_ms
        trace = trace_by_key.get(target.key)
        rows.append(
            PredictionRow(
                batch_size=target.batch_size,
                context_len=target.context_len,
                measured_tpot_ms=target.measured_tpot_ms,
                pred_tpot_ms=pred,
                pct_error=percent_error(target.measured_tpot_ms, pred),
                attention_ms=attention_ms,
                gemm_linear_ms=gemm_lookup.value,
                small_kernel_ms=small_ms,
                runtime_residual_ms=runtime_residual_ms,
                attention_status=attention_status,
                gemm_status=gemm_lookup.status,
                small_kernel_status=small_status,
                trace_decode_step_ms=trace.decode_step_ms if trace else None,
                trace_attention_ms=trace.attention_ms if trace else None,
                trace_gemm_linear_ms=trace.gemm_linear_ms if trace else None,
                trace_small_kernel_ms=trace.small_kernel_ms if trace else None,
                diagnostic_reason=";".join(diagnostics),
            )
        )
    return rows


def write_prediction_csv(path: Path, rows: Sequence[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_size",
        "context_len",
        "measured_tpot_ms",
        "pred_tpot_ms",
        "pct_error",
        "attention_ms",
        "gemm_linear_ms",
        "small_kernel_ms",
        "runtime_residual_ms",
        "attention_status",
        "gemm_status",
        "small_kernel_status",
        "trace_decode_step_ms",
        "trace_attention_ms",
        "trace_gemm_linear_ms",
        "trace_small_kernel_ms",
        "diagnostic_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "batch_size": row.batch_size,
                "context_len": row.context_len,
                "measured_tpot_ms": _fmt(row.measured_tpot_ms),
                "pred_tpot_ms": _fmt(row.pred_tpot_ms),
                "pct_error": _fmt(row.pct_error),
                "attention_ms": _fmt(row.attention_ms),
                "gemm_linear_ms": _fmt(row.gemm_linear_ms),
                "small_kernel_ms": _fmt(row.small_kernel_ms),
                "runtime_residual_ms": _fmt(row.runtime_residual_ms),
                "attention_status": row.attention_status,
                "gemm_status": row.gemm_status,
                "small_kernel_status": row.small_kernel_status,
                "trace_decode_step_ms": _fmt_optional(row.trace_decode_step_ms),
                "trace_attention_ms": _fmt_optional(row.trace_attention_ms),
                "trace_gemm_linear_ms": _fmt_optional(row.trace_gemm_linear_ms),
                "trace_small_kernel_ms": _fmt_optional(row.trace_small_kernel_ms),
                "diagnostic_reason": row.diagnostic_reason,
            })


def write_report(
    path: Path,
    *,
    rows: Sequence[PredictionRow],
    decode_profile_path: Path,
    attention_profile_path: Path,
    gemm_summary_path: Path,
    small_kernel_summary_path: Path,
) -> None:
    summary = summarize_errors(rows)
    trace_summary = summarize_trace_errors(rows)
    worst = sorted(rows, key=lambda row: row.pct_error, reverse=True)[:8]
    small_status_counts = _count_status(row.small_kernel_status for row in rows)
    lines = [
        "# Kernel-Composed Llama-3.1-8B H100 TPOT",
        "",
        "## Inputs",
        "",
        f"- Decode validation target: `{decode_profile_path}`",
        f"- Attention profile: `{attention_profile_path}`",
        f"- GEMM NCU summary: `{gemm_summary_path}`",
        f"- Small-kernel summary: `{small_kernel_summary_path}`",
        "",
        "Measured TPOT is used only as the validation target.",
        "",
        "## Error Summary",
        "",
        f"- Rows: `{summary.rows}`",
        f"- MAPE: `{summary.mape:.2f}%`",
        f"- Median APE: `{summary.median_ape:.2f}%`",
        f"- Max APE: `{summary.max_ape:.2f}%`",
        f"- Small-kernel status counts: `{small_status_counts}`",
        "",
    ]
    if trace_summary.rows:
        lines.extend([
            "## Trace Cross-Check",
            "",
            "The trace decode-step value is diagnostic only; the decode profile "
            "remains the validation target for this run.",
            "",
            f"- Rows with trace target: `{trace_summary.rows}`",
            f"- Pred vs trace MAPE: `{trace_summary.mape:.2f}%`",
            f"- Pred vs trace median APE: `{trace_summary.median_ape:.2f}%`",
            f"- Pred vs trace max APE: `{trace_summary.max_ape:.2f}%`",
            "",
        ])

    lines.extend([
        "## Worst Rows",
        "",
        "| B | T | measured | pred | APE | attention | GEMM | small | flags |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in worst:
        lines.append(
            "| "
            f"{row.batch_size} | {row.context_len} | "
            f"{row.measured_tpot_ms:.3f} | {row.pred_tpot_ms:.3f} | "
            f"{row.pct_error:.2f}% | {row.attention_ms:.3f} | "
            f"{row.gemm_linear_ms:.3f} | {row.small_kernel_ms:.3f} | "
            f"{row.diagnostic_reason or 'ok'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def summarize_errors(rows: Sequence[PredictionRow]) -> ErrorSummary:
    errors = [row.pct_error for row in rows if math.isfinite(row.pct_error)]
    if not errors:
        return ErrorSummary(rows=0, mape=math.nan, median_ape=math.nan, max_ape=math.nan)
    return ErrorSummary(
        rows=len(errors),
        mape=sum(errors) / len(errors),
        median_ape=median(errors),
        max_ape=max(errors),
    )


def summarize_trace_errors(rows: Sequence[PredictionRow]) -> ErrorSummary:
    errors = [
        percent_error(row.trace_decode_step_ms, row.pred_tpot_ms)
        for row in rows
        if row.trace_decode_step_ms is not None and row.trace_decode_step_ms != 0.0
    ]
    if not errors:
        return ErrorSummary(rows=0, mape=math.nan, median_ape=math.nan, max_ape=math.nan)
    return ErrorSummary(
        rows=len(errors),
        mape=sum(errors) / len(errors),
        median_ape=median(errors),
        max_ape=max(errors),
    )


def percent_error(actual: float, predicted: float) -> float:
    if actual == 0.0:
        raise ValueError("actual value must be non-zero")
    return abs(predicted - actual) / abs(actual) * 100.0


def _kernel_row_ms(
    row: dict[str, str],
    *,
    direct_columns: Iterable[str] = (),
) -> float:
    for column in direct_columns:
        value = row.get(column)
        if value:
            return float(value)

    if row.get("ncu_gpu_time_ms_sum"):
        calls = int(float(row.get("calls_per_decode_step") or 1))
        return float(row["ncu_gpu_time_ms_sum"]) * calls

    if row.get("decode_step_contribution_ms_median"):
        return float(row["decode_step_contribution_ms_median"])

    if row.get("latency_ms_median"):
        calls = int(float(row.get("calls_per_decode_step") or 1))
        return float(row["latency_ms_median"]) * calls

    raise KeyError(f"no supported latency columns in row: {sorted(row)}")


def _lerp_log(target: int, low: int, high: int, low_value: float, high_value: float) -> float:
    if low == high:
        return low_value
    x = math.log2(target)
    x0 = math.log2(low)
    x1 = math.log2(high)
    weight = (x - x0) / (x1 - x0)
    return low_value + weight * (high_value - low_value)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else _fmt(value)


def _count_status(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def main() -> None:
    args = parse_args()
    targets = load_decode_targets(args.decode_profile, gpu=args.gpu)
    if not targets:
        raise SystemExit(f"No decode validation rows found for gpu={args.gpu!r}")

    rows = build_prediction_rows(
        targets,
        attention_ms_by_key=load_attention_profile(args.attention_profile, gpu=args.gpu),
        gemm_ms_by_batch=load_gemm_summary(args.gemm_summary),
        small_ms_by_key=load_small_kernel_summary(
            args.small_kernel_summary,
            include_diagnostic=args.include_diagnostic_small_kernels,
        ),
        trace_by_key=load_trace_summary(args.trace_summary),
        runtime_residual_ms=args.runtime_residual_ms,
    )
    write_prediction_csv(args.output, rows)
    write_report(
        args.report_output,
        rows=rows,
        decode_profile_path=args.decode_profile,
        attention_profile_path=args.attention_profile,
        gemm_summary_path=args.gemm_summary,
        small_kernel_summary_path=args.small_kernel_summary,
    )

    summary = summarize_errors(rows)
    print(
        f"kernel-composed TPOT: rows={summary.rows} "
        f"MAPE={summary.mape:.2f}% median={summary.median_ape:.2f}% "
        f"max={summary.max_ape:.2f}%"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
