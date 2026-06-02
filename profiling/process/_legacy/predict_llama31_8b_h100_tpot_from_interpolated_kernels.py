"""Predict benchmark-turn TPOT from interpolated per-kernel NCU components.

This experiment keeps the predictor fully kernel-composed.  XGBoost, when
selected, is used only to interpolate individual kernel timing tables.  Measured
benchmark TPOT is loaded only after component prediction, as the validation
target.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Protocol, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.process._legacy.predict_llama31_8b_h100_tpot_from_kernels import (  # noqa: E402
    DEFAULT_ATTENTION_PROFILE,
    DEFAULT_GEMM_SUMMARY,
    DEFAULT_SMALL_KERNEL_SUMMARY,
    _kernel_row_ms,
)


DEFAULT_BENCHMARK_TURNS = Path(
    "profile_data/results/benchmark_turns_llama31_8b_h100_vllm.csv"
)
DEFAULT_OUTPUT = Path(
    "profile_data/results/llama31_8b_h100_interpolated_kernel_tpot_predictions.csv"
)
DEFAULT_REPORT = Path(
    "profile_data/results/llama31_8b_h100_interpolated_kernel_tpot_report.md"
)


@dataclass(frozen=True)
class KernelSample:
    batch_size: int
    context_len: int | None
    value_ms: float


@dataclass(frozen=True)
class BenchmarkTarget:
    batch_size: int
    context_len: int
    measured_tpot_ms: float
    profile: str
    concurrency: int
    turn_index: int
    primary_eval: bool
    diagnostic_reason: str
    row: dict[str, str]


@dataclass(frozen=True)
class ComponentPrediction:
    value_ms: float
    status: str


@dataclass(frozen=True)
class TpotPredictionRow:
    component_model: str
    profile: str
    concurrency: int
    turn_index: int
    primary_eval: bool
    batch_size: int
    context_len: int
    measured_tpot_ms: float
    pred_tpot_ms: float
    pct_error: float
    signed_error_ms: float
    attention_ms: float
    gemm_ms: float
    small_kernel_ms: float
    runtime_residual_ms: float
    attention_status: str
    gemm_status: str
    small_kernel_status: str
    diagnostic_reason: str


@dataclass(frozen=True)
class ErrorSummary:
    rows: int
    mape: float
    median_ape: float
    max_ape: float
    mean_signed_error_ms: float
    median_signed_error_ms: float


class ComponentModel(Protocol):
    name: str

    def predict(self, batch_size: int, context_len: int | None = None) -> ComponentPrediction:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-turns", type=Path, default=DEFAULT_BENCHMARK_TURNS)
    parser.add_argument("--attention-profile", type=Path, default=DEFAULT_ATTENTION_PROFILE)
    parser.add_argument("--gemm-summary", type=Path, default=DEFAULT_GEMM_SUMMARY)
    parser.add_argument(
        "--small-kernel-summary",
        type=Path,
        default=DEFAULT_SMALL_KERNEL_SUMMARY,
    )
    parser.add_argument(
        "--component-model",
        choices=("log_interp", "xgboost", "both"),
        default="xgboost",
        help=(
            "Per-kernel interpolation model to evaluate. Use 'both' only for "
            "local diagnostics; dashboard artifacts use xgboost as the "
            "canonical path."
        ),
    )
    parser.add_argument(
        "--include-diagnostic-small-kernels",
        action="store_true",
        help="Include diagnostic Torch-reference small-kernel rows.",
    )
    parser.add_argument(
        "--runtime-residual-ms",
        type=float,
        default=0.0,
        help="Explicit reported residual added after kernel composition.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


class LogLinearKernelModel:
    name = "log_interp"

    def __init__(self, samples: Sequence[KernelSample]):
        if not samples:
            raise ValueError("component model requires at least one sample")
        self._samples = tuple(samples)
        self._is_2d = any(sample.context_len is not None for sample in samples)
        self._exact = {
            self._key(sample.batch_size, sample.context_len): sample.value_ms
            for sample in samples
        }
        self._b_min = min(sample.batch_size for sample in samples)
        self._b_max = max(sample.batch_size for sample in samples)
        contexts = [sample.context_len for sample in samples if sample.context_len is not None]
        self._t_min = min(contexts) if contexts else None
        self._t_max = max(contexts) if contexts else None

    def predict(self, batch_size: int, context_len: int | None = None) -> ComponentPrediction:
        key = self._key(batch_size, context_len if self._is_2d else None)
        exact = self._exact.get(key)
        if exact is not None:
            return ComponentPrediction(exact, "exact")
        value = (
            self._predict_2d(batch_size, context_len)
            if self._is_2d
            else self._predict_1d(batch_size)
        )
        status = "log_linear_interpolated"
        if self._outside_range(batch_size, context_len):
            status = "log_linear_extrapolated"
        return ComponentPrediction(max(0.0, value), status)

    def _predict_1d(self, batch_size: int) -> float:
        points = sorted(
            (_log2(sample.batch_size), sample.value_ms)
            for sample in self._samples
        )
        x = _log2(batch_size)
        if len(points) == 1:
            return points[0][1]
        if x <= points[0][0]:
            return _line(points[0], points[1], x)
        if x >= points[-1][0]:
            return _line(points[-2], points[-1], x)
        for low, high in zip(points, points[1:]):
            if low[0] <= x <= high[0]:
                return _line(low, high, x)
        return min(points, key=lambda point: abs(point[0] - x))[1]

    def _predict_2d(self, batch_size: int, context_len: int | None) -> float:
        if context_len is None:
            raise ValueError("2D component requires context_len")
        x = _log2(batch_size)
        y = _log2(context_len)
        rows = []
        targets = []
        for sample in _nearest_samples(self._samples, batch_size, context_len, k=8):
            sx = _log2(sample.batch_size)
            sy = _log2(sample.context_len or 1)
            rows.append([1.0, sx, sy])
            targets.append(sample.value_ms)
        coeffs, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
        return float(np.asarray([1.0, x, y]) @ coeffs)

    def _outside_range(self, batch_size: int, context_len: int | None) -> bool:
        if batch_size < self._b_min or batch_size > self._b_max:
            return True
        if self._is_2d:
            if context_len is None:
                return True
            return bool(
                self._t_min is not None
                and self._t_max is not None
                and (context_len < self._t_min or context_len > self._t_max)
            )
        return False

    @staticmethod
    def _key(batch_size: int, context_len: int | None) -> tuple[int, int | None]:
        return (batch_size, context_len)


class XGBoostKernelModel:
    name = "xgboost"

    def __init__(self, samples: Sequence[KernelSample]):
        if not samples:
            raise ValueError("component model requires at least one sample")
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is required for --component-model xgboost") from exc

        self._samples = tuple(samples)
        self._is_2d = any(sample.context_len is not None for sample in samples)
        self._exact = {
            self._key(sample.batch_size, sample.context_len): sample.value_ms
            for sample in samples
        }
        self._b_min = min(sample.batch_size for sample in samples)
        self._b_max = max(sample.batch_size for sample in samples)
        contexts = [sample.context_len for sample in samples if sample.context_len is not None]
        self._t_min = min(contexts) if contexts else None
        self._t_max = max(contexts) if contexts else None

        x_train = np.asarray([
            self._features(sample.batch_size, sample.context_len)
            for sample in samples
        ])
        y_train = np.asarray([math.log1p(max(0.0, sample.value_ms)) for sample in samples])
        self._model = XGBRegressor(
            booster="gbtree",
            objective="reg:squarederror",
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=0,
            n_jobs=1,
            verbosity=0,
        )
        self._model.fit(x_train, y_train)

    def predict(self, batch_size: int, context_len: int | None = None) -> ComponentPrediction:
        query_context = context_len if self._is_2d else None
        exact = self._exact.get(self._key(batch_size, query_context))
        if exact is not None:
            return ComponentPrediction(exact, "exact")
        pred = self._model.predict(np.asarray([self._features(batch_size, query_context)]))[0]
        status = "xgboost_interpolated"
        if self._outside_range(batch_size, query_context):
            status = "xgboost_extrapolated"
        return ComponentPrediction(max(0.0, math.expm1(float(pred))), status)

    def _features(self, batch_size: int, context_len: int | None) -> list[float]:
        b = _log2(batch_size)
        if not self._is_2d:
            return [b, float(batch_size)]
        if context_len is None:
            raise ValueError("2D component requires context_len")
        t = _log2(context_len)
        return [b, t, b * t, _log2(max(1, batch_size * context_len))]

    def _outside_range(self, batch_size: int, context_len: int | None) -> bool:
        if batch_size < self._b_min or batch_size > self._b_max:
            return True
        if self._is_2d:
            if context_len is None:
                return True
            return bool(
                self._t_min is not None
                and self._t_max is not None
                and (context_len < self._t_min or context_len > self._t_max)
            )
        return False

    @staticmethod
    def _key(batch_size: int, context_len: int | None) -> tuple[int, int | None]:
        return (batch_size, context_len)


def load_benchmark_targets(path: Path) -> list[BenchmarkTarget]:
    targets: list[BenchmarkTarget] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            targets.append(BenchmarkTarget(
                batch_size=int(row["batch_size"]),
                context_len=int(row["context_len"]),
                measured_tpot_ms=float(row["tpot_meas_ms"] or row["decode_step_ms"]),
                profile=row.get("profile", ""),
                concurrency=int(row.get("concurrency") or row["batch_size"]),
                turn_index=int(row.get("turn_index") or 0),
                primary_eval=_bool(row.get("primary_eval")),
                diagnostic_reason=row.get("diagnostic_reason", ""),
                row=dict(row),
            ))
    return targets


def load_attention_components(path: Path) -> dict[str, list[KernelSample]]:
    samples: list[KernelSample] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(KernelSample(
                batch_size=int(row["batch_size"]),
                context_len=int(row["context_len"]),
                value_ms=_kernel_row_ms(
                    row,
                    direct_columns=(
                        "ncu_flash_full_model_ms_sum",
                        "attention_ms",
                        "flash_full_model_ms_median",
                        "flash_full_model_ms",
                    ),
                ),
            ))
    return {"attention": samples}


def load_gemm_components(path: Path) -> dict[str, list[KernelSample]]:
    samples_by_op: dict[str, list[KernelSample]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            op = row["op_name"]
            samples_by_op.setdefault(op, []).append(KernelSample(
                batch_size=int(row["batch_size"]),
                context_len=None,
                value_ms=_kernel_row_ms(row),
            ))
    return samples_by_op


def load_small_kernel_components(
    path: Path,
    *,
    include_diagnostic: bool = False,
) -> dict[str, list[KernelSample]]:
    samples_by_kernel: dict[str, list[KernelSample]] = {}
    if not path.exists():
        return samples_by_kernel
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                not include_diagnostic
                and "source_status" in row
                and row["source_status"] != "source_of_truth"
            ):
                continue
            kernel_name = row["kernel_name"]
            samples_by_kernel.setdefault(kernel_name, []).append(KernelSample(
                batch_size=int(row["batch_size"]),
                context_len=int(row["context_len"]),
                value_ms=_kernel_row_ms(row),
            ))
    return samples_by_kernel


def build_component_models(
    components: dict[str, list[KernelSample]],
    *,
    model_name: str,
) -> dict[str, ComponentModel]:
    model_cls: type[ComponentModel]
    if model_name == "log_interp":
        model_cls = LogLinearKernelModel
    elif model_name == "xgboost":
        model_cls = XGBoostKernelModel
    else:
        raise ValueError(f"unknown component model: {model_name}")
    return {name: model_cls(samples) for name, samples in sorted(components.items())}


def build_prediction_rows(
    targets: Sequence[BenchmarkTarget],
    *,
    model_name: str,
    attention_models: dict[str, ComponentModel],
    gemm_models: dict[str, ComponentModel],
    small_kernel_models: dict[str, ComponentModel],
    runtime_residual_ms: float = 0.0,
) -> list[TpotPredictionRow]:
    rows: list[TpotPredictionRow] = []
    attention_model = attention_models["attention"]
    for target in targets:
        attention = attention_model.predict(target.batch_size, target.context_len)
        gemm = _sum_component_predictions(gemm_models, target.batch_size, None)
        small = _sum_component_predictions(
            small_kernel_models,
            target.batch_size,
            target.context_len,
        )
        pred = (
            attention.value_ms
            + gemm.value_ms
            + small.value_ms
            + runtime_residual_ms
        )
        rows.append(TpotPredictionRow(
            component_model=model_name,
            profile=target.profile,
            concurrency=target.concurrency,
            turn_index=target.turn_index,
            primary_eval=target.primary_eval,
            batch_size=target.batch_size,
            context_len=target.context_len,
            measured_tpot_ms=target.measured_tpot_ms,
            pred_tpot_ms=pred,
            pct_error=percent_error(target.measured_tpot_ms, pred),
            signed_error_ms=pred - target.measured_tpot_ms,
            attention_ms=attention.value_ms,
            gemm_ms=gemm.value_ms,
            small_kernel_ms=small.value_ms,
            runtime_residual_ms=runtime_residual_ms,
            attention_status=attention.status,
            gemm_status=gemm.status,
            small_kernel_status=small.status,
            diagnostic_reason=target.diagnostic_reason,
        ))
    return rows


def write_prediction_csv(path: Path, rows: Sequence[TpotPredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "component_model",
        "profile",
        "concurrency",
        "turn_index",
        "primary_eval",
        "batch_size",
        "context_len",
        "measured_tpot_ms",
        "pred_tpot_ms",
        "pct_error",
        "signed_error_ms",
        "attention_ms",
        "gemm_ms",
        "small_kernel_ms",
        "runtime_residual_ms",
        "attention_status",
        "gemm_status",
        "small_kernel_status",
        "diagnostic_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "component_model": row.component_model,
                "profile": row.profile,
                "concurrency": row.concurrency,
                "turn_index": row.turn_index,
                "primary_eval": "true" if row.primary_eval else "false",
                "batch_size": row.batch_size,
                "context_len": row.context_len,
                "measured_tpot_ms": _fmt(row.measured_tpot_ms),
                "pred_tpot_ms": _fmt(row.pred_tpot_ms),
                "pct_error": _fmt(row.pct_error),
                "signed_error_ms": _fmt(row.signed_error_ms),
                "attention_ms": _fmt(row.attention_ms),
                "gemm_ms": _fmt(row.gemm_ms),
                "small_kernel_ms": _fmt(row.small_kernel_ms),
                "runtime_residual_ms": _fmt(row.runtime_residual_ms),
                "attention_status": row.attention_status,
                "gemm_status": row.gemm_status,
                "small_kernel_status": row.small_kernel_status,
                "diagnostic_reason": row.diagnostic_reason,
            })


def write_report(
    path: Path,
    *,
    rows: Sequence[TpotPredictionRow],
    benchmark_turns: Path,
    attention_profile: Path,
    gemm_summary: Path,
    small_kernel_summary: Path,
    small_kernel_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Interpolated Kernel-Composed TPOT",
        "",
        "## Inputs",
        "",
        f"- Benchmark turns: `{benchmark_turns}`",
        f"- Attention NCU summary: `{attention_profile}`",
        f"- GEMM NCU summary: `{gemm_summary}`",
        f"- Small-kernel NCU summary: `{small_kernel_summary}`",
        "",
        "Measured benchmark TPOT is used only as the validation target. "
        "Component models are trained only on NCU kernel timing rows.",
        "",
        "## Error Summary",
        "",
        "| component model | slice | rows | MAPE | median APE | max APE | mean signed err | median signed err |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in sorted({row.component_model for row in rows}):
        model_rows = [row for row in rows if row.component_model == model_name]
        for label, subset in (
            ("primary", [row for row in model_rows if row.primary_eval]),
            ("all", model_rows),
            ("diagnostic", [row for row in model_rows if not row.primary_eval]),
        ):
            summary = summarize_errors(subset)
            lines.append(
                "| "
                f"{model_name} | {label} | {summary.rows} | "
                f"{summary.mape:.2f}% | {summary.median_ape:.2f}% | "
                f"{summary.max_ape:.2f}% | {summary.mean_signed_error_ms:.3f} | "
                f"{summary.median_signed_error_ms:.3f} |"
            )

    lines.extend([
        "",
        "## Component Models",
        "",
        f"- Small-kernel component models included: `{small_kernel_count}`",
        "- GEMM is modeled per projection family and summed.",
        "- Small kernels are modeled per kernel name and summed.",
        "- Runtime residual is explicit; no global TPOT residual model is trained.",
        "",
        "## Status Counts",
        "",
    ])
    for model_name in sorted({row.component_model for row in rows}):
        model_rows = [row for row in rows if row.component_model == model_name]
        lines.extend([
            f"### {model_name}",
            "",
            f"- Attention: `{_counter_text(row.attention_status for row in model_rows)}`",
            f"- GEMM: `{_counter_text(row.gemm_status for row in model_rows)}`",
            f"- Small kernels: `{_counter_text(row.small_kernel_status for row in model_rows)}`",
            "",
        ])

    lines.extend([
        "## Worst Primary Rows",
        "",
        "| model | profile | c | turn | B | T | measured | pred | APE | components |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    primary_rows = [row for row in rows if row.primary_eval]
    for row in sorted(primary_rows, key=lambda item: item.pct_error, reverse=True)[:12]:
        lines.append(
            "| "
            f"{row.component_model} | {row.profile} | {row.concurrency} | "
            f"{row.turn_index} | {row.batch_size} | {row.context_len} | "
            f"{row.measured_tpot_ms:.3f} | {row.pred_tpot_ms:.3f} | "
            f"{row.pct_error:.2f}% | "
            f"attn={row.attention_ms:.3f}, gemm={row.gemm_ms:.3f}, "
            f"small={row.small_kernel_ms:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def summarize_errors(rows: Sequence[TpotPredictionRow]) -> ErrorSummary:
    if not rows:
        return ErrorSummary(
            rows=0,
            mape=math.nan,
            median_ape=math.nan,
            max_ape=math.nan,
            mean_signed_error_ms=math.nan,
            median_signed_error_ms=math.nan,
        )
    errors = [row.pct_error for row in rows]
    signed = [row.signed_error_ms for row in rows]
    return ErrorSummary(
        rows=len(rows),
        mape=sum(errors) / len(errors),
        median_ape=median(errors),
        max_ape=max(errors),
        mean_signed_error_ms=sum(signed) / len(signed),
        median_signed_error_ms=median(signed),
    )


def percent_error(actual: float, predicted: float) -> float:
    if actual == 0.0:
        raise ValueError("actual value must be non-zero")
    return abs(predicted - actual) / abs(actual) * 100.0


def _sum_component_predictions(
    models: dict[str, ComponentModel],
    batch_size: int,
    context_len: int | None,
) -> ComponentPrediction:
    total = 0.0
    statuses = []
    for name, model in sorted(models.items()):
        pred = model.predict(batch_size, context_len)
        total += pred.value_ms
        statuses.append(f"{name}:{pred.status}")
    return ComponentPrediction(total, ";".join(statuses) if statuses else "none")


def _nearest_samples(
    samples: Sequence[KernelSample],
    batch_size: int,
    context_len: int,
    *,
    k: int,
) -> list[KernelSample]:
    b_values = [_log2(sample.batch_size) for sample in samples]
    t_values = [_log2(sample.context_len or 1) for sample in samples]
    b_scale = max(1e-9, max(b_values) - min(b_values))
    t_scale = max(1e-9, max(t_values) - min(t_values))
    b = _log2(batch_size)
    t = _log2(context_len)
    return sorted(
        samples,
        key=lambda sample: (
            ((_log2(sample.batch_size) - b) / b_scale) ** 2
            + ((_log2(sample.context_len or 1) - t) / t_scale) ** 2
        ),
    )[: min(k, len(samples))]


def _line(low: tuple[float, float], high: tuple[float, float], x: float) -> float:
    if high[0] == low[0]:
        return low[1]
    weight = (x - low[0]) / (high[0] - low[0])
    return low[1] + weight * (high[1] - low[1])


def _log2(value: int) -> float:
    return math.log2(max(1, int(value)))


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _counter_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _model_names(requested: str) -> list[str]:
    if requested == "both":
        return ["log_interp", "xgboost"]
    return [requested]


def main() -> None:
    args = parse_args()
    targets = load_benchmark_targets(args.benchmark_turns)
    attention_components = load_attention_components(args.attention_profile)
    gemm_components = load_gemm_components(args.gemm_summary)
    small_components = load_small_kernel_components(
        args.small_kernel_summary,
        include_diagnostic=args.include_diagnostic_small_kernels,
    )

    all_rows: list[TpotPredictionRow] = []
    for model_name in _model_names(args.component_model):
        rows = build_prediction_rows(
            targets,
            model_name=model_name,
            attention_models=build_component_models(
                attention_components,
                model_name=model_name,
            ),
            gemm_models=build_component_models(gemm_components, model_name=model_name),
            small_kernel_models=build_component_models(
                small_components,
                model_name=model_name,
            ),
            runtime_residual_ms=args.runtime_residual_ms,
        )
        all_rows.extend(rows)

    write_prediction_csv(args.output, all_rows)
    write_report(
        args.report_output,
        rows=all_rows,
        benchmark_turns=args.benchmark_turns,
        attention_profile=args.attention_profile,
        gemm_summary=args.gemm_summary,
        small_kernel_summary=args.small_kernel_summary,
        small_kernel_count=len(small_components),
    )

    for model_name in _model_names(args.component_model):
        model_rows = [row for row in all_rows if row.component_model == model_name]
        primary = summarize_errors([row for row in model_rows if row.primary_eval])
        all_summary = summarize_errors(model_rows)
        print(
            f"{model_name}: primary rows={primary.rows} "
            f"MAPE={primary.mape:.2f}% median={primary.median_ape:.2f}% "
            f"max={primary.max_ape:.2f}% | all rows={all_summary.rows} "
            f"MAPE={all_summary.mape:.2f}%"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
