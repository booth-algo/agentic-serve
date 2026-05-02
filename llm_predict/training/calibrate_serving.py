"""Fit serving-system calibration from dashboard benchmark data.

This intentionally excludes legacy chat-short/chat-medium profiles. The
calibration target is current canonical single-turn/stress plus high
concurrency and multi-turn/cache-aware analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from llm_predict.cache_aware import (
        aggregate_turn_cache_feature,
        weighted_median,
    )
    from llm_predict.composer import Composer
    from llm_predict.configs.model_configs import MODEL_CONFIGS, get_model
    from llm_predict.framework_corrections import ttft_validation_scope
    from llm_predict.serving import predict_serving
    from llm_predict.serving_calibration import clear_calibration_cache
    from llm_predict.validate import _HW_MAP, _actual_isl_osl, _resolve_model
else:
    from ..cache_aware import (
        aggregate_turn_cache_feature,
        weighted_median,
    )
    from ..composer import Composer
    from ..configs.model_configs import MODEL_CONFIGS, get_model
    from ..framework_corrections import ttft_validation_scope
    from ..serving import predict_serving
    from ..serving_calibration import clear_calibration_cache
    from ..validate import _HW_MAP, _actual_isl_osl, _resolve_model


ROOT = Path(__file__).resolve().parents[2]
BENCH_DATA = ROOT / "inference-benchmark" / "dashboard" / "public" / "data.json"
OUTPUT_JSON = ROOT / "llm_predict" / "data" / "serving_calibration.json"
OUTPUT_REPORT = ROOT / "llm_predict" / "data" / "serving_calibration_report.md"

SINGLE_GPU_HARDWARE = {"H100", "A100", "RTX3090", "RTX2080Ti"}
FULL_PREFILL_PROFILES = {
    "chat-singleturn",
    "prefill-heavy",
    "decode-heavy",
    "random-1k",
}
LEGACY_EXCLUDED_PROFILES = {"chat-short", "chat-medium", "chat-long"}


@dataclass
class Record:
    gpu: str
    backend: str
    backend_version: str
    model: str
    is_moe: bool
    profile: str
    mode: str | None
    concurrency: int
    isl: int
    osl: int
    ttft_meas: float | None
    tpot_meas: float | None
    e2el_meas: float | None
    ttft_kernel_ms: float
    tpot_raw_ms: float
    decode_total_raw_ms: float
    per_turn: list[dict[str, Any]] | None


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mape(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    return sum(abs(pred - meas) / meas * 100 for pred, meas in values if meas > 0) / len(values)


def _status(n: int, n_profiles: int, n_long: int, ttft_mape: float,
            is_moe: bool = False) -> str:
    if is_moe:
        if n >= 4 and n_profiles >= 3 and ttft_mape <= 30:
            return "medium_confidence"
        return "low_confidence"
    if n >= 4 and n_profiles >= 3 and n_long >= 1 and ttft_mape <= 15:
        return "high_confidence"
    if n >= 3 and n_profiles >= 2 and ttft_mape <= 20:
        return "medium_confidence"
    return "low_confidence"


def _load_records() -> list[Record]:
    with open(BENCH_DATA) as f:
        data = json.load(f)

    composers: dict[str, Composer] = {}
    records: list[Record] = []
    for entry in data:
        cfg_block = entry.get("config", {})
        profile = cfg_block.get("profile")
        if profile in LEGACY_EXCLUDED_PROFILES:
            continue

        gpu = _HW_MAP.get(entry.get("hardware", ""), entry.get("hardware", ""))
        if gpu not in SINGLE_GPU_HARDWARE:
            continue

        model = _resolve_model(cfg_block.get("model", ""))
        if model is None or model not in MODEL_CONFIGS:
            continue

        summary = entry.get("summary", {})
        if summary.get("successful_requests", 0) <= 0:
            continue

        composer = composers.setdefault(gpu, Composer(gpu))
        model_cfg = get_model(model)
        isl, osl = _actual_isl_osl(summary)
        raw_pred = predict_serving(
            composer, model_cfg, gpu, isl, osl,
            int(cfg_block.get("concurrency", 1)),
            backend=None,
            model_key=model,
        )

        records.append(Record(
            gpu=gpu,
            backend=cfg_block.get("backend", ""),
            backend_version=entry.get("engineVersion", ""),
            model=model,
            is_moe=model_cfg.is_moe,
            profile=profile,
            mode=cfg_block.get("mode"),
            concurrency=int(cfg_block.get("concurrency", 1)),
            isl=isl,
            osl=osl,
            ttft_meas=summary.get("median_ttft_ms"),
            tpot_meas=summary.get("median_tpot_ms"),
            e2el_meas=summary.get("median_e2el_ms"),
            ttft_kernel_ms=raw_pred.ttft_kernel_ms,
            tpot_raw_ms=raw_pred.tpot_ms,
            decode_total_raw_ms=raw_pred.decode_total_ms,
            per_turn=entry.get("perTurn"),
        ))
    return records


def _group_key(record: Record) -> tuple[str, str, str, str]:
    return record.gpu, record.backend, record.backend_version, record.model


def _error_pct(predicted: float, measured: float | None) -> float | None:
    if measured is None or measured <= 0 or predicted <= 0:
        return None
    return abs(predicted - measured) / measured * 100.0


def _full_prefill_prediction(record: Record) -> dict[str, float]:
    ttft_ms = record.ttft_kernel_ms
    tpot_ms = record.tpot_raw_ms
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2el_ms": ttft_ms + record.decode_total_raw_ms,
        "total_context_tokens": float(record.isl),
        "new_prefill_tokens": float(record.isl),
        "cache_hit_rate": 0.0,
    }


def _cache_aware_prediction(record: Record,
                            composers: dict[str, Composer],
                            raw_cache: dict[tuple[Any, ...], Any] | None = None) -> dict[str, float] | None:
    feature = aggregate_turn_cache_feature(record.per_turn)
    if feature is None:
        return None

    composer = composers.setdefault(record.gpu, Composer(record.gpu))
    cfg = get_model(record.model)
    ttft_values: list[tuple[float, float]] = []
    tpot_values: list[tuple[float, float]] = []
    e2el_values: list[tuple[float, float]] = []
    ctx_values: list[tuple[float, float]] = []
    new_values: list[tuple[float, float]] = []
    hit_values: list[tuple[float, float]] = []

    cache_key = (
        record.gpu, record.model, record.concurrency,
        feature.total_context_tokens, feature.new_prefill_tokens,
        feature.output_tokens,
    )
    raw_pred = raw_cache.get(cache_key) if raw_cache is not None else None
    if raw_pred is None:
        raw_pred = predict_serving(
            composer, cfg, record.gpu,
            feature.total_context_tokens,
            feature.output_tokens,
            record.concurrency,
            backend=None,
            model_key=record.model,
            total_context_tokens=feature.total_context_tokens,
            new_prefill_tokens=feature.new_prefill_tokens,
            cached_context_tokens=feature.cached_context_tokens,
            cache_hit_rate=feature.cache_hit_rate,
        )
        if raw_cache is not None:
            raw_cache[cache_key] = raw_pred
    ttft_ms = raw_pred.ttft_kernel_ms
    tpot_ms = raw_pred.tpot_ms
    e2el_ms = ttft_ms + raw_pred.decode_total_ms
    weight = float(feature.successful)
    ttft_values.append((ttft_ms, weight))
    tpot_values.append((tpot_ms, weight))
    e2el_values.append((e2el_ms, weight))
    ctx_values.append((feature.total_context_tokens, weight))
    new_values.append((feature.new_prefill_tokens, weight))
    hit_values.append((feature.cache_hit_rate, weight))

    return {
        "ttft_ms": weighted_median(ttft_values),
        "tpot_ms": weighted_median(tpot_values),
        "e2el_ms": weighted_median(e2el_values),
        "total_context_tokens": weighted_median(ctx_values),
        "new_prefill_tokens": weighted_median(new_values),
        "cache_hit_rate": weighted_median(hit_values),
    }


def _fit_calibrations(records: list[Record]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        if record.profile not in FULL_PREFILL_PROFILES:
            continue
        by_key[_group_key(record)].append(record)

    calibrations: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        gpu, backend, version, model = key
        c1_rows = [
            r for r in rows
            if r.concurrency == 1 and r.ttft_meas and r.ttft_meas > 0
        ]
        if not c1_rows:
            continue

        points = [(r.ttft_kernel_ms, float(r.ttft_meas)) for r in c1_rows]
        ttft_mape = _mape(points)
        n_profiles = len({r.profile for r in c1_rows})
        n_long = sum(1 for r in c1_rows if r.isl >= 1000)
        status = _status(len(c1_rows), n_profiles, n_long, ttft_mape, rows[0].is_moe)

        tpot_points = [
            (r.tpot_raw_ms, float(r.tpot_meas))
            for r in rows
            if r.tpot_meas and r.tpot_meas > 0 and r.tpot_raw_ms > 0
        ]
        e2el_points = [
            (r.ttft_kernel_ms + r.decode_total_raw_ms, float(r.e2el_meas))
            for r in rows
            if r.e2el_meas and r.e2el_meas > 0 and r.ttft_kernel_ms > 0
        ]

        calibration: dict[str, Any] = {
            "gpu": gpu,
            "backend": backend,
            "backend_version": version,
            "model": model,
            "calibration_status": status,
            "fit_scope": "canonical_full_prefill_profiles_only",
            "profiles": sorted({r.profile for r in c1_rows}),
            "n_c1": len(c1_rows),
            "n_profiles_c1": n_profiles,
            "n_long_c1": n_long,
            "raw_ttft_mape": round(ttft_mape, 2),
            "raw_tpot_mape": round(_mape(tpot_points), 2),
        "raw_e2el_mape": round(_mape(e2el_points), 2),
            "notes": (
                f"fit excludes legacy profiles {sorted(LEGACY_EXCLUDED_PROFILES)}; "
                f"status={status}; artifact is diagnostic only"
            ),
        }

        calibrations.append(calibration)
    return calibrations


def _prefix_cache_summary(records: list[Record],
                          calibrations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cal = {
        (c["gpu"], c["backend"], c["backend_version"], c["model"]): c
        for c in calibrations
    }
    composers: dict[str, Composer] = {}
    raw_cache: dict[tuple[Any, ...], Any] = {}
    summaries: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)

    for record in records:
        if ttft_validation_scope(record.profile, record.mode) != "prefix_cache_affected":
            continue

        cal = by_cal.get(_group_key(record))
        if not cal:
            continue
        full_pred = _full_prefill_prediction(record)
        cache_raw = _cache_aware_prediction(
            record, composers,
            raw_cache=raw_cache,
        )
        if not cache_raw:
            continue

        row: dict[str, float] = {
            "full_ctx": cache_raw["total_context_tokens"],
            "new_tokens": cache_raw["new_prefill_tokens"],
            "cache_hit_rate": cache_raw["cache_hit_rate"],
        }
        for metric, measured in (
            ("ttft", record.ttft_meas),
            ("tpot", record.tpot_meas),
            ("e2el", record.e2el_meas),
        ):
            full_err = _error_pct(full_pred[f"{metric}_ms"], measured)
            raw_err = _error_pct(cache_raw[f"{metric}_ms"], measured)
            if full_err is not None:
                row[f"full_{metric}_err"] = full_err
            if raw_err is not None:
                row[f"cache_raw_{metric}_err"] = raw_err
        summaries[(record.gpu, record.backend, record.model, record.profile)].append(row)

    out = []
    for key, values in sorted(summaries.items()):
        gpu, backend, model, profile = key
        def med_field(field: str) -> float:
            return round(_median([v[field] for v in values if field in v]), 1)

        out.append({
            "gpu": gpu,
            "backend": backend,
            "model": model,
            "profile": profile,
            "n_rows": len(values),
            "median_full_ctx": round(_median([v["full_ctx"] for v in values]), 1),
            "median_new_tokens": round(_median([v["new_tokens"] for v in values]), 1),
            "median_cache_hit_rate": round(_median([v["cache_hit_rate"] for v in values]), 3),
            "full_prefill_ttft_mape": med_field("full_ttft_err"),
            "cache_raw_ttft_mape": med_field("cache_raw_ttft_err"),
            "full_prefill_tpot_mape": med_field("full_tpot_err"),
            "cache_raw_tpot_mape": med_field("cache_raw_tpot_err"),
            "full_prefill_e2el_mape": med_field("full_e2el_err"),
            "cache_raw_e2el_mape": med_field("cache_raw_e2el_err"),
        })
    return out


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Serving Calibration Report",
        "",
        "Calibration excludes legacy `chat-short`, `chat-medium`, and `chat-long`.",
        "The active scope is canonical single-turn/stress, high concurrency, and multi-turn cache analysis.",
        "",
        "## Calibration Coverage",
        "",
        "| GPU | Backend | Version | Model | Status | C=1 rows | Profiles | Long rows | Raw TTFT MAPE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cal in payload["calibrations"]:
        lines.append(
            f"| {cal['gpu']} | {cal['backend']} | {cal['backend_version']} | "
            f"{cal['model']} | {cal['calibration_status']} | {cal['n_c1']} | "
            f"{cal['n_profiles_c1']} | {cal['n_long_c1']} | {cal['raw_ttft_mape']}% |"
        )

    lines += [
        "",
        "## Prefix Cache Multi-turn Summary",
        "",
        "| GPU | Backend | Model | Profile | Rows | Median ctx | Median new | Cache hit | Full E2EL | Cache-aware E2EL | Cache-aware TTFT | Cache-aware TPOT |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["prefix_cache_summary"]:
        lines.append(
            f"| {row['gpu']} | {row['backend']} | {row['model']} | {row['profile']} | "
            f"{row['n_rows']} | {row['median_full_ctx']} | {row['median_new_tokens']} | "
            f"{row['median_cache_hit_rate']} | {row['full_prefill_e2el_mape']}% | "
            f"{row['cache_raw_e2el_mape']}% | {row['cache_raw_ttft_mape']}% | "
            f"{row['cache_raw_tpot_mape']}% |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Calibration artifacts are diagnostic only; serving predictions do not consume empirical multipliers.",
        "- Multi-turn TTFT should be evaluated against cache-aware TTFT, not cumulative full-prefill TTFT.",
        "- Prefix-cache rows without `perTurn` remain unsupported rather than using inferred cache state.",
        "- MoE decode gaps remain visible as raw analytical error until a kernel-level MoE model is added.",
        "",
    ]
    path.write_text("\n".join(lines))


def calibrate(output_json: Path = OUTPUT_JSON,
              output_report: Path = OUTPUT_REPORT) -> dict[str, Any]:
    records = _load_records()
    calibrations = _fit_calibrations(records)
    prefix_summary = _prefix_cache_summary(records, calibrations)
    payload = {
        "schema_version": 1,
        "source": str(BENCH_DATA),
        "excluded_profiles": sorted(LEGACY_EXCLUDED_PROFILES),
        "full_prefill_profiles": sorted(FULL_PREFILL_PROFILES),
        "calibrations": calibrations,
        "prefix_cache_summary": prefix_summary,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    _write_report(payload, output_report)
    clear_calibration_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit serving calibration artifact.")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()

    payload = calibrate(args.output_json, args.output_report)
    high = sum(c["calibration_status"] == "high_confidence" for c in payload["calibrations"])
    medium = sum(c["calibration_status"] == "medium_confidence" for c in payload["calibrations"])
    low = sum(c["calibration_status"] == "low_confidence" for c in payload["calibrations"])
    print(
        f"Wrote {args.output_json} and {args.output_report} "
        f"({high} high, {medium} medium, {low} low confidence calibrations)"
    )


if __name__ == "__main__":
    main()
