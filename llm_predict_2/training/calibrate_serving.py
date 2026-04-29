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
    from llm_predict_2.cache_aware import (
        TurnCacheFeature,
        aggregate_turn_cache_feature,
        weighted_median,
    )
    from llm_predict_2.composer import Composer
    from llm_predict_2.configs.model_configs import MODEL_CONFIGS, get_model
    from llm_predict_2.framework_corrections import (
        decode_correction_factor,
        moe_decode_correction_factor,
        ttft_queue_factor,
        ttft_validation_scope,
    )
    from llm_predict_2.serving import predict_serving
    from llm_predict_2.serving_calibration import clear_calibration_cache
    from llm_predict_2.validate import _HW_MAP, _actual_isl_osl, _resolve_model
else:
    from ..cache_aware import (
        TurnCacheFeature,
        aggregate_turn_cache_feature,
        weighted_median,
    )
    from ..composer import Composer
    from ..configs.model_configs import MODEL_CONFIGS, get_model
    from ..framework_corrections import (
        decode_correction_factor,
        moe_decode_correction_factor,
        ttft_queue_factor,
        ttft_validation_scope,
    )
    from ..serving import predict_serving
    from ..serving_calibration import clear_calibration_cache
    from ..validate import _HW_MAP, _actual_isl_osl, _resolve_model


ROOT = Path(__file__).resolve().parents[2]
BENCH_DATA = ROOT / "inference-benchmark" / "dashboard" / "public" / "data.json"
OUTPUT_JSON = ROOT / "llm_predict_2" / "data" / "serving_calibration.json"
OUTPUT_REPORT = ROOT / "llm_predict_2" / "data" / "serving_calibration_report.md"

SINGLE_GPU_HARDWARE = {"H100", "A100", "RTX3090", "RTX2080Ti"}
FULL_PREFILL_PROFILES = {
    "chat-singleturn",
    "prefill-heavy",
    "decode-heavy",
    "random-1k",
}
LEGACY_EXCLUDED_PROFILES = {"chat-short", "chat-medium", "chat-long"}
APPLY_CALIBRATION_STATUSES = {"high_confidence", "medium_confidence"}


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


def _fit_alpha_beta(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Robust 1D fit: grid alpha, median residual beta, minimize MAPE."""
    if not points:
        return 1.0, 0.0, 0.0

    best: tuple[float, float, float] | None = None
    for i in range(0, 301):
        alpha = i / 100.0
        beta = _median([meas - alpha * kernel for kernel, meas in points])
        err = _mape([(alpha * kernel + beta, meas) for kernel, meas in points])
        if best is None or err < best[0]:
            best = (err, alpha, beta)
    assert best is not None
    err, alpha, beta = best
    return alpha, beta, err


def _status(n: int, n_profiles: int, n_long: int, fit_mape: float,
            is_moe: bool = False) -> str:
    if is_moe:
        if n >= 4 and n_profiles >= 3 and fit_mape <= 30:
            return "medium_confidence"
        return "low_confidence"
    if n >= 4 and n_profiles >= 3 and n_long >= 1 and fit_mape <= 15:
        return "high_confidence"
    if n >= 3 and n_profiles >= 2 and fit_mape <= 20:
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


def _factor_entry(values: list[float]) -> dict[str, Any]:
    return {
        "factor": round(_median(values), 4),
        "n": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _calibration_applies(calibration: dict[str, Any]) -> bool:
    return calibration.get("calibration_status") in APPLY_CALIBRATION_STATUSES


def _factor_from_table(table: dict[str, Any] | None,
                       concurrency: int,
                       key: str = "factor",
                       default: float = 1.0) -> float:
    if not table:
        return default
    parsed: dict[int, float] = {}
    for raw_conc, value in table.items():
        try:
            conc = int(raw_conc)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            factor = value.get(key)
        else:
            factor = value if key == "factor" else None
        if factor is not None:
            parsed[conc] = float(factor)
    if not parsed:
        return default
    points = sorted(parsed)
    if concurrency <= points[0]:
        return parsed[points[0]]
    if concurrency >= points[-1]:
        return parsed[points[-1]]
    for left, right in zip(points, points[1:]):
        if left <= concurrency <= right:
            span = right - left
            weight = (concurrency - left) / span if span else 0.0
            return parsed[left] + weight * (parsed[right] - parsed[left])
    return default


def _decode_factor_from_calibration(calibration: dict[str, Any],
                                    record: Record) -> float:
    if not _calibration_applies(calibration):
        return 1.0
    if record.is_moe:
        model_profiles = calibration.get("moe_decode_factors_by_profile", {}).get(
            record.model, {}
        )
        if isinstance(model_profiles, dict):
            profile_table = model_profiles.get(record.profile)
            if profile_table:
                return _factor_from_table(profile_table, record.concurrency)
        model_block = calibration.get("moe_decode_factors", {}).get(record.model, {})
        if model_block:
            return _factor_from_table(model_block, record.concurrency)
        factor, _ = moe_decode_correction_factor(
            record.gpu, record.backend, record.backend_version,
            record.model, record.concurrency, record.profile
        )
        return factor
    profile_table = calibration.get("decode_factors_by_profile", {}).get(
        record.profile
    )
    if profile_table:
        return _factor_from_table(profile_table, record.concurrency)
    decode_table = calibration.get("decode_factors", {})
    if decode_table:
        return _factor_from_table(decode_table, record.concurrency)
    factor, _ = decode_correction_factor(
        record.gpu, record.backend, record.concurrency,
        record.backend_version, record.model, record.profile
    )
    return factor


def _queue_factor_from_calibration(calibration: dict[str, Any],
                                   record: Record) -> float:
    if not _calibration_applies(calibration):
        return 1.0
    profile_table = calibration.get("ttft_queue_factors_by_profile", {}).get(
        record.profile
    )
    if profile_table:
        return _factor_from_table(profile_table, record.concurrency)
    queue_table = calibration.get("ttft_queue_factors", {})
    if queue_table:
        return _factor_from_table(queue_table, record.concurrency)
    factor, _ = ttft_queue_factor(
        record.gpu, record.backend, record.concurrency,
        record.backend_version, record.model, record.profile
    )
    return factor


def _prefix_factor_from_calibration(calibration: dict[str, Any],
                                    record: Record,
                                    key: str) -> float:
    if not _calibration_applies(calibration):
        return 1.0
    profile_block = calibration.get("prefix_cache_factors_by_profile", {}).get(
        record.profile, {}
    )
    return _factor_from_table(profile_block, record.concurrency, key=key)


def _error_pct(predicted: float, measured: float | None) -> float | None:
    if measured is None or measured <= 0 or predicted <= 0:
        return None
    return abs(predicted - measured) / measured * 100.0


def _full_prefill_prediction(record: Record,
                             calibration: dict[str, Any]) -> dict[str, float]:
    ttft = calibration["ttft_correction"]
    alpha = float(ttft["alpha"])
    beta = float(ttft["beta_ms"])
    queue_factor = _queue_factor_from_calibration(calibration, record)
    decode_factor = _decode_factor_from_calibration(calibration, record)
    ttft_ms = (alpha * record.ttft_kernel_ms + beta) * queue_factor
    tpot_ms = record.tpot_raw_ms * decode_factor
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2el_ms": ttft_ms + tpot_ms * record.osl,
        "total_context_tokens": float(record.isl),
        "new_prefill_tokens": float(record.isl),
        "cache_hit_rate": 0.0,
    }


def _feature_from_prior(record: Record,
                        calibration: dict[str, Any]) -> TurnCacheFeature | None:
    profile_block = calibration.get("prefix_cache_priors_by_profile", {}).get(
        record.profile
    )
    if not isinstance(profile_block, dict):
        return None
    new_tokens = profile_block.get("new_prefill_tokens")
    if new_tokens is None:
        new_fraction = profile_block.get("new_prefill_fraction")
        if new_fraction is not None:
            new_tokens = round(record.isl * float(new_fraction))
    if new_tokens is None:
        return None
    new_tokens = max(1, min(int(round(float(new_tokens))), record.isl))
    cached_tokens = max(0, record.isl - new_tokens)
    return TurnCacheFeature(
        turn_index=-1,
        successful=1,
        total_context_tokens=record.isl,
        new_prefill_tokens=new_tokens,
        cached_context_tokens=cached_tokens,
        cache_hit_rate=cached_tokens / max(1, record.isl),
        output_tokens=max(1, record.osl),
    )


def _infer_new_prefill_tokens(record: Record,
                              calibration: dict[str, Any],
                              composer: Composer) -> int | None:
    if not record.ttft_meas or record.ttft_meas <= 0:
        return None
    ttft = calibration["ttft_correction"]
    alpha = float(ttft["alpha"])
    beta = float(ttft["beta_ms"])
    queue_factor = _queue_factor_from_calibration(calibration, record)
    cfg = get_model(record.model)

    def predicted(tokens: int) -> float:
        kernel = composer.predict_ttft_ms(cfg, tokens, kv_len=record.isl)
        return (alpha * kernel + beta) * queue_factor

    lo = 1
    hi = max(1, record.isl)
    for _ in range(24):
        mid = (lo + hi) // 2
        if predicted(mid) < record.ttft_meas:
            lo = mid + 1
        else:
            hi = mid
    candidates = range(max(1, lo - 4), min(record.isl, lo + 4) + 1)
    best = min(candidates, key=lambda tokens: abs(predicted(tokens) - record.ttft_meas))
    return int(best)


def _fit_prefix_cache_priors(records: list[Record],
                             calibrations: list[dict[str, Any]]) -> None:
    by_cal = {
        (c["gpu"], c["backend"], c["backend_version"], c["model"]): c
        for c in calibrations
    }
    composers: dict[str, Composer] = {}
    values: dict[tuple[str, str, str, str], dict[str, list[dict[str, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        if record.per_turn:
            continue
        if record.concurrency != 1:
            continue
        if ttft_validation_scope(record.profile, record.mode) != "prefix_cache_affected":
            continue
        calibration = by_cal.get(_group_key(record))
        if not calibration or not _calibration_applies(calibration):
            continue
        composer = composers.setdefault(record.gpu, Composer(record.gpu))
        new_tokens = _infer_new_prefill_tokens(record, calibration, composer)
        if new_tokens is None:
            continue
        values[_group_key(record)][record.profile].append({
            "new_prefill_tokens": float(new_tokens),
            "new_prefill_fraction": float(new_tokens) / max(1, record.isl),
            "total_context_tokens": float(record.isl),
            "cache_hit_rate": max(0.0, (record.isl - new_tokens) / max(1, record.isl)),
        })

    for calibration in calibrations:
        key = (
            calibration["gpu"], calibration["backend"],
            calibration["backend_version"], calibration["model"],
        )
        profile_values = values.get(key)
        if not profile_values:
            continue
        calibration["prefix_cache_priors_by_profile"] = {
            profile: {
                "n": len(items),
                "new_prefill_tokens": round(_median([
                    item["new_prefill_tokens"] for item in items
                ]), 1),
                "new_prefill_fraction": round(_median([
                    item["new_prefill_fraction"] for item in items
                ]), 4),
                "median_total_context_tokens": round(_median([
                    item["total_context_tokens"] for item in items
                ]), 1),
                "median_cache_hit_rate": round(_median([
                    item["cache_hit_rate"] for item in items
                ]), 4),
                "fit_source": "c1_ttft_inverse_no_per_turn",
            }
            for profile, items in sorted(profile_values.items())
        }


def _cache_aware_prediction(record: Record,
                            calibration: dict[str, Any],
                            composers: dict[str, Composer],
                            apply_prefix_contention: bool,
                            raw_cache: dict[tuple[Any, ...], Any] | None = None) -> dict[str, float] | None:
    feature = aggregate_turn_cache_feature(record.per_turn)
    if feature is None:
        feature = _feature_from_prior(record, calibration)
    if feature is None:
        return None

    ttft = calibration["ttft_correction"]
    alpha = float(ttft["alpha"])
    beta = float(ttft["beta_ms"])
    queue_factor = _queue_factor_from_calibration(calibration, record)
    decode_factor = _decode_factor_from_calibration(calibration, record)
    prefix_ttft = (
        _prefix_factor_from_calibration(calibration, record, "ttft_factor")
        if apply_prefix_contention else 1.0
    )
    prefix_decode = (
        _prefix_factor_from_calibration(calibration, record, "decode_factor")
        if apply_prefix_contention else 1.0
    )

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
            apply_prefix_contention=False,
        )
        if raw_cache is not None:
            raw_cache[cache_key] = raw_pred
    ttft_ms = (alpha * raw_pred.ttft_kernel_ms + beta) * queue_factor
    tpot_ms = raw_pred.tpot_ms * decode_factor
    if apply_prefix_contention:
        ttft_ms *= prefix_ttft
        tpot_ms *= prefix_decode
    e2el_ms = ttft_ms + tpot_ms * feature.output_tokens
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
        alpha, beta, fit_mape = _fit_alpha_beta(points)
        n_profiles = len({r.profile for r in c1_rows})
        n_long = sum(1 for r in c1_rows if r.isl >= 1000)
        status = _status(len(c1_rows), n_profiles, n_long, fit_mape, rows[0].is_moe)

        queue_values: dict[int, list[float]] = defaultdict(list)
        decode_values: dict[int, list[float]] = defaultdict(list)
        moe_values: dict[int, list[float]] = defaultdict(list)
        queue_values_by_profile: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        decode_values_by_profile: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        moe_values_by_profile: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            corrected_base = alpha * row.ttft_kernel_ms + beta
            if (not row.is_moe and row.ttft_meas and corrected_base > 0
                    and row.ttft_meas > 0):
                if row.concurrency == 1:
                    queue_values[row.concurrency].append(1.0)
                    queue_values_by_profile[row.profile][row.concurrency].append(1.0)
                else:
                    factor = row.ttft_meas / corrected_base
                    queue_values[row.concurrency].append(factor)
                    queue_values_by_profile[row.profile][row.concurrency].append(factor)

            if row.tpot_meas and row.tpot_meas > 0 and row.tpot_raw_ms > 0:
                if row.is_moe:
                    factor = row.tpot_meas / row.tpot_raw_ms
                    moe_values[row.concurrency].append(factor)
                    moe_values_by_profile[row.profile][row.concurrency].append(factor)
                else:
                    factor = row.tpot_meas / row.tpot_raw_ms
                    decode_values[row.concurrency].append(factor)
                    decode_values_by_profile[row.profile][row.concurrency].append(factor)

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
            "ttft_correction": {
                "alpha": round(alpha, 4),
                "beta_ms": round(beta, 4),
                "fit_mape": round(fit_mape, 2),
            },
            "notes": (
                f"fit excludes legacy profiles {sorted(LEGACY_EXCLUDED_PROFILES)}; "
                f"status={status}"
            ),
        }

        if queue_values:
            calibration["ttft_queue_factors"] = {
                str(conc): _factor_entry(values)
                for conc, values in sorted(queue_values.items())
            }
            calibration["ttft_queue_factors_by_profile"] = {
                profile: {
                    str(conc): _factor_entry(values)
                    for conc, values in sorted(profile_values.items())
                }
                for profile, profile_values in sorted(queue_values_by_profile.items())
            }
        if decode_values:
            calibration["decode_factors"] = {
                str(conc): _factor_entry(values)
                for conc, values in sorted(decode_values.items())
            }
            calibration["decode_factors_by_profile"] = {
                profile: {
                    str(conc): _factor_entry(values)
                    for conc, values in sorted(profile_values.items())
                }
                for profile, profile_values in sorted(decode_values_by_profile.items())
            }
        if moe_values:
            calibration["moe_decode_factors"] = {
                model: {
                    str(conc): _factor_entry(values)
                    for conc, values in sorted(moe_values.items())
                }
            }
            calibration["moe_decode_factors_by_profile"] = {
                model: {
                    profile: {
                        str(conc): _factor_entry(values)
                        for conc, values in sorted(profile_values.items())
                    }
                    for profile, profile_values in sorted(moe_values_by_profile.items())
                }
            }

        calibrations.append(calibration)
    return calibrations


def _prefix_entry(ttft_values: list[float],
                  decode_values: list[float],
                  meta: list[dict[str, float]]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "n": max(len(ttft_values), len(decode_values)),
        "ttft_factor": round(_median(ttft_values), 4) if ttft_values else 1.0,
        "decode_factor": round(_median(decode_values), 4) if decode_values else 1.0,
    }
    if ttft_values:
        entry["ttft_min"] = round(min(ttft_values), 4)
        entry["ttft_max"] = round(max(ttft_values), 4)
    if decode_values:
        entry["decode_min"] = round(min(decode_values), 4)
        entry["decode_max"] = round(max(decode_values), 4)
    if meta:
        entry["median_cache_hit_rate"] = round(_median([m["cache_hit_rate"] for m in meta]), 4)
        entry["median_total_context_tokens"] = round(_median([m["total_context_tokens"] for m in meta]), 1)
        entry["median_new_prefill_tokens"] = round(_median([m["new_prefill_tokens"] for m in meta]), 1)
    return entry


def _fit_prefix_cache_factors(records: list[Record],
                              calibrations: list[dict[str, Any]]) -> None:
    by_cal = {
        (c["gpu"], c["backend"], c["backend_version"], c["model"]): c
        for c in calibrations
    }
    composers: dict[str, Composer] = {}
    values: dict[tuple[str, str, str, str], dict[str, dict[int, dict[str, list[Any]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "ttft": [],
            "decode": [],
            "meta": [],
        })))
    )
    raw_cache: dict[tuple[Any, ...], Any] = {}

    for record in records:
        if ttft_validation_scope(record.profile, record.mode) != "prefix_cache_affected":
            continue
        calibration = by_cal.get(_group_key(record))
        if not calibration:
            continue
        cache_pred = _cache_aware_prediction(
            record, calibration, composers, apply_prefix_contention=False,
            raw_cache=raw_cache,
        )
        if not cache_pred:
            continue

        bucket = values[_group_key(record)][record.profile][record.concurrency]
        if record.ttft_meas and record.ttft_meas > 0 and cache_pred["ttft_ms"] > 0:
            bucket["ttft"].append(record.ttft_meas / cache_pred["ttft_ms"])
        if record.tpot_meas and record.tpot_meas > 0 and cache_pred["tpot_ms"] > 0:
            bucket["decode"].append(record.tpot_meas / cache_pred["tpot_ms"])
        bucket["meta"].append({
            "cache_hit_rate": cache_pred["cache_hit_rate"],
            "total_context_tokens": cache_pred["total_context_tokens"],
            "new_prefill_tokens": cache_pred["new_prefill_tokens"],
        })

    for calibration in calibrations:
        key = (
            calibration["gpu"], calibration["backend"],
            calibration["backend_version"], calibration["model"],
        )
        profile_values = values.get(key)
        if not profile_values:
            continue
        calibration["prefix_cache_factors_by_profile"] = {
            profile: {
                str(concurrency): _prefix_entry(
                    bucket["ttft"], bucket["decode"], bucket["meta"]
                )
                for concurrency, bucket in sorted(concurrency_values.items())
            }
            for profile, concurrency_values in sorted(profile_values.items())
        }


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
        full_pred = _full_prefill_prediction(record, cal)
        cache_raw = _cache_aware_prediction(
            record, cal, composers, apply_prefix_contention=False,
            raw_cache=raw_cache,
        )
        cache_cal = _cache_aware_prediction(
            record, cal, composers, apply_prefix_contention=True,
            raw_cache=raw_cache,
        )
        if not cache_raw or not cache_cal:
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
            cal_err = _error_pct(cache_cal[f"{metric}_ms"], measured)
            if full_err is not None:
                row[f"full_{metric}_err"] = full_err
            if raw_err is not None:
                row[f"cache_raw_{metric}_err"] = raw_err
            if cal_err is not None:
                row[f"cache_cal_{metric}_err"] = cal_err
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
            "cache_contention_ttft_mape": med_field("cache_cal_ttft_err"),
            "full_prefill_tpot_mape": med_field("full_tpot_err"),
            "cache_raw_tpot_mape": med_field("cache_raw_tpot_err"),
            "cache_contention_tpot_mape": med_field("cache_cal_tpot_err"),
            "full_prefill_e2el_mape": med_field("full_e2el_err"),
            "cache_raw_e2el_mape": med_field("cache_raw_e2el_err"),
            "cache_contention_e2el_mape": med_field("cache_cal_e2el_err"),
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
        "| GPU | Backend | Version | Model | Status | C=1 rows | Profiles | Long rows | TTFT fit MAPE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cal in payload["calibrations"]:
        ttft = cal["ttft_correction"]
        lines.append(
            f"| {cal['gpu']} | {cal['backend']} | {cal['backend_version']} | "
            f"{cal['model']} | {cal['calibration_status']} | {cal['n_c1']} | "
            f"{cal['n_profiles_c1']} | {cal['n_long_c1']} | {ttft['fit_mape']}% |"
        )

    lines += [
        "",
        "## Prefix Cache Multi-turn Summary",
        "",
        "| GPU | Backend | Model | Profile | Rows | Median ctx | Median new | Cache hit | Full E2EL | Cache raw E2EL | Cache+contention E2EL | TTFT raw→cal | TPOT raw→cal |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["prefix_cache_summary"]:
        lines.append(
            f"| {row['gpu']} | {row['backend']} | {row['model']} | {row['profile']} | "
            f"{row['n_rows']} | {row['median_full_ctx']} | {row['median_new_tokens']} | "
            f"{row['median_cache_hit_rate']} | {row['full_prefill_e2el_mape']}% | "
            f"{row['cache_raw_e2el_mape']}% | {row['cache_contention_e2el_mape']}% | "
            f"{row['cache_raw_ttft_mape']}%→{row['cache_contention_ttft_mape']}% | "
            f"{row['cache_raw_tpot_mape']}%→{row['cache_contention_tpot_mape']}% |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `low_confidence` calibrations are recorded for coverage visibility but are not applied by default.",
        "- Multi-turn TTFT should be evaluated against cache-aware TTFT, not cumulative full-prefill TTFT.",
        "- Prefix-cache rows without `perTurn` use a C=1 TTFT-inverted cache prior before contention factors are fitted.",
        "- Prefix-cache contention factors are fitted at GPU/backend/model/profile/concurrency granularity; this v1 is a fitted artifact, not a holdout cache-residency model.",
        "- MoE decode factors are model-specific because fused expert kernels do not follow dense GEMM timing.",
        "",
    ]
    path.write_text("\n".join(lines))


def calibrate(output_json: Path = OUTPUT_JSON,
              output_report: Path = OUTPUT_REPORT) -> dict[str, Any]:
    records = _load_records()
    calibrations = _fit_calibrations(records)
    _fit_prefix_cache_priors(records, calibrations)
    _fit_prefix_cache_factors(records, calibrations)
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
