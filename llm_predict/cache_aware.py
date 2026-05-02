"""Cache-aware helpers for multi-turn serving prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .composer import Composer
from .configs.model_configs import ModelConfig
from .serving import ServingPrediction, decode_interval_count, predict_serving


def _optional_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return None


def _optional_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            return value_float
    return None


@dataclass(frozen=True)
class TurnCacheFeature:
    turn_index: int
    successful: int
    total_context_tokens: int
    new_prefill_tokens: int
    cached_context_tokens: int
    cache_hit_rate: float
    output_tokens: int


def derive_turn_cache_features(per_turn: list[dict[str, Any]] | None) -> list[TurnCacheFeature]:
    """Derive aggregate cache features from benchmark perTurn summaries."""
    if not per_turn:
        return []

    features: list[TurnCacheFeature] = []
    previous_context = 0
    turns = sorted(per_turn, key=lambda row: int(row.get("turn_index", 0)))
    for index, turn in enumerate(turns):
        total_context = _optional_int(
            turn, "median_input_tokens", "avg_input_tokens"
        ) or 0
        output_tokens = _optional_int(
            turn, "median_output_tokens", "avg_output_tokens"
        ) or 0
        successful = int(turn.get("successful", turn.get("num_requests", 1)) or 0)
        if total_context <= 0 or output_tokens <= 0 or successful <= 0:
            continue

        measured_new_tokens = _optional_int(
            turn, "median_new_prefill_tokens", "avg_new_prefill_tokens"
        )
        measured_cached_tokens = _optional_int(
            turn, "median_cached_context_tokens", "avg_cached_context_tokens"
        )
        measured_hit_rate = _optional_float(
            turn, "median_cache_hit_rate", "avg_cache_hit_rate"
        )

        new_tokens = (
            measured_new_tokens
            if measured_new_tokens is not None
            else max(1, total_context - previous_context)
        )
        new_tokens = min(new_tokens, total_context)
        cached_tokens = (
            measured_cached_tokens
            if measured_cached_tokens is not None
            else max(0, total_context - new_tokens)
        )
        cached_tokens = max(0, min(cached_tokens, total_context))
        hit_rate = (
            measured_hit_rate
            if measured_hit_rate is not None
            else cached_tokens / total_context
        )
        hit_rate = max(0.0, min(hit_rate, 1.0))
        features.append(TurnCacheFeature(
            turn_index=int(turn.get("turn_index", index)),
            successful=successful,
            total_context_tokens=total_context,
            new_prefill_tokens=new_tokens,
            cached_context_tokens=cached_tokens,
            cache_hit_rate=hit_rate,
            output_tokens=max(1, output_tokens),
        ))
        previous_context = total_context
    return features


def aggregate_turn_cache_feature(per_turn: list[dict[str, Any]] | None) -> TurnCacheFeature | None:
    """Return the successful-request-weighted representative cache turn."""
    features = derive_turn_cache_features(per_turn)
    if not features:
        return None
    weighted = [(feature, float(feature.successful)) for feature in features]
    total_context = int(round(weighted_median([
        (feature.total_context_tokens, weight) for feature, weight in weighted
    ])))
    new_tokens = int(round(weighted_median([
        (feature.new_prefill_tokens, weight) for feature, weight in weighted
    ])))
    output_tokens = int(round(weighted_median([
        (feature.output_tokens, weight) for feature, weight in weighted
    ])))
    new_tokens = max(1, min(new_tokens, total_context))
    cached_tokens = max(0, total_context - new_tokens)
    return TurnCacheFeature(
        turn_index=-1,
        successful=sum(feature.successful for feature in features),
        total_context_tokens=max(1, total_context),
        new_prefill_tokens=new_tokens,
        cached_context_tokens=cached_tokens,
        cache_hit_rate=cached_tokens / max(1, total_context),
        output_tokens=max(1, output_tokens),
    )


def weighted_median(values: list[tuple[float, float]]) -> float:
    clean = [
        (float(value), max(0.0, float(weight)))
        for value, weight in values
        if math.isfinite(float(value)) and math.isfinite(float(weight)) and weight > 0
    ]
    if not clean:
        return 0.0
    clean.sort(key=lambda item: item[0])
    total_weight = sum(weight for _, weight in clean)
    threshold = total_weight / 2.0
    cumulative = 0.0
    for index, (value, weight) in enumerate(clean):
        cumulative += weight
        if cumulative > threshold:
            return value
        if math.isclose(cumulative, threshold):
            if index + 1 < len(clean):
                return (value + clean[index + 1][0]) / 2.0
            return value
    return clean[-1][0]


def weighted_mean(values: list[tuple[float, float]]) -> float:
    clean = [
        (float(value), max(0.0, float(weight)))
        for value, weight in values
        if math.isfinite(float(value)) and math.isfinite(float(weight)) and weight > 0
    ]
    total_weight = sum(weight for _, weight in clean)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in clean) / total_weight


def _measured_error(predicted: float, measured: float | None) -> float | None:
    if measured is None or measured <= 0:
        return None
    if predicted <= 0:
        return None
    return round(abs(predicted - measured) / min(predicted, measured) * 100.0, 1)


def _round_optional(value: float | None, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def predict_multiturn_from_per_turn(
    composer: Composer,
    cfg: ModelConfig,
    gpu: str,
    per_turn: list[dict[str, Any]] | None,
    concurrency: int,
    tensor_parallel_size: int = 1,
    backend: str | None = None,
    backend_version: str | None = None,
    model_key: str | None = None,
    profile: str | None = None,
) -> ServingPrediction | None:
    features = derive_turn_cache_features(per_turn)
    if not features:
        return None

    raw_by_turn: dict[int, dict[str, Any]] = {}
    if per_turn:
        raw_by_turn = {
            int(row.get("turn_index", index)): row
            for index, row in enumerate(per_turn)
        }

    turn_rows: list[tuple[TurnCacheFeature, ServingPrediction]] = []
    for feature in features:
        pred = predict_serving(
            composer, cfg, gpu,
            feature.total_context_tokens,
            feature.output_tokens,
            max(1, feature.successful),
            tensor_parallel_size=tensor_parallel_size,
            backend=backend,
            backend_version=backend_version,
            model_key=model_key,
            profile=profile,
            total_context_tokens=feature.total_context_tokens,
            new_prefill_tokens=feature.new_prefill_tokens,
            cached_context_tokens=feature.cached_context_tokens,
            cache_hit_rate=feature.cache_hit_rate,
            cache_feature_source="per_turn",
        )
        turn_rows.append((feature, pred))

    total_successful = sum(feature.successful for feature, _ in turn_rows)
    if total_successful <= 0:
        return None

    weighted_decode_total = weighted_mean([
        (pred.decode_total_ms, feature.successful)
        for feature, pred in turn_rows
    ])
    output_token_weight = sum(
        decode_interval_count(feature.output_tokens) * feature.successful
        for feature, _ in turn_rows
    )
    ttft_ms = weighted_mean([
        (pred.ttft_ms, feature.successful)
        for feature, pred in turn_rows
    ])
    tpot_ms = (
        sum(pred.decode_total_ms * feature.successful for feature, pred in turn_rows)
        / max(1.0, float(output_token_weight))
    )

    first_pred = turn_rows[0][1]
    turn_predictions: list[dict[str, Any]] = []
    for feature, pred in turn_rows:
        raw_turn = raw_by_turn.get(feature.turn_index, {})
        measured_ttft = _optional_float(raw_turn, "median_ttft_ms")
        measured_tpot = _optional_float(raw_turn, "median_tpot_ms")
        measured_e2el = _optional_float(raw_turn, "median_e2el_ms")
        turn_row: dict[str, Any] = {
            "turn_index": feature.turn_index,
            "successful": feature.successful,
            "total_context_tokens": feature.total_context_tokens,
            "new_prefill_tokens": feature.new_prefill_tokens,
            "cached_context_tokens": feature.cached_context_tokens,
            "cache_hit_rate": round(feature.cache_hit_rate, 4),
            "output_tokens": feature.output_tokens,
            "ttft_pred": round(pred.ttft_ms, 2),
            "tpot_pred": round(pred.tpot_ms, 2),
            "e2el_pred": round(pred.e2el_ms, 2),
            "ttft_kernel_ms": round(pred.ttft_kernel_ms, 2),
            "ttft_floor_ms": round(pred.ttft_floor_ms, 2),
            "ttft_first_decode_ms": round(pred.ttft_first_decode_ms, 2),
            "ttft_queue_ms": round(pred.ttft_queue_ms, 2),
        }
        if measured_ttft is not None:
            turn_row["ttft_meas"] = round(measured_ttft, 2)
            turn_row["ttft_err"] = _round_optional(
                _measured_error(pred.ttft_ms, measured_ttft), 1
            )
        if measured_tpot is not None:
            turn_row["tpot_meas"] = round(measured_tpot, 2)
            turn_row["tpot_err"] = _round_optional(
                _measured_error(pred.tpot_ms, measured_tpot), 1
            )
        if measured_e2el is not None:
            turn_row["e2el_meas"] = round(measured_e2el, 2)
            turn_row["e2el_err"] = _round_optional(
                _measured_error(pred.e2el_ms, measured_e2el), 1
            )
        turn_predictions.append({
            key: value for key, value in turn_row.items()
            if value is not None
        })

    return ServingPrediction(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2el_ms=ttft_ms + weighted_decode_total,
        decode_total_ms=weighted_decode_total,
        bs_eff=weighted_mean([
            (pred.bs_eff, feature.successful)
            for feature, pred in turn_rows
        ]),
        concurrency=concurrency,
        ttft_kernel_ms=weighted_mean([
            (pred.ttft_kernel_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        ttft_base_ms=weighted_mean([
            (pred.ttft_base_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        ttft_floor_ms=weighted_mean([
            (pred.ttft_floor_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        ttft_first_decode_ms=weighted_mean([
            (pred.ttft_first_decode_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        ttft_queue_ms=weighted_mean([
            (pred.ttft_queue_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        calibration_status=first_pred.calibration_status,
        total_context_tokens=int(round(weighted_mean([
            (feature.total_context_tokens, feature.successful)
            for feature, _ in turn_rows
        ]))),
        new_prefill_tokens=int(round(weighted_mean([
            (feature.new_prefill_tokens, feature.successful)
            for feature, _ in turn_rows
        ]))),
        cached_context_tokens=int(round(weighted_mean([
            (feature.cached_context_tokens, feature.successful)
            for feature, _ in turn_rows
        ]))),
        cache_hit_rate=weighted_mean([
            (feature.cache_hit_rate, feature.successful)
            for feature, _ in turn_rows
        ]),
        cache_aware_applied=any(pred.cache_aware_applied for _, pred in turn_rows),
        cache_feature_source="per_turn",
        cache_prediction_regime=(
            "prefix_cached_prefill"
            if any(pred.cache_aware_applied for _, pred in turn_rows)
            else "full_prefill"
        ),
        ttft_prediction_supported=all(
            pred.ttft_prediction_supported for _, pred in turn_rows
        ),
        multiturn_prediction_mode="per_turn_aggregated",
        predicted_turn_count=len(turn_rows),
        total_successful_turn_requests=total_successful,
        mean_predicted_turn_ttft_ms=weighted_mean([
            (pred.ttft_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        mean_predicted_turn_tpot_ms=weighted_mean([
            (pred.tpot_ms, feature.successful)
            for feature, pred in turn_rows
        ]),
        multiturn_turn_predictions=turn_predictions,
    )
