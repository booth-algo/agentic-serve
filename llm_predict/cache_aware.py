"""Cache-aware helpers for multi-turn serving prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .composer import Composer
from .configs.model_configs import ModelConfig
from .serving import ServingPrediction, predict_serving


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


def predict_multiturn_from_per_turn(
    composer: Composer,
    cfg: ModelConfig,
    gpu: str,
    per_turn: list[dict[str, Any]] | None,
    concurrency: int,
    backend: str | None = None,
    backend_version: str | None = None,
    model_key: str | None = None,
    profile: str | None = None,
    apply_prefix_contention: bool = True,
) -> ServingPrediction | None:
    feature = aggregate_turn_cache_feature(per_turn)
    if feature is None:
        return None
    return predict_serving(
        composer, cfg, gpu,
        feature.total_context_tokens,
        feature.output_tokens,
        concurrency,
        backend=backend,
        backend_version=backend_version,
        model_key=model_key,
        profile=profile,
        total_context_tokens=feature.total_context_tokens,
        new_prefill_tokens=feature.new_prefill_tokens,
        cached_context_tokens=feature.cached_context_tokens,
        cache_hit_rate=feature.cache_hit_rate,
        apply_prefix_contention=apply_prefix_contention,
    )
