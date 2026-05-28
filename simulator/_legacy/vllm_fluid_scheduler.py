"""Fluid/wave vLLM serving-scheduler approximation.

This module intentionally does not try to clone vLLM request order.  It turns a
benchmark turn into a no-fit scheduler pressure estimate: how many decode waves
fit in KV, how much uncached prefill debt exists, and how much of that debt can
intrude into decode-priority steps through leftover token budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .vllm_prefix_cache_residency import block_aligned_tokens, blocks_for_tokens


FluidCachePolicy = Literal["optimistic", "fluid_mid", "pessimistic"]


@dataclass(frozen=True)
class VllmFluidSchedulerConfig:
    max_num_batched_tokens: int = 8_192
    max_num_seqs: int = 512
    cache_block_size: int = 16
    available_gpu_kv_blocks: int = 27_651
    max_model_len: int = 32_768
    decode_counts_against_token_budget: bool = True


@dataclass(frozen=True)
class VllmFluidTurnInput:
    scheduled_request_count: int
    context_len: int
    output_tokens: int
    cached_context_tokens: int
    new_prefill_tokens: int
    cache_hit_rate: float = 0.0


@dataclass(frozen=True)
class VllmFluidCacheEnvelope:
    policy: FluidCachePolicy
    effective_cached_tokens: int
    uncached_prefill_tokens_per_request: int
    requested_cached_tokens_per_request: int
    requested_cached_blocks: int
    capacity_feasible_cached_tokens: int
    cache_fit_ratio: float
    classification: str


@dataclass(frozen=True)
class VllmFluidTurnSummary:
    cache_policy: FluidCachePolicy
    scheduled_request_count: int
    context_len: int
    output_tokens: int
    effective_cached_tokens: int
    uncached_prefill_tokens_per_request: int
    fresh_prefill_debt_tokens: int
    decode_slots: int
    active_decode_capacity: int
    kv_limited_decode_capacity: int
    decode_wave_factor: float
    decode_waves: int
    decode_steps: int
    mixed_decode_prefill_steps: int
    decode_only_steps: int
    prefill_only_steps: int
    estimated_steps: int
    prefill_tokens_in_mixed_steps: int
    prefill_tokens_after_decode: int
    leftover_prefill_tokens_per_decode_step: int
    max_decode_batch: int
    mean_decode_batch: float
    mean_decode_batch_ratio: float
    total_scheduled_tokens: int
    resident_blocks_per_request: int
    requested_resident_blocks: int
    max_used_kv_blocks: int
    min_free_kv_blocks: int
    kv_pressure: float
    prefill_debt_per_decode_token: float
    cache_fit_ratio: float
    requested_cached_blocks: int
    capacity_feasible_cached_tokens: int
    cache_residency_classification: str
    scheduler_sensitivity: str


@dataclass(frozen=True)
class VllmFluidTurnResult:
    optimistic: VllmFluidTurnSummary
    fluid_mid: VllmFluidTurnSummary
    pessimistic: VllmFluidTurnSummary

    @property
    def envelopes(self) -> tuple[VllmFluidTurnSummary, ...]:
        return (self.optimistic, self.fluid_mid, self.pessimistic)

    def by_policy(self, policy: FluidCachePolicy) -> VllmFluidTurnSummary:
        if policy == "optimistic":
            return self.optimistic
        if policy == "fluid_mid":
            return self.fluid_mid
        if policy == "pessimistic":
            return self.pessimistic
        raise ValueError(f"unknown fluid cache policy: {policy}")


def simulate_fluid_turn(
    turn: VllmFluidTurnInput,
    config: VllmFluidSchedulerConfig | None = None,
) -> VllmFluidTurnResult:
    """Estimate scheduler shape envelopes for one aggregate benchmark turn."""
    cfg = config or VllmFluidSchedulerConfig()
    normalized = _normalize_turn(turn)
    cache_envelopes = _cache_envelopes(normalized, cfg)
    return VllmFluidTurnResult(
        optimistic=_summary_for_cache(cache_envelopes["optimistic"], normalized, cfg),
        fluid_mid=_summary_for_cache(cache_envelopes["fluid_mid"], normalized, cfg),
        pessimistic=_summary_for_cache(cache_envelopes["pessimistic"], normalized, cfg),
    )


def _normalize_turn(turn: VllmFluidTurnInput) -> VllmFluidTurnInput:
    context_len = max(1, int(turn.context_len))
    scheduled = max(1, int(turn.scheduled_request_count))
    output_tokens = max(1, int(turn.output_tokens))
    cached = max(0, min(int(turn.cached_context_tokens), context_len))
    if cached <= 0 and turn.cache_hit_rate > 0.0:
        cached = max(0, min(int(round(context_len * turn.cache_hit_rate)), context_len))
    new_prefill = max(0, min(int(turn.new_prefill_tokens), context_len))
    return VllmFluidTurnInput(
        scheduled_request_count=scheduled,
        context_len=context_len,
        output_tokens=output_tokens,
        cached_context_tokens=cached,
        new_prefill_tokens=new_prefill,
        cache_hit_rate=max(0.0, min(float(turn.cache_hit_rate), 1.0)),
    )


def _cache_envelopes(
    turn: VllmFluidTurnInput,
    cfg: VllmFluidSchedulerConfig,
) -> dict[FluidCachePolicy, VllmFluidCacheEnvelope]:
    block_size = max(1, int(cfg.cache_block_size))
    available_blocks = max(0, int(cfg.available_gpu_kv_blocks))
    requested_cached = block_aligned_tokens(turn.cached_context_tokens, block_size)
    requested_cached_blocks = (
        turn.scheduled_request_count * blocks_for_tokens(requested_cached, block_size)
    )
    resident_blocks = _requested_resident_blocks(turn, cfg)
    cache_fit_ratio = min(1.0, available_blocks / max(1, resident_blocks))
    kv_pressure = resident_blocks / max(1, available_blocks)
    cache_survival = 1.0 if resident_blocks <= available_blocks else 0.0
    capacity_feasible = min(
        requested_cached,
        (available_blocks // turn.scheduled_request_count) * block_size,
    )
    optimistic = requested_cached
    fluid_mid = block_aligned_tokens(
        round(requested_cached * min(cache_fit_ratio, cache_survival)),
        block_size,
    )
    pessimistic = requested_cached if resident_blocks <= available_blocks else 0
    return {
        "optimistic": _cache_envelope(
            policy="optimistic",
            effective_cached=optimistic,
            requested_cached=requested_cached,
            requested_cached_blocks=requested_cached_blocks,
            capacity_feasible=capacity_feasible,
            cache_fit_ratio=1.0,
            turn=turn,
        ),
        "fluid_mid": _cache_envelope(
            policy="fluid_mid",
            effective_cached=fluid_mid,
            requested_cached=requested_cached,
            requested_cached_blocks=requested_cached_blocks,
            capacity_feasible=capacity_feasible,
            cache_fit_ratio=cache_fit_ratio,
            turn=turn,
        ),
        "pessimistic": _cache_envelope(
            policy="pessimistic",
            effective_cached=pessimistic,
            requested_cached=requested_cached,
            requested_cached_blocks=requested_cached_blocks,
            capacity_feasible=capacity_feasible,
            cache_fit_ratio=1.0 if pessimistic == requested_cached else 0.0,
            turn=turn,
        ),
    }


def _cache_envelope(
    *,
    policy: FluidCachePolicy,
    effective_cached: int,
    requested_cached: int,
    requested_cached_blocks: int,
    capacity_feasible: int,
    cache_fit_ratio: float,
    turn: VllmFluidTurnInput,
) -> VllmFluidCacheEnvelope:
    effective_cached = max(0, min(int(effective_cached), turn.context_len))
    uncached = max(0, turn.context_len - effective_cached)
    if requested_cached <= 0:
        classification = "no_benchmark_cache"
    elif effective_cached >= requested_cached:
        classification = "cache_matches"
    elif effective_cached <= 0:
        classification = "benchmark_cache_overoptimistic"
    else:
        classification = "cache_residency_partial"
    return VllmFluidCacheEnvelope(
        policy=policy,
        effective_cached_tokens=effective_cached,
        uncached_prefill_tokens_per_request=uncached,
        requested_cached_tokens_per_request=requested_cached,
        requested_cached_blocks=requested_cached_blocks,
        capacity_feasible_cached_tokens=capacity_feasible,
        cache_fit_ratio=cache_fit_ratio,
        classification=classification,
    )


def _summary_for_cache(
    cache: VllmFluidCacheEnvelope,
    turn: VllmFluidTurnInput,
    cfg: VllmFluidSchedulerConfig,
) -> VllmFluidTurnSummary:
    scheduled = turn.scheduled_request_count
    max_tokens = max(1, int(cfg.max_num_batched_tokens))
    max_seqs = max(1, int(cfg.max_num_seqs))
    resident_blocks_per_request = _resident_blocks_per_request(turn, cfg)
    available_blocks = max(0, int(cfg.available_gpu_kv_blocks))
    if resident_blocks_per_request <= 0:
        kv_limited_capacity = scheduled
    elif available_blocks <= 0:
        kv_limited_capacity = scheduled
    else:
        kv_limited_capacity = max(1, available_blocks // resident_blocks_per_request)
    token_limited_capacity = max_tokens if cfg.decode_counts_against_token_budget else scheduled
    raw_active_decode_capacity = max(
        1,
        min(scheduled, max_seqs, token_limited_capacity, kv_limited_capacity),
    )
    requested_resident_blocks = _requested_resident_blocks(turn, cfg)
    kv_pressure = requested_resident_blocks / max(1, available_blocks)
    active_decode_capacity = max(1, int(raw_active_decode_capacity))
    decode_slots = scheduled * turn.output_tokens
    decode_steps = _ceil_div(decode_slots, active_decode_capacity)
    decode_waves = _ceil_div(scheduled, active_decode_capacity)
    decode_wave_factor = max(1.0, scheduled / active_decode_capacity)
    fresh_prefill_debt = scheduled * cache.uncached_prefill_tokens_per_request
    decode_visible_prefill_debt = fresh_prefill_debt
    leftover_per_decode_step = (
        max(0, max_tokens - active_decode_capacity)
        if cfg.decode_counts_against_token_budget
        else max_tokens
    )
    if decode_visible_prefill_debt > 0 and leftover_per_decode_step > 0:
        mixed_steps = min(
            decode_steps,
            _ceil_div(decode_visible_prefill_debt, leftover_per_decode_step),
        )
    else:
        mixed_steps = 0
    prefill_in_mixed = min(
        decode_visible_prefill_debt,
        mixed_steps * leftover_per_decode_step,
    )
    remaining_prefill = max(0, fresh_prefill_debt - prefill_in_mixed)
    prefill_only = _ceil_div(remaining_prefill, max_tokens)
    estimated_steps = decode_steps + prefill_only
    decode_only = max(0, decode_steps - mixed_steps)
    max_used = min(available_blocks, active_decode_capacity * resident_blocks_per_request)
    min_free = available_blocks - max_used
    mean_decode_batch = decode_slots / estimated_steps if estimated_steps else 0.0
    prefill_per_decode = fresh_prefill_debt / max(1, decode_slots)
    return VllmFluidTurnSummary(
        cache_policy=cache.policy,
        scheduled_request_count=scheduled,
        context_len=turn.context_len,
        output_tokens=turn.output_tokens,
        effective_cached_tokens=cache.effective_cached_tokens,
        uncached_prefill_tokens_per_request=cache.uncached_prefill_tokens_per_request,
        fresh_prefill_debt_tokens=fresh_prefill_debt,
        decode_slots=decode_slots,
        active_decode_capacity=active_decode_capacity,
        kv_limited_decode_capacity=kv_limited_capacity,
        decode_wave_factor=decode_wave_factor,
        decode_waves=decode_waves,
        decode_steps=decode_steps,
        mixed_decode_prefill_steps=mixed_steps,
        decode_only_steps=decode_only,
        prefill_only_steps=prefill_only,
        estimated_steps=estimated_steps,
        prefill_tokens_in_mixed_steps=prefill_in_mixed,
        prefill_tokens_after_decode=remaining_prefill,
        leftover_prefill_tokens_per_decode_step=leftover_per_decode_step,
        max_decode_batch=active_decode_capacity,
        mean_decode_batch=mean_decode_batch,
        mean_decode_batch_ratio=mean_decode_batch / scheduled if scheduled else 0.0,
        total_scheduled_tokens=decode_slots + fresh_prefill_debt,
        resident_blocks_per_request=resident_blocks_per_request,
        requested_resident_blocks=requested_resident_blocks,
        max_used_kv_blocks=max_used,
        min_free_kv_blocks=min_free,
        kv_pressure=kv_pressure,
        prefill_debt_per_decode_token=prefill_per_decode,
        cache_fit_ratio=cache.cache_fit_ratio,
        requested_cached_blocks=cache.requested_cached_blocks,
        capacity_feasible_cached_tokens=cache.capacity_feasible_cached_tokens,
        cache_residency_classification=cache.classification,
        scheduler_sensitivity=_scheduler_sensitivity(
            kv_pressure=kv_pressure,
            decode_wave_factor=decode_wave_factor,
            prefill_debt_per_decode_token=prefill_per_decode,
            mixed_steps=mixed_steps,
            decode_steps=decode_steps,
            cache_classification=cache.classification,
        ),
    )


def _scheduler_sensitivity(
    *,
    kv_pressure: float,
    decode_wave_factor: float,
    prefill_debt_per_decode_token: float,
    mixed_steps: int,
    decode_steps: int,
    cache_classification: str,
) -> str:
    if kv_pressure >= 1.2:
        return "kv_scheduler_sensitive"
    if cache_classification in {
        "cache_residency_partial",
        "benchmark_cache_overoptimistic",
    }:
        return "cache_residency_unstable"
    mixed_fraction = mixed_steps / max(1, decode_steps)
    if prefill_debt_per_decode_token >= 32.0 or mixed_fraction >= 0.25:
        return "mixed_prefill_likely"
    if decode_wave_factor >= 1.25:
        return "decode_wave_sensitive"
    return "decode_clean"


def _resident_blocks_per_request(
    turn: VllmFluidTurnInput,
    cfg: VllmFluidSchedulerConfig,
) -> int:
    resident_tokens = min(
        max(1, int(cfg.max_model_len)),
        turn.context_len + max(0, turn.output_tokens),
    )
    return blocks_for_tokens(resident_tokens, max(1, int(cfg.cache_block_size)))


def _requested_resident_blocks(
    turn: VllmFluidTurnInput,
    cfg: VllmFluidSchedulerConfig,
) -> int:
    return turn.scheduled_request_count * _resident_blocks_per_request(turn, cfg)


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0
    return int(math.ceil(numerator / max(1, denominator)))
