from __future__ import annotations

import pytest

from simulator._legacy.vllm_fluid_scheduler import (
    VllmFluidSchedulerConfig,
    VllmFluidTurnInput,
    simulate_fluid_turn,
)


def test_fluid_scheduler_limits_active_decode_capacity_by_kv_blocks() -> None:
    result = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=4,
            context_len=32,
            output_tokens=2,
            cached_context_tokens=32,
            new_prefill_tokens=0,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=8,
            cache_block_size=16,
            available_gpu_kv_blocks=4,
        ),
    )

    mid = result.fluid_mid
    assert mid.resident_blocks_per_request == 3
    assert mid.active_decode_capacity == 1
    assert mid.decode_wave_factor == pytest.approx(4.0)
    assert mid.decode_waves == 4


def test_fluid_scheduler_emits_cache_envelopes() -> None:
    result = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=4,
            context_len=64,
            output_tokens=1,
            cached_context_tokens=48,
            new_prefill_tokens=16,
            cache_hit_rate=0.75,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=16,
            cache_block_size=16,
            available_gpu_kv_blocks=8,
        ),
    )

    assert result.optimistic.effective_cached_tokens == 48
    assert result.fluid_mid.effective_cached_tokens == 0
    assert result.pessimistic.effective_cached_tokens == 0
    assert result.fluid_mid.cache_residency_classification == (
        "benchmark_cache_overoptimistic"
    )
    assert result.pessimistic.cache_residency_classification == (
        "benchmark_cache_overoptimistic"
    )


def test_fluid_scheduler_accounts_for_mixed_and_prefill_only_steps() -> None:
    result = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=4,
            context_len=8,
            output_tokens=1,
            cached_context_tokens=6,
            new_prefill_tokens=2,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=6,
            max_num_seqs=8,
            cache_block_size=16,
            available_gpu_kv_blocks=64,
        ),
    )

    mid = result.fluid_mid
    assert mid.active_decode_capacity == 4
    assert mid.decode_steps == 1
    assert mid.leftover_prefill_tokens_per_decode_step == 2
    assert mid.fresh_prefill_debt_tokens == 32
    assert mid.mixed_decode_prefill_steps == 1
    assert mid.prefill_tokens_in_mixed_steps == 2
    assert mid.prefill_only_steps == 5
    assert mid.estimated_steps == 6


def test_fluid_scheduler_makes_prefill_visible_near_kv_pressure_wall() -> None:
    result = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=8,
            context_len=64,
            output_tokens=8,
            cached_context_tokens=48,
            new_prefill_tokens=16,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=8,
            cache_block_size=16,
            available_gpu_kv_blocks=36,
        ),
    )

    mid = result.fluid_mid
    assert mid.kv_pressure >= 1.0
    assert mid.effective_cached_tokens == 0
    assert mid.active_decode_capacity == 7
    assert mid.decode_wave_factor == pytest.approx(8 / 7)
    assert mid.decode_steps == 10
    assert mid.mixed_decode_prefill_steps == 10
    assert mid.prefill_tokens_in_mixed_steps == 90
    assert mid.prefill_only_steps == 27
    assert mid.estimated_steps == 37


def test_fluid_scheduler_classifies_decode_clean_rows() -> None:
    result = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=2,
            context_len=16,
            output_tokens=1,
            cached_context_tokens=16,
            new_prefill_tokens=0,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=8,
            cache_block_size=16,
            available_gpu_kv_blocks=64,
        ),
    )

    assert result.fluid_mid.scheduler_sensitivity == "decode_clean"
