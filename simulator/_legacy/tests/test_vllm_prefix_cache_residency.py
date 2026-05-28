from simulator._legacy.vllm_prefix_cache_residency import (
    VllmPrefixCacheResidencyConfig,
    VllmPrefixCacheResidencyInput,
    block_aligned_tokens,
    blocks_for_tokens,
    estimate_prefix_cache_residency,
)


def test_block_helpers_match_vllm_full_block_cache_hits() -> None:
    assert block_aligned_tokens(6568, 16) == 6560
    assert block_aligned_tokens(5904, 16) == 5904
    assert blocks_for_tokens(17, 16) == 2


def test_c40_benchmark_cache_faithful_matches_engine_cache_truth() -> None:
    result = estimate_prefix_cache_residency(
        VllmPrefixCacheResidencyInput(
            request_count=40,
            context_len=7188,
            cached_context_tokens=6568,
            new_prefill_tokens=144,
            mode="benchmark_cache",
        ),
        VllmPrefixCacheResidencyConfig(
            cache_block_size=16,
            available_gpu_kv_blocks=27_769,
        ),
    )

    assert result.cache_residency_classification == "cache_matches"
    assert result.engine_effective_cached_tokens == 6560
    assert result.engine_uncached_prefill_tokens == 628
    assert result.requested_cached_blocks == 16_400


def test_c80_benchmark_cache_faithful_loses_cache_when_residency_exceeds_budget() -> None:
    result = estimate_prefix_cache_residency(
        VllmPrefixCacheResidencyInput(
            request_count=80,
            context_len=6188,
            cached_context_tokens=5904,
            new_prefill_tokens=138,
            mode="benchmark_cache",
        ),
        VllmPrefixCacheResidencyConfig(
            cache_block_size=16,
            available_gpu_kv_blocks=27_769,
        ),
    )

    assert result.cache_residency_classification == "benchmark_cache_overoptimistic"
    assert result.requested_cached_blocks == 29_520
    assert result.engine_effective_cached_tokens == 0
    assert result.engine_uncached_prefill_tokens == 6188
    assert result.capacity_feasible_cached_tokens == 5552


def test_synthetic_shared_prefix_reproduces_mean_engine_cache_truth() -> None:
    result = estimate_prefix_cache_residency(
        VllmPrefixCacheResidencyInput(
            request_count=80,
            context_len=6188,
            cached_context_tokens=5904,
            new_prefill_tokens=138,
            mode="synthetic_shared_prefix",
            shared_prefix_tokens=1024,
        ),
        VllmPrefixCacheResidencyConfig(
            cache_block_size=16,
            available_gpu_kv_blocks=27_769,
        ),
    )

    assert result.cache_residency_classification == "trace_shape_mismatch"
    assert result.engine_effective_cached_tokens == 1011.2
    assert result.engine_uncached_prefill_tokens == 414_144
