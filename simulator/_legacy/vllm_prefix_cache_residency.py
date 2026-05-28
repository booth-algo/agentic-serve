"""Effective vLLM prefix-cache residency estimator.

This module models the narrow cache behavior needed by the TPOT predictor.  It
is not a full KV cache simulator; it mirrors the vLLM boundary that matters for
our benchmark rows:

* ``KVCacheManager.get_computed_blocks()`` returns full block-aligned prefix
  cache hits.
* ``FullAttentionManager.find_longest_cache_hit()`` walks block hashes from the
  prompt start and stops at the first miss.
* ``KVCacheManager.allocate_slots()`` can only use cached blocks that survive
  the finite GPU KV block budget.

The v0 policy is intentionally conservative.  For benchmark-cache-faithful rows
we treat a simultaneous turn's cached prefixes as useful only if the requested
resident cached prefix working set fits in the available KV block pool.  When it
does not fit, the front of each prefix is considered unstable/evictable, and the
longest-prefix lookup returns zero useful computed tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


PrefixCacheMode = Literal["benchmark_cache", "synthetic_shared_prefix"]


@dataclass(frozen=True)
class VllmPrefixCacheResidencyConfig:
    cache_block_size: int = 16
    available_gpu_kv_blocks: int = 27_769


@dataclass(frozen=True)
class VllmPrefixCacheResidencyInput:
    request_count: int
    context_len: int
    cached_context_tokens: int
    new_prefill_tokens: int
    mode: PrefixCacheMode = "benchmark_cache"
    shared_prefix_tokens: int = 0


@dataclass(frozen=True)
class VllmPrefixCacheResidencyResult:
    mode: str
    request_count: int
    context_len: int
    benchmark_cached_context_tokens: int
    benchmark_new_prefill_tokens: int
    cache_block_size: int
    available_gpu_kv_blocks: int
    requested_cached_blocks: int
    requested_cached_tokens: int
    capacity_feasible_cached_tokens: int
    engine_effective_cached_tokens: float
    engine_uncached_prefill_tokens: int
    cache_residency_classification: str


def estimate_prefix_cache_residency(
    workload: VllmPrefixCacheResidencyInput,
    config: VllmPrefixCacheResidencyConfig | None = None,
) -> VllmPrefixCacheResidencyResult:
    """Estimate engine-visible prefix-cache hits for one aggregate turn."""
    cfg = config or VllmPrefixCacheResidencyConfig()
    if workload.mode == "synthetic_shared_prefix":
        return _estimate_synthetic_shared_prefix(workload, cfg)
    return _estimate_benchmark_cache(workload, cfg)


def block_aligned_tokens(tokens: int, block_size: int) -> int:
    """Return vLLM-style full-block-aligned cacheable tokens."""
    return max(0, int(tokens)) // max(1, int(block_size)) * max(1, int(block_size))


def blocks_for_tokens(tokens: int, block_size: int) -> int:
    """Return ceil-div block residency for allocated KV tokens."""
    tokens = max(0, int(tokens))
    block_size = max(1, int(block_size))
    return int(math.ceil(tokens / block_size)) if tokens else 0


def _estimate_benchmark_cache(
    workload: VllmPrefixCacheResidencyInput,
    cfg: VllmPrefixCacheResidencyConfig,
) -> VllmPrefixCacheResidencyResult:
    request_count = max(1, int(workload.request_count))
    context_len = max(1, int(workload.context_len))
    block_size = max(1, int(cfg.cache_block_size))
    available_blocks = max(0, int(cfg.available_gpu_kv_blocks))
    requested_per_request = block_aligned_tokens(
        min(int(workload.cached_context_tokens), context_len),
        block_size,
    )
    requested_blocks_per_request = requested_per_request // block_size
    requested_blocks = request_count * requested_blocks_per_request
    capacity_feasible = min(
        requested_per_request,
        available_blocks // request_count * block_size,
    )

    if requested_blocks <= available_blocks:
        effective_cached = float(requested_per_request)
        classification = "cache_matches"
    elif requested_per_request <= 0:
        effective_cached = 0.0
        classification = "no_benchmark_cache"
    else:
        # vLLM prefix lookup is a longest-prefix walk.  Once KV pressure evicts
        # the leading block(s), later resident suffix blocks do not help this
        # request's prefix hit length.
        effective_cached = 0.0
        classification = "benchmark_cache_overoptimistic"

    uncached_per_request = max(0, context_len - int(effective_cached))
    return VllmPrefixCacheResidencyResult(
        mode=workload.mode,
        request_count=request_count,
        context_len=context_len,
        benchmark_cached_context_tokens=max(0, int(workload.cached_context_tokens)),
        benchmark_new_prefill_tokens=max(0, int(workload.new_prefill_tokens)),
        cache_block_size=block_size,
        available_gpu_kv_blocks=available_blocks,
        requested_cached_blocks=requested_blocks,
        requested_cached_tokens=requested_per_request * request_count,
        capacity_feasible_cached_tokens=capacity_feasible,
        engine_effective_cached_tokens=effective_cached,
        engine_uncached_prefill_tokens=uncached_per_request,
        cache_residency_classification=classification,
    )


def _estimate_synthetic_shared_prefix(
    workload: VllmPrefixCacheResidencyInput,
    cfg: VllmPrefixCacheResidencyConfig,
) -> VllmPrefixCacheResidencyResult:
    request_count = max(1, int(workload.request_count))
    context_len = max(1, int(workload.context_len))
    block_size = max(1, int(cfg.cache_block_size))
    available_blocks = max(0, int(cfg.available_gpu_kv_blocks))
    shared_tokens = block_aligned_tokens(
        min(int(workload.shared_prefix_tokens), context_len),
        block_size,
    )
    shared_blocks = shared_tokens // block_size
    shared_fits = shared_blocks <= available_blocks
    cached_followers = max(0, request_count - 1)
    effective_total = shared_tokens * cached_followers if shared_fits else 0
    effective_mean = effective_total / request_count
    uncached_total = request_count * context_len - effective_total
    capacity_feasible = min(
        shared_tokens,
        available_blocks * block_size,
    )
    if shared_tokens <= 0:
        classification = "no_shared_prefix"
    elif shared_fits:
        classification = "trace_shape_mismatch"
    else:
        classification = "benchmark_cache_overoptimistic"

    return VllmPrefixCacheResidencyResult(
        mode=workload.mode,
        request_count=request_count,
        context_len=context_len,
        benchmark_cached_context_tokens=max(0, int(workload.cached_context_tokens)),
        benchmark_new_prefill_tokens=max(0, int(workload.new_prefill_tokens)),
        cache_block_size=block_size,
        available_gpu_kv_blocks=available_blocks,
        requested_cached_blocks=shared_blocks,
        requested_cached_tokens=shared_tokens,
        capacity_feasible_cached_tokens=capacity_feasible,
        engine_effective_cached_tokens=effective_mean,
        engine_uncached_prefill_tokens=uncached_total,
        cache_residency_classification=classification,
    )
