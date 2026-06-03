#!/usr/bin/env python3
"""Analytic vLLM KV-block pool estimate for configs with no measured pool.

vLLM v1 sizes its paged-KV pool at startup as the GPU memory left after weights
and a fixed activation / non-torch reserve, divided by the per-block KV footprint:

    budget_bytes      = total_memory * gpu_mem_util - weight_bytes/tp - RESERVE
    bytes_per_block   = block_size * kv_bytes_per_token / kv_shards   (per GPU)
    available_blocks  = floor(budget_bytes / bytes_per_block)

``kv_shards = min(tp, kv_heads)`` because GQA shards the KV heads across ranks.
``weight_bytes = n_params * bytes_per_param`` is the real HBM footprint (so this is
quant-aware: MXFP4 experts make bytes_per_param < 2).

RESERVE is the one estimated constant. It is the vLLM non-torch + peak-activation
memory carve-out; it is NOT fit to any accuracy target. Calibrated against the three
configs whose pools we know exactly:

    H100   80GiB  util .90  tp1  ->  reserve 4.10 GB   (measured pool 27250)
    A100   40GiB  util .85  tp1  ->  reserve 2.71 GB   (measured pool  8458)
    H100x2 80GiB  util .90  tp2  ->  reserve 3.81 GB   (measured pool 62416)

A single RESERVE = 3.5 GB reproduces all three within ~5%. Configs with a MEASURED
pool keep their exact value in the deployment JSON (manifest ``kv_pool`` status
``measured``); only configs without one fall back to this estimate (status
``derived``). The pool is the pressure-signal denominator — a first-cut quantity for
uncalibrated configs.
"""
from __future__ import annotations

import math

RESERVE_BYTES = 3.5e9  # vLLM non-torch + peak-activation reserve (see module docstring)


def available_kv_blocks(
    total_memory_bytes: float,
    gpu_mem_util: float,
    weight_bytes: float,
    tp: int,
    kv_bytes_per_token: float,
    kv_heads: int,
    block_size: int = 16,
    reserve_bytes: float = RESERVE_BYTES,
) -> int:
    """vLLM v1 paged-KV pool size in blocks (per the startup allocation; see module docstring)."""
    tp = max(1, int(tp))
    kv_shards = min(tp, max(1, int(kv_heads)))
    budget = total_memory_bytes * gpu_mem_util - weight_bytes / tp - reserve_bytes
    bytes_per_block = block_size * kv_bytes_per_token / kv_shards
    if budget <= 0 or bytes_per_block <= 0:
        return 1
    return max(1, math.floor(budget / bytes_per_block))
