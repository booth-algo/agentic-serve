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
memory carve-out; it is NOT fit to any accuracy target.

THE RULE (audit-v2 G6, stated 2026-06-10): RESERVE = the mean of the back-solved
reserves of the three configs whose pools we know exactly, rounded to 0.1 GB.
Back-solve: ``reserve_i = total*util - weights/tp - pool_blocks*bytes_per_block``
(Llama-3.1-8B: weights 16.06 GB bf16, bytes_per_block = 16*131072/min(tp,8)):

    H100   80GiB  util .90  tp1  pool 27250  ->  reserve 4.102 GB
    A100   40GiB  util .85  tp1  pool  8458  ->  reserve 2.710 GB
    H100x2 80GiB  util .90  tp2  pool 62416  ->  reserve 3.831 GB
                                       mean  =  3.548 GB  ->  3.5 GB

The single RESERVE = 3.5 GB reproduces the three known pools within 5%
(+1.05% / -4.46% / +0.51%) — pinned by
simulator/tests/test_deployment_configs.py::test_reserve_rule_reproduces_known_pools.
KNOWN AMPLIFICATION: the per-config envelope (2.71..4.10 GB, i.e. +-0.6..0.8 GB
around 3.5) is a small absolute error on 80 GiB pools but +-30..60% of SMALL pools
(e.g. RTX3090 24 GiB: 1117 derived blocks) — treat derived small-GPU pools as
first-cut only. Configs with a MEASURED pool keep their exact value in the
deployment JSON (manifest ``kv_pool`` status ``measured``); only configs without
one fall back to this estimate (status ``derived``). The pool is the
pressure-signal denominator — a first-cut quantity for uncalibrated configs.
"""
from __future__ import annotations

import math

# vLLM non-torch + peak-activation reserve. Rule: mean of the 3 back-solved known-pool
# reserves (4.102/2.710/3.831 GB) = 3.548 -> rounded to 3.5 (see module docstring).
RESERVE_BYTES = 3.5e9


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
