"""Concurrency-aware steady-state model for serving_e2e.

Pure-function module implementing steady-state batch size estimation,
TTFT queuing model, and KV cache saturation detection. No I/O or
XGBoost dependencies — keeps serving_e2e.py clean.
"""
from __future__ import annotations

import math

_TTFT_QUEUE_ALPHA: dict[str, float] = {
    "A100": 1.0,
    "RTX3090": 1.0,
    "RTX2080Ti": 1.0,
    "H100": 1.0,
}

_GPU_MEMORY_GB: dict[str, float] = {
    "A100": 40.0,
    "RTX3090": 24.0,
    "RTX2080Ti": 22.0,
    "H100": 80.0,
}


def effective_decode_bs(concurrency: int, ttft_ms: float, tpot_ms: float,
                        osl: int) -> float:
    if concurrency <= 1 or osl <= 0:
        return 1.0
    decode_time = tpot_ms * osl
    total_time = ttft_ms + decode_time
    if total_time <= 0:
        return 1.0
    decode_fraction = decode_time / total_time
    bs_eff = concurrency * decode_fraction
    return max(1.0, min(float(concurrency), bs_eff))


def iterative_bs_eff(concurrency: int,
                     tpot_at_bs: callable,
                     ttft_ms: float,
                     osl: int,
                     max_iter: int = 5,
                     tol: float = 0.01,
                     damping: float = 0.7) -> float:
    if concurrency <= 1:
        return 1.0

    tpot_1 = tpot_at_bs(1.0)
    bs = effective_decode_bs(concurrency, ttft_ms, tpot_1, osl)

    for _ in range(max_iter):
        tpot_bs = tpot_at_bs(bs)
        bs_new = effective_decode_bs(concurrency, ttft_ms, tpot_bs, osl)
        bs_damped = damping * bs_new + (1.0 - damping) * bs
        if abs(bs_damped - bs) / max(bs, 1e-9) < tol:
            bs = bs_damped
            break
        bs = bs_damped

    return max(1.0, min(float(concurrency), bs))


def ttft_queuing_factor(concurrency: int, ttft_ms: float, tpot_ms: float,
                        osl: int, gpu: str = "A100") -> float:
    if concurrency <= 1:
        return 1.0
    # Workload-aware threshold: longer decode = higher C before prefill
    # queue saturates. k=1.8 calibrated from A100 Llama-8B chat-short/medium/long.
    PREFILL_SATURATION_K = 1.8
    LOW_SLOPE = 0.025
    PLATEAU_BASE = 8.0
    PLATEAU_EXP = 0.1
    c_thresh = max(5, osl * tpot_ms / max(ttft_ms, 1e-9) / PREFILL_SATURATION_K)
    if concurrency < c_thresh:
        return 1.0 + LOW_SLOPE * (concurrency - 1)
    return PLATEAU_BASE * (concurrency / c_thresh) ** PLATEAU_EXP


def estimate_kv_memory_gb(bs_eff: float, max_kv_len: int, n_layers: int,
                          kv_heads: int, head_dim: int,
                          dtype_bytes: int = 2) -> float:
    kv_bytes = 2 * n_layers * kv_heads * head_dim * max_kv_len * dtype_bytes
    return bs_eff * kv_bytes / (1024 ** 3)


def estimate_model_memory_gb(n_layers: int, d: int, ffn: int,
                             dtype_bytes: int = 2) -> float:
    per_layer = (4 * d * d + 3 * d * ffn + 2 * d) * dtype_bytes
    return n_layers * per_layer / (1024 ** 3)


def is_saturated(bs_eff: float, max_kv_len: int, n_layers: int,
                 kv_heads: int, head_dim: int, d: int, ffn: int,
                 gpu: str = "A100") -> tuple[bool, str]:
    gpu_mem = _GPU_MEMORY_GB.get(gpu, 40.0)
    model_mem = estimate_model_memory_gb(n_layers, d, ffn)
    kv_mem = estimate_kv_memory_gb(bs_eff, max_kv_len, n_layers,
                                    kv_heads, head_dim)
    total = model_mem + kv_mem
    if total > gpu_mem * 0.95:
        return True, (
            "KV cache %.1fGB + model %.1fGB = %.1fGB > %.1fGB GPU memory"
            % (kv_mem, model_mem, total, gpu_mem)
        )
    return False, ""


__all__ = [
    "effective_decode_bs", "iterative_bs_eff", "ttft_queuing_factor",
    "estimate_kv_memory_gb", "estimate_model_memory_gb", "is_saturated",
]
