"""Serving predictor: analytical concurrency model on top of kernel predictions.

Composes TTFT, TPOT, E2EL at arbitrary concurrency using Little's Law
steady-state batch size, iterative solver, and TTFT queuing.
"""

import math
from dataclasses import dataclass

from .configs.model_configs import ModelConfig
from .composer import Composer
from .framework_corrections import (
    decode_correction_factor,
    framework_correction,
    get_calibration_status,
    moe_decode_correction_factor,
    prefix_cache_contention_factors,
    ttft_queue_factor,
)

_DECODE_CORRECTION: dict[str, dict] = {
    "A100":      {"alpha_base": 1.0, "exponent": 0.0},
    "RTX3090":   {"alpha_base": 1.0, "exponent": 0.0},
    "RTX2080Ti": {"alpha_base": 1.0, "exponent": 0.0},
    "H100":      {"alpha_base": 1.0, "exponent": 0.0},
}

_TTFT_QUEUE: dict[str, dict] = {
    "A100":      {"K": 0.3, "c_thresh": 30},
    "RTX3090":   {"K": 0.3, "c_thresh": 30},
    "RTX2080Ti": {"K": 0.3, "c_thresh": 20},
    "H100":      {"K": 0.6, "c_thresh": 100},
}


@dataclass
class ServingPrediction:
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float
    decode_total_ms: float
    bs_eff: float
    concurrency: int
    ttft_kernel_ms: float = 0.0
    ttft_base_ms: float = 0.0
    ttft_queue_factor: float = 1.0
    ttft_correction_applied: bool = False
    ttft_queue_applied: bool = False
    decode_correction_factor: float = 1.0
    decode_correction_applied: bool = False
    moe_decode_correction_applied: bool = False
    calibration_status: str = "missing"
    total_context_tokens: int = 0
    new_prefill_tokens: int = 0
    cached_context_tokens: int = 0
    cache_hit_rate: float = 0.0
    cache_aware_applied: bool = False
    prefix_cache_ttft_factor: float = 1.0
    prefix_cache_decode_factor: float = 1.0
    prefix_cache_contention_applied: bool = False


def _decode_alpha(gpu: str, bs: float) -> float:
    params = _DECODE_CORRECTION.get(gpu, {"alpha_base": 1.0, "exponent": 0.0})
    alpha = params["alpha_base"] * (bs ** params["exponent"])
    floor = params.get("alpha_floor")
    if floor is not None:
        alpha = max(alpha, floor)
    return alpha


def _ttft_queuing_factor(gpu: str, concurrency: int) -> float:
    params = _TTFT_QUEUE.get(gpu, {"K": 0.3, "c_thresh": 30})
    if concurrency <= 1:
        return 1.0
    c = min(concurrency, params["c_thresh"])
    return 1.0 + params["K"] * math.log(c)


def _integrate_decode_ms(composer: Composer, cfg: ModelConfig,
                         isl: int, osl: int, bs: float,
                         n_points: int = 8) -> float:
    if osl <= 0:
        return 0.0
    total = 0.0
    for i in range(n_points):
        t = (i + 0.5) * osl / n_points
        kv_len = isl + int(t)
        step_us = composer.predict_decode_step_us(cfg, kv_len, bs=max(1, int(bs)))
        total += step_us
    return (total * osl / n_points) / 1000.0


def _iterative_bs_eff(composer: Composer, cfg: ModelConfig,
                      gpu: str, isl: int, osl: int,
                      concurrency: int, max_iter: int = 5,
                      damping: float = 0.3,
                      ttft_ms_for_batch: float | None = None,
                      ttft_prefill_tokens: int | None = None,
                      ttft_kv_len: int | None = None) -> float:
    if concurrency <= 1:
        return 1.0

    ttft_ms = ttft_ms_for_batch
    if ttft_ms is None:
        prefill_tokens = ttft_prefill_tokens if ttft_prefill_tokens is not None else isl
        kv_len = ttft_kv_len if ttft_kv_len is not None else isl
        ttft_ms = composer.predict_ttft_ms(cfg, prefill_tokens, kv_len=kv_len)
    bs = 1.0
    for _ in range(max_iter):
        decode_ms = _integrate_decode_ms(composer, cfg, isl, osl, bs)
        alpha = _decode_alpha(gpu, bs)
        decode_ms *= alpha
        decode_frac = decode_ms / (ttft_ms + decode_ms) if (ttft_ms + decode_ms) > 0 else 0.5
        bs_new = concurrency * decode_frac
        bs_new = max(1.0, min(bs_new, float(concurrency)))
        bs = (1.0 - damping) * bs_new + damping * bs
    return bs


def predict_serving(composer: Composer, cfg: ModelConfig,
                    gpu: str, isl: int, osl: int,
                    concurrency: int = 1,
                    backend: str | None = None,
                    backend_version: str | None = None,
                    model_key: str | None = None,
                    profile: str | None = None,
                    total_context_tokens: int | None = None,
                    new_prefill_tokens: int | None = None,
                    cached_context_tokens: int | None = None,
                    cache_hit_rate: float | None = None,
                    apply_prefix_contention: bool = True) -> ServingPrediction:
    total_context = max(1, int(total_context_tokens if total_context_tokens is not None else isl))
    if new_prefill_tokens is None:
        if cached_context_tokens is not None:
            prefill_tokens = total_context - int(cached_context_tokens)
        elif cache_hit_rate is not None:
            hit_rate = max(0.0, min(1.0, float(cache_hit_rate)))
            prefill_tokens = round(total_context * (1.0 - hit_rate))
        else:
            prefill_tokens = total_context
    else:
        prefill_tokens = int(new_prefill_tokens)
    prefill_tokens = max(1, min(prefill_tokens, total_context))

    if cached_context_tokens is None or new_prefill_tokens is not None:
        cached_context = max(0, total_context - prefill_tokens)
    else:
        cached_context = max(0, min(int(cached_context_tokens), total_context - 1))

    if cache_hit_rate is None:
        derived_cache_hit_rate = cached_context / total_context
    else:
        derived_cache_hit_rate = max(0.0, min(1.0, float(cache_hit_rate)))

    cache_aware = prefill_tokens < total_context or derived_cache_hit_rate > 0.0
    ttft_kernel = composer.predict_ttft_ms(
        cfg, prefill_tokens, kv_len=total_context
    )
    calibration_model = model_key or cfg.name
    if backend:
        calibration_status = get_calibration_status(
            gpu, backend, backend_version, calibration_model
        )
        ttft_base, corr_applied = framework_correction(
            gpu, backend, ttft_kernel, backend_version, calibration_model
        )
        queue_factor, queue_applied = ttft_queue_factor(
            gpu, backend, concurrency, backend_version, calibration_model, profile
        )
    else:
        calibration_status = "raw_kernel"
        ttft_base = ttft_kernel
        corr_applied = False
        queue_factor = _ttft_queuing_factor(gpu, concurrency)
        queue_applied = False
    ttft_ms = ttft_base * queue_factor

    bs_eff = _iterative_bs_eff(
        composer, cfg, gpu, total_context, osl, concurrency,
        ttft_ms_for_batch=ttft_base,
        ttft_prefill_tokens=prefill_tokens,
        ttft_kv_len=total_context,
    )
    alpha = _decode_alpha(gpu, bs_eff)
    decode_total_ms = _integrate_decode_ms(
        composer, cfg, total_context, osl, bs_eff
    ) * alpha
    if backend:
        if cfg.is_moe:
            decode_factor, decode_applied = moe_decode_correction_factor(
                gpu, backend, backend_version, calibration_model, concurrency, profile
            )
            moe_decode_applied = decode_applied
        else:
            decode_factor, decode_applied = decode_correction_factor(
                gpu, backend, concurrency, backend_version, calibration_model, profile
            )
            moe_decode_applied = False
        decode_total_ms *= decode_factor
    else:
        decode_factor = 1.0
        decode_applied = False
        moe_decode_applied = False

    prefix_ttft_factor = 1.0
    prefix_decode_factor = 1.0
    prefix_applied = False
    if backend and cache_aware and apply_prefix_contention:
        prefix_ttft_factor, prefix_decode_factor, prefix_applied = (
            prefix_cache_contention_factors(
                gpu, backend, backend_version, calibration_model,
                concurrency, profile
            )
        )
        if prefix_applied:
            ttft_ms *= prefix_ttft_factor
            decode_total_ms *= prefix_decode_factor

    tpot_ms = decode_total_ms / max(osl, 1)
    e2el_ms = ttft_ms + decode_total_ms

    return ServingPrediction(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2el_ms=e2el_ms,
        decode_total_ms=decode_total_ms,
        bs_eff=bs_eff,
        concurrency=concurrency,
        ttft_kernel_ms=ttft_kernel,
        ttft_base_ms=ttft_base,
        ttft_queue_factor=queue_factor,
        ttft_correction_applied=corr_applied,
        ttft_queue_applied=queue_applied,
        decode_correction_factor=decode_factor,
        decode_correction_applied=decode_applied,
        moe_decode_correction_applied=moe_decode_applied,
        calibration_status=calibration_status,
        total_context_tokens=total_context,
        new_prefill_tokens=prefill_tokens,
        cached_context_tokens=cached_context,
        cache_hit_rate=derived_cache_hit_rate,
        cache_aware_applied=cache_aware,
        prefix_cache_ttft_factor=prefix_ttft_factor,
        prefix_cache_decode_factor=prefix_decode_factor,
        prefix_cache_contention_applied=prefix_applied,
    )
