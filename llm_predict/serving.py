"""Serving predictor: analytical concurrency model on top of kernel predictions."""

from dataclasses import dataclass
from typing import Any

from .configs.model_configs import ModelConfig
from .composer import Composer
from .framework_corrections import get_calibration_status


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
    ttft_floor_ms: float = 0.0
    ttft_first_decode_ms: float = 0.0
    ttft_queue_ms: float = 0.0
    calibration_status: str = "missing"
    total_context_tokens: int = 0
    new_prefill_tokens: int = 0
    cached_context_tokens: int = 0
    cache_hit_rate: float = 0.0
    cache_aware_applied: bool = False
    cache_feature_source: str = "none"
    cache_prediction_regime: str = "full_prefill"
    ttft_prediction_supported: bool = True
    unsupported_reason: str | None = None
    multiturn_prediction_mode: str | None = None
    predicted_turn_count: int = 0
    total_successful_turn_requests: int = 0
    mean_predicted_turn_ttft_ms: float = 0.0
    mean_predicted_turn_tpot_ms: float = 0.0
    multiturn_turn_predictions: list[dict[str, Any]] | None = None


def decode_interval_count(output_tokens: int) -> int:
    """Return post-TTFT decode intervals for benchmark TPOT/E2EL semantics."""
    return max(0, int(output_tokens) - 1)


def _integrate_decode_ms(composer: Composer, cfg: ModelConfig,
                         isl: int, decode_steps: int, bs: float,
                         tp: int = 1, n_points: int = 8) -> float:
    if decode_steps <= 0:
        return 0.0
    total = 0.0
    for i in range(n_points):
        t = (i + 0.5) * decode_steps / n_points
        kv_offset = 1 + min(decode_steps - 1, int(t))
        kv_len = isl + kv_offset
        step_us = composer.predict_decode_step_us(cfg, kv_len, bs=max(1, int(bs)),
                                                   tensor_parallel_size=tp)
        total += step_us
    return (total * decode_steps / n_points) / 1000.0


def _iterative_bs_eff(composer: Composer, cfg: ModelConfig,
                      isl: int, decode_steps: int,
                      concurrency: int, tp: int = 1, max_iter: int = 5,
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
        ttft_ms = composer.predict_ttft_ms(cfg, prefill_tokens, kv_len=kv_len,
                                            tensor_parallel_size=tp)
    bs = 1.0
    for _ in range(max_iter):
        decode_ms = _integrate_decode_ms(composer, cfg, isl, decode_steps, bs, tp=tp)
        decode_frac = decode_ms / (ttft_ms + decode_ms) if (ttft_ms + decode_ms) > 0 else 0.5
        bs_new = concurrency * decode_frac
        bs_new = max(1.0, min(bs_new, float(concurrency)))
        bs = (1.0 - damping) * bs_new + damping * bs
    return bs


def predict_serving(composer: Composer, cfg: ModelConfig,
                    gpu: str, isl: int, osl: int,
                    concurrency: int = 1,
                    tensor_parallel_size: int = 1,
                    backend: str | None = None,
                    backend_version: str | None = None,
                    model_key: str | None = None,
                    profile: str | None = None,
                    total_context_tokens: int | None = None,
                    new_prefill_tokens: int | None = None,
                    cached_context_tokens: int | None = None,
                    cache_hit_rate: float | None = None,
                    cache_feature_source: str | None = None,
                    cache_prediction_regime: str | None = None,
                    ttft_prediction_supported: bool = True,
                    unsupported_reason: str | None = None,
                    _sim_ttft_ms: float | None = None,
                    _sim_tpot_ms: float | None = None,
                    _sim_e2el_ms: float | None = None) -> ServingPrediction:
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
    resolved_feature_source = cache_feature_source or ("provided" if cache_aware else "none")
    resolved_regime = cache_prediction_regime or (
        "prefix_cached_prefill" if cache_aware else "full_prefill"
    )
    ttft_kernel = composer.predict_ttft_ms(
        cfg, prefill_tokens, kv_len=total_context,
        tensor_parallel_size=tensor_parallel_size,
    )
    calibration_model = model_key or cfg.name
    if backend:
        calibration_status = get_calibration_status(
            gpu, backend, backend_version, calibration_model
        )
    else:
        calibration_status = "raw_kernel"
    ttft_base = ttft_kernel

    # Fixed per-forward-pass overhead: TP all-reduce barriers + scheduler +
    # CUDA graph / kernel launch. Scales with n_layers and TP.
    # TP all-reduce: ~5 reduces per layer (QKV-attn, O, gate+up, down).
    # Barrier latency ~5us per reduce at TP>1.
    tp_barrier_us = 5.0 * 5.0 * cfg.n_layers if tensor_parallel_size > 1 else 0.0
    scheduler_overhead_us = 500.0  # scheduler loop + kernel launch
    ttft_floor_ms = (tp_barrier_us + scheduler_overhead_us) / 1000.0

    # First decode step: after prefill completes, one decode iteration
    # runs before the first output token appears.
    ttft_first_decode_ms = composer.predict_decode_step_us(
        cfg, kv_len=total_context, bs=1,
        tensor_parallel_size=tensor_parallel_size,
    ) / 1000.0

    # Queue model for simultaneous arrivals in continuous batching.
    # Scheduler interleaves prefill steps and decode steps. With independent
    # prompts (different user messages per request), each request prefills
    # separately. The median request waits for ~concurrency/2 prefills to
    # complete ahead of it, with decode interleaving between each.
    ttft_queue_ms = 0.0
    if concurrency > 1:
        kv_mid = total_context + max(0, int(osl) // 2)
        m = float(concurrency) / 2.0  # median position in queue

        decode_step_ms = composer.predict_decode_step_us(
            cfg, kv_len=kv_mid, bs=max(1, int(m)),
            tensor_parallel_size=tensor_parallel_size,
        ) / 1000.0

        pref_queue = m * ttft_kernel
        decode_queue = m * (m + 1.0) / 2.0 * composer.predict_decode_step_us(
            cfg, kv_len=kv_mid, bs=1,
            tensor_parallel_size=tensor_parallel_size,
        ) / 1000.0
        ttft_queue_ms = pref_queue + decode_queue
    ttft_ms = ttft_kernel + ttft_floor_ms + ttft_first_decode_ms + ttft_queue_ms
    decode_steps = decode_interval_count(osl)

    bs_eff = _iterative_bs_eff(
        composer, cfg, total_context, decode_steps, concurrency,
        tp=tensor_parallel_size,
        ttft_ms_for_batch=ttft_base,
        ttft_prefill_tokens=prefill_tokens,
        ttft_kv_len=total_context,
    )
    decode_total_ms = _integrate_decode_ms(
        composer, cfg, total_context, decode_steps, bs_eff,
        tp=tensor_parallel_size,
    )

    tpot_ms = decode_total_ms / max(decode_steps, 1)
    e2el_ms = ttft_ms + decode_total_ms

    # Override with event-driven simulation values when provided.
    if _sim_ttft_ms is not None:
        ttft_ms = float(_sim_ttft_ms)
    if _sim_tpot_ms is not None:
        tpot_ms = float(_sim_tpot_ms)
    if _sim_e2el_ms is not None:
        e2el_ms = float(_sim_e2el_ms)

    return ServingPrediction(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2el_ms=e2el_ms,
        decode_total_ms=decode_total_ms,
        bs_eff=bs_eff,
        concurrency=concurrency,
        ttft_kernel_ms=ttft_kernel,
        ttft_base_ms=ttft_base,
        ttft_floor_ms=ttft_floor_ms,
        ttft_first_decode_ms=ttft_first_decode_ms,
        ttft_queue_ms=ttft_queue_ms,
        calibration_status=calibration_status,
        total_context_tokens=total_context,
        new_prefill_tokens=prefill_tokens,
        cached_context_tokens=cached_context,
        cache_hit_rate=derived_cache_hit_rate,
        cache_aware_applied=cache_aware,
        cache_feature_source=resolved_feature_source,
        cache_prediction_regime=resolved_regime,
        ttft_prediction_supported=ttft_prediction_supported,
        unsupported_reason=unsupported_reason,
    )
