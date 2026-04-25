"""serving_e2e — ISL/OSL and concurrency-aware latency predictor.

Provides `predict_serving_e2e(pred, cfg, isl, osl, concurrency=1)`
returning `{ttft_ms, tpot_ms, e2el_ms, decode_ms, bs_eff, saturated}`.

See `.claude/paper/concurrency_model_plan.md` for the full design.
"""
from __future__ import annotations

from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

from . import composer, model_specs, concurrency_model

_DECODE_CORRECTION: dict[str, float] = {
    "A100": 1.497,
    "RTX3090": 0.832,
    "RTX2080Ti": 0.605,
    "H100": 1.50,
}

_GEMM_INTERP_THRESHOLD = 128


def _predict_gemm_decode(pred, M: float, N: int, K: int) -> float:
    """GEMM prediction safe for decode batch sizes (M=1..500+).
    
    XGBoost was trained on M=1 and M>=128. At 1 < M < 128 it produces
    non-monotonic values. Linearly interpolates between M=1 and M=128
    in that range — physically correct since small-M GEMMs are
    bandwidth-bound with near-constant latency.
    """
    if M <= 1:
        return max(0.0, pred.predict_gemm(M=1, N=N, K=K))
    if M >= _GEMM_INTERP_THRESHOLD:
        return max(0.0, pred.predict_gemm(M=int(round(M)), N=N, K=K))
    t1 = max(0.0, pred.predict_gemm(M=1, N=N, K=K))
    t128 = max(0.0, pred.predict_gemm(M=_GEMM_INTERP_THRESHOLD, N=N, K=K))
    frac = (M - 1) / (_GEMM_INTERP_THRESHOLD - 1)
    return t1 + (t128 - t1) * frac


def _decode_step_ms(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                    kv_cache_len: int, bs: float = 1.0, tp: int = 1,
                    shard_tp: bool = True) -> float:
    d, h, kv, ffn = cfg.d, cfg.h, cfg.kv, cfg.ffn
    h_d = cfg.head_dim
    if shard_tp and tp > 1:
        h_local = max(1, h // tp)
        kv_local = max(1, kv // tp)
        ffn_local = max(1, ffn // tp)
    else:
        h_local, kv_local, ffn_local = h, kv, ffn

    M = bs
    step_ms = 0.0

    qkv_n = (h_local + 2 * kv_local) * h_d
    step_ms += _predict_gemm_decode(pred, M=M, N=qkv_n, K=d)
    step_ms += _predict_gemm_decode(pred, M=M, N=d, K=h_local * h_d)

    decode_attn_ms = pred.predict_attention_decode(
        bs=max(1, int(round(bs))), kv_cache_len=kv_cache_len,
        n_heads=h_local, head_dim=h_d, kv_heads=kv_local,
    )
    if decode_attn_ms > 0:
        step_ms += decode_attn_ms

    if cfg.is_moe:
        tokens_per_expert = max(1, int(M * cfg.k / max(cfg.E, 1)))
        for _ in range(cfg.k):
            step_ms += _predict_gemm_decode(pred, M=tokens_per_expert, N=ffn_local, K=d)
            step_ms += _predict_gemm_decode(pred, M=tokens_per_expert, N=ffn_local, K=d)
            step_ms += _predict_gemm_decode(pred, M=tokens_per_expert, N=d, K=ffn_local)
    else:
        step_ms += _predict_gemm_decode(pred, M=M, N=ffn_local, K=d)
        step_ms += _predict_gemm_decode(pred, M=M, N=ffn_local, K=d)
        step_ms += _predict_gemm_decode(pred, M=M, N=d, K=ffn_local)

    step_ms += max(0.0, pred.predict_elementwise("rmsnorm", int(M) * d))
    step_ms += max(0.0, pred.predict_elementwise("rmsnorm", int(M) * d))
    step_ms += max(0.0, pred.predict_elementwise("silu", int(M) * ffn_local))
    step_ms += max(0.0, pred.predict_elementwise("mul", int(M) * ffn_local))

    return step_ms * cfg.n_layers


def _integrate_decode_ms(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                         isl: int, osl: int, bs: float = 1.0,
                         tp: int = 1, shard_tp: bool = True) -> float:
    if osl <= 0:
        return 0.0
    n_samples = 8
    if osl < n_samples:
        kv_lens = list(range(isl, isl + osl))
        per_step = [_decode_step_ms(pred, cfg, kv, bs=bs, tp=tp, shard_tp=shard_tp)
                    for kv in kv_lens]
        return float(sum(per_step))

    kv_samples = [int(isl + (osl - 1) * i / (n_samples - 1))
                  for i in range(n_samples)]
    per_sample = [_decode_step_ms(pred, cfg, kv, bs=bs, tp=tp, shard_tp=shard_tp)
                  for kv in kv_samples]
    decode_ms = 0.0
    for i in range(n_samples - 1):
        seg_avg = 0.5 * (per_sample[i] + per_sample[i + 1])
        seg_tokens = (osl - 1) / (n_samples - 1)
        decode_ms += seg_avg * seg_tokens
    return decode_ms


def predict_serving_e2e(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                        isl: int, osl: int, bs: int = 1,
                        concurrency: int = 1,
                        tp: int = 1, shard_tp: bool = True,
                        include_lm_head: bool = True) -> dict:
    """Predict TTFT + decode -> E2EL for a (model, gpu, isl, osl, concurrency).

    Returns dict with:
        ttft_ms, tpot_ms, decode_ms, e2el_ms, bs_eff, saturated
    """
    ttft_bs1 = composer.predict_ttft_ms(
        pred, cfg, seq=isl, bs=1, tp=tp, shard_tp=shard_tp,
        include_lm_head=include_lm_head,
    )

    if osl <= 0:
        return {
            "ttft_ms": float(ttft_bs1), "tpot_ms": 0.0,
            "decode_ms": 0.0, "e2el_ms": float(ttft_bs1),
            "bs_eff": 1.0, "saturated": False,
        }

    alpha_base = _DECODE_CORRECTION.get(pred.gpu, 1.0)

    if concurrency <= 1:
        decode_ms_raw = _integrate_decode_ms(pred, cfg, isl, osl,
                                              bs=float(bs), tp=tp, shard_tp=shard_tp)
        decode_ms = decode_ms_raw * alpha_base
        tpot_ms = decode_ms / max(osl, 1)
        return {
            "ttft_ms": float(ttft_bs1), "tpot_ms": float(tpot_ms),
            "decode_ms": float(decode_ms), "e2el_ms": float(ttft_bs1 + decode_ms),
            "bs_eff": 1.0, "saturated": False,
        }

    def _alpha_at_bs(bs_val: float) -> float:
        if bs_val <= 1:
            return alpha_base
        return alpha_base * (bs_val ** -0.13)

    def tpot_at_bs(bs_val: float) -> float:
        raw = _integrate_decode_ms(pred, cfg, isl, osl,
                                    bs=bs_val, tp=tp, shard_tp=shard_tp)
        return (raw * _alpha_at_bs(bs_val)) / max(osl, 1)

    bs_eff = concurrency_model.iterative_bs_eff(
        concurrency=concurrency, tpot_at_bs=tpot_at_bs,
        ttft_ms=ttft_bs1, osl=osl,
    )

    decode_ms = _integrate_decode_ms(pred, cfg, isl, osl,
                                      bs=bs_eff, tp=tp, shard_tp=shard_tp) * _alpha_at_bs(bs_eff)
    tpot_ms = decode_ms / max(osl, 1)

    tpot_1 = tpot_at_bs(1.0)
    ttft_factor = concurrency_model.ttft_queuing_factor(
        concurrency, ttft_bs1, tpot_1, osl, gpu=pred.gpu,
    )
    ttft_ms = ttft_bs1 * ttft_factor

    saturated, _ = concurrency_model.is_saturated(
        bs_eff=bs_eff, max_kv_len=isl + osl,
        n_layers=cfg.n_layers, kv_heads=cfg.kv,
        head_dim=cfg.head_dim, d=cfg.d, ffn=cfg.ffn, gpu=pred.gpu,
    )

    return {
        "ttft_ms": float(ttft_ms), "tpot_ms": float(tpot_ms),
        "decode_ms": float(decode_ms), "e2el_ms": float(ttft_ms + decode_ms),
        "bs_eff": float(bs_eff), "saturated": saturated,
    }


__all__ = ["predict_serving_e2e"]
