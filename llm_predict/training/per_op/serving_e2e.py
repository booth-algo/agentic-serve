"""Per-op serving_e2e — ISL/OSL-aware predictor using per-op XGBoost.

Ablation counterpart to the per-kernel `serving_e2e.py`. Uses a single
`PerOpPredictor` (perop_v5_shape.pkl) instead of family-specific kernel
pkls. Composes per-op predictions (attn, ffn, norm_pre, norm_post)
across layers for both prefill and decode.

Uses feature_spec.compute_features (28-dim v5, includes kv_cache_len)
rather than the runtime features.py (26-dim, pre-v5) to match the
trained pkl.
"""
from __future__ import annotations

from llm_predict.predictors.per_op.predictor import PerOpPredictor
from llm_predict.training.per_op.feature_spec import compute_features
from llm_predict.training.per_kernel import model_specs

LAYER_OPS = ("attn", "ffn", "norm_pre", "norm_post")

DECODE_CORRECTION: dict[str, float] = {
    "A100": 1.0,
    "RTX3090": 1.0,
    "RTX2080Ti": 1.0,
    "H100": 1.0,
}


def _predict_layer_us(pred: PerOpPredictor, cfg: model_specs.ModelConfig,
                      bs: int, seq_len: int, kv_cache_len: int) -> float:
    total_us = 0.0
    for op in LAYER_OPS:
        row = {
            "op": op,
            "n_tokens": bs * seq_len,
            "bs": bs,
            "seq": seq_len,
            "d": cfg.d,
            "h": cfg.h,
            "kv": cfg.kv,
            "ffn": cfg.ffn,
            "E": cfg.E,
            "k": cfg.k,
            "kv_cache_len": kv_cache_len,
        }
        feats = compute_features(row)
        val = pred.predict(feats)
        if val is not None and val > 0:
            total_us += val
    return total_us


def _decode_step_ms(pred: PerOpPredictor, cfg: model_specs.ModelConfig,
                    kv_cache_len: int, bs: int = 1) -> float:
    layer_us = _predict_layer_us(pred, cfg, bs=bs, seq_len=1,
                                 kv_cache_len=kv_cache_len)
    return layer_us * cfg.n_layers / 1000.0


def predict_serving_e2e_perop(pred: PerOpPredictor, cfg: model_specs.ModelConfig,
                              isl: int, osl: int, bs: int = 1,
                              gpu: str = "RTX2080Ti") -> dict:
    layer_us = _predict_layer_us(pred, cfg, bs=bs, seq_len=isl, kv_cache_len=0)
    ttft_ms = layer_us * cfg.n_layers / 1000.0

    if osl <= 0:
        return {
            "ttft_ms": float(ttft_ms),
            "tpot_ms": 0.0,
            "decode_ms": 0.0,
            "e2el_ms": float(ttft_ms),
        }

    n_samples = 8
    if osl < n_samples:
        kv_lens = list(range(isl, isl + osl))
        per_step = [_decode_step_ms(pred, cfg, kv, bs=bs) for kv in kv_lens]
        decode_ms = float(sum(per_step))
    else:
        kv_samples = [int(isl + (osl - 1) * i / (n_samples - 1))
                      for i in range(n_samples)]
        per_sample = [_decode_step_ms(pred, cfg, kv, bs=bs) for kv in kv_samples]
        decode_ms = 0.0
        for i in range(n_samples - 1):
            seg_avg = 0.5 * (per_sample[i] + per_sample[i + 1])
            seg_tokens = (osl - 1) / (n_samples - 1)
            decode_ms += seg_avg * seg_tokens

    alpha = DECODE_CORRECTION.get(gpu, 1.0)
    decode_ms *= alpha

    tpot_ms = decode_ms / max(osl, 1)
    e2el_ms = ttft_ms + decode_ms

    return {
        "ttft_ms": float(ttft_ms),
        "tpot_ms": float(tpot_ms),
        "decode_ms": float(decode_ms),
        "e2el_ms": float(e2el_ms),
    }


__all__ = ["predict_serving_e2e_perop"]
