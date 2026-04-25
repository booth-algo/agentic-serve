"""Per-kernel → layer → TTFT composition.

Port of `/root/per-kernel-rebuild/compose_percat.py` using the runtime
`PerKernelPredictor` API instead of raw pkl loads, so any trainer/predictor
feature-spec drift is surfaced here (dogfooding).

For a given (ModelConfig, GPU, input_seq_len, batch, tp), enumerate the ops
in one transformer layer, call the four family predictors per op, sum, and
multiply by n_layers. Returns predicted TTFT in ms.

No measured-TTFT loader here — `validate.py` provides the comparison loop.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

from . import model_specs


def predict_layer_ms(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                       seq: int, bs: int = 1, tp: int = 1, shard_tp: bool = True) -> float:
    """Sum of per-kernel predicted latencies for ONE layer at this (bs, seq)."""
    d, h, kv, ffn = cfg.d, cfg.h, cfg.kv, cfg.ffn
    E, k = cfg.E, cfg.k
    h_d = cfg.head_dim
    if shard_tp and tp > 1:
        h_local   = max(1, h // tp)
        kv_local  = max(1, kv // tp)
        ffn_local = max(1, ffn // tp)
    else:
        h_local, kv_local, ffn_local = h, kv, ffn

    M = bs * seq
    layer_ms = 0.0

    # Attention
    qkv_n = (h_local + 2 * kv_local) * h_d
    layer_ms += max(0.0, pred.predict_gemm(M=M, N=qkv_n, K=d))          # QKV fused
    layer_ms += max(0.0, pred.predict_gemm(M=M, N=d, K=h_local * h_d))  # O
    fa = pred.predict_attention_prefill(bs=bs, seq=seq, n_heads=h_local, head_dim=h_d,
                                         kv_heads=kv_local)
    if fa > 0:
        layer_ms += fa

    # FFN
    if cfg.is_moe:
        tokens_per_expert = max(1, int(M * k / max(E, 1)))
        for _ in range(k):
            layer_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=ffn_local, K=d))
            layer_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=ffn_local, K=d))
            layer_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=d, K=ffn_local))
    else:
        layer_ms += max(0.0, pred.predict_gemm(M=M, N=ffn_local, K=d))  # gate
        layer_ms += max(0.0, pred.predict_gemm(M=M, N=ffn_local, K=d))  # up
        layer_ms += max(0.0, pred.predict_gemm(M=M, N=d, K=ffn_local))  # down

    # Elementwise
    layer_ms += max(0.0, pred.predict_elementwise("rmsnorm", M * d))
    layer_ms += max(0.0, pred.predict_elementwise("rmsnorm", M * d))
    layer_ms += max(0.0, pred.predict_elementwise("silu", M * ffn_local))
    layer_ms += max(0.0, pred.predict_elementwise("mul",  M * ffn_local))

    # Misc ops per layer (calibrated from A100 Llama-8B measured counts:
    # ~5 reduce, ~4 cast, ~4 copy, ~5 splitk_reduce per forward-pass layer).
    # Softmax inside attention is a row-wise reduce over (M, h, seq).
    layer_ms += 2 * max(0.0, pred.predict_misc("reduce", numel=M * h_local * seq))
    # Norm statistics (mean, variance) for the two rmsnorms — reduce over d.
    layer_ms += 3 * max(0.0, pred.predict_misc("reduce", numel=M * d))

    # Dtype-cast kernels around reductions (bf16 ↔ fp32).
    layer_ms += 4 * max(0.0, pred.predict_misc("cast", numel=M * d))

    # Copy kernels (attention internals, residual buffers, KV writeback).
    layer_ms += 4 * max(0.0, pred.predict_misc("copy", numel=M * d))

    # Split-K reducer kernels — one per primary GEMM when cuBLAS splits.
    layer_ms += max(0.0, pred.predict_misc("splitk_reduce", M=M, N=qkv_n, K=d))
    layer_ms += max(0.0, pred.predict_misc("splitk_reduce", M=M, N=d, K=h_local * h_d))
    if cfg.is_moe:
        layer_ms += max(0.0, pred.predict_misc("splitk_reduce", M=M, N=ffn_local, K=d))
        layer_ms += max(0.0, pred.predict_misc("splitk_reduce", M=M, N=d, K=ffn_local))
    else:
        layer_ms += 2 * max(0.0, pred.predict_misc("splitk_reduce", M=M, N=ffn_local, K=d))
        layer_ms += max(0.0, pred.predict_misc("splitk_reduce", M=M, N=d, K=ffn_local))

    return layer_ms


def predict_ttft_ms(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                     seq: int, bs: int = 1, tp: int = 1, shard_tp: bool = True,
                     include_lm_head: bool = True) -> float:
    """Predict TTFT (prefill only) in ms: per_layer × n_layers + optional LM head."""
    per_layer = predict_layer_ms(pred, cfg, seq=seq, bs=bs, tp=tp, shard_tp=shard_tp)
    total = per_layer * cfg.n_layers
    if include_lm_head:
        total += max(0.0, pred.predict_gemm(M=bs * seq, N=cfg.vocab, K=cfg.d))
        total += max(0.0, pred.predict_elementwise("rmsnorm", bs * seq * cfg.d))
    return total


def predict_ttft_batch(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                        seqs: Iterable[int], bs: int = 1, tp: int = 1,
                        shard_tp: bool = True) -> np.ndarray:
    return np.array([predict_ttft_ms(pred, cfg, seq=int(s), bs=bs, tp=tp, shard_tp=shard_tp)
                     for s in seqs], dtype=float)


# Naming-convention alias (locked 2026-04-25; see .claude/paper/predictor_roadmap.md):
# `microbench_ttft` is the prefill-only TTFT track. Use the alias in new code so
# the intent is explicit; the legacy `predict_ttft_ms` stays for backwards compat.
predict_microbench_ttft = predict_ttft_ms

__all__ = ["predict_layer_ms", "predict_ttft_ms", "predict_microbench_ttft",
           "predict_ttft_batch"]
