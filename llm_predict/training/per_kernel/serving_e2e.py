"""serving_e2e — ISL/OSL-aware predictor (Phase 1 scaffolding).

Provides `predict_serving_e2e(pred, cfg, isl, osl, bs=1)` returning
`{ttft_ms, tpot_ms, e2el_ms, decode_ms}`.

Phase 1 contract:
- `osl == 0`: returns `ttft_ms = composer.predict_ttft_ms(seq=isl, bs=bs, ...)`,
  decode/tpot/e2el = (ttft, 0, ttft). Output identical to current
  microbench_ttft pipeline so existing seq128 reports stay reproducible.
- `osl > 0`: uses `pred.predict_attention_decode` (memory-bandwidth
  approximation in Phase 1; trained pkl in Phase 2) plus per-layer fixed
  ffn/norm cost from a single decode-step pass. Decode time is integrated
  over kv_cache_len in [isl, isl+osl] via 8-point trapezoidal quadrature
  to avoid O(osl) inner loops for long OSLs.

Phase 2 will retrain per-op pkls with `kv_cache_len > 0` rows and replace
the analytical decode-step approximation with learned predictions. Phase 3
wires this into `validate.py --mode serving_e2e --profile <name>`.

See `.claude/paper/predictor_roadmap.md` (item 3) for the full plan and
`composer.predict_microbench_ttft` for the prefill-only counterpart.
"""
from __future__ import annotations

from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

from . import composer, model_specs

# Per-GPU decode overhead correction factor, calibrated from bench data.
# α = measured_decode / predicted_decode, where decode = e2el - ttft.
# Sources:
#   A100: mean across chat-{short,medium,long} Llama-8B conc=1 → 1.497
#         (bandwidth approx underpredicts; missing dispatch/sync overhead)
#   RTX2080Ti: chat-short Llama-8B conc=1 → 0.605
#         (per-op decode_attn pkl overpredicts; torch.profiler overhead)
#   RTX3090: mean across chat-{short,medium,long} Llama-8B conc=1 -> 0.832
# Applied multiplicatively to predicted decode_ms. Option 2 (ncu decode
# profiling) will eliminate the need for this factor.
_DECODE_CORRECTION: dict[str, float] = {
    "A100": 1.497,
    "RTX3090": 0.832,
    "RTX2080Ti": 0.605,
    "H100": 1.50,
}


def _decode_step_ms(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                    kv_cache_len: int, bs: int = 1, tp: int = 1,
                    shard_tp: bool = True) -> float:
    """Latency for ONE decode step at the given kv_cache_len."""
    d, h, kv, ffn = cfg.d, cfg.h, cfg.kv, cfg.ffn
    h_d = cfg.head_dim
    if shard_tp and tp > 1:
        h_local = max(1, h // tp)
        kv_local = max(1, kv // tp)
        ffn_local = max(1, ffn // tp)
    else:
        h_local, kv_local, ffn_local = h, kv, ffn

    M = bs  # decode step processes 1 token per request
    step_ms = 0.0

    # Attention QKV/O projections at seq=1 effective.
    qkv_n = (h_local + 2 * kv_local) * h_d
    step_ms += max(0.0, pred.predict_gemm(M=M, N=qkv_n, K=d))
    step_ms += max(0.0, pred.predict_gemm(M=M, N=d, K=h_local * h_d))

    # Decode attention: O(bs * kv_cache_len * kv_heads * head_dim) memory-bound.
    decode_attn_ms = pred.predict_attention_decode(
        bs=bs, kv_cache_len=kv_cache_len, n_heads=h_local,
        head_dim=h_d, kv_heads=kv_local,
    )
    if decode_attn_ms > 0:
        step_ms += decode_attn_ms

    # FFN.
    if cfg.is_moe:
        tokens_per_expert = max(1, int(M * cfg.k / max(cfg.E, 1)))
        for _ in range(cfg.k):
            step_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=ffn_local, K=d))
            step_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=ffn_local, K=d))
            step_ms += max(0.0, pred.predict_gemm(M=tokens_per_expert, N=d, K=ffn_local))
    else:
        step_ms += max(0.0, pred.predict_gemm(M=M, N=ffn_local, K=d))
        step_ms += max(0.0, pred.predict_gemm(M=M, N=ffn_local, K=d))
        step_ms += max(0.0, pred.predict_gemm(M=M, N=d, K=ffn_local))

    # Elementwise.
    step_ms += max(0.0, pred.predict_elementwise("rmsnorm", M * d))
    step_ms += max(0.0, pred.predict_elementwise("rmsnorm", M * d))
    step_ms += max(0.0, pred.predict_elementwise("silu", M * ffn_local))
    step_ms += max(0.0, pred.predict_elementwise("mul", M * ffn_local))

    return step_ms * cfg.n_layers


def predict_serving_e2e(pred: PerKernelPredictor, cfg: model_specs.ModelConfig,
                        isl: int, osl: int, bs: int = 1, tp: int = 1,
                        shard_tp: bool = True,
                        include_lm_head: bool = True) -> dict:
    """Predict TTFT + decode -> E2EL for a (model, gpu, isl, osl, bs) workload.

    Returns dict with:
        ttft_ms   - prefill latency at seq=isl
        tpot_ms   - mean per-token decode latency over [0, osl]
        decode_ms - total decode time across osl tokens
        e2el_ms   - ttft_ms + decode_ms

    When osl == 0, decode/tpot are 0 and e2el == ttft (identical numerics
    to composer.predict_ttft_ms).
    """
    ttft_ms = composer.predict_ttft_ms(
        pred, cfg, seq=isl, bs=bs, tp=tp, shard_tp=shard_tp,
        include_lm_head=include_lm_head,
    )

    if osl <= 0:
        return {
            "ttft_ms": float(ttft_ms),
            "tpot_ms": 0.0,
            "decode_ms": 0.0,
            "e2el_ms": float(ttft_ms),
        }

    # Integrate decode_step_ms over kv_cache_len in [isl, isl+osl-1] via
    # 8-point trapezoidal. Decode attention scales linearly with kv_cache_len
    # (memory bandwidth bound), so trapezoidal is exact for the attn term and
    # a tight approximation for the constant ffn/norm terms.
    n_samples = 8
    if osl < n_samples:
        # Few tokens — sum exactly.
        kv_lens = list(range(isl, isl + osl))
        per_step = [_decode_step_ms(pred, cfg, kv, bs=bs, tp=tp, shard_tp=shard_tp)
                    for kv in kv_lens]
        decode_ms = float(sum(per_step))
    else:
        # Sample at n_samples points across [isl, isl+osl-1], integrate.
        kv_samples = [int(isl + (osl - 1) * i / (n_samples - 1))
                      for i in range(n_samples)]
        per_sample = [_decode_step_ms(pred, cfg, kv, bs=bs, tp=tp, shard_tp=shard_tp)
                      for kv in kv_samples]
        # Trapezoidal over (n_samples - 1) segments scales the average step
        # time by the total osl (units: ms/token * tokens = ms).
        decode_ms = 0.0
        for i in range(n_samples - 1):
            seg_avg = 0.5 * (per_sample[i] + per_sample[i + 1])
            seg_tokens = (osl - 1) / (n_samples - 1)
            decode_ms += seg_avg * seg_tokens

    # Apply per-GPU decode correction factor (Option 1 calibration).
    alpha = _DECODE_CORRECTION.get(pred.gpu, 1.0)
    decode_ms *= alpha

    tpot_ms = decode_ms / max(osl, 1)
    e2el_ms = ttft_ms + decode_ms

    return {
        "ttft_ms": float(ttft_ms),
        "tpot_ms": float(tpot_ms),
        "decode_ms": float(decode_ms),
        "e2el_ms": float(e2el_ms),
    }


__all__ = ["predict_serving_e2e"]
