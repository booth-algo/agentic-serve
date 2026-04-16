"""MoE dispatch corrections.

The per-op and per-category predictors both underestimate MoE models because:
  1. vLLM's fused_moe kernel bundles routing + gate/up/down + combine in one
     CUDA kernel whose latency scales differently from the sum of per-expert
     GEMMs our predictors construct.
  2. Expert imbalance: some experts receive more tokens than the average
     tok·k/E, bottlenecking the whole batch.
  3. MoE decode has additional HBM traffic (weight loading thrashes cache
     when experts are spread across many small weight tensors).

These corrections are post-prediction multiplicative/additive scaling factors
calibrated against measured Mixtral-8x7B and gpt-oss-20b/120b TTFT on A100.
"""


def post_prediction_correction(
    pred_us: float,
    n_tokens: int,
    d_model: int,
    num_experts: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """Apply targeted post-prediction corrections for known weak regimes.

    1. MoE decode: expert weight loading causes ~2.5× more HBM traffic than
       analytical model predicts (cache thrashing from separate weight tensors).
    2. Small-seq overhead: kernel launch + MoE routing has a minimum floor.

    NOTE: Constants here are calibrated from A100 measurements. For other GPUs
    the scale factors should be refit — see `calibration/framework.py` for the
    framework-overhead counterpart.
    """
    corrected = pred_us

    # MoE decode correction: when tok is small and model has many experts,
    # FFN weight loading is much slower than bandwidth model predicts
    is_decode = (seq_len <= 1 and n_tokens <= 8)
    if is_decode and num_experts >= 8:
        moe_decode_scale = 1.0 + 0.15 * num_experts
        corrected = pred_us * min(moe_decode_scale, 4.0)

    # Small-seq minimum floor: kernel launch overhead
    if n_tokens * seq_len <= 64:
        min_floor_us = 150.0 + d_model * 0.1
        if num_experts > 0:
            min_floor_us += 300.0 * min(num_experts, 8)
        corrected = max(corrected, min_floor_us)

    return corrected
