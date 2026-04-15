"""
Post-prediction corrections and allreduce calibration for per-op predictor.

These functions apply targeted corrections for known weak prediction regimes
and load hardware-specific calibration data for allreduce latency.
"""

import os
import json


def post_prediction_correction(pred_us, n_tokens, d_model, num_experts, batch_size, seq_len):
    """Apply targeted post-prediction corrections for known weak regimes.

    1. MoE decode: expert weight loading causes ~2.5x more HBM traffic than
       analytical model predicts (cache thrashing from separate weight tensors).
    2. Small-seq overhead: kernel launch + MoE routing has a minimum floor.
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


_allreduce_calibration_cache = None


def get_allreduce_calibration():
    global _allreduce_calibration_cache
    if _allreduce_calibration_cache is not None:
        return _allreduce_calibration_cache
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cal_path = os.path.join(base, 'profiling', 'data', 'A100', 'allreduce_calibration.json')
    if os.path.isfile(cal_path):
        with open(cal_path) as f:
            data = json.load(f)
        _allreduce_calibration_cache = data.get('model', {})
    return _allreduce_calibration_cache


def calibrated_allreduce_latency(size_bytes):
    cal = get_allreduce_calibration()
    if cal:
        peak_bw = cal.get('peak_bw_gbps', 297.2) * 1e9
        min_lat = cal.get('min_latency_ms', 0.184) / 1e3
        return max(min_lat, 2 * size_bytes / peak_bw)
    return None
