"""NCCL all-reduce latency calibration.

Uses empirical allreduce_calibration.json (per GPU) to return per-call latency
for a given transfer size. Paper analytical model (roofline BW) consistently
underestimates measured NCCL allreduce by 2-3× at small sizes due to minimum
latency overhead (ring setup, sync barriers). This module uses the empirical
table instead.

For TP>1 transformers, 2 allreduces per layer (after O-proj, after FFN-down).
"""

import json
import os
from typing import Optional


_cache = None


def get_allreduce_calibration(gpu: str = "A100"):
    """Load allreduce calibration table for a given GPU.

    Returns the `model` dict inside the JSON, containing:
      - peak_bw_gbps : float
      - min_latency_ms : float

    Returns None if no calibration is available for the given GPU.
    """
    global _cache
    if _cache is not None and _cache.get("_gpu") == gpu:
        return _cache
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cal_path = os.path.join(base, "profiling", "data", gpu, "allreduce_calibration.json")
    if not os.path.isfile(cal_path):
        return None
    with open(cal_path) as f:
        data = json.load(f)
    model = data.get("model", {})
    model["_gpu"] = gpu
    _cache = model
    return _cache


def calibrated_allreduce_latency(size_bytes: int, gpu: str = "A100") -> Optional[float]:
    """Return allreduce latency in seconds for `size_bytes` on `gpu`.

    Uses `max(min_latency, 2·size / peak_bw)` — the factor of 2 reflects the
    ring algorithm's 2·(N-1)/N ≈ 2 bandwidth-multiplier for large N.

    Returns None if no calibration data is available.
    """
    cal = get_allreduce_calibration(gpu)
    if cal is None:
        return None
    peak_bw = cal.get("peak_bw_gbps", 297.2) * 1e9
    min_lat = cal.get("min_latency_ms", 0.184) / 1e3
    return max(min_lat, 2 * size_bytes / peak_bw)
