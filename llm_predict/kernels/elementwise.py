"""Elementwise kernel predictors: rmsnorm, silu_mul, rotary_emb, residual_add.

All memory-bound. Model: latency_us = floor_us + bytes_moved / effective_bw
One (floor_us, effective_bw) pair per kernel type per GPU, calibrated from
a handful of ncu measurements.
"""

import json
from pathlib import Path

from ..configs.gpu_specs import GpuSpec, get_gpu

DATA_DIR = Path(__file__).parent.parent / "data" / "elementwise"

_DEFAULTS = {
    "rmsnorm":      {"reads": 1, "writes": 1},
    "silu_mul":     {"reads": 2, "writes": 1},
    "rotary_emb":   {"reads": 1, "writes": 1},
    "residual_add": {"reads": 2, "writes": 1},
}


class ElementwisePredictor:
    def __init__(self, gpu: str, kernel_type: str):
        if kernel_type not in _DEFAULTS:
            raise ValueError(
                f"Unknown kernel type: {kernel_type}. "
                f"Available: {list(_DEFAULTS)}")
        self.gpu_name = gpu
        self.gpu = get_gpu(gpu)
        self.kernel_type = kernel_type
        self._io = _DEFAULTS[kernel_type]
        self._floor_us: float = self.gpu.kernel_floor_us
        self._eff_bw_gb_s: float = self.gpu.hbm_bw_gb_s * 0.8
        self._load()

    def _load(self):
        cal_path = DATA_DIR / f"{self.gpu_name}.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            if self.kernel_type in cal:
                entry = cal[self.kernel_type]
                self._floor_us = entry.get("floor_us", self._floor_us)
                self._eff_bw_gb_s = entry.get("eff_bw_gb_s", self._eff_bw_gb_s)

    def bytes_moved(self, tensor_elements: int, dtype_bytes: int = 2) -> float:
        n_tensors = self._io["reads"] + self._io["writes"]
        return tensor_elements * dtype_bytes * n_tensors

    def predict(self, tensor_elements: int, dtype_bytes: int = 2) -> float:
        bm = self.bytes_moved(tensor_elements, dtype_bytes)
        return self._floor_us + bm / (self._eff_bw_gb_s * 1e3)


class RmsNormPredictor(ElementwisePredictor):
    def __init__(self, gpu: str):
        super().__init__(gpu, "rmsnorm")

    def predict_from_shape(self, hidden_dim: int, seq_len: int,
                           bs: int = 1, dtype_bytes: int = 2) -> float:
        return self.predict(hidden_dim * seq_len * bs, dtype_bytes)


class SiluMulPredictor(ElementwisePredictor):
    def __init__(self, gpu: str):
        super().__init__(gpu, "silu_mul")

    def predict_from_shape(self, intermediate_size: int, seq_len: int,
                           bs: int = 1, dtype_bytes: int = 2) -> float:
        return self.predict(intermediate_size * seq_len * bs, dtype_bytes)


class RotaryEmbPredictor(ElementwisePredictor):
    def __init__(self, gpu: str):
        super().__init__(gpu, "rotary_emb")

    def predict_from_shape(self, n_heads: int, head_dim: int, seq_len: int,
                           bs: int = 1, dtype_bytes: int = 2) -> float:
        return self.predict(n_heads * head_dim * seq_len * bs, dtype_bytes)


class ResidualAddPredictor(ElementwisePredictor):
    def __init__(self, gpu: str):
        super().__init__(gpu, "residual_add")

    def predict_from_shape(self, hidden_dim: int, seq_len: int,
                           bs: int = 1, dtype_bytes: int = 2) -> float:
        return self.predict(hidden_dim * seq_len * bs, dtype_bytes)
