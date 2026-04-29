"""Flash attention predictor: roofline + XGBoost residual.

Prefill: compute-bound, O(seq^2 * n_heads * head_dim) with tiling.
Decode:  memory-bound, scanning KV cache.
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from ..configs.gpu_specs import GpuSpec, get_gpu
from .roofline import roofline_us

DATA_DIR = Path(__file__).parent.parent / "data" / "flash_attn"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def _flash_attn_flops(seq_len: int, kv_len: int, n_heads: int,
                      head_dim: int) -> float:
    return 2.0 * seq_len * kv_len * n_heads * head_dim


def _flash_attn_bytes(seq_len: int, kv_len: int, n_heads: int,
                      head_dim: int, dtype_bytes: int = 2) -> float:
    q_bytes = seq_len * n_heads * head_dim * dtype_bytes
    kv_bytes = 2 * kv_len * n_heads * head_dim * dtype_bytes
    o_bytes = seq_len * n_heads * head_dim * dtype_bytes
    return q_bytes + kv_bytes + o_bytes


class FlashAttnPredictor:
    def __init__(self, gpu: str):
        self.gpu_name = gpu
        self.gpu = get_gpu(gpu)
        self._table: Optional[dict] = None
        self._xgb = None
        self._load()

    def _load(self):
        table_path = DATA_DIR / f"{self.gpu_name}.csv"
        if table_path.exists():
            self._table = {}
            import csv
            with open(table_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row["seq_len"]), int(row["n_heads"]),
                           int(row["head_dim"]), int(row["causal"]))
                    self._table[key] = float(row["latency_us"])

        model_path = MODEL_DIR / f"flash_{self.gpu_name}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self._xgb = pickle.load(f)

    def predict(self, seq_len: int, n_heads: int, head_dim: int,
                causal: bool = True, kv_len: Optional[int] = None) -> float:
        if kv_len is None:
            kv_len = seq_len

        if self._table:
            exact = self._table.get((seq_len, n_heads, head_dim, int(causal)))
            if exact is not None:
                return exact

        flops = _flash_attn_flops(seq_len, kv_len, n_heads, head_dim)
        bytes_moved = _flash_attn_bytes(seq_len, kv_len, n_heads, head_dim)
        baseline = roofline_us(flops, bytes_moved, self.gpu)

        if self._xgb is not None:
            features = self._make_features(
                seq_len, kv_len, n_heads, head_dim, causal, baseline)
            log_residual = self._xgb.predict(features.reshape(1, -1))[0]
            return baseline * math.exp(log_residual)

        return baseline

    def _make_features(self, seq_len: int, kv_len: int, n_heads: int,
                       head_dim: int, causal: bool,
                       baseline_us: float) -> np.ndarray:
        flops = _flash_attn_flops(seq_len, kv_len, n_heads, head_dim)
        bytes_moved = _flash_attn_bytes(seq_len, kv_len, n_heads, head_dim)
        oi = flops / bytes_moved if bytes_moved > 0 else 0.0
        return np.array([
            math.log2(max(seq_len, 1)),
            math.log2(max(kv_len, 1)),
            math.log2(max(n_heads, 1)),
            math.log2(max(head_dim, 1)),
            float(causal),
            oi,
            math.log(max(baseline_us, 1e-6)),
        ], dtype=np.float32)
