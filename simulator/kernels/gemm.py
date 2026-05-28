"""GEMM predictor: table for M<=2, roofline + XGBoost residual otherwise.

Training target: log(measured_us / roofline_us)
Prediction:      roofline_us * exp(xgb.predict(features))
"""

import math
import pickle
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from ..configs.gpu_specs import GpuSpec, get_gpu
from .roofline import gemm_roofline_us

DATA_DIR = Path(__file__).parent.parent / "data" / "gemm"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
_LOGGER = logging.getLogger(__name__)
_MAX_PREDICT_CACHE = 4096
_MAX_INTERPOLATION_WARNINGS = 1024


class GemmPredictor:
    def __init__(self, gpu: str):
        self.gpu_name = gpu
        self.gpu = get_gpu(gpu)
        self._table: Optional[dict[tuple[int, int, int], float]] = None
        self._m_by_nk: dict[tuple[int, int], list[int]] = {}
        self._predict_cache: OrderedDict[tuple[int, int, int, int], float] = OrderedDict()
        self._interpolation_miss_warned: set[tuple[int, int, int]] = set()
        self._xgb = None
        self._load()

    def _load(self):
        table_path = DATA_DIR / f"{self.gpu_name}.csv"
        if table_path.exists():
            self._table = {}
            self._m_by_nk = {}
            import csv
            with open(table_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row["M"]), int(row["N"]), int(row["K"]))
                    self._table[key] = float(row["latency_us"])
                    self._m_by_nk.setdefault((key[1], key[2]), []).append(key[0])
            for values in self._m_by_nk.values():
                values.sort()

        model_path = MODEL_DIR / f"gemm_{self.gpu_name}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self._xgb = pickle.load(f)

    def predict(self, M: int, N: int, K: int, dtype_bytes: int = 2) -> float:
        cache_key = (M, N, K, dtype_bytes)
        cached = self._predict_cache.get(cache_key)
        if cached is not None:
            self._predict_cache.move_to_end(cache_key)
            return cached

        if self._table and M <= 2:
            exact = self._table.get((M, N, K))
            if exact is not None:
                self._remember_prediction(cache_key, exact)
                return exact

        if self._table:
            interp = self._interpolate_M(M, N, K)
            if interp is not None:
                self._remember_prediction(cache_key, interp)
                return interp

        baseline = gemm_roofline_us(M, N, K, self.gpu, dtype_bytes)

        if self._xgb is not None:
            features = self._make_features(M, N, K, dtype_bytes, baseline)
            log_residual = self._xgb.predict(features.reshape(1, -1))[0]
            predicted = baseline * math.exp(log_residual)
            self._remember_prediction(cache_key, predicted)
            return predicted

        self._remember_prediction(cache_key, baseline)
        return baseline

    def _remember_prediction(self, key: tuple[int, int, int, int],
                             value: float) -> None:
        self._predict_cache[key] = value
        self._predict_cache.move_to_end(key)
        if len(self._predict_cache) > _MAX_PREDICT_CACHE:
            self._predict_cache.popitem(last=False)

    def _interpolate_M(self, M: int, N: int, K: int) -> Optional[float]:
        if self._table is None:
            return None
        ms = self._m_by_nk.get((N, K), [])
        if len(ms) < 2:
            return None
        if M < ms[0] or M > ms[-1]:
            return None
        for i in range(len(ms) - 1):
            if ms[i] <= M <= ms[i + 1]:
                lo, hi = ms[i], ms[i + 1]
                t = (M - lo) / (hi - lo) if hi != lo else 0.0
                v_lo = self._table[(lo, N, K)]
                v_hi = self._table[(hi, N, K)]
                return v_lo + t * (v_hi - v_lo)
        key = (M, N, K)
        if (key not in self._interpolation_miss_warned
                and len(self._interpolation_miss_warned) < _MAX_INTERPOLATION_WARNINGS):
            self._interpolation_miss_warned.add(key)
            _LOGGER.warning(
                "GEMM interpolation failed despite in-range M grid; "
                "falling back to roofline/residual for gpu=%s M=%s N=%s K=%s",
                self.gpu_name, M, N, K,
            )
        return None

    def _make_features(self, M: int, N: int, K: int,
                       dtype_bytes: int, baseline_us: float) -> np.ndarray:
        flops = 2.0 * M * N * K
        bytes_moved = (M * K + K * N + M * N) * dtype_bytes
        oi = flops / bytes_moved if bytes_moved > 0 else 0.0
        return np.array([
            math.log2(max(M, 1)),
            math.log2(max(N, 1)),
            math.log2(max(K, 1)),
            oi,
            math.log(max(baseline_us, 1e-6)),
        ], dtype=np.float32)
