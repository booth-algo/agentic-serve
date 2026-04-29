"""Table-backed GEMM predictor with M-interpolation and roofline fallback.

Replaces XGBoost predict_gemm with:
1. Exact lookup when (M,N,K) is in the profiled table
2. M-interpolation when (N,K) is known but M differs
3. Roofline analytical fallback when outside coverage
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np


class GemmTable:
    """Per-GPU table of profiled GEMM shapes with interpolation."""

    def __init__(self, gpu: str, table_path: Path):
        self.gpu = gpu
        self._table: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self._load(table_path)

    def _load(self, path: Path) -> None:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                M = int(row["M"])
                N = int(row["N"])
                K = int(row["K"])
                ms = float(row["measured_ms"])
                if math.isnan(ms):
                    continue
                key = (N, K)
                if key not in self._table:
                    self._table[key] = []
                self._table[key].append((M, ms))
        # Sort each (N,K) entry by M for binary search interpolation
        for key in self._table:
            self._table[key].sort()
        total = sum(len(v) for v in self._table.values())
        print(f"[GemmTable] {self.gpu}: {len(self._table)} (N,K) groups, {total} total entries")

    def predict(self, M: float, N: int, K: int) -> Optional[float]:
        """Predict GEMM time in ms. Returns None if outside coverage."""
        key = (N, K)
        if key not in self._table:
            return None

        entries = self._table[key]
        M_int = int(round(M))

        # Exact match
        for m, ms in entries:
            if m == M_int:
                return ms

        # Interpolation: find bracketing M values
        ms_below = ms_above = None
        m_below = m_above = None
        for m, ms in entries:
            if m <= M_int:
                m_below, ms_below = m, ms
            if m >= M_int and m_above is None:
                m_above, ms_above = m, ms

        if ms_below is not None and ms_above is not None and m_below != m_above:
            # Log-linear interpolation in M (GEMM scales ~linearly in M for large M,
            # but sub-linearly for small M due to launch overhead)
            frac = (M - m_below) / (m_above - m_below)
            return ms_below + frac * (ms_above - ms_below)

        # Extrapolation from nearest point using linear scaling
        if ms_below is not None:
            return ms_below * (M / max(m_below, 1))
        if ms_above is not None:
            return ms_above * (M / max(m_above, 1))

        return None

    def coverage_distance(self, M: int, N: int, K: int) -> float:
        """Return 0 if exact match, >0 if interpolated, inf if outside."""
        key = (N, K)
        if key not in self._table:
            return float("inf")
        entries = self._table[key]
        ms_list = [m for m, _ in entries]
        if M in ms_list:
            return 0.0
        if ms_list[0] <= M <= ms_list[-1]:
            return min(abs(M - m) for m in ms_list) / max(M, 1)
        return float("inf")


def _roofline_gemm_ms(M: float, N: int, K: int,
                       peak_tflops: float = 100.0,
                       mem_bw_gb_s: float = 2000.0,
                       dtype_bytes: int = 2) -> float:
    """Analytical roofline estimate for GEMM latency."""
    flops = 2.0 * M * N * K
    bytes_io = (M * K + K * N + M * N) * dtype_bytes
    compute_ms = flops / (peak_tflops * 1e9)
    memory_ms = bytes_io / (mem_bw_gb_s * 1e6)
    return max(compute_ms, memory_ms)


_GPU_SPECS = {
    "H100": {"peak_tflops": 990, "mem_bw_gb_s": 3350},  # bf16 tensor core
    "A100": {"peak_tflops": 312, "mem_bw_gb_s": 2039},
    "RTX3090": {"peak_tflops": 142, "mem_bw_gb_s": 936},
    "RTX2080Ti": {"peak_tflops": 107, "mem_bw_gb_s": 616},
}


class HybridGemmPredictor:
    """Table lookup + roofline fallback."""

    def __init__(self, gpu: str, table_path: Path):
        self.gpu = gpu
        self.table = GemmTable(gpu, table_path)
        specs = _GPU_SPECS.get(gpu, {"peak_tflops": 100, "mem_bw_gb_s": 2000})
        self._peak_tflops = specs["peak_tflops"]
        self._mem_bw = specs["mem_bw_gb_s"]

    def predict(self, M: float, N: int, K: int) -> float:
        table_ms = self.table.predict(M, N, K)
        if table_ms is not None:
            return table_ms
        return _roofline_gemm_ms(M, N, K,
                                  peak_tflops=self._peak_tflops,
                                  mem_bw_gb_s=self._mem_bw)

    def is_covered(self, M: int, N: int, K: int) -> bool:
        return self.table.coverage_distance(M, N, K) < float("inf")
