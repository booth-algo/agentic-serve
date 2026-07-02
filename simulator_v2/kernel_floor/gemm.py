# simulator_v2/kernel_floor/gemm.py

"""GEMM kernel cost (us) from the measured NCU table, interpolated/extrapolated
along the (M, N) axes so vLLM's fused wider-N matmuls price off the table rather
than a roofline. Lookup order in `GemmTable.gemm_us`. Native unit us."""

import csv
from dataclasses import dataclass
from pathlib import Path

from simulator_v2.core.types import GpuConfig
from simulator_v2.core.mode import Mode, mode


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class GemmTable:
    """Measured GEMM latencies (us) keyed by (M, N, K).

    Duplicate (M, N, K) rows are collapsed by MIN on load (see load_gemm_table):
    the raw NCU table carries repeat runs with a wide intra-pair spread
    (~20-56%), and a real vLLM decode step runs its kernels warm and
    back-to-back (CUDA-graphed), landing near the fast end of that spread.
    Validated on the H100 decode trace: min centers the composed floor
    (mean ratio ~1.0) where averaging biased it ~14% hot.
    """
    cells: dict[tuple[int, int, int], float]
    m_by_k: dict[int, tuple[int, ...]]  # ascending measured M for each K
    n_by_k: dict[int, tuple[int, ...]]  # ascending measured N for each K

    def gemm_us(
        self, m: int, n: int, k: int, gpu: GpuConfig, *, dtype_bytes: int = 2
    ) -> float:
        """Cost (us) for a GEMM (M, N, K), in order of preference:

          1. exact (M, N, K) measured           -> return the measured cell
          2. N is a measured column for this K   -> interpolate in M
          3. N is off-column, K has >=2 columns  -> interp/extrap in N
                                                    (each N-node priced via M-interp)
          4. otherwise                           -> analytic roofline fallback

        We default back to the roofline (case 4) when the table can't place the
        query: K absent from the table, M outside the measured M-range, the K
        slice has only a single N column (can't interp in N), or a needed grid
        cell is missing. Cases 2-3 also fall through to roofline if M-interp hits
        an absent cell (non-rectangular slice).
        """
        m, n, k = int(m), int(n), int(k)
        exact = self.cells.get((m, n, k))
        if exact is not None:
            return exact

        m_axis = self.m_by_k.get(k)
        n_axis = self.n_by_k.get(k)
        if not m_axis or not n_axis or m < m_axis[0] or m > m_axis[-1]:
            return _roofline_us(m, n, k, gpu, dtype_bytes)

        # N on a measured column -> only interpolate in M.
        if n in n_axis:
            v = self._interp_in_m(m, n, k, m_axis)
            return v if v is not None else _roofline_us(m, n, k, gpu, dtype_bytes)

        # N off the measured columns -> need >=2 columns to interp/extrap in N.
        if len(n_axis) < 2:
            return _roofline_us(m, n, k, gpu, dtype_bytes)
        v = _linear_on_axis(n_axis, lambda n_node: self._interp_in_m(m, n_node, k, m_axis), n)
        return v if v is not None else _roofline_us(m, n, k, gpu, dtype_bytes)

    def _interp_in_m(self, m: int, n: int, k: int, m_axis: tuple[int, ...]) -> float | None:
        """Cost at (m, n, k) by linear interpolation in M on a measured N column.
        m is assumed within [m_axis[0], m_axis[-1]]. Returns None if a needed cell
        is absent (non-rectangular slice)."""
        exact = self.cells.get((m, n, k))
        if exact is not None:
            return exact
        for lo, hi in zip(m_axis, m_axis[1:]):
            if lo <= m <= hi:
                a = self.cells.get((lo, n, k))
                b = self.cells.get((hi, n, k))
                if a is None or b is None:
                    return None
                t = 0.0 if hi == lo else (m - lo) / (hi - lo)
                return a + t * (b - a)
        return None


@mode(Mode.BACKTEST)
def _linear_on_axis(axis: tuple[int, ...], value_fn, x: int) -> float | None:
    """Linear interp/extrap of value_fn over a sorted axis, evaluated at x.

    Below/above the measured range it extrapolates with the slope of the two
    nearest nodes; above-range results are floored at the edge value (a GEMM
    can't get cheaper as N grows past the measured edge), and all results at 0.
    """
    if x <= axis[0]:
        lo, hi = axis[0], axis[1]
    elif x >= axis[-1]:
        lo, hi = axis[-2], axis[-1]
    else:
        lo = hi = axis[-1]
        for a, b in zip(axis, axis[1:]):
            if a <= x <= b:
                lo, hi = a, b
                break
    v_lo, v_hi = value_fn(lo), value_fn(hi)
    if v_lo is None or v_hi is None:
        return None
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    v = v_lo + t * (v_hi - v_lo)
    if x > axis[-1]:
        v = max(v, v_hi)
    return max(v, 0.0)


@mode(Mode.BACKTEST)
def _roofline_us(m: int, n: int, k: int, gpu: GpuConfig, dtype_bytes: int) -> float:
    """Analytic GEMM roofline in us: max(compute-bound, memory-bound)."""
    flops = 2.0 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * dtype_bytes
    compute_us = flops / (gpu.peak_flops_per_s * gpu.util_flops) * 1e6
    memory_us = bytes_moved / (gpu.peak_bw_bytes_per_s * gpu.util_bw) * 1e6
    return max(compute_us, memory_us)


@mode(Mode.BACKTEST)
def load_gemm_table(path: Path, *, reduce: str = "min") -> GemmTable:
    """forward_pass GEMM CSV (cols: M, N, K, dtype_bytes, latency_us) -> GemmTable.

    `reduce` collapses duplicate (M, N, K) rows: "min" (default -- the warmest
    run, closest to the back-to-back, CUDA-graphed kernels of a real decode step,
    where launch overhead and cold-cache effects are gone) or "mean" (neutral
    average; biases the composed decode floor ~14% hot on the H100 trace).
    """
    sums: dict[tuple[int, int, int], list[float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            sums.setdefault(key, []).append(float(row["latency_us"]))
    if not sums:
        raise RuntimeError(f"no usable GEMM rows in {path}")
    if reduce == "min":
        cells = {key: min(vals) for key, vals in sums.items()}
    elif reduce == "mean":
        cells = {key: sum(vals) / len(vals) for key, vals in sums.items()}
    else:
        raise ValueError(f"unknown reduce={reduce!r}; use 'mean' or 'min'")

    m_by_k: dict[int, set[int]] = {}
    n_by_k: dict[int, set[int]] = {}
    for (m, n, k) in cells:
        m_by_k.setdefault(k, set()).add(m)
        n_by_k.setdefault(k, set()).add(n)
    return GemmTable(
        cells=cells,
        m_by_k={k: tuple(sorted(ms)) for k, ms in m_by_k.items()},
        n_by_k={k: tuple(sorted(ns)) for k, ns in n_by_k.items()},
    )
