"""Per-step wall-time lookup for cached-prefill steps on H100 / Llama-3.1-8B.

Reads ``profiling/results/cached_prefill_v3_H100.csv`` — a 25-point measured
grid produced by ``profiling/profile/vllm/cuda_events/cached_prefill_steps_v3.py``
— and exposes ``cached_prefill_step_ms(U, P)`` via bilinear interpolation in
log space.

Why this exists (see plan):
- The roofline's ``T_upper = chunk × prefill_per_token`` is a pure-FLOPs upper
  bound. It misses per-chunk fixed cost (small-chunk GEMM, scheduler Python,
  kernel launch even with CUDA graphs) and FA3-over-P bandwidth cost.
- This file is the only kernel-composition input we plug back in. No fitting
  — just interpolation between measured anchor points.

Grid coverage:
  U ∈ {64, 128, 256, 512, 1024}            (pending new prefill for the session)
  P ∈ {512, 1024, 2048, 4096, 8192}        (already-cached KV)
  prefill_ms ∈ {12.4 … 25.7}                (per-step wall time at chunk=16)

Outside the grid we clamp to the nearest edge — extrapolation in log space
beyond 1024 → 8192 is unreliable.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path


DEFAULT_CSV = Path("profiling/results/cached_prefill_v3_H100.csv")


@dataclass(frozen=True)
class CachedPrefillGrid:
    """Sorted unique U / P axes plus the prefill_ms value at each (U, P)."""
    u_axis: tuple[float, ...]
    p_axis: tuple[float, ...]
    # grid[i][j] = prefill_ms at U=u_axis[i], P=p_axis[j]
    grid: tuple[tuple[float, ...], ...]

    def lookup(self, u: float, p: float) -> float:
        """Bilinear interp in log space; clamps out-of-grid queries to the
        nearest edge.
        """
        if not self.u_axis or not self.p_axis:
            raise RuntimeError("empty cached-prefill grid")

        u_clamped = max(self.u_axis[0], min(self.u_axis[-1], float(u)))
        p_clamped = max(self.p_axis[0], min(self.p_axis[-1], float(p)))
        log_u = math.log(u_clamped)
        log_p = math.log(p_clamped)

        i = _bracket_index([math.log(x) for x in self.u_axis], log_u)
        j = _bracket_index([math.log(x) for x in self.p_axis], log_p)

        u0_log = math.log(self.u_axis[i])
        u1_log = math.log(self.u_axis[i + 1])
        p0_log = math.log(self.p_axis[j])
        p1_log = math.log(self.p_axis[j + 1])

        du = (log_u - u0_log) / max(1e-12, u1_log - u0_log)
        dp = (log_p - p0_log) / max(1e-12, p1_log - p0_log)

        a00 = self.grid[i][j]
        a10 = self.grid[i + 1][j]
        a01 = self.grid[i][j + 1]
        a11 = self.grid[i + 1][j + 1]

        return (
            a00 * (1 - du) * (1 - dp)
            + a10 * du * (1 - dp)
            + a01 * (1 - du) * dp
            + a11 * du * dp
        )


def _bracket_index(axis: list[float], value: float) -> int:
    """Return the lower index of the bracketing pair (i, i+1) for `value` in
    a sorted `axis`. Clamps so i ∈ [0, len(axis) − 2].
    """
    last = len(axis) - 2
    if value <= axis[0]:
        return 0
    if value >= axis[-1]:
        return last
    # Linear scan; axes are tiny (5 entries) — no need for bisect.
    for i in range(last + 1):
        if axis[i] <= value <= axis[i + 1]:
            return i
    return last


def load_grid(path: Path = DEFAULT_CSV) -> CachedPrefillGrid:
    """Parse the CSV into a CachedPrefillGrid. Cached at module level."""
    rows: list[tuple[float, float, float]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append((float(r["U"]), float(r["P"]), float(r["prefill_ms"])))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    u_axis = tuple(sorted({u for u, _, _ in rows}))
    p_axis = tuple(sorted({p for _, p, _ in rows}))
    by_pair = {(u, p): ms for u, p, ms in rows}
    grid = tuple(
        tuple(by_pair[(u, p)] for p in p_axis) for u in u_axis
    )
    # Validate every (u, p) on the grid is present
    for u in u_axis:
        for p in p_axis:
            if (u, p) not in by_pair:
                raise RuntimeError(
                    f"missing measurement for U={u}, P={p} in {path}"
                )
    return CachedPrefillGrid(u_axis=u_axis, p_axis=p_axis, grid=grid)


@cache
def _default_grid() -> CachedPrefillGrid:
    return load_grid(DEFAULT_CSV)


def cached_prefill_step_ms(u_tokens: float, p_tokens: float) -> float:
    """Per-scheduler-step wall time for a session contributing ``u_tokens``
    of new prefill against ``p_tokens`` of already-cached KV.
    """
    return _default_grid().lookup(u_tokens, p_tokens)
