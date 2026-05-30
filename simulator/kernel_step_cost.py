"""Per-step decode wall-time from the measured H100 / Llama-3.1-8B kernel grid.

Reads ``profiling/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv``
— a measured (batch_size, context_len) → ``decode_step_ms`` grid (validated at
7.4% MAPE; see ``profiling/docs/prediction_pipeline.yaml`` ``decode`` block) —
and exposes ``decode_step_ms(B, T)`` via bilinear interpolation in log space.

Why this exists:
- The closed-form roofline ``T_min`` underweights the per-step *fixed cost*
  (CUDA-graph launch, GEMM, sampling — ~6.5 ms floor at small batch) and is the
  decode-step *engine* time, which the measured grid gives directly. This is the
  physically-correct lower bound for per-turn TPOT (the ``kernel_step`` floor in
  the amplifier law — see the project memory ``tpot-amplifier-pressure-law``).
- It is the second kernel-composition input we plug back in (alongside
  ``cached_prefill_lookup``). No fitting — interpolation between measured anchors.

Grid coverage (large sweep):
  B ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}
  T ∈ {512, 1024, 2048, 4096, 8192, 16384}
The grid is *triangular*: high-B × high-T cells are absent (they OOM on a single
H100). For queries whose bilinear corners fall in that region we fill the
missing corner from the analytic decode roofline ``fixed_floor + B·T·kv_bpt/bw``,
which matches the measured grid at the coverage boundary (~19 ms at B=128,
T=2048) and extends it physically into the OOM region. ``fixed_floor`` is the
measured small-batch step time (grid value at B=1, T=512).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams


DEFAULT_CSV = Path(
    "profiling/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv"
)


@dataclass(frozen=True)
class DecodeStepGrid:
    """Sorted batch / context axes plus the measured decode_step_ms at each
    present (B, T). Missing (OOM) cells are simply absent from ``cells``.
    """

    b_axis: tuple[float, ...]
    t_axis: tuple[float, ...]
    cells: dict[tuple[float, float], float]
    fixed_floor_ms: float  # measured small-batch step time (B=1, T=min)

    def _analytic(self, b: float, t: float, params: RooflineParams) -> float:
        """``fixed_floor + bandwidth_term`` — the decode roofline anchored to the
        measured small-batch floor. Used to fill OOM corners and to extrapolate
        beyond the grid. compute term included for the (rare) compute-bound case.
        """
        kv = float(params.kv_bytes_per_token)
        bw = params.peak_bw_bytes_per_s * params.util_bw
        bandwidth_ms = (b * t * kv) / bw * 1e3
        compute_ms = (
            2.0 * float(params.n_params) * b
            / (params.peak_flops_per_s * params.util_flops) * 1e3
        )
        return max(self.fixed_floor_ms + bandwidth_ms, compute_ms)

    def lookup(self, b: float, t: float, params: RooflineParams) -> float:
        """Bilinear interp in log space over the measured grid. Corners that
        are absent (OOM region) are filled from the analytic decode roofline,
        which is continuous with the measured grid at the coverage boundary.
        Queries outside the axes clamp to the nearest edge before bracketing.
        """
        if not self.b_axis or not self.t_axis:
            raise RuntimeError("empty decode-step grid")

        b_q = max(self.b_axis[0], min(self.b_axis[-1], float(b)))
        t_q = max(self.t_axis[0], min(self.t_axis[-1], float(t)))
        log_b = math.log(b_q)
        log_t = math.log(t_q)

        i = _bracket_index([math.log(x) for x in self.b_axis], log_b)
        j = _bracket_index([math.log(x) for x in self.t_axis], log_t)

        b0, b1 = self.b_axis[i], self.b_axis[i + 1]
        t0, t1 = self.t_axis[j], self.t_axis[j + 1]

        db = (log_b - math.log(b0)) / max(1e-12, math.log(b1) - math.log(b0))
        dt = (log_t - math.log(t0)) / max(1e-12, math.log(t1) - math.log(t0))

        def corner(bb: float, tt: float) -> float:
            v = self.cells.get((bb, tt))
            return v if v is not None else self._analytic(bb, tt, params)

        a00 = corner(b0, t0)
        a10 = corner(b1, t0)
        a01 = corner(b0, t1)
        a11 = corner(b1, t1)

        interp = (
            a00 * (1 - db) * (1 - dt)
            + a10 * db * (1 - dt)
            + a01 * (1 - db) * dt
            + a11 * db * dt
        )
        # Beyond the grid's outer edge (e.g. T or B above the max axis value),
        # the analytic roofline is the more reliable extrapolant. Take the max
        # so we never under-count the bandwidth term past the measured region.
        if float(b) > self.b_axis[-1] or float(t) > self.t_axis[-1]:
            return max(interp, self._analytic(float(b), float(t), params))
        return interp


def _bracket_index(axis: list[float], value: float) -> int:
    """Lower index of the bracketing pair (i, i+1) for ``value`` in a sorted
    ``axis``. Clamps so i ∈ [0, len(axis) − 2].
    """
    last = len(axis) - 2
    if value <= axis[0]:
        return 0
    if value >= axis[-1]:
        return last
    for i in range(last + 1):
        if axis[i] <= value <= axis[i + 1]:
            return i
    return last


def load_grid(path: Path = DEFAULT_CSV) -> DecodeStepGrid:
    """Parse the wide-summary CSV into a DecodeStepGrid. Rows with a zero /
    non-``ok`` decode_step_ms are treated as absent (OOM cells).
    """
    cells: dict[tuple[float, float], float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                ms = float(r["decode_step_ms"])
            except (KeyError, ValueError):
                continue
            if ms <= 0.0 or r.get("validation_status") not in (None, "", "ok"):
                continue
            cells[(float(r["batch_size"]), float(r["context_len"]))] = ms
    if not cells:
        raise RuntimeError(f"no usable decode-step rows in {path}")
    b_axis = tuple(sorted({b for b, _ in cells}))
    t_axis = tuple(sorted({t for _, t in cells}))
    floor = cells[(b_axis[0], t_axis[0])]
    return DecodeStepGrid(
        b_axis=b_axis, t_axis=t_axis, cells=cells, fixed_floor_ms=floor
    )


@cache
def _default_grid() -> DecodeStepGrid:
    return load_grid(DEFAULT_CSV)


def decode_step_ms(
    batch: float, context_tokens: float, params: RooflineParams | None = None
) -> float:
    """Measured decode-step wall time for ``batch`` running requests each at
    ``context_tokens`` of resident KV. Uses the measured kernel grid where
    covered and the analytic decode roofline (anchored to the grid) in the
    OOM region / beyond the grid edge.
    """
    return _default_grid().lookup(
        max(1.0, float(batch)),
        max(1.0, float(context_tokens)),
        params or RooflineParams(),
    )
