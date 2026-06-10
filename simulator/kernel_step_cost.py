"""Per-step decode wall-time from the measured H100 / Llama-3.1-8B kernel grid.

Reads ``profile_data/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv``
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
The grid is *triangular*: high-B × high-T cells are absent because the sweep's
KV-footprint cap (B·(T+128) ≤ 500k tokens) skipped them — NOT because they OOM
(audit-v2 S12: the old "OOM" label here was wrong). ``load_grid`` additionally
drops 7 interior rows flagged ``validation_status='check'`` by the trace
tooling; a 2026-06-10 event-timed re-measurement (GPU 6, ``decode_steps.py``;
``decode_profile_H100_2026-06-10_s12recheck.csv``) shows those dropped values
sat in non-monotone dips absent from the re-measured surface, so dropping them
stays correct (see the L3 de-fit entry). For queries whose bilinear corners fall
in an absent region we fill the missing corner from the analytic decode
roofline ``fixed_floor + B·T·kv_bpt/bw``, which matches the measured grid at
the coverage boundary (~19 ms at B=128, T=2048) and extends it physically
beyond the cap. ``fixed_floor`` is the measured small-batch step time (the min
over the B=1 row — see ``load_grid``).

Per-deployment grids (e.g. tp2): ``build_simulator_rows`` swaps ``_default_grid``
to the CSV named by the deployment's ``data.decode_grid`` manifest entry. Dated
raw run CSVs are append-only; ``profiling/process/build_decode_grid.py`` merges
them (newest run wins per cell) into the grid artifact a deployment points at.
The 2026-06-10 H100x2 merged grid (54 cells, dense B×T rectangle + T=24576 tail
up to the real 998,656-token KV pool) removes the analytic fill from every
reachable tp2 decode state — the fill's linear-in-``b·ctx`` pricing over-priced
the previously unmeasured region by a median 1.08× (1.10–1.24× where
B·T ≥ 200k; measured marginal KV cost at T=16384 is ~11–17 ms per 1M tokens vs
the fill's 21.0 — the real kernel is SUB-linear in ``b·ctx``).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams


DEFAULT_CSV = Path(
    "profile_data/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv"
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
    analytic_only: bool = False    # no measured cells -> lookup() returns the full decode roofline
    launch_floor_ms: float = 0.0   # analytic_only floor: per-step launch/sampling cost NOT explained by HBM reads

    def _decode_roofline_full(self, b: float, t: float, params: RooflineParams) -> float:
        """Full decode-step roofline for an UNCALIBRATED config (no measured grid).

        Unlike ``_analytic`` (which fills only OOM corners of a measured grid and so
        leans on the grid's measured ``fixed_floor`` to capture weight reads), this is
        self-contained and must model the weight read explicitly — otherwise a 70B model
        would be predicted like the 8B grid floor. Decode reads, per step:

            FFN/proj GEMM   : max(weight_bytes/tp / bw, 2*n_params/tp*b / flops)   # mem- or compute-bound
            attention KV    : b * ctx * kv_per_gpu / bw                            # separate kernel -> adds
            launch floor    : CUDA-graph launch + sampling (HBM-independent)

        ``weight_bytes = n_params * bytes_per_param`` is the real HBM footprint (quant-aware:
        MXFP4 experts give bytes_per_param < 2). ``launch_floor_ms`` is anchored to the measured
        8B grid floor minus its own weight+KV reads (see ``default_launch_floor_ms``), so this
        reproduces the measured Llama-3.1-8B/H100 floor (~6.5 ms) exactly at (b=1, t=min).
        """
        tp = max(1, int(params.tensor_parallel))
        kv_shards = min(tp, max(1, int(params.kv_heads)))
        bw = params.peak_bw_bytes_per_s * params.util_bw
        weight_bytes = float(params.n_params) * float(params.bytes_per_param) / tp
        kv = float(params.kv_bytes_per_token) / kv_shards
        gemm_ms = max(
            weight_bytes / bw * 1e3,
            2.0 * (float(params.n_params) / tp) * b / (params.peak_flops_per_s * params.util_flops) * 1e3,
        )
        attn_ms = (b * t * kv) / bw * 1e3
        return self.launch_floor_ms + gemm_ms + attn_ms

    def _analytic(self, b: float, t: float, params: RooflineParams) -> float:
        """``fixed_floor + bandwidth_term`` — the decode roofline anchored to the
        measured small-batch floor. Used to fill OOM corners and to extrapolate
        beyond the grid. compute term included for the (rare) compute-bound case.

        Tensor-parallel aware: under TP the KV cache is sharded by head and the
        weights are sharded across ranks, so each GPU streams only 1/tp of the
        weights and 1/min(tp, kv_heads) of the KV per token from its OWN HBM
        (``peak_bw`` stays per-GPU). tp=1 is byte-identical to the single-GPU form.
        """
        tp = max(1, int(params.tensor_parallel))
        kv_shards = min(tp, max(1, int(params.kv_heads)))
        kv = float(params.kv_bytes_per_token) / kv_shards
        bw = params.peak_bw_bytes_per_s * params.util_bw
        bandwidth_ms = (b * t * kv) / bw * 1e3
        compute_ms = (
            2.0 * (float(params.n_params) / tp) * b
            / (params.peak_flops_per_s * params.util_flops) * 1e3
        )
        return max(self.fixed_floor_ms + bandwidth_ms, compute_ms)

    def lookup(self, b: float, t: float, params: RooflineParams) -> float:
        """Bilinear interp in log space over the measured grid. Corners that
        are absent (OOM region) are filled from the analytic decode roofline,
        which is continuous with the measured grid at the coverage boundary.
        Queries outside the axes clamp to the nearest edge before bracketing.
        """
        if self.analytic_only:
            return self._decode_roofline_full(max(1.0, float(b)), max(1.0, float(t)), params)
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
    """Parse a decode-grid CSV into a DecodeStepGrid. Rows with a zero
    decode_step_ms or a non-``ok`` ``validation_status`` (e.g. the tp1 trace
    sweep's 7 ``'check'`` rows — measured but flagged by the off-repo bucketing
    cross-check; re-measured 2026-06-10, drop retained — audit-v2 S12) are
    treated as absent cells and later analytic-filled by ``lookup``.
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
    # ``fixed_floor`` = the small-batch launch+weights floor that anchors the analytic decode roofline
    # (``_analytic``) for cells beyond the measured grid. Take the MIN over the smallest-batch row, not
    # the single (B_min, T_min) cell: that cell can carry a one-off warm-up overhead (e.g. H100x2 B=1
    # T=512 = 9.1 ms vs the row's true 4.7 ms floor at T=2048), which would inflate EVERY analytic-fill
    # cell. The min over the B_min row is the robust measured floor. No-op for well-behaved grids whose
    # T_min cell already is the row minimum (e.g. H100 tp1: 6.56 vs 6.55 — within rounding).
    b0 = b_axis[0]
    floor = min(cells[(b0, t)] for t in t_axis if (b0, t) in cells)
    return DecodeStepGrid(
        b_axis=b_axis, t_axis=t_axis, cells=cells, fixed_floor_ms=floor
    )


@cache
def _default_grid() -> DecodeStepGrid:
    return load_grid(DEFAULT_CSV)


@cache
def default_launch_floor_ms() -> float:
    """Per-step launch/sampling cost, anchored to the measured Llama-3.1-8B/H100 grid floor.

    The measured small-batch floor (~6.5 ms at b=1, t=512) is weight-read + min-KV-read +
    launch. Subtracting the modelled HBM reads (with the default 8B/H100 params) leaves the
    HBM-independent launch + sampling overhead — used as the floor for analytic (uncalibrated)
    configs so they continue from the same physical anchor instead of a magic constant.
    """
    grid = _default_grid()
    p = RooflineParams()  # Llama-3.1-8B / H100 defaults — the config the floor was measured on
    bw = p.peak_bw_bytes_per_s * p.util_bw
    b0, t0 = grid.b_axis[0], grid.t_axis[0]
    weight_ms = float(p.n_params) * float(p.bytes_per_param) / bw * 1e3
    attn_ms = (b0 * t0 * float(p.kv_bytes_per_token)) / bw * 1e3
    # Non-negativity guard only (a launch floor cannot be < 0). With the 8B/H100 defaults the
    # residual is ~1.37 ms for every config (fixed_floor 6.55 - weight 5.15 - attn 0.02), so this
    # max() never binds. De-fit 2026-06-05: the retired 0.3 was an unexplained magic literal; 0.0 is
    # the physical floor and a proven no-op (TPOT byte-identical on every analytic config).
    return max(0.0, grid.fixed_floor_ms - weight_ms - attn_ms)


def analytic_grid(launch_floor_ms: float | None = None) -> DecodeStepGrid:
    """A cell-free DecodeStepGrid whose ``lookup`` returns the full decode roofline.

    For configs with NO measured decode grid (a different GPU and/or model): instead of
    borrowing the H100 8B grid, ``decode_step_ms`` then scales physically with the config's
    own RooflineParams (weight bytes, bandwidth, KV). Swap it in via ``_default_grid`` exactly
    like a measured grid (see build_simulator_rows). The axes are nominal (unused for analytic
    lookups). ``launch_floor_ms`` defaults to the 8B-anchored value.
    """
    floor = default_launch_floor_ms() if launch_floor_ms is None else launch_floor_ms
    return DecodeStepGrid(
        b_axis=(1.0, 256.0), t_axis=(1.0, 16384.0), cells={},
        fixed_floor_ms=floor, analytic_only=True, launch_floor_ms=floor,
    )


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
