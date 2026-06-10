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
stays correct (see the L3 de-fit entry). Absent cells are filled two ways:

* INTERIOR holes (bracketed by measured cells in at least one axis — the
  dropped 'check' rows and sweep gaps): the analytic decode roofline
  ``fixed_floor + B·T·kv_bpt/bw``, which matches the measured grid at the
  coverage boundary (~19 ms at B=128, T=2048) and was validated against the
  2026-06-10 S12 re-measurement (it lands on the re-measured monotone surface).
* BEYOND the measured hull (past the row's last T *and* the column's last B,
  or past the outer axes): a measured-scaling power law — the local log-log
  slope of the grid's own last two cells along each axis, clamped to [0, 1]
  (0 = monotone-flat) and additionally CAPPED at the linear roofline
  increment above the measured edge (``v_edge + Δ(B·T·kv/bw)``): the power
  law scales the WHOLE step time, but the fixed launch/weight component must
  not grow with B·T and the kernel is at most linear in ``b·ctx``. Max of
  the row/col extrapolants. No free parameters: every exponent is a ratio of
  measured cells and the cap is the roofline. De-fit 2026-06-10 (campaign
  L3): the previous linear ``b·ctx`` fill over-priced the real SUB-linear
  kernel — held-out on the 35 newly measured tp2 cells (old 19-cell grid as
  input) the linear fill scores median 1.084× / worst 1.241× (B·T ≥ 200k
  median 1.098×) while this extrapolation scores median 1.067× / worst
  1.111× (B·T ≥ 200k median 1.049×) with no new under-pricing (min 0.925 vs
  linear 0.930). The max-combination is the only variant that improves every
  statistic without under-pricing (geo/min under-price to 0.85/0.77 worst);
  the roofline cap bounds the movement of reachable tp1 states (KV pool
  436k tokens) to ≲11% while leaving the tp2 gains untouched.

``fixed_floor`` is the measured small-batch step time (the min over the B=1
row — see ``load_grid``).

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

Analytic-only configs (no measured grid at all) price the full decode roofline
via ``analytic_grid``; their launch floor is no longer the single H100-derived
1.37 ms but the residual of the NEAREST MEASURED GPU CLASS (audit-v2 G5):
H100 tp1 1.37 / H100x2 tp2 1.82 / A100 tp1 2.06 ms, each derived from that
config's own measured grid floor minus its modelled HBM reads — see
``launch_floor_for``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from functools import cache, cached_property
from pathlib import Path
from typing import NamedTuple

from simulator.closed_form_tpot import RooflineParams


DEFAULT_CSV = Path(
    "profile_data/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv"
)

# The other measured decode grids + the params their configs run with — consumed by the
# per-config launch-floor derivation (audit-v2 G5, ``launch_floor_for``). Grid CSVs are the
# same artifacts the deployment manifests point at; the A100 roofline JSON is the calibrated
# per-GPU params file (read at runtime so a re-anchored A100 util flows through automatically).
H100X2_GRID_CSV = Path("profile_data/results/decode_profile_H100x2_merged_2026-06-10.csv")
A100_GRID_CSV = Path("profile_data/results/decode_profile_A100_2026-06-02.csv")
A100_ROOFLINE_JSON = Path("profile_data/kernels/roofline_params_A100_llama31_8b.json")


@dataclass(frozen=True)
class DecodeStepGrid:
    """Sorted batch / context axes plus the measured decode_step_ms at each
    present (B, T). Cells the sweep skipped (KV-footprint cap) or dropped
    (the 'check' rows — audit-v2 S12) are simply absent from ``cells``.
    """

    b_axis: tuple[float, ...]
    t_axis: tuple[float, ...]
    cells: dict[tuple[float, float], float]
    fixed_floor_ms: float  # measured small-batch step time (min over the B_min row — see load_grid)
    analytic_only: bool = False    # no measured cells -> lookup() returns the full decode roofline
    # analytic_only floor: per-step launch/sampling cost NOT explained by HBM reads.
    # None -> resolved per query from the nearest measured GPU class (launch_floor_for, audit-v2 G5).
    launch_floor_ms: float | None = None

    @cached_property
    def _rows(self) -> dict[float, tuple[float, ...]]:
        """Measured context lengths per batch row, ascending."""
        rows: dict[float, list[float]] = {}
        for b, t in self.cells:
            rows.setdefault(b, []).append(t)
        return {b: tuple(sorted(ts)) for b, ts in rows.items()}

    @cached_property
    def _cols(self) -> dict[float, tuple[float, ...]]:
        """Measured batch sizes per context column, ascending."""
        cols: dict[float, list[float]] = {}
        for b, t in self.cells:
            cols.setdefault(t, []).append(b)
        return {t: tuple(sorted(bs)) for t, bs in cols.items()}

    def _decode_roofline_full(self, b: float, t: float, params: RooflineParams) -> float:
        """Full decode-step roofline for an UNCALIBRATED config (no measured grid).

        Unlike ``_analytic`` (which fills only uncovered corners of a measured grid and so
        leans on the grid's measured ``fixed_floor`` to capture weight reads), this is
        self-contained and must model the weight read explicitly — otherwise a 70B model
        would be predicted like the 8B grid floor. Decode reads, per step:

            FFN/proj GEMM   : max(weight_bytes/tp / bw, 2*n_params/tp*b / flops)   # mem- or compute-bound
            attention KV    : b * ctx * kv_per_gpu / bw                            # separate kernel -> adds
            launch floor    : CUDA-graph launch + sampling (HBM-independent)

        ``weight_bytes = n_params * bytes_per_param`` is the real HBM footprint (quant-aware:
        MXFP4 experts give bytes_per_param < 2). ``launch_floor_ms`` is anchored to a measured
        grid floor minus its own weight+KV reads; when it is None it resolves per query to the
        NEAREST MEASURED GPU CLASS for ``params`` (``launch_floor_for`` — audit-v2 G5), so e.g.
        an analytic A100 config gets the A100-grid residual (~2.06 ms), not H100's 1.37.
        """
        launch_ms = (
            self.launch_floor_ms if self.launch_floor_ms is not None
            else launch_floor_for(params)
        )
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
        return launch_ms + gemm_ms + attn_ms

    def _analytic(self, b: float, t: float, params: RooflineParams) -> float:
        """``fixed_floor + bandwidth_term`` — the decode roofline anchored to the
        measured small-batch floor. Used to fill INTERIOR holes (cells bracketed
        by measurements — sweep gaps and the S12 dropped 'check' rows, where the
        2026-06-10 re-measurement validated this fill); beyond-hull cells use
        ``_beyond_hull`` instead. compute term included for the (rare)
        compute-bound case.

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
        return max(self.fixed_floor_ms + bandwidth_ms, _compute_ms(b, params))

    def _kv_lin_ms(self, b: float, t: float, params: RooflineParams) -> float:
        """The linear KV-read roofline term ``B·T·kv_per_gpu / bw`` in ms — the
        shape the retired fill priced; now only the CAP on extrapolation growth.
        """
        tp = max(1, int(params.tensor_parallel))
        kv_shards = min(tp, max(1, int(params.kv_heads)))
        kv = float(params.kv_bytes_per_token) / kv_shards
        bw = params.peak_bw_bytes_per_s * params.util_bw
        return (b * t * kv) / bw * 1e3

    @staticmethod
    def _measured_scaling(v_edge: float, v_prev: float, x_edge: float,
                          x_prev: float, x: float, lin_edge: float,
                          lin_x: float) -> float:
        """Extrapolate past the measured edge with the grid's OWN local scaling:
        the log-log slope of the last two measured values along the axis,

            alpha = log(v_edge / v_prev) / log(x_edge / x_prev),  clamped to [0, 1],
            v(x)  = v_edge · (x / x_edge)^alpha,

        then CAPPED at the linear roofline increment above the edge,
        ``v_edge + (lin_x − lin_edge)``. All bounds are physical, not tuned:
        alpha ≥ 0 is monotonicity (a decode step cannot get cheaper as B/T
        grows beyond the edge — protects against the measured negative B=1→2
        increments at long context); the cap is the at-most-linear-in-``b·ctx``
        kernel (2026-06-10 sub-linearity measurement) applied to the INCREMENT
        only, because the power law scales the whole step time while the fixed
        launch/weight component must not grow with B·T (uncapped, a measured
        near-linear B-slope would over-shoot the roofline by the floor share).
        No free parameters: alpha is a ratio of measured cells.
        """
        alpha = math.log(v_edge / v_prev) / math.log(x_edge / x_prev)
        alpha = min(1.0, max(0.0, alpha))
        return min(v_edge * (x / x_edge) ** alpha, v_edge + (lin_x - lin_edge))

    def _beyond_hull(self, bb: float, tt: float,
                     params: RooflineParams) -> float | None:
        """Sub-linear fill for a grid node OUTSIDE the measured hull — beyond
        the last measured T of its batch row AND beyond the last measured B of
        its context column (i.e. the unmeasured high-B×high-T region, where the
        linear analytic fill over-prices the sub-linear kernel — de-fit
        2026-06-10). Extrapolates the measured scaling along both axes via
        ``_measured_scaling`` and takes the max (the conservative combination — the
        only one that removed the over-price without introducing under-pricing
        on the 35 held-out tp2 cells; see the module docstring). Returns None
        for INTERIOR holes (bracketed by measurements in either axis): those
        keep the S12-validated analytic fill.
        """
        row = self._rows.get(bb, ())
        col = self._cols.get(tt, ())
        row_beyond = len(row) >= 2 and tt > row[-1]
        col_beyond = len(col) >= 2 and bb > col[-1]
        if not (row_beyond and col_beyond):
            return None
        v_row = self._measured_scaling(
            self.cells[(bb, row[-1])], self.cells[(bb, row[-2])],
            row[-1], row[-2], tt,
            self._kv_lin_ms(bb, row[-1], params), self._kv_lin_ms(bb, tt, params))
        v_col = self._measured_scaling(
            self.cells[(col[-1], tt)], self.cells[(col[-2], tt)],
            col[-1], col[-2], bb,
            self._kv_lin_ms(col[-1], tt, params), self._kv_lin_ms(bb, tt, params))
        return max(v_row, v_col, _compute_ms(bb, params))

    def _beyond_axes(self, b: float, t: float, b_q: float, t_q: float,
                     edge_value: float, params: RooflineParams) -> float:
        """Measured-scaling extrapolation past the OUTER axes. Replaces the
        retired ``max(interp, _analytic)`` linear extrapolant (de-fit
        2026-06-10: linear ``b·ctx`` over-prices the sub-linear kernel). Scales
        the clamped-edge value by the power law of the corner-filled surface at
        the last two axis nodes — same measured-ratio exponents as
        ``_beyond_hull``, evaluated on the lookup surface so it stays
        continuous at the edge.
        """
        v = edge_value
        if t > self.t_axis[-1] and len(self.t_axis) >= 2:
            v_e = self.lookup(b_q, self.t_axis[-1], params)
            v_p = self.lookup(b_q, self.t_axis[-2], params)
            v *= self._measured_scaling(
                v_e, v_p, self.t_axis[-1], self.t_axis[-2], t,
                self._kv_lin_ms(b_q, self.t_axis[-1], params),
                self._kv_lin_ms(b_q, t, params)) / v_e
        if b > self.b_axis[-1] and len(self.b_axis) >= 2:
            v_e = self.lookup(self.b_axis[-1], t_q, params)
            v_p = self.lookup(self.b_axis[-2], t_q, params)
            v *= self._measured_scaling(
                v_e, v_p, self.b_axis[-1], self.b_axis[-2], b,
                self._kv_lin_ms(self.b_axis[-1], t_q, params),
                self._kv_lin_ms(b, t_q, params)) / v_e
        return max(v, _compute_ms(b, params))

    def lookup(self, b: float, t: float, params: RooflineParams) -> float:
        """Bilinear interp in log space over the measured grid. Absent corners
        are filled from the measured-scaling extrapolation when they lie beyond
        the measured hull (``_beyond_hull``) and from the analytic decode
        roofline when they are interior holes — both continuous with the
        measured grid at the coverage boundary. Queries past the outer axes
        extrapolate with the measured edge scaling (``_beyond_axes``).
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
            if v is not None:
                return v
            v = self._beyond_hull(bb, tt, params)
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
        # Beyond the grid's outer edge (T or B above the max axis value) the
        # measured edge scaling is the extrapolant — the retired
        # max(interp, _analytic) re-imposed the linear-in-b·ctx over-price.
        if float(b) > self.b_axis[-1] or float(t) > self.t_axis[-1]:
            return self._beyond_axes(float(b), float(t), b_q, t_q, interp, params)
        return interp


def _compute_ms(b: float, params: RooflineParams) -> float:
    """Compute-bound decode-step lower bound (per-rank GEMM FLOPs at batch b)."""
    tp = max(1, int(params.tensor_parallel))
    return (
        2.0 * (float(params.n_params) / tp) * b
        / (params.peak_flops_per_s * params.util_flops) * 1e3
    )


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


def launch_floor_residual_ms(grid: DecodeStepGrid, params: RooflineParams) -> float:
    """Launch-floor residual of a measured grid: floor − modelled HBM reads (audit-v2 G5).

    The measured small-batch floor is weight-read + min-KV-read + launch. Subtracting the
    HBM reads modelled with the params of the config the grid was MEASURED ON leaves the
    HBM-independent launch + sampling (+ tp collective-launch) overhead. The attn term is
    evaluated at (B_min, T_min of the axis) — the original derivation convention; it differs
    from the floor cell's own T by ≤ 0.07 ms (the term is ~0.02–0.09 ms everywhere).
    Non-negativity guard only (a launch floor cannot be < 0); it never binds on the
    measured grids. De-fit 2026-06-05: the retired 0.3 was an unexplained magic literal;
    0.0 is the physical floor and a proven no-op.
    """
    tp = max(1, int(params.tensor_parallel))
    kv_shards = min(tp, max(1, int(params.kv_heads)))
    bw = params.peak_bw_bytes_per_s * params.util_bw
    b0, t0 = grid.b_axis[0], grid.t_axis[0]
    weight_ms = float(params.n_params) * float(params.bytes_per_param) / tp / bw * 1e3
    attn_ms = (b0 * t0 * float(params.kv_bytes_per_token) / kv_shards) / bw * 1e3
    return max(0.0, grid.fixed_floor_ms - weight_ms - attn_ms)


@cache
def default_launch_floor_ms() -> float:
    """Per-step launch/sampling cost, anchored to the measured Llama-3.1-8B/H100 grid floor.

    With the 8B/H100 defaults the residual is ~1.37 ms (fixed_floor 6.55 − weight 5.15 −
    attn 0.02). tp=1 makes ``launch_floor_residual_ms`` byte-identical to the original
    unsharded derivation.
    """
    # RooflineParams() = Llama-3.1-8B / H100 defaults — the config the grid was measured on.
    return launch_floor_residual_ms(_default_grid(), RooflineParams())


class _LaunchFloorClass(NamedTuple):
    name: str
    tensor_parallel: int
    peak_bw_bytes_per_s: float
    residual_ms: float


@cache
def _launch_floor_classes() -> tuple[_LaunchFloorClass, ...]:
    """One launch-floor residual per MEASURED decode grid (audit-v2 G5).

    The audit found the single H100-derived 1.37 ms residual is NOT config-independent:
    each measured grid leaves a different residual (H100 tp1 1.37, H100x2 tp2 1.82 from the
    re-measured 4.40 ms floor, A100 tp1 2.06). Each entry pairs a grid with the params of
    the config it was measured on (A100 params come from the calibrated roofline JSON so a
    future util re-anchor flows through). Grids missing on disk are skipped — the H100
    default always exists, so resolution never comes up empty.
    """
    classes = [_LaunchFloorClass(
        "H100", 1, RooflineParams().peak_bw_bytes_per_s, default_launch_floor_ms())]
    if H100X2_GRID_CSV.exists():
        p2 = replace(RooflineParams(), tensor_parallel=2)  # H100x2 inherits H100 peaks/utils
        classes.append(_LaunchFloorClass(
            "H100x2", 2, p2.peak_bw_bytes_per_s,
            launch_floor_residual_ms(load_grid(H100X2_GRID_CSV), p2)))
    if A100_GRID_CSV.exists() and A100_ROOFLINE_JSON.exists():
        pa = RooflineParams.from_json(A100_ROOFLINE_JSON)
        classes.append(_LaunchFloorClass(
            "A100", 1, pa.peak_bw_bytes_per_s,
            launch_floor_residual_ms(load_grid(A100_GRID_CSV), pa)))
    return tuple(classes)


def launch_floor_for(params: RooflineParams) -> float:
    """Launch-floor residual for an ANALYTIC (uncalibrated) config: the nearest
    measured GPU class — the documented G5 choice for configs without their own grid.

    Nearest = match tp regime first (single-GPU vs tensor-parallel: the tp collective
    launch is part of the floor — H100x2 1.82 vs H100 1.37 on the same silicon), then
    the closest peak HBM bandwidth in log space (GPU generation: A100 2.06 vs H100
    1.37). Falls back across regimes when a regime has no measured class.
    """
    classes = _launch_floor_classes()
    tp = max(1, int(params.tensor_parallel))
    pool = [c for c in classes if (c.tensor_parallel >= 2) == (tp >= 2)] or list(classes)
    target = math.log(max(1.0, float(params.peak_bw_bytes_per_s)))
    best = min(pool, key=lambda c: abs(math.log(c.peak_bw_bytes_per_s) - target))
    return best.residual_ms


def analytic_grid(launch_floor_ms: float | None = None) -> DecodeStepGrid:
    """A cell-free DecodeStepGrid whose ``lookup`` returns the full decode roofline.

    For configs with NO measured decode grid (a different GPU and/or model): instead of
    borrowing the H100 8B grid, ``decode_step_ms`` then scales physically with the config's
    own RooflineParams (weight bytes, bandwidth, KV). Swap it in via ``_default_grid`` exactly
    like a measured grid (see build_simulator_rows). The axes are nominal (unused for analytic
    lookups). ``launch_floor_ms`` default (None) resolves PER QUERY to the nearest measured
    GPU class for the query's params (``launch_floor_for`` — audit-v2 G5); pass an explicit
    value to pin it.
    """
    # fixed_floor_ms is unused on the analytic-only path (only _decode_roofline_full runs);
    # mirror the explicit launch floor for introspection, 0.0 when dynamically resolved.
    return DecodeStepGrid(
        b_axis=(1.0, 256.0), t_axis=(1.0, 16384.0), cells={},
        fixed_floor_ms=launch_floor_ms if launch_floor_ms is not None else 0.0,
        analytic_only=True, launch_floor_ms=launch_floor_ms,
    )


def decode_step_ms(
    batch: float, context_tokens: float, params: RooflineParams | None = None
) -> float:
    """Measured decode-step wall time for ``batch`` running requests each at
    ``context_tokens`` of resident KV. Uses the measured kernel grid where
    covered, the analytic decode roofline (anchored to the grid) for interior
    holes, and the grid's own measured scaling (sub-linear in ``b·ctx``) beyond
    the measured hull / grid edge.
    """
    return _default_grid().lookup(
        max(1.0, float(batch)),
        max(1.0, float(context_tokens)),
        params or RooflineParams(),
    )
