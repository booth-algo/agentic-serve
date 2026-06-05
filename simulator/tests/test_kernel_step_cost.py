"""Tests for the measured decode-step kernel grid lookup."""

from __future__ import annotations

from pathlib import Path

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import (
    _default_grid,
    decode_step_ms,
    load_grid,
)


def test_grid_loads_with_expected_axes() -> None:
    g = _default_grid()
    assert g.b_axis[0] == 1.0
    assert g.t_axis[0] == 512.0
    assert 256.0 in g.b_axis
    assert 16384.0 in g.t_axis
    # measured small-batch floor is ~6.5 ms
    assert 6.0 < g.fixed_floor_ms < 7.0


def test_fixed_floor_is_robust_to_warmup_outlier(tmp_path: Path) -> None:
    # fixed_floor anchors the analytic decode roofline for cells beyond the measured grid; it must be
    # the MIN over the smallest-batch row, not the single (B_min, T_min) cell, so a one-off warm-up
    # overhead in that cell does not inflate every analytic-fill cell (the H100x2 B=1,T=512=9.1 vs the
    # row's true 4.7 floor was the dominant tp2 high-conc TPOT over-prediction).
    csv = tmp_path / "grid.csv"
    csv.write_text(
        "batch_size,context_len,decode_step_ms,validation_status\n"
        "1,512,9.10,ok\n"      # warm-up outlier at (B_min, T_min)
        "1,2048,4.68,ok\n"     # the real small-batch floor
        "1,8192,5.49,ok\n"
        "256,512,10.5,ok\n"
        "256,2048,16.25,ok\n"
    )
    g = load_grid(csv)
    assert g.fixed_floor_ms == 4.68          # row minimum, NOT the 9.10 corner
    # well-behaved grid (corner already the row min) is unaffected
    csv2 = tmp_path / "grid2.csv"
    csv2.write_text(
        "batch_size,context_len,decode_step_ms,validation_status\n"
        "1,512,6.55,ok\n1,2048,6.60,ok\n256,512,10.0,ok\n"
    )
    assert load_grid(csv2).fixed_floor_ms == 6.55


def test_lookup_matches_grid_at_measured_points() -> None:
    g = _default_grid()
    # exact grid points should round-trip (interp weight collapses to the node)
    assert abs(decode_step_ms(1, 512) - g.cells[(1.0, 512.0)]) < 1e-6
    assert abs(decode_step_ms(128, 2048) - g.cells[(128.0, 2048.0)]) < 1e-6


def test_step_time_rises_with_batch_and_context() -> None:
    # bandwidth-bound region: more batch / context => longer step
    assert decode_step_ms(128, 2048) > decode_step_ms(1, 512)
    assert decode_step_ms(64, 8192) > decode_step_ms(64, 1024)


def test_oom_corner_filled_by_analytic_is_monotone() -> None:
    """High-B × high-T cells are absent from the grid; the analytic fill must
    keep the surface monotone and finite (no NaN/zero hole).
    """
    v = decode_step_ms(256, 16384)  # far OOM corner
    assert v > decode_step_ms(256, 1024)
    assert v > decode_step_ms(1, 16384)
    assert v < 1000.0  # sane magnitude, not a blow-up


def test_floor_holds_at_tiny_workload() -> None:
    # batch=1, short ctx => essentially the fixed floor, never below it
    g = _default_grid()
    assert decode_step_ms(1, 512) >= g.fixed_floor_ms - 1e-6


def test_clamps_below_axis_minimum() -> None:
    # sub-1 batch / sub-min context clamp rather than extrapolate downward
    assert decode_step_ms(0.1, 1.0) == decode_step_ms(1, 512)


def test_analytic_fill_continuous_at_boundary() -> None:
    """The analytic fill should be close to the measured value at the last
    covered cell, so the measured→analytic handoff has no large step.
    """
    g = _default_grid()
    p = RooflineParams()
    measured = g.cells[(128.0, 2048.0)]
    analytic = g._analytic(128.0, 2048.0, p)
    assert abs(measured - analytic) / measured < 0.20


def test_load_grid_skips_oom_cells() -> None:
    g = load_grid()
    # the (256, 4096) cell OOMs and must be absent
    assert (256.0, 4096.0) not in g.cells
    # but a low-corner cell is present
    assert (1.0, 512.0) in g.cells
