"""Tests for the measured decode-step kernel grid lookup."""

from __future__ import annotations

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
