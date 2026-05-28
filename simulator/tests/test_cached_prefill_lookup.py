"""Tests for the cached-prefill kernel lookup."""

from __future__ import annotations

from simulator.cached_prefill_lookup import (
    cached_prefill_step_ms,
    load_grid,
)


# Known anchors from cached_prefill_v3_H100.csv (first / last rows).
ANCHOR_64_512 = 12.3691
ANCHOR_128_512 = 12.5523
ANCHOR_1024_512 = 14.0405
ANCHOR_64_8192 = 24.2667
ANCHOR_1024_8192 = 25.6611


def test_grid_loads_with_expected_shape() -> None:
    grid = load_grid()
    assert grid.u_axis == (64.0, 128.0, 256.0, 512.0, 1024.0)
    assert grid.p_axis == (512.0, 1024.0, 2048.0, 4096.0, 8192.0)
    # 5×5 measured grid → 25 prefill_ms values
    assert sum(len(row) for row in grid.grid) == 25


def test_anchors_return_measured_values_exactly() -> None:
    """At grid corners, interpolation should hit the measured CSV value."""
    assert abs(cached_prefill_step_ms(64, 512) - ANCHOR_64_512) < 0.01
    assert abs(cached_prefill_step_ms(1024, 512) - ANCHOR_1024_512) < 0.01
    assert abs(cached_prefill_step_ms(64, 8192) - ANCHOR_64_8192) < 0.01
    assert abs(cached_prefill_step_ms(1024, 8192) - ANCHOR_1024_8192) < 0.01


def test_below_grid_u_clamps_to_lowest_u() -> None:
    """U=16 below the grid (smallest is 64) → behaves like U=64."""
    assert (
        abs(cached_prefill_step_ms(16, 512) - cached_prefill_step_ms(64, 512))
        < 0.01
    )


def test_above_grid_p_clamps_to_highest_p() -> None:
    """P=20000 above the grid (largest is 8192) → behaves like P=8192."""
    assert (
        abs(cached_prefill_step_ms(64, 20000) - cached_prefill_step_ms(64, 8192))
        < 0.01
    )


def test_interpolation_between_two_u_anchors_at_fixed_p() -> None:
    """U=96 sits between U=64 and U=128 → log-interpolated value falls in
    the [ANCHOR_64_512, ANCHOR_128_512] range.
    """
    out = cached_prefill_step_ms(96, 512)
    lo = min(ANCHOR_64_512, ANCHOR_128_512)
    hi = max(ANCHOR_64_512, ANCHOR_128_512)
    assert lo - 0.01 <= out <= hi + 0.01, out


def test_monotone_increase_in_p_at_fixed_u() -> None:
    """Larger P should give larger per-step time (FA3 over P + KV reads)."""
    vals = [cached_prefill_step_ms(256, p) for p in (512, 1024, 2048, 4096, 8192)]
    for a, b in zip(vals, vals[1:]):
        assert b > a - 0.01, vals


def test_default_module_function_matches_explicit_grid_lookup() -> None:
    grid = load_grid()
    assert (
        abs(cached_prefill_step_ms(256, 2048) - grid.lookup(256, 2048))
        < 1e-6
    )


def test_typical_osworld_workload_returns_finite_ms() -> None:
    """Spot-check: osworld c=160 turn 5 looks like U~100, P~9000.
    Should yield ~20-30 ms (close to the P=8192 column).
    """
    out = cached_prefill_step_ms(100, 9000)
    assert 15.0 < out < 35.0, out
