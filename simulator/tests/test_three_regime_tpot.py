"""Tests for the three-regime TPOT classifier + predictor."""

from __future__ import annotations

from simulator.closed_form_tpot import RooflineParams
from simulator.three_regime_tpot import (
    K_SUSTAIN,
    Regime,
    TurnObservation,
    classify_cell,
    compute_t_max,
    physical_t_upper_ms,
    predict_cell_tpot,
)


def _t(idx: int, cached: float, new: float, output: float, n: int) -> TurnObservation:
    return TurnObservation(
        turn_index=idx,
        cached_context_tokens=cached,
        new_prefill_tokens=new,
        output_tokens=output,
        request_count=n,
    )


def test_k_sustain_is_two() -> None:
    assert K_SUSTAIN == 2


def test_physical_t_upper_is_about_205ms() -> None:
    params = RooflineParams()
    assert 195 < physical_t_upper_ms(params) < 215


def test_compute_t_max_caps_at_physical_t_upper() -> None:
    params = RooflineParams()
    # llm-d says 500 ms → cap to physical ~205 ms
    assert compute_t_max(500.0, params) == physical_t_upper_ms(params)
    # llm-d says 28 ms (chat) → use the llm-d value
    assert compute_t_max(28.0, params) == 28.0
    # llm-d missing → fall back to physical
    assert compute_t_max(None, params) == physical_t_upper_ms(params)


# ----------------------------------------------------- regime classification


def test_flat_regime_when_pressure_stays_below_one() -> None:
    """chat-like: tiny ctx, low c → cohort fits in pool throughout."""
    params = RooflineParams()
    turns = [_t(i, cached=100.0 + 10 * i, new=50.0, output=30.0, n=5) for i in range(10)]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.FLAT
    assert cls.jump_turn is None
    assert cls.peak_pressure < 1.0


def test_saturating_regime_when_pressure_sustained_at_end() -> None:
    """agentic-like: ctx grows past pool, never recovers."""
    params = RooflineParams()
    # ctx grows linearly; pressure crosses 1 mid-trajectory and stays.
    turns = [_t(i, cached=400.0 * i, new=100.0, output=25.0, n=80) for i in range(20)]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.SATURATING
    assert cls.jump_turn is not None and cls.jump_turn > 0
    assert cls.late_pressure >= 1.0


def test_perturbing_regime_when_pressure_recovers_via_completion() -> None:
    """osworld-like: brief spike at turn 5, then request_count drops, recovering."""
    params = RooflineParams()
    # ctx ramps up, peaks at turn 4 with pressure ≥ 1, then request_count
    # collapses and ctx growth slows so pressure drops back below 1.
    turns = [
        _t(0, 0,    100, 80, 160),
        _t(1, 1200, 100, 80, 142),
        _t(2, 2000, 100, 80, 136),
        # Pressure crosses 1 here (4200×123 / 27250 ≈ 1.18)
        _t(3, 4000, 100, 80, 123),
        _t(4, 4400, 100, 80, 104),
        # Bulk completion drops request_count + much smaller ctx → recovery
        _t(5, 4500, 100, 80, 47),
        _t(6, 4600, 100, 80, 46),
        _t(7, 4700, 100, 80, 45),
        _t(8, 4800, 100, 80, 44),
    ]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.PERTURBING, cls
    # Spike turns should be in the perturbation set
    assert any(t in cls.perturbation_turns for t in (3, 4))
    assert cls.late_pressure < 1.0


def test_single_saturated_turn_is_perturbing_not_saturating() -> None:
    """K_SUSTAIN = 2 prevents 1-turn blips from being classified as saturating."""
    params = RooflineParams()
    turns = [
        _t(0, 100, 100, 30, 80),
        _t(1, 200, 100, 30, 80),
        # Single saturated turn
        _t(2, 8000, 100, 30, 80),
        # Recovery
        _t(3, 200, 100, 30, 5),
        _t(4, 300, 100, 30, 5),
    ]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.PERTURBING


# ----------------------------------------------------- per-turn prediction


def test_predict_flat_emits_t_min_everywhere() -> None:
    params = RooflineParams()
    turns = [_t(i, 100.0, 50.0, 30.0, 5) for i in range(6)]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.FLAT
    t_min = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    out = predict_cell_tpot(cls, turns, t_min, t_max_ms=205.0)
    assert out == t_min


def test_predict_saturating_ramps_from_t_min_to_t_max() -> None:
    params = RooflineParams()
    turns = [_t(i, cached=400.0 * i, new=100.0, output=25.0, n=80) for i in range(20)]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.SATURATING
    t_min = [10.0] * len(turns)
    out = predict_cell_tpot(
        cls, turns, t_min, t_max_ms=205.0, use_kernel_lookup=False,
    )
    # Before jump → T_min
    assert out[0] == 10.0
    # After jump → ramps up
    assert out[-1] > out[cls.jump_turn]
    # Ramp tops out at T_max on the final turn
    assert abs(out[-1] - 205.0) < 1.0


def test_predict_perturbing_spikes_to_t_max_at_perturbation_turns() -> None:
    params = RooflineParams()
    turns = [
        _t(0, 100, 100, 30, 80),
        _t(1, 200, 100, 30, 80),
        _t(2, 8000, 100, 30, 80),     # spike turn
        _t(3, 200, 100, 30, 5),
        _t(4, 300, 100, 30, 5),
    ]
    cls = classify_cell(turns, params)
    assert cls.regime == Regime.PERTURBING
    assert 2 in cls.perturbation_turns
    t_min = [7.0] * 5
    out = predict_cell_tpot(
        cls, turns, t_min, t_max_ms=200.0, use_kernel_lookup=False,
    )
    assert out[2] == 200.0   # spike
    assert out[0] == 7.0     # baseline
    assert out[4] == 7.0     # back to baseline


def test_burst_completion_reclassifies_saturating_to_perturbing() -> None:
    """osworld c=160-like: pressure sustained ≥ 1 throughout AND late_pressure ≥ 1
    by workload rules, BUT one big request_count drop (104 → 47) means bulk
    completion. Should classify PERTURBING, not SATURATING.
    """
    params = RooflineParams()
    cohort_c = 160
    turns = []
    # Pressure ramps up over first 4 turns with full cohort, hits ≥ 1
    for i in range(5):
        turns.append(_t(i, cached=2000.0 + 1500 * i, new=100.0, output=80.0, n=160 - 10 * i))
    # Sharp drop in request_count at turn 5 (104 → 47 mimicking osworld)
    for i in range(5, 12):
        turns.append(_t(i, cached=9000.0 + 200 * (i - 5), new=100.0, output=80.0, n=47 - i + 5))
    cls = classify_cell(turns, params, cohort_c=cohort_c)
    # Late pressure is still ≥ 1 — but burst dominates, classifier sees
    # PERTURBING.
    assert cls.regime == Regime.PERTURBING, cls
    assert cls.jump_turn is None
    assert cls.peak_pressure >= 1.0


def test_no_burst_keeps_saturating() -> None:
    """Without a burst-completion drop, sustained pressure stays SATURATING."""
    params = RooflineParams()
    cohort_c = 80
    # Pressure grows monotonically with ctx; request_count only declines
    # gradually (no burst), like swe c=80.
    turns = [
        _t(i, cached=400.0 * i, new=100.0, output=25.0, n=max(40, 80 - i))
        for i in range(20)
    ]
    cls = classify_cell(turns, params, cohort_c=cohort_c)
    assert cls.regime == Regime.SATURATING
    assert cls.jump_turn is not None


def test_coverage_adjustment_inflates_perturbing_t_max() -> None:
    """When coverage < 1, the per-turn T_max for PERTURBING grows beyond
    the kernel step time alone (closer to client ITL).
    """
    from simulator.three_regime_tpot import _t_max_per_turn
    params = RooflineParams()
    turn = TurnObservation(
        turn_index=5, cached_context_tokens=5000, new_prefill_tokens=100,
        output_tokens=80, request_count=160,
    )
    # Without coverage (1.0): T_max = kernel step time directly
    t_no_cov = _t_max_per_turn(turn, params, 200.0, True, Regime.PERTURBING, None)
    # With coverage=0.5: T_max ≈ 2× the kernel step time (capped by 200ms)
    t_with_cov = _t_max_per_turn(turn, params, 200.0, True, Regime.PERTURBING, 0.5)
    assert t_with_cov > t_no_cov, (t_no_cov, t_with_cov)
    # Coverage 0.5 should roughly double the prediction
    assert 1.5 < (t_with_cov / t_no_cov) <= 2.0 + 0.01


def test_coverage_unused_for_saturating_regime() -> None:
    """SATURATING uses pressure-scaled T_max (T_lower → physical T_upper as
    pressure grows). Coverage is ignored (sustained step-skipping ITL is
    already captured by the pressure scaling).
    """
    from simulator.three_regime_tpot import _t_max_per_turn
    params = RooflineParams()
    # Construct a turn with pressure ≈ 1.5 (moderate overshoot)
    # ctx_mid ≈ 11000, blocks ≈ 688, with request_count=55 → pressure = 1.39
    turn = TurnObservation(
        turn_index=5, cached_context_tokens=11000, new_prefill_tokens=100,
        output_tokens=80, request_count=55,
    )
    t_no_cov = _t_max_per_turn(
        turn, params, 205.0, True, Regime.SATURATING, None, t_lower_ms=20.0,
    )
    t_with_cov = _t_max_per_turn(
        turn, params, 205.0, True, Regime.SATURATING, 0.3, t_lower_ms=20.0,
    )
    # Coverage doesn't change SATURATING T_max
    assert t_no_cov == t_with_cov
    # Pressure-scaled: between T_lower (20) and T_upper (~205); ≈ 20 + 0.39 × 185 ≈ 92
    assert 60 < t_no_cov < 130, t_no_cov


def test_empty_cell_returns_flat_with_zero_pressure() -> None:
    params = RooflineParams()
    cls = classify_cell([], params)
    assert cls.regime == Regime.FLAT
    assert cls.peak_pressure == 0.0
