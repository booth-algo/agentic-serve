"""Tests for the three-regime augmenter (cell-level classification + emit)."""

from __future__ import annotations

from profiling.process.emitters.augment_simulator_predictions_with_three_regime import (
    predict_cell,
)
from simulator.closed_form_tpot import RooflineParams


def _turn(idx: int, cached: float, new: float, output: float, n: int,
          tpot_pred: float, tpot_pred_llm_d: float | None = None) -> dict:
    d = {
        "turn_index": idx,
        "cached_context_tokens": cached,
        "new_prefill_tokens": new,
        "output_tokens": output,
        "scheduled_requests": n,
        "tpot_pred": tpot_pred,
    }
    if tpot_pred_llm_d is not None:
        d["tpot_pred_llm_d"] = tpot_pred_llm_d
    return d


def test_flat_cell_emits_t_min_everywhere() -> None:
    """chat-like at c=5: pressure stays below 1 → predict T_min throughout."""
    params = RooflineParams()
    turns = [_turn(i, cached=100.0 + 10 * i, new=50.0, output=30.0, n=5,
                   tpot_pred=6.7 + 0.05 * i, tpot_pred_llm_d=6.9) for i in range(8)]
    regime, preds, info = predict_cell(turns, cohort_c=5, params=params)
    assert regime == "FLAT"
    # Predictions equal T_min per turn
    assert preds == [t["tpot_pred"] for t in turns]
    assert info["peak_pressure"] < 1.0
    assert info["jump_turn"] is None


def test_saturating_cell_ramps_from_t_min_to_t_max() -> None:
    """ctx grows to overflow KV throughout → sustained saturation.

    For SATURATING, T_max uses the physical asymptote (~205 ms), NOT the
    llm-d per-cell mean (which is dragged down by pre-saturation low-tpot
    turns). The 3-turn sharp ramp reaches the plateau quickly.
    """
    params = RooflineParams()
    turns = [_turn(i, cached=400.0 * i, new=100.0, output=25.0, n=80,
                   tpot_pred=15.0 + 0.1 * i, tpot_pred_llm_d=200.0) for i in range(20)]
    regime, preds, info = predict_cell(
        turns, cohort_c=80, params=params, use_kernel_lookup=False,
    )
    assert regime == "SATURATING"
    assert info["jump_turn"] is not None
    # First turn (pre-jump) = T_min
    assert preds[0] == turns[0]["tpot_pred"]
    # T_max for SATURATING = physical ~205 ms, ignoring llm-d=200 cap
    assert 195 < info["t_max_ms"] < 215
    # Last turn plateaued at T_max (constant ceiling under use_kernel_lookup=False)
    assert abs(preds[-1] - info["t_max_ms"]) < 1.0


def test_perturbing_cell_spikes_to_t_max_at_pressure_turns() -> None:
    """ctx crosses pool capacity briefly, then request_count drops → recovers."""
    params = RooflineParams()
    turns = [
        _turn(0, 0,    100, 80, 160, tpot_pred=10.0, tpot_pred_llm_d=60.0),
        _turn(1, 1200, 100, 80, 142, tpot_pred=11.0, tpot_pred_llm_d=60.0),
        _turn(2, 2000, 100, 80, 136, tpot_pred=12.0, tpot_pred_llm_d=60.0),
        _turn(3, 4000, 100, 80, 123, tpot_pred=14.0, tpot_pred_llm_d=60.0),
        _turn(4, 4400, 100, 80, 104, tpot_pred=15.0, tpot_pred_llm_d=60.0),
        # Bulk completion drops request_count + ctx growth slows
        _turn(5, 4500, 100, 80, 47,  tpot_pred=15.0, tpot_pred_llm_d=60.0),
        _turn(6, 4600, 100, 80, 46,  tpot_pred=15.0, tpot_pred_llm_d=60.0),
        _turn(7, 4700, 100, 80, 45,  tpot_pred=15.0, tpot_pred_llm_d=60.0),
    ]
    regime, preds, info = predict_cell(
        turns, cohort_c=160, params=params, use_kernel_lookup=False,
    )
    assert regime == "PERTURBING"
    # Last turn = T_min (recovered)
    assert preds[-1] == turns[-1]["tpot_pred"]
    # Some turn = T_max = min(physical_T_upper, 60) = 60
    # (constant-cap regime, use_kernel_lookup=False)
    assert any(abs(p - 60.0) < 0.01 for p in preds)


def test_t_max_capped_by_llmd_only_for_perturbing_regime() -> None:
    """PERTURBING uses min(physical, llm-d) — chat-like brushes capacity then
    recovers, so the workload-specific saturation envelope (llm-d) caps the
    peak. SATURATING uses physical_T_upper directly (this is tested above).
    """
    params = RooflineParams()
    # Build a perturbing cell: pressure crosses 1 mid-trajectory then drops
    # via active drop in last turns.
    turns = []
    # Early turns: pressure crosses 1 (large ctx × full cohort)
    for i in range(4):
        turns.append(_turn(i, cached=6000.0 + 200 * i, new=100.0, output=25.0, n=80,
                            tpot_pred=10.0, tpot_pred_llm_d=28.0))
    # Drop active sharply to force recovery (pressure drops below 1 since 10 << capacity)
    for i in range(4, 10):
        turns.append(_turn(i, cached=6800.0 + 50 * (i - 4), new=100.0, output=25.0, n=10,
                            tpot_pred=10.0, tpot_pred_llm_d=28.0))
    regime, preds, info = predict_cell(turns, cohort_c=80, params=params)
    assert regime == "PERTURBING", regime
    # T_max for PERTURBING = min(physical=205, llm-d=28) = 28
    assert abs(info["t_max_ms"] - 28.0) < 0.5


def test_t_max_falls_back_to_physical_when_llmd_missing() -> None:
    """No tpot_pred_llm_d → use physical T_upper ≈ 205 ms."""
    params = RooflineParams()
    turns = [_turn(i, cached=600.0 * i, new=100.0, output=25.0, n=80,
                   tpot_pred=10.0, tpot_pred_llm_d=None) for i in range(20)]
    regime, preds, info = predict_cell(turns, cohort_c=80, params=params)
    assert regime == "SATURATING"
    assert 195 < info["t_max_ms"] < 215


def test_preds_align_to_input_turn_order_not_sort_order() -> None:
    """Predictions must come back in the same order as input turns, not sorted."""
    params = RooflineParams()
    # Build a saturating cell but feed turns in reversed order
    raw_turns = [_turn(i, cached=600.0 * i, new=100.0, output=25.0, n=80,
                       tpot_pred=10.0 + i, tpot_pred_llm_d=200.0) for i in range(20)]
    shuffled = list(reversed(raw_turns))
    regime, preds, info = predict_cell(shuffled, cohort_c=80, params=params)
    assert regime == "SATURATING"
    # Predictions should be aligned with the shuffled input (turn 9 first → high; turn 0 last → low)
    # Turn 0 prediction should be near T_min[0] = 10.0 (pre-jump)
    assert preds[-1] == 10.0
