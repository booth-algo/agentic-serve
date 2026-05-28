"""Tests for the augmenter's turn-history-aware sustained-saturation tracking."""

from __future__ import annotations

from profiling.process.emitters.augment_simulator_predictions_with_two_roofline import (
    K_SUSTAIN,
    predict_cell_turns,
)
from simulator.closed_form_tpot import RooflineParams


def _turn(idx: int, cached: float, new: float, output: float, active: int) -> dict:
    return {
        "turn_index": idx,
        "cached_context_tokens": cached,
        "new_prefill_tokens": new,
        "output_tokens": output,
        "engine_max_decode_batch": active,
    }


def test_k_sustain_is_two() -> None:
    assert K_SUSTAIN == 2


def test_no_pressure_cell_predicts_lower_roofline() -> None:
    """Active << capacity → consecutive_saturated never increments → use active."""
    params = RooflineParams()
    turns = [_turn(i, cached=100.0 + 20 * i, new=50.0, output=30.0, active=5) for i in range(10)]
    preds = predict_cell_turns(turns, cohort_c=5, params=params)
    # Decode-bound roofline at c=5 should be in the 4–10 ms range
    for p in preds:
        assert 3.0 < p < 15.0, p


def test_sustained_saturation_uses_cohort_c() -> None:
    """When active stays ≥ capacity for K_SUSTAIN turns, cohort c drives pressure.

    Synthetic cell mimicking swe c=80 late-turn dynamics: active and
    capacity_batch shrink together, so per-turn pressure_active stays just at
    1.0; only the turn-history logic should drive predictions toward T_upper.
    """
    params = RooflineParams()
    cohort_c = 80
    # Construct turns where active == capacity_batch every turn (pressure_active = 1.0).
    # ctx grows linearly so per_session_blocks grows; capacity_batch shrinks.
    turns = []
    for i in range(15):
        cached = 5000.0 + 200 * i
        psb = (cached + 100 + 12) / 16  # ~= per_session_blocks
        capacity = max(1, int(params.available_kv_blocks // max(1, psb)))
        turns.append(_turn(i, cached=cached, new=100.0, output=25.0, active=capacity))
    preds = predict_cell_turns(turns, cohort_c=cohort_c, params=params)

    # First K_SUSTAIN-1 turns: not yet sustained → low predictions
    assert preds[0] < preds[2], (preds[0], preds[2])
    # From turn K_SUSTAIN onward: cohort c drives pressure → should be higher
    assert preds[K_SUSTAIN] > preds[0], (preds[K_SUSTAIN], preds[0])
    # By the end the prediction should be meaningfully above T_lower
    assert preds[-1] > 30.0, preds[-1]


def test_recovering_cell_resets_after_active_drops_below_capacity() -> None:
    """Active drops sharply (sessions completed) → pressure_active < 1 → reset.

    Mimics chat c=320 around turns 4–10: active falls from cohort to a much
    smaller number, KV pool is no longer contended.
    """
    params = RooflineParams()
    cohort_c = 320
    # Build a trajectory that pushes pressure ≥ 1 for the first 3 turns, then
    # active drops sharply (sessions complete) so pressure falls below 1.
    turns: list[dict] = []
    for i in range(3):
        # Tight cell: active ~= capacity_batch → pressure ≈ 1
        cached = 1200.0 + 100 * i
        psb = (cached + 300 + 100) / 16
        capacity = max(1, int(params.available_kv_blocks // max(1, psb)))
        turns.append(_turn(i, cached=cached, new=300.0, output=200.0, active=capacity))
    # Now active drops to a third of capacity → recovery
    for i in range(3, 8):
        cached = 1500.0 + 100 * i
        psb = (cached + 300 + 100) / 16
        capacity = max(1, int(params.available_kv_blocks // max(1, psb)))
        turns.append(_turn(i, cached=cached, new=300.0, output=200.0, active=max(1, capacity // 3)))

    preds = predict_cell_turns(turns, cohort_c=cohort_c, params=params)

    # During sustained phase (turn 2+), prediction should be elevated above T_lower
    assert preds[2] > preds[0], (preds[2], preds[0])
    # After recovery, prediction should drop substantially (no more cohort-c amp)
    assert preds[-1] < preds[2], (preds[-1], preds[2])


def test_single_blip_does_not_trigger_sustained() -> None:
    """One isolated saturated turn must not flip to sustained-saturation regime."""
    params = RooflineParams()
    cohort_c = 80
    turns = []
    # 2 low-pressure turns
    for i in range(2):
        turns.append(_turn(i, cached=100.0 + 20 * i, new=50.0, output=30.0, active=cohort_c))
    # 1 saturated turn (active = capacity, single blip)
    big_cached = 4400.0
    psb = (big_cached + 50 + 15) / 16
    capacity = max(1, int(params.available_kv_blocks // psb))
    turns.append(_turn(2, cached=big_cached, new=50.0, output=30.0, active=capacity))
    # Then back to below capacity (active much smaller)
    for i in range(3, 6):
        turns.append(_turn(i, cached=4500.0, new=50.0, output=30.0, active=cohort_c // 4))

    preds = predict_cell_turns(turns, cohort_c=cohort_c, params=params)

    # The post-blip turns should NOT be in sustained-saturation regime
    # — they should return to roughly T_lower since the blip was isolated.
    assert preds[-1] < 60.0, preds[-1]


def test_missing_active_telemetry_falls_back_safely() -> None:
    """When engine_max_decode_batch is missing, augmenter must not crash."""
    params = RooflineParams()
    turns = [
        {"turn_index": 0, "cached_context_tokens": 100.0,
         "new_prefill_tokens": 50.0, "output_tokens": 30.0},
        {"turn_index": 1, "cached_context_tokens": 200.0,
         "new_prefill_tokens": 50.0, "output_tokens": 30.0},
    ]
    preds = predict_cell_turns(turns, cohort_c=4, params=params)
    assert len(preds) == 2
    for p in preds:
        assert p > 0
