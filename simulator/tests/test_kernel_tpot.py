"""Tests for the kernel-composition per-turn TPOT predictor."""

from __future__ import annotations

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_tpot import (
    OUT_KNEE_HI,
    OUT_KNEE_LO,
    P_HI_LONG,
    P_HI_SHORT,
    P_LO,
    KernelTurnInput,
    _smoothstep,
    predict_cell_tpot,
    predict_turn_tpot,
    saturated_ceiling_ms,
)


def _turn(cached: float, new: float, out: float, sched: float) -> KernelTurnInput:
    return KernelTurnInput(
        cached_context_tokens=cached,
        new_prefill_tokens=new,
        output_tokens=out,
        scheduled_requests=sched,
    )


# ------------------------------------------------------------ smoothstep
def test_smoothstep_endpoints_and_midpoint() -> None:
    assert _smoothstep(0.0, P_LO, P_HI_LONG) == 0.0
    assert _smoothstep(10.0, P_LO, P_HI_LONG) == 1.0
    mid = _smoothstep((P_LO + P_HI_LONG) / 2, P_LO, P_HI_LONG)
    assert abs(mid - 0.5) < 1e-9


def test_output_gated_step_is_steeper_for_short_output() -> None:
    """Short-output (swe/terminal) workloads reach the saturation ceiling at a
    lower pressure (steeper step) than long-output (chat/osworld) ones, and
    cap higher — so under the same high-pressure cohort a short-output turn
    predicts above a long-output one.
    """
    assert P_HI_SHORT < P_HI_LONG
    assert OUT_KNEE_LO < OUT_KNEE_HI
    short = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=160))
    long = predict_turn_tpot(_turn(cached=8000, new=100, out=200, sched=160))
    assert short > long


def test_tiny_output_turn_does_not_saturate() -> None:
    """A high-pressure turn producing very few output tokens can't sustain
    saturation — it predicts near the (low) kernel step, not the ceiling. The
    same workload with enough output tokens does saturate.
    """
    # tiny output (≈9 tok): full cohort, but the turn finishes before the
    # eviction queue builds → near kernel step, far below the ceiling.
    tiny = predict_turn_tpot(_turn(cached=1400, new=100, out=9, sched=320))
    # enough output (≥ sustain HI) at the same cohort + bigger context → saturates.
    sustained = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=320))
    assert tiny < 40.0, tiny
    assert sustained > 150.0, sustained
    assert tiny < sustained


# ------------------------------------------------------------ saturated ceiling
def test_saturated_ceiling_inverse_in_output() -> None:
    """Short-output turns saturate higher than long-output turns."""
    short = saturated_ceiling_ms(27)   # swe/terminal-like
    long = saturated_ceiling_ms(87)    # osworld-like
    assert short > long
    # matches the measured anchors within a few ms
    assert 220 < short < 260
    assert 120 < long < 160


def test_saturated_ceiling_capped() -> None:
    # a 1-token output would explode the 1/output term; the cap holds it
    assert saturated_ceiling_ms(1) == 260.0


# ------------------------------------------------------------ regime behavior
def test_flat_regime_returns_kernel_step() -> None:
    """Low pressure (small ctx, low concurrency) => prediction ≈ kernel step,
    well below the saturated ceiling.
    """
    out = predict_turn_tpot(_turn(cached=200, new=100, out=80, sched=5))
    assert out < 20.0  # decode-step floor territory


def test_saturating_turn_approaches_output_ceiling() -> None:
    """High pressure (big ctx, large cohort) => prediction near T_upper(output)."""
    # short output => high ceiling
    hi = predict_turn_tpot(_turn(cached=9000, new=100, out=27, sched=160))
    # long output => lower ceiling, same pressure regime
    lo = predict_turn_tpot(_turn(cached=9000, new=100, out=87, sched=160))
    assert hi > lo
    assert hi > 150.0  # well into the amplified regime
    assert lo < hi


def test_prediction_monotone_in_pressure() -> None:
    """Holding workload fixed, more concurrent sessions => higher TPOT."""
    base = predict_turn_tpot(_turn(cached=5000, new=100, out=40, sched=20))
    more = predict_turn_tpot(_turn(cached=5000, new=100, out=40, sched=200))
    assert more > base


def test_prediction_bounded_by_kernel_step_and_ceiling() -> None:
    t = _turn(cached=6000, new=120, out=50, sched=120)
    pred = predict_turn_tpot(t)
    assert pred >= predict_turn_tpot(_turn(6000, 120, 50, 1)) - 1e-6  # >= floor
    assert pred <= saturated_ceiling_ms(50) + 1e-6                    # <= ceiling


def test_predict_cell_returns_per_turn_list() -> None:
    turns = [_turn(100 * i, 100, 60, 40) for i in range(5)]
    out = predict_cell_tpot(turns)
    assert len(out) == 5
    assert all(x > 0 for x in out)


def test_params_passthrough_does_not_crash() -> None:
    p = RooflineParams()
    assert predict_turn_tpot(_turn(1000, 100, 60, 40), p) > 0
