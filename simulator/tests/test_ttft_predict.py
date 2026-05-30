"""Tests for the forward queue-wait TTFT predictor."""

from __future__ import annotations

from simulator.closed_form_tpot import RooflineParams
from simulator.ttft_predict import (
    PROFILE_DIST,
    _baseline_prefill_ms,
    predict_cell_ttft,
    predict_turn_ttft,
)


def _turn(idx: int, cached: float, new: float, out: float, sched: float | None = None) -> dict:
    d = {
        "turn_index": idx,
        "cached_context_tokens": cached,
        "new_prefill_tokens": new,
        "output_tokens": out,
    }
    if sched is not None:
        d["scheduled_requests"] = sched
    return d


# --------------------------------------------------------------- baseline
def test_single_session_is_the_prefill_floor() -> None:
    """sched=1, tiny context => pressure≈0 => TTFT == queue-free prefill baseline."""
    p = RooflineParams()
    pred = predict_turn_ttft(cached=500, new_prefill=200, output=30, sched=1, tpot=12.0, params=p)
    base = _baseline_prefill_ms(200, 500, p)
    # sub term ~ RESIDUAL*pressure*1*step is tiny at sched=1; over term is 0.
    assert pred >= base
    assert pred < base + 30.0  # only a small residual at sched=1


def test_large_prefill_uses_roofline_not_clamped_grid() -> None:
    """new_prefill beyond the grid (1024) must grow with U (roofline), not clamp."""
    p = RooflineParams()
    small = _baseline_prefill_ms(1024, 0, p)
    big = _baseline_prefill_ms(4096, 0, p)
    assert big > small * 2.0  # 4x tokens -> roofline-linear, well above the clamp


# ----------------------------------------------------------- monotonicity
def test_ttft_monotone_in_concurrency() -> None:
    """More concurrent sessions => more queue wait => higher TTFT."""
    lo = predict_turn_ttft(4000, 200, 40, sched=20, tpot=20.0)
    hi = predict_turn_ttft(4000, 200, 40, sched=300, tpot=20.0)
    assert hi > lo


def test_oversubscription_term_dominates_at_high_pressure() -> None:
    """Above capacity the backlog (pressure-1)*out*tpot drives TTFT up steeply."""
    # large cached => few blocks of headroom => high pressure at moderate sched.
    mid = predict_turn_ttft(9000, 200, 80, sched=120, tpot=150.0)
    top = predict_turn_ttft(9000, 200, 80, sched=320, tpot=150.0)
    assert top > mid + 1000.0  # unbounded queue: big jump, not a bounded ceiling


# ------------------------------------------------------------ forward path
def test_forward_path_ignores_measured_scheduled() -> None:
    """Forward prediction must not read scheduled_requests."""
    with_sched = [_turn(i, 2000 + 1000 * i, 150, 28, sched=999) for i in range(8)]
    without = [_turn(i, 2000 + 1000 * i, 150, 28) for i in range(8)]
    a = predict_cell_ttft(with_sched, "swebench-multiturn-synth", 160)
    b = predict_cell_ttft(without, "swebench-multiturn-synth", 160)
    assert a == b


def test_oracle_path_uses_measured_scheduled() -> None:
    hi = [_turn(i, 2000 + 1000 * i, 150, 28, sched=320) for i in range(8)]
    lo = [_turn(i, 2000 + 1000 * i, 150, 28, sched=20) for i in range(8)]
    a = predict_cell_ttft(hi, "swebench-multiturn-synth", 320, oracle=True)
    b = predict_cell_ttft(lo, "swebench-multiturn-synth", 320, oracle=True)
    assert a != b
    assert sum(a) > sum(b)  # bigger measured cohort -> more queueing


def test_empty_cell_returns_empty() -> None:
    assert predict_cell_ttft([], "swebench-multiturn-synth", 160) == []


def test_unknown_profile_falls_back_to_concurrency() -> None:
    """No survival curve -> sched_hat == concurrency; predictor still runs."""
    out = predict_cell_ttft([_turn(0, 1000, 200, 40)], "nonexistent-profile", 64)
    assert len(out) == 1 and out[0] > 0


def test_tpot_override_is_used() -> None:
    """A larger turn-decode (out*tpot) raises TTFT in the oversubscribed regime."""
    turns = [_turn(0, 9000, 200, 80, sched=320)]
    lo = predict_cell_ttft(turns, "osworld-multiturn-synth", 320, oracle=True, tpot_preds=[50.0])
    hi = predict_cell_ttft(turns, "osworld-multiturn-synth", 320, oracle=True, tpot_preds=[200.0])
    assert hi[0] > lo[0]


def test_all_profiles_have_distribution() -> None:
    for prof in ("swebench-multiturn-synth", "osworld-multiturn-synth",
                 "terminalbench-multiturn-synth", "chat-multiturn-synth"):
        assert prof in PROFILE_DIST
