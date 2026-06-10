"""Tests for the session regime classifier + its stepping-ramp window hint."""

from __future__ import annotations

from simulator._legacy.session_regime_classifier import (
    JUMP_FLOOR,
    PRESSURE_ONSET,
    _pcross,
    _smoothstep,
    classify_session,
    session_ramp_window,
)


def _turn(idx: int, cached: float, new: float, out: float, sched: float) -> dict:
    return {
        "turn_index": idx,
        "cached_context_tokens": cached,
        "new_prefill_tokens": new,
        "output_tokens": out,
        "scheduled_requests": sched,
    }


def _saturate_cell() -> list[dict]:
    # Constant full cohort (non-draining), monotone context growth, short output.
    return [_turn(i, 2000 + 1000 * i, 100, 28, 160) for i in range(12)]


def _flat_cell() -> list[dict]:
    # Low concurrency, small context -> never loads the pool.
    return [_turn(i, 200 + 50 * i, 100, 80, 4) for i in range(10)]


def _chat_like_perturb() -> list[dict]:
    # Draining cohort, output-heavy (ratio<1), no prefill dominance.
    scheds = [160, 130, 100, 70, 45, 30, 20, 15, 12, 10]
    return [_turn(i, 2000 + 300 * i, 100, 200, s) for i, s in enumerate(scheds)]


def _osworld_like_perturb() -> list[dict]:
    # Draining cohort BUT prefill-heavy + long output + loads the pool.
    scheds = [200, 160, 120, 80, 55, 35, 25, 18, 14, 10]
    return [_turn(i, 2000 + 200 * i, 1200, 90, s) for i, s in enumerate(scheds)]


# ----------------------------------------------------------------- smoothstep
def test_smoothstep_degenerate_window() -> None:
    # Zero-width window: 0 at/below the knot, 1 strictly above it.
    assert _smoothstep(2.0, 2.0, 2.0) == 0.0  # x <= lo
    assert _smoothstep(2.5, 2.0, 2.0) == 1.0  # x > lo
    assert _smoothstep(1.5, 2.0, 2.0) == 0.0  # x < lo


def test_pcross_returns_first_crossing_turn_index() -> None:
    ts = _saturate_cell()
    # pressures rise with context; the first crossing exists and is an int.
    from simulator._legacy.session_regime_classifier import _pressure

    pcs = [_pressure(t) for t in ts]
    assert _pcross(pcs, ts, PRESSURE_ONSET) is not None
    assert _pcross(pcs, ts, 99.0) is None  # never reaches an absurd threshold


# ----------------------------------------------------------- class + window
def test_saturate_emits_window_and_pcross_jump() -> None:
    r = classify_session(_saturate_cell())
    assert r["class"] == "SATURATE"
    assert r["jump_turn"] is not None and r["jump_turn"] >= JUMP_FLOOR
    assert r["jump_start"] is not None and r["jump_end"] is not None
    assert r["jump_end"] >= r["jump_start"] >= JUMP_FLOOR
    assert 0.0 < r["confidence"] <= 1.0


def test_flat_cell_has_no_hint() -> None:
    r = classify_session(_flat_cell())
    assert r["class"] == "FLAT"
    assert r["jump_turn"] is None
    assert r["jump_start"] is None and r["jump_end"] is None
    assert r["confidence"] == 0.0


def test_chat_like_perturb_is_hint_noop() -> None:
    """Output-heavy draining PERTURB (chat) must NOT get a window — the soft hint
    has to leave it byte-identical to the pressure path."""
    r = classify_session(_chat_like_perturb())
    assert r["class"] == "PERTURB_RETURN"
    assert r["confidence"] == 0.0
    assert r["jump_start"] is None


def test_osworld_like_perturb_gets_stepping_window() -> None:
    """Prefill-heavy long-output PERTURB (osworld) DOES get a stepping window even
    though its class is PERTURB_RETURN — this is the osworld-stepping fix."""
    r = classify_session(_osworld_like_perturb())
    assert r["class"] == "PERTURB_RETURN"
    assert r["confidence"] > 0.0
    assert r["jump_start"] is not None and r["jump_end"] is not None
    assert r["jump_end"] >= r["jump_start"] >= JUMP_FLOOR


def test_session_ramp_window_matches_classify() -> None:
    cell = _saturate_cell()
    win = session_ramp_window(cell)
    full = classify_session(cell)
    assert win == {k: full[k] for k in ("jump_start", "jump_end", "confidence")}


def test_classify_accepts_cell_dict_via_helper() -> None:
    win = session_ramp_window({"multiturn_turn_predictions": _saturate_cell()})
    assert win["confidence"] > 0.0


def test_higher_concurrency_has_higher_confidence() -> None:
    """Confidence tracks the measured jump-timing reliability band [40,160]."""
    lo = classify_session([_turn(i, 2000 + 1000 * i, 100, 28, 60) for i in range(12)])
    hi = classify_session([_turn(i, 2000 + 1000 * i, 100, 28, 200) for i in range(12)])
    assert lo["class"] == hi["class"] == "SATURATE"
    assert hi["confidence"] >= lo["confidence"]
