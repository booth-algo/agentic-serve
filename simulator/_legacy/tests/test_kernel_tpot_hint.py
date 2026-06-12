"""Tests for the classifier stepping-ramp soft-hint kernel predictor."""

from __future__ import annotations

from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator._legacy.kernel_tpot_hint import predict_cell_tpot_hinted


def _kti(cached: float, new: float, out: float, sched: float) -> KernelTurnInput:
    return KernelTurnInput(
        cached_context_tokens=cached,
        new_prefill_tokens=new,
        output_tokens=out,
        scheduled_requests=sched,
    )


def _saturate_cell() -> list[KernelTurnInput]:
    return [_kti(2000 + 1000 * i, 100, 28, 160) for i in range(12)]


def _chat_like_cell() -> list[KernelTurnInput]:
    # Draining, output-heavy (ratio<1): classifier confidence 0 -> hint no-op.
    scheds = [160, 130, 100, 70, 45, 30, 20, 15, 12, 10]
    return [_kti(2000 + 300 * i, 100, 200, s) for i, s in enumerate(scheds)]


def _flat_cell() -> list[KernelTurnInput]:
    return [_kti(200 + 50 * i, 100, 80, 4) for i in range(10)]


def _lagging_cell() -> list[KernelTurnInput]:
    # Prefill-heavy, long-output, high cohort (the osworld pattern) with a slowly
    # climbing pressure: the turn-space ramp window outpaces the pressure path, so
    # the timing hint has real room to pull the lagging mid-session turns up.
    return [_kti(500 + 300 * i, 1200, 90, 160) for i in range(12)]


def test_hint_is_byte_identical_for_chat_like_cell() -> None:
    """A confidence-0 cell must come back exactly equal to the production kernel."""
    cell = _chat_like_cell()
    assert predict_cell_tpot_hinted(cell) == predict_cell_tpot(cell)


def test_hint_is_byte_identical_for_flat_cell() -> None:
    cell = _flat_cell()
    assert predict_cell_tpot_hinted(cell) == predict_cell_tpot(cell)


def test_hint_never_lowers_below_baseline() -> None:
    """The pull is one-sided: every hinted turn is >= the pressure-path baseline."""
    cell = _saturate_cell()
    base = predict_cell_tpot(cell)
    hint = predict_cell_tpot_hinted(cell)
    assert len(hint) == len(base)
    assert all(h >= b - 1e-9 for h, b in zip(hint, base))


def test_hint_lifts_a_lagging_cell() -> None:
    """When the pure pressure path rises late (osworld-like prefill-heavy cell),
    the timing hint pulls the lagging mid-session turns up toward the ceiling."""
    cell = _lagging_cell()
    base = predict_cell_tpot(cell)
    hint = predict_cell_tpot_hinted(cell)
    assert max(h - b for h, b in zip(hint, base)) > 5.0


def test_empty_cell_returns_empty() -> None:
    assert predict_cell_tpot_hinted([]) == []
