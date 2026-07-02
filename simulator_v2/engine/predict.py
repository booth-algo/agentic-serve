# simulator_v2/engine/predict.py

from simulator_v2.core.types import TurnPrediction
from simulator_v2.engine import amplifier, queue_sim
from simulator_v2.core.mode import Mode, mode


@mode(Mode.SHARED)
def _tpot_ms(hw, turns, concurrency, *, cohort=None) -> list[float]:
    """Per-turn TPOT: the saturation amplifier (decode floor ramped toward the
    KV-pressure ceiling). Uses each turn's own `scheduled_requests`, not the
    global concurrency."""
    del concurrency  # amplifier reads per-turn turn.scheduled_requests
    context_spread = cohort.context_spread if cohort is not None else 1.0
    return [amplifier.tpot_ms(hw, turn, context_spread=context_spread) for turn in turns]


@mode(Mode.SHARED)
def _ttft_ms(hw, turns, concurrency, *, trajectories=None, shared_prefix_tokens=0.0) -> list[float]:
    """Per-turn TTFT: the queue sim replays the cohort through the scheduler and
    times each request's first token (prefill under contention)."""
    return queue_sim.predict_ttft(
        hw, turns, concurrency, trajectories=trajectories,
        shared_prefix_tokens=shared_prefix_tokens,
    )


@mode(Mode.SHARED)
def predict(hw, turns, concurrency, *, cohort=None, trajectories=None, shared_prefix_tokens=0.0) -> list[TurnPrediction]:
    """Predict per-turn TTFT, TPOT, and E2EL for one benchmark cell."""
    tpots = _tpot_ms(hw, turns, concurrency, cohort=cohort)
    ttfts = _ttft_ms(hw, turns, concurrency, trajectories=trajectories, shared_prefix_tokens=shared_prefix_tokens)
    return [
        TurnPrediction(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2el_ms=ttft_ms + tpot_ms * turn.osl_tokens,
        )
        for tpot_ms, ttft_ms, turn in zip(tpots, ttfts, turns)
    ]
