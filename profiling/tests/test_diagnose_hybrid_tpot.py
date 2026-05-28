"""Tests for diagnose_hybrid_tpot.py — physical formulas + regime labels."""

from __future__ import annotations

import json
from pathlib import Path

from profiling.process.predictors.diagnose_hybrid_tpot import (
    MAX_NUM_BATCHED_TOKENS,
    diagnose_turn,
    label_regime,
    load_sim_predictions,
    physical_decode_step_ms,
    physical_prefill_per_token_ms,
)
from simulator.closed_form_tpot import RooflineParams


# ---------------------------------------------------- physical formulas


def test_chunk_cap_is_vllm_v1_default() -> None:
    """8192 is vLLM v1 default. Bench traces confirm this is the cap in effect."""
    assert MAX_NUM_BATCHED_TOKENS == 8192


def test_physical_decode_step_low_ctx_dominated_by_weights() -> None:
    """At small ctx, weights bytes dominate the bandwidth read."""
    p = RooflineParams()
    weights_only = physical_decode_step_ms(running=1, ctx=1, p=p)
    weights_plus_kv = physical_decode_step_ms(running=1, ctx=1000, p=p)
    assert weights_only < weights_plus_kv
    # 16 GB weights / 3.11 TB/s ≈ 5.15 ms
    assert 4.0 < weights_only < 6.5


def test_physical_decode_step_scales_with_running_and_ctx() -> None:
    p = RooflineParams()
    base = physical_decode_step_ms(running=10, ctx=1000, p=p)
    bigger_batch = physical_decode_step_ms(running=20, ctx=1000, p=p)
    bigger_ctx = physical_decode_step_ms(running=10, ctx=2000, p=p)
    # Doubling running roughly doubles the KV-read term (not the weights term)
    assert base < bigger_batch < 2 * base
    assert base < bigger_ctx < 2 * base


def test_prefill_per_tok_is_a_few_dozen_microseconds() -> None:
    """8B model @ 989 TFLOPS × 0.65 util → ~25 μs/token (~0.025 ms)."""
    p = RooflineParams()
    ms = physical_prefill_per_token_ms(p)
    assert 0.015 < ms < 0.040, ms


# ------------------------------------------------------- regime labels


def _p() -> RooflineParams:
    return RooflineParams(available_kv_blocks=27250, cache_block_size=16)


def test_regime_no_pressure() -> None:
    r = label_regime(
        waiting_max=0, per_session_blocks=10, capacity_batch=2725,
        running=80, concurrency=80, p=_p(),
    )
    assert r == "no_pressure"


def test_regime_kv_admission_throttle() -> None:
    r = label_regime(
        waiting_max=15, per_session_blocks=400, capacity_batch=68,
        running=68, concurrency=80, p=_p(),
    )
    assert r == "kv_admission_throttle"


def test_regime_single_request_near_limit() -> None:
    # per_session = 25000 / 27250 = 0.917 → near
    r = label_regime(
        waiting_max=0, per_session_blocks=25000, capacity_batch=1,
        running=1, concurrency=1, p=_p(),
    )
    assert r == "single_request_near_limit"


def test_regime_single_request_exceeds_limit() -> None:
    r = label_regime(
        waiting_max=0, per_session_blocks=30000, capacity_batch=0,
        running=1, concurrency=1, p=_p(),
    )
    assert r == "single_request_exceeds_limit"


def test_regime_mild_pressure_is_the_catch_all() -> None:
    # running below c, no big wait queue — neither no_pressure nor throttle
    r = label_regime(
        waiting_max=1, per_session_blocks=100, capacity_batch=272,
        running=50, concurrency=80, p=_p(),
    )
    assert r == "mild_pressure"


# --------------------------------------------------- end-to-end on a turn


def test_diagnose_turn_chat_c5_no_pressure() -> None:
    """no-pressure cell from real data: residual should be near 1."""
    turn = {
        "turn_index": 2,
        "tpot_meas": 6.9172,
        "cached_context_tokens": 480,
        "new_prefill_tokens": 318,
        "output_tokens": 139,
        "total_context_tokens": 830,
        "engine_total_step_ms": 945.314,
        "engine_steps": 140,
        "engine_max_decode_batch": 5.0,
        "engine_capacity_waiting_requests": 0.0,
        "engine_total_prefill_tokens": 1590,
        "engine_mixed_steps": 0,
    }
    p = RooflineParams()
    out = diagnose_turn("chat-multiturn-synth", 5, turn, p)
    assert out.regime == "no_pressure"
    assert out.tpot_residual is not None
    # tpot_meas / predicted should be near 1
    assert 0.7 < out.tpot_residual < 1.6
    # step prediction roughly matches engine observation
    assert 0.6 < out.step_residual < 1.5


def test_diagnose_turn_handles_missing_engine_fields() -> None:
    turn = {
        "turn_index": 0,
        "tpot_meas": 10.0,
        "cached_context_tokens": 0,
        "new_prefill_tokens": 100,
        "output_tokens": 50,
        "total_context_tokens": 100,
        # All engine_* fields missing — common for cells without traces
    }
    out = diagnose_turn("probe", 1, turn, RooflineParams())
    assert out.step_residual is None
    assert out.engine_steps is None
    assert out.tpot_residual is not None  # tpot_meas + ctx are enough


# ------------------------------------------------------- real-data smoke


def test_load_sim_predictions_real_data() -> None:
    """Smoke test against the actual dashboard JSON if present."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_path = repo_root / "inference-benchmark/dashboard/public/simulator-predictions.json"
    if not sim_path.exists():
        return  # skip gracefully
    rows = load_sim_predictions(sim_path)
    assert len(rows) > 500, len(rows)
    # All have a regime label
    regimes = {r.regime for r in rows}
    assert "no_pressure" in regimes
    # Sanity: tpot_meas is positive where present
    for r in rows[:50]:
        if r.tpot_meas_ms is not None:
            assert r.tpot_meas_ms > 0
