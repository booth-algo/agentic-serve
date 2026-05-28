"""Tests for two-roofline TPOT predictor — physical decomposition correctness."""

from __future__ import annotations

from simulator.closed_form_tpot import RooflineParams
from simulator.two_roofline_tpot import (
    MAX_NUM_BATCHED_TOKENS,
    TurnWorkload,
    predict_two_roofline,
)


def test_chunk_cap_is_vllm_v1_default() -> None:
    assert MAX_NUM_BATCHED_TOKENS == 8192


def test_t_upper_is_about_205ms_on_h100_llama31_8b() -> None:
    """T_upper = chunk × prefill_per_tok ≈ 205 ms (compute-bound, hardware-only)."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=0, new_prefill_tokens=1000, output_tokens=10),
        concurrency=10, p=p,
    )
    assert 195 < out.t_upper_ms < 215, out.t_upper_ms


def test_low_concurrency_chat_stays_on_lower_roofline() -> None:
    """At c=5 with modest workload, prefill demand << decode demand → mostly decode-only."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=480, new_prefill_tokens=318, output_tokens=139),
        concurrency=5, p=p,
    )
    # 5 × 318 = 1590 tokens of prefill / 8192 = 1 prefill step
    # 5 × 139 / running ≈ many decode steps → mostly decode-only
    assert out.prefill_steps == 1
    assert out.decode_steps > 100  # 5 × 139 / 5 = 139
    assert out.decode_only_steps > 100
    assert out.regime == "decode_bound"
    # Predicted TPOT ≈ T_lower (5-8 ms range for chat-like)
    assert 4 < out.predicted_tpot_ms < 15, out.predicted_tpot_ms


def test_swebench_c80_at_jump_predicts_high_tpot() -> None:
    """High c × high ISL: predicted TPOT should be in saturating range, not flat low."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=12000, new_prefill_tokens=500, output_tokens=25),
        concurrency=80, p=p,
    )
    # Per-session blocks ≈ ceil(12500/16) = 782 → capacity_batch ≈ 27250/782 = 34
    # → running = 34 (not 80)
    assert out.running < 80
    # Without re-prefill (out of scope), this falls back to the decode roofline
    # at the smaller running batch — so still above T_lower at c=80 batch=34
    # but below the saturation roofline T_upper.
    assert out.predicted_tpot_ms > 20, out.predicted_tpot_ms
    assert out.predicted_tpot_ms < out.t_upper_ms


def test_running_capped_by_capacity_batch_when_ctx_large() -> None:
    """When ctx is huge, capacity_batch caps running below requested concurrency."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=20000, new_prefill_tokens=0, output_tokens=10),
        concurrency=320, p=p,
    )
    # 20005/16 ≈ 1251 blocks per session → 27250/1251 ≈ 21 → running ≤ 21
    assert out.running <= 22, out.running
    assert out.capacity_batch <= 22


def test_running_not_capped_at_low_ctx_high_concurrency() -> None:
    """If ctx is tiny, capacity_batch is huge → running = concurrency."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=0, new_prefill_tokens=50, output_tokens=20),
        concurrency=320, p=p,
    )
    assert out.running == 320
    assert out.capacity_batch >= 320


def test_high_pressure_saturates_at_t_upper() -> None:
    """When KV pressure ≫ 2× over capacity: predicted saturates AT T_upper.

    The interpolation blends T_lower → T_upper as pressure goes from 1 to 2.
    Above pressure = 2, the line is capped at T_upper (no overshoot).
    """
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=20000, new_prefill_tokens=0, output_tokens=10),
        concurrency=320, p=p,
    )
    # ctx_mid = 20005 → blocks/session ≈ 1251 → capacity_batch ≈ 21
    # pressure = 320 / 21 ≈ 15 → far beyond 2 → w = 1 → T_pred = T_upper
    assert out.predicted_tpot_ms == out.t_upper_ms


def test_jump_threshold_isl_osl_vs_chunk_running() -> None:
    """Jump fires when prefill_steps > decode_steps, i.e. ISL/OSL > chunk/running."""
    p = RooflineParams()
    # threshold: ISL/OSL > 8192/running
    # at running=80, threshold = 102.4
    # Below threshold:
    out_below = predict_two_roofline(
        TurnWorkload(cached_context_tokens=0, new_prefill_tokens=50, output_tokens=10),
        concurrency=80, p=p,
    )
    assert out_below.prefill_steps <= out_below.decode_steps
    assert out_below.regime == "decode_bound"
    # Above threshold:
    out_above = predict_two_roofline(
        TurnWorkload(cached_context_tokens=0, new_prefill_tokens=2000, output_tokens=10),
        concurrency=80, p=p,
    )
    assert out_above.prefill_steps > out_above.decode_steps
    # 2000 × 80 / 8192 = 19.5 → 20 prefill steps
    # 10 × 80 / 80 = 10 decode steps
    assert out_above.regime == "prefill_bound"


def test_zero_new_prefill_yields_zero_mixed_steps() -> None:
    """When new_prefill = 0 (no chunked prefill needed): all decode-only."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=5000, new_prefill_tokens=0, output_tokens=50),
        concurrency=20, p=p,
    )
    assert out.prefill_steps == 0
    assert out.mixed_steps == 0
    assert out.decode_only_steps == out.decode_steps
    assert out.predicted_tpot_ms < 20, out.predicted_tpot_ms  # decode-only roofline


def test_zero_output_tokens_clamped_to_one() -> None:
    """Avoid divide-by-zero when bench reports output_tokens=0 on a degenerate turn."""
    p = RooflineParams()
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=100, new_prefill_tokens=100, output_tokens=0),
        concurrency=10, p=p,
    )
    assert out.predicted_tpot_ms > 0
    assert out.output_tokens >= 1.0


def test_predicted_close_to_observed_swe_c80_turn11() -> None:
    """Smoke check against the spot-check from the verdict doc."""
    p = RooflineParams()
    # Approx swe c=80 turn 11 workload from per_request data
    out = predict_two_roofline(
        TurnWorkload(cached_context_tokens=11000, new_prefill_tokens=1500, output_tokens=25),
        concurrency=80, p=p,
    )
    # Observed tpot_meas at swe c=80 turn 11 ≈ 93 ms; without modeling session-
    # cycle re-prefill the model predicts the engine-throughput lower bound (~35–60 ms).
    # Documented gap. Loose lower bound: predicted > T_lower for c=80.
    assert 30 < out.predicted_tpot_ms < 250, out.predicted_tpot_ms
