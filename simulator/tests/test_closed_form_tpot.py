"""Tests for the closed-form TPOT / TTFT predictor.

Covers: roofline math at c=1 baseline; bandwidth-bound vs compute-bound
classification; load monotonicity in c; cached-prefix shortens TTFT;
sidecar-bucket → TurnInput parsing; llm-d config fit endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.closed_form_tpot import (
    ClosedFormTpotPredictor,
    LlmDInferenceSimConfig,
    RooflineParams,
    TurnInput,
    TurnPrediction,
    fit_llm_d_config,
    iter_sidecar_inputs,
    turn_input_from_sidecar,
)


H100_PARAMS_PATH = Path("profiling/data/roofline_params_H100_llama31_8b.json")


def _predictor(kv_pressure_enabled: bool = False) -> ClosedFormTpotPredictor:
    if H100_PARAMS_PATH.exists():
        return ClosedFormTpotPredictor.from_json(
            H100_PARAMS_PATH, kv_pressure_enabled=kv_pressure_enabled
        )
    return ClosedFormTpotPredictor(
        RooflineParams(), kv_pressure_enabled=kv_pressure_enabled
    )


# ---------------------------------------------------------------- math anchors


def test_decode_step_at_c1_matches_hand_calc() -> None:
    """At c=1, ctx≈0, decode is compute-bound ≈ 12.5 ms (H100 BF16, 8B params)."""
    pred = _predictor()
    inp = TurnInput(
        profile="probe",
        concurrency=1,
        turn_index=0,
        output_tokens=2.0,          # ctx_mid = 0 + 1 = 1 token
        new_prefill_tokens=0.0,
        cached_context_tokens=0.0,
    )
    out = pred.predict(inp)
    # 2 · 8.03e9 · 1 / (989e12 · 0.65) = 2.5e-5 s = 0.025 ms compute
    # weights bw: 2 · 8.03e9 / (3.35e12 · 0.93) = ~5.16 ms
    # → bandwidth-bound at c=1.
    assert out.classification == "bandwidth_bound"
    assert 4.5 < out.tpot_ms < 6.0, f"tpot_ms={out.tpot_ms}"


def test_decode_step_at_high_c_with_long_ctx_is_bandwidth_bound() -> None:
    """c=40, ctx=7200 → 53 GB / step → ~17 ms, bandwidth-bound."""
    pred = _predictor()
    inp = TurnInput(
        profile="probe",
        concurrency=40,
        turn_index=12,
        output_tokens=22.0,            # ctx_mid = 7188 + 11 ≈ 7199
        new_prefill_tokens=7188.0,
        cached_context_tokens=0.0,
    )
    out = pred.predict(inp)
    assert out.classification == "bandwidth_bound"
    assert 14.0 < out.tpot_ms < 20.0, f"tpot_ms={out.tpot_ms}"


def test_tpot_monotonic_in_concurrency() -> None:
    """Once weights are amortized, tpot grows with c (more KV reads)."""
    pred = _predictor()
    base = dict(profile="p", turn_index=0, output_tokens=20.0,
                new_prefill_tokens=4000.0, cached_context_tokens=0.0)
    tpots = [pred.predict(TurnInput(concurrency=c, **base)).tpot_ms
             for c in (1, 8, 40, 160)]
    assert all(tpots[i] <= tpots[i + 1] for i in range(len(tpots) - 1)), tpots


def test_cached_prefix_shortens_ttft() -> None:
    """Cached tokens reduce the fresh prefill burst → smaller TTFT."""
    pred = _predictor()
    no_cache = pred.predict(TurnInput(
        profile="p", concurrency=8, turn_index=1, output_tokens=10.0,
        new_prefill_tokens=2000.0, cached_context_tokens=0.0,
    ))
    half_cached = pred.predict(TurnInput(
        profile="p", concurrency=8, turn_index=1, output_tokens=10.0,
        new_prefill_tokens=2000.0, cached_context_tokens=1000.0,
    ))
    full_cached = pred.predict(TurnInput(
        profile="p", concurrency=8, turn_index=1, output_tokens=10.0,
        new_prefill_tokens=2000.0, cached_context_tokens=2000.0,
    ))
    assert no_cache.ttft_ms > half_cached.ttft_ms > full_cached.ttft_ms

    # Hold TOTAL context constant; cache only shifts work from prefill→cached.
    # In that case TPOT really is independent of the cache split.
    fixed_total_no_cache = pred.predict(TurnInput(
        profile="p", concurrency=8, turn_index=1, output_tokens=10.0,
        new_prefill_tokens=2000.0, cached_context_tokens=0.0,
    ))
    fixed_total_half = pred.predict(TurnInput(
        profile="p", concurrency=8, turn_index=1, output_tokens=10.0,
        new_prefill_tokens=1000.0, cached_context_tokens=1000.0,
    ))
    assert fixed_total_no_cache.tpot_ms == pytest.approx(fixed_total_half.tpot_ms)


def test_fresh_prefill_clamped_at_zero() -> None:
    """cached > new_prefill should not produce negative prefill tokens."""
    pred = _predictor()
    out = pred.predict(TurnInput(
        profile="p", concurrency=4, turn_index=0, output_tokens=8.0,
        new_prefill_tokens=500.0, cached_context_tokens=10_000.0,
    ))
    assert out.ttft_ms == pytest.approx(0.0)


def test_e2e_equals_ttft_plus_decode_tokens_times_tpot() -> None:
    pred = _predictor()
    inp = TurnInput(
        profile="p", concurrency=4, turn_index=0, output_tokens=50.0,
        new_prefill_tokens=1000.0, cached_context_tokens=0.0,
    )
    out = pred.predict(inp)
    expected = out.ttft_ms + (50.0 - 1.0) * out.tpot_ms
    assert out.e2e_ms == pytest.approx(expected)


def test_output_tokens_one_makes_e2e_equal_ttft() -> None:
    """O=1 → no decode tokens → e2e == TTFT."""
    pred = _predictor()
    out = pred.predict(TurnInput(
        profile="p", concurrency=2, turn_index=0, output_tokens=1.0,
        new_prefill_tokens=100.0, cached_context_tokens=0.0,
    ))
    assert out.e2e_ms == pytest.approx(out.ttft_ms)


def test_zero_concurrency_rejected() -> None:
    pred = _predictor()
    with pytest.raises(ValueError):
        pred.predict(TurnInput(
            profile="p", concurrency=0, turn_index=0, output_tokens=1.0,
            new_prefill_tokens=10.0, cached_context_tokens=0.0,
        ))


# ----------------------------------------------------- sidecar bucket parsing


def test_turn_input_from_sidecar_averages_distributions() -> None:
    bucket = {
        "profile": "swebench-multiturn-synth",
        "concurrency": 40,
        "turn_index": 12,
        "request_count": 4,
        "output_tokens": [20, 22, 24, 22],
        "new_prefill_tokens": [7000, 7100, 7200, 7300],
        "cached_context_tokens": [0, 0, 0, 0],
    }
    inp = turn_input_from_sidecar(bucket)
    assert inp.profile == "swebench-multiturn-synth"
    assert inp.concurrency == 40
    assert inp.turn_index == 12
    assert inp.output_tokens == pytest.approx(22.0)
    assert inp.new_prefill_tokens == pytest.approx(7150.0)
    assert inp.cached_context_tokens == 0.0


def test_iter_sidecar_inputs_skips_non_mapping_entries() -> None:
    payload = {
        "good": {
            "profile": "p", "concurrency": 1, "turn_index": 0,
            "output_tokens": [10], "new_prefill_tokens": [50],
            "cached_context_tokens": [0],
        },
        "bad": "not a dict",
    }
    inputs = list(iter_sidecar_inputs(payload))
    assert len(inputs) == 1
    assert inputs[0].profile == "p"


# ----------------------------------------------------- llm-d config fitting


def test_fit_llm_d_config_extracts_base_itl_and_load_factor() -> None:
    pred = _predictor()
    inputs = [
        TurnInput(profile="p", concurrency=c, turn_index=0,
                  output_tokens=20.0, new_prefill_tokens=1000.0,
                  cached_context_tokens=0.0)
        for c in (1, 40, 160)
    ]
    predictions = [pred.predict(i) for i in inputs]
    cfg = fit_llm_d_config(predictions)
    assert isinstance(cfg, LlmDInferenceSimConfig)
    assert cfg.max_num_seqs == 160
    assert cfg.inter_token_latency_ms == pytest.approx(predictions[0].tpot_ms)
    assert cfg.time_factor_under_load >= 1.0
    assert cfg.prefill_overhead_ms == pytest.approx(predictions[0].ttft_ms)


def test_fit_llm_d_config_to_llm_d_dict_uses_go_duration_strings() -> None:
    cfg = LlmDInferenceSimConfig(
        prefill_overhead_ms=12.5,
        prefill_time_per_token_ms=0.045,
        inter_token_latency_ms=8.2,
        time_factor_under_load=2.3,
        max_num_seqs=320,
    )
    d = cfg.to_llm_d_dict()
    assert d["prefill-overhead"] == "12.500ms"
    assert d["prefill-time-per-token"] == "0.0450ms"
    assert d["inter-token-latency"] == "8.200ms"
    assert d["time-factor-under-load"] == 2.3
    assert d["max-num-seqs"] == 320


def test_fit_llm_d_config_empty_raises() -> None:
    with pytest.raises(ValueError):
        fit_llm_d_config([])


# ----------------------------------------------------- params loading sanity


def test_h100_params_load_with_published_utilizations(tmp_path) -> None:
    """The roofline_params JSON at the published location loads cleanly."""
    if not H100_PARAMS_PATH.exists():
        pytest.skip("roofline_params_H100_llama31_8b.json not present")
    params = RooflineParams.from_json(H100_PARAMS_PATH)
    assert params.util_flops == pytest.approx(0.65)
    assert params.util_bw == pytest.approx(0.93)
    assert params.n_params == 8_030_000_000


def test_params_roundtrip_to_json(tmp_path) -> None:
    params = RooflineParams()
    path = tmp_path / "params.json"
    params.to_json(path)
    loaded = RooflineParams.from_json(path)
    assert loaded == params


def test_invalid_utilizations_rejected() -> None:
    with pytest.raises(ValueError):
        ClosedFormTpotPredictor(RooflineParams(util_flops=0.0))
    with pytest.raises(ValueError):
        ClosedFormTpotPredictor(RooflineParams(util_bw=-0.1))


# ----------------------------------------------------------- KV pressure


def test_kv_pressure_disabled_by_default_skips_wave_factor() -> None:
    """When the predictor is constructed without `kv_pressure_enabled=True`,
    the closed-form behaves as a pure roofline — no wave amplification even
    at high c × long ctx where pressure would otherwise fire.
    """
    pred = _predictor()  # default: kv_pressure_enabled=False
    out = pred.predict(TurnInput(
        profile="probe", concurrency=160, turn_index=27,
        output_tokens=90.0,
        new_prefill_tokens=12000.0,
        cached_context_tokens=0.0,
    ))
    assert out.wave_factor == pytest.approx(1.0)
    assert out.effective_decode_batch == 160


def test_kv_pressure_at_c80_long_ctx_amplifies_tpot() -> None:
    """c=80, ctx≈12500 → 80 × ceil(12500/16) = 80 × 782 = 62560 blocks needed
    > 27250 available.  B_eff ≈ 27250//782 = 34, wave_factor ≈ 80/34 ≈ 2.35.
    TPOT amplifies vs the unpressured baseline.
    """
    pred = _predictor(kv_pressure_enabled=True)
    inp = TurnInput(
        profile="probe", concurrency=80, turn_index=29,
        output_tokens=22.0,
        new_prefill_tokens=12500.0,
        cached_context_tokens=0.0,
    )
    out = pred.predict(inp)
    assert out.effective_decode_batch == 34, out.effective_decode_batch
    assert 2.1 < out.wave_factor < 2.5, out.wave_factor
    # Unamplified tpot at B=34 ctx≈12500 ≈ 22-25 ms;
    # amplified by ~2.35 → ~54 ms.
    assert 40.0 < out.tpot_ms < 70.0, out.tpot_ms


def test_kv_pressure_under_capacity_keeps_wave_factor_one() -> None:
    """c=4, ctx=1000 → 4 × ceil(1000/16) = 4 × 63 = 252 blocks ≪ 27651.
    B_eff = c = 4, wave_factor = 1.0, TPOT unchanged vs unpressured.
    """
    pred = _predictor(kv_pressure_enabled=True)
    inp = TurnInput(
        profile="probe", concurrency=4, turn_index=0,
        output_tokens=10.0,
        new_prefill_tokens=1000.0,
        cached_context_tokens=0.0,
    )
    out = pred.predict(inp)
    assert out.effective_decode_batch == 4
    assert out.wave_factor == pytest.approx(1.0)


def test_kv_pressure_uses_block_size_for_ceiling() -> None:
    """ctx=17 with block_size=16 → 2 blocks per request (not 1)."""
    # Tiny KV budget so any non-trivial ctx hits pressure deterministically.
    params = RooflineParams(available_kv_blocks=10, cache_block_size=16)
    pred = ClosedFormTpotPredictor(params, kv_pressure_enabled=True)
    out = pred.predict(TurnInput(
        profile="probe", concurrency=10, turn_index=0,
        output_tokens=1.0,            # ctx_mid = 17 + 0 = 17 tokens
        new_prefill_tokens=17.0,
        cached_context_tokens=0.0,
    ))
    # capacity_batch = 10 // ceil(17/16) = 10 // 2 = 5
    assert out.effective_decode_batch == 5
    assert out.wave_factor == pytest.approx(10 / 5)


def test_kv_capacity_loads_from_json(tmp_path) -> None:
    """A JSON file with `available_kv_blocks: 100` overrides the default."""
    params_path = tmp_path / "p.json"
    params_path.write_text(json.dumps({
        "n_params": 8_030_000_000,
        "peak_flops_per_s": 989e12,
        "peak_bw_bytes_per_s": 3.35e12,
        "kv_bytes_per_token": 131072.0,
        "util_flops": 0.65,
        "util_bw": 0.93,
        "bytes_per_param": 2.0,
        "available_kv_blocks": 100,
        "cache_block_size": 16,
    }))
    params = RooflineParams.from_json(params_path)
    assert params.available_kv_blocks == 100
    assert params.cache_block_size == 16
    # kv_capacity_bytes = 100 * 16 * 131072 = ~210 MB
    assert params.kv_capacity_bytes == 100 * 16 * 131072


def test_shared_prefix_fraction_dedupes_kv_reads_and_relieves_pressure() -> None:
    """At c=160 with cached=11000 and shared_prefix=1.0, the cached blocks
    are deduped to one set across the cohort → no wave amplification and
    bandwidth reflects 1× the prefix (not 160×).
    """
    pred = _predictor(kv_pressure_enabled=True)
    inp_shared = TurnInput(
        profile="osworld-like", concurrency=160, turn_index=27,
        output_tokens=90.0,
        new_prefill_tokens=92.0,
        cached_context_tokens=11263.0,
        shared_prefix_fraction=1.0,
    )
    inp_unique = TurnInput(
        profile="swe-like", concurrency=160, turn_index=27,
        output_tokens=90.0,
        new_prefill_tokens=92.0,
        cached_context_tokens=11263.0,
        shared_prefix_fraction=0.0,
    )
    shared = pred.predict(inp_shared)
    unique = pred.predict(inp_unique)
    # Shared: capacity easily met (one prefix copy) → no wave amplification.
    assert shared.wave_factor == pytest.approx(1.0)
    # Unique: 160 × per-session cached blocks easily exceeds capacity.
    assert unique.wave_factor > 2.0
    # Shared TPOT << unique TPOT, by a large margin.
    assert shared.tpot_ms < 0.5 * unique.tpot_ms


def test_partial_shared_prefix_scales_smoothly() -> None:
    """Half-shared prefix → fewer dedup savings than fully shared but more
    than fully per-session.
    """
    pred = _predictor(kv_pressure_enabled=True)
    inputs = [
        TurnInput(
            profile="probe", concurrency=80, turn_index=10,
            output_tokens=40.0, new_prefill_tokens=200.0,
            cached_context_tokens=5000.0, shared_prefix_fraction=f,
        )
        for f in (0.0, 0.5, 1.0)
    ]
    tpots = [pred.predict(i).tpot_ms for i in inputs]
    # Monotonically decreasing as more of the prefix is deduped.
    assert tpots[0] >= tpots[1] >= tpots[2]


def test_chat_workload_does_not_regress_under_kv_pressure() -> None:
    """Chat profile workloads (low c, short ctx) must have wave_factor=1."""
    pred = _predictor(kv_pressure_enabled=True)
    # Representative chat numbers from sidecar: c<=160, ctx ~1000-2000.
    for c in (1, 10, 40, 80, 120, 160):
        out = pred.predict(TurnInput(
            profile="chat-multiturn-synth", concurrency=c, turn_index=2,
            output_tokens=200.0,
            new_prefill_tokens=800.0,
            cached_context_tokens=400.0,
        ))
        assert out.wave_factor == pytest.approx(1.0), \
            f"c={c} unexpectedly amplified by {out.wave_factor}"
