"""Tests for the llm-d-inference-sim config emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiling.process.emitters.emit_llm_d_inference_sim_config import (
    _collect_profile_summaries,
    emit,
    fit_profile_config,
)
from simulator.closed_form_tpot import (
    ClosedFormTpotPredictor,
    RooflineParams,
)


def _predictor() -> ClosedFormTpotPredictor:
    return ClosedFormTpotPredictor(RooflineParams())


def test_collect_profile_summaries_groups_concurrencies_and_means() -> None:
    per_turn = {
        "p__1__0": {
            "profile": "p", "concurrency": 1, "turn_index": 0,
            "output_tokens": [10, 12],
            "new_prefill_tokens": [100, 200],
            "cached_context_tokens": [0, 0],
        },
        "p__40__0": {
            "profile": "p", "concurrency": 40, "turn_index": 0,
            "output_tokens": [20, 22],
            "new_prefill_tokens": [50, 60],
            "cached_context_tokens": [10, 20],
        },
        "other__8__0": {
            "profile": "other", "concurrency": 8, "turn_index": 0,
            "output_tokens": [5],
            "new_prefill_tokens": [30],
            "cached_context_tokens": [0],
        },
    }
    out = _collect_profile_summaries(per_turn)
    assert sorted(out.keys()) == ["other", "p"]
    p = out["p"]
    assert p["concurrencies"] == [1, 40]
    assert p["mean_output"] == pytest.approx((10 + 12 + 20 + 22) / 4)
    assert p["mean_new_prefill"] == pytest.approx((100 + 200 + 50 + 60) / 4)
    assert p["max_new_prefill"] == 200
    assert p["mean_cached"] == pytest.approx((0 + 0 + 10 + 20) / 4)


def test_fit_profile_config_produces_valid_llm_d_fields() -> None:
    pred = _predictor()
    summary = {
        "concurrencies": [1, 8, 40],
        "mean_output": 20.0,
        "mean_new_prefill": 1000.0,
        "mean_cached": 500.0,
        "max_new_prefill": 5000.0,
    }
    cfg = fit_profile_config(pred, summary, profile_name="probe")
    # Required llm-d-inference-sim flags
    assert cfg["prefill-overhead"].endswith("ms")
    assert cfg["prefill-time-per-token"].endswith("ms")
    assert cfg["inter-token-latency"].endswith("ms")
    assert isinstance(cfg["time-factor-under-load"], (int, float))
    assert cfg["time-factor-under-load"] >= 1.0
    assert cfg["max-num-seqs"] == 40
    # prefill_time_per_token at H100 8B BF16 should be ~0.025 ms/token
    slope_ms = float(cfg["prefill-time-per-token"].rstrip("ms"))
    assert 0.01 < slope_ms < 0.05, slope_ms


def test_fit_profile_config_single_concurrency_load_factor_is_one() -> None:
    pred = _predictor()
    summary = {
        "concurrencies": [1],
        "mean_output": 20.0,
        "mean_new_prefill": 1000.0,
        "mean_cached": 0.0,
        "max_new_prefill": 1000.0,
    }
    cfg = fit_profile_config(pred, summary, profile_name="probe")
    assert cfg["time-factor-under-load"] == pytest.approx(1.0)
    assert cfg["max-num-seqs"] == 1


def test_emit_end_to_end_writes_expected_structure(tmp_path) -> None:
    sidecar = tmp_path / "per_request.json"
    sidecar.write_text(json.dumps({
        "per_turn": {
            "p__1__0": {
                "profile": "p", "concurrency": 1, "turn_index": 0,
                "output_tokens": [10], "new_prefill_tokens": [100],
                "cached_context_tokens": [0],
            },
            "p__40__0": {
                "profile": "p", "concurrency": 40, "turn_index": 0,
                "output_tokens": [20], "new_prefill_tokens": [200],
                "cached_context_tokens": [50],
            },
        },
    }))
    params_path = tmp_path / "params.json"
    RooflineParams().to_json(params_path)
    out_path = tmp_path / "configs.json"
    emit(
        per_request_json=sidecar,
        roofline_params_json=params_path,
        output_path=out_path,
        hardware_key="TestHW_TestModel",
    )
    payload = json.loads(out_path.read_text())
    assert "TestHW_TestModel" in payload
    assert "p" in payload["TestHW_TestModel"]
    cfg = payload["TestHW_TestModel"]["p"]
    # Surface-level keys present
    for k in ("prefill-overhead", "prefill-time-per-token",
              "inter-token-latency", "time-factor-under-load", "max-num-seqs"):
        assert k in cfg


def test_emit_missing_profiles_raises(tmp_path) -> None:
    sidecar = tmp_path / "per_request.json"
    sidecar.write_text(json.dumps({"per_turn": {}}))
    params_path = tmp_path / "params.json"
    RooflineParams().to_json(params_path)
    with pytest.raises(SystemExit):
        emit(
            per_request_json=sidecar,
            roofline_params_json=params_path,
            output_path=tmp_path / "out.json",
            hardware_key="X",
        )
