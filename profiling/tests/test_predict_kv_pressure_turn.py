"""Tests for predict_kv_pressure_turn.py — the diagnostic that asks whether
the physics predicts the same turn where TPOT actually jumps.
"""

from __future__ import annotations

import json
from pathlib import Path

from profiling.process.predictors.predict_kv_pressure_turn import (
    JumpDetection,
    TurnWorkload,
    detect_jump_turn,
    load_measured_tpot_series,
    load_per_request_workload,
    predict_jump_turn,
)
from simulator.closed_form_tpot import RooflineParams


# ------------------------------------------------------------ prediction


def test_predict_jump_at_linear_cached_growth() -> None:
    """cached_context grows linearly with turn → demand crosses the budget at a
    predictable turn.
    """
    # Tiny KV budget so the math is hand-checkable.
    params = RooflineParams(available_kv_blocks=100, cache_block_size=16)
    # blocks_per_req(N) = ceil((cached(N) + 0 + 0.5*4) / 16) = ceil((32N + 2)/16)
    # at c=4: total_demand(N) = 4 × blocks_per_req(N)
    # For N=0: ceil(2/16)=1, demand=4. N=1: ceil(34/16)=3, demand=12.
    # N=10: ceil(322/16)=21, demand=84. N=12: ceil(386/16)=25, demand=100 (=cap, no jump).
    # N=13: ceil(418/16)=27, demand=108 (>cap, JUMP).
    workload = [
        TurnWorkload(
            turn_index=n,
            cached_context_tokens=float(32 * n),
            new_prefill_tokens=0.0,
            output_tokens=4.0,
            request_count=4,
        )
        for n in range(20)
    ]
    result = predict_jump_turn(workload, concurrency=4, params=params)
    assert result.predicted_jump_turn == 13, result
    assert result.demand_at_jump is not None and result.demand_at_jump > 100
    assert result.available_blocks == 100


def test_predict_no_jump_when_demand_under_capacity() -> None:
    """Small per-session contexts never exceed the budget → no jump."""
    params = RooflineParams(available_kv_blocks=1000, cache_block_size=16)
    workload = [
        TurnWorkload(
            turn_index=n,
            cached_context_tokens=float(10 * n),
            new_prefill_tokens=10.0,
            output_tokens=20.0,
            request_count=2,
        )
        for n in range(20)
    ]
    result = predict_jump_turn(workload, concurrency=2, params=params)
    assert result.predicted_jump_turn is None
    assert result.demand_at_jump is None


# ------------------------------------------------------------- detection


def test_detect_jump_flat_then_spike() -> None:
    """[10, 10, 11, 35, 36, 37] → baseline ≈ 10, jump at turn 3 (3.5×), persists."""
    series = [(0, 10.0), (1, 10.0), (2, 11.0), (3, 35.0), (4, 36.0), (5, 37.0)]
    result = detect_jump_turn(series)
    assert result.detected_jump_turn == 3, result
    assert result.baseline_ms is not None and 9.5 < result.baseline_ms < 11.5
    assert result.tpot_at_jump_ms == 35.0
    assert result.tentative is False


def test_detect_no_jump_monotonic_rise() -> None:
    """Slow rise that never hits 3× baseline."""
    series = [(i, 10.0 + 0.8 * i) for i in range(10)]  # max 17.2, never 30
    result = detect_jump_turn(series)
    assert result.detected_jump_turn is None, result


def test_detect_single_turn_spike_is_not_a_jump() -> None:
    """[10, 10, 10, 40, 10, 10] — spike at turn 3 fails persistence."""
    series = [(0, 10.0), (1, 10.0), (2, 10.0), (3, 40.0), (4, 10.0), (5, 10.0)]
    result = detect_jump_turn(series)
    assert result.detected_jump_turn is None, result


def test_detect_tentative_jump_when_last_turn_crosses() -> None:
    """[10, 10, 10, 11, 35] — turn 4 fires (3.5× baseline) but no neighbor → tentative."""
    series = [(0, 10.0), (1, 10.0), (2, 10.0), (3, 11.0), (4, 35.0)]
    result = detect_jump_turn(series)
    assert result.detected_jump_turn == 4
    assert result.tentative is True


def test_detect_handles_none_values() -> None:
    """None entries are skipped, then the rule fires on remaining."""
    series = [(0, 10.0), (1, None), (2, 10.0), (3, 12.0), (4, 35.0), (5, 36.0)]
    result = detect_jump_turn(series)
    # baseline_n=3 non-None values = [10, 10, 12] → median 10; jump_floor=30.
    # turn 4 = 35 >= 30 ✓; turn 5 = 36 >= 20 (2.0*10) ✓ → detected.
    assert result.detected_jump_turn == 4


# --------------------------------------------------------- loaders / I/O


def test_load_per_request_workload_groups_and_sorts(tmp_path: Path) -> None:
    payload = {
        "per_turn": {
            "p__1__1": {
                "profile": "p", "concurrency": 1, "turn_index": 1,
                "output_tokens": [10, 12],
                "new_prefill_tokens": [100, 200],
                "cached_context_tokens": [0, 50],
                "request_count": 2,
            },
            "p__1__0": {
                "profile": "p", "concurrency": 1, "turn_index": 0,
                "output_tokens": [10],
                "new_prefill_tokens": [100],
                "cached_context_tokens": [0],
                "request_count": 1,
            },
        }
    }
    path = tmp_path / "per_request.json"
    path.write_text(json.dumps(payload))
    result = load_per_request_workload(path)
    assert ("p", 1) in result
    turns = result[("p", 1)]
    assert [t.turn_index for t in turns] == [0, 1]
    assert turns[1].output_tokens == 11.0  # mean of [10, 12]
    assert turns[1].cached_context_tokens == 25.0


def test_load_measured_tpot_series_unwraps_hardware_key(tmp_path: Path) -> None:
    payload = {
        "H100": [
            {
                "profile": "p", "concurrency": 4,
                "multiturn_turn_predictions": [
                    {"turn_index": 1, "tpot_meas": 12.5},
                    {"turn_index": 0, "tpot_meas": 10.0},
                    {"turn_index": 2, "tpot_meas": None},
                ],
            }
        ]
    }
    path = tmp_path / "sim.json"
    path.write_text(json.dumps(payload))
    out = load_measured_tpot_series(path)
    assert ("p", 4) in out
    series = out[("p", 4)]
    assert [(0, 10.0), (1, 12.5), (2, None)] == series


# ------------------------------------------------------- real-data anchor


def test_real_data_swebench_c80_has_predicted_and_detected_jump_within_3() -> None:
    """Anchor against the real artifacts in profiling/results + dashboard public.

    Skips if either file is missing — keeps the test useful when devs run a
    fresh checkout but doesn't break CI if the artifacts haven't been refreshed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    per_req = repo_root / "profiling/results/benchmark_per_request_llama31_8b_h100_vllm.json"
    sim_preds = repo_root / "inference-benchmark/dashboard/public/simulator-predictions.json"
    if not per_req.exists() or not sim_preds.exists():
        return  # skip gracefully

    workloads = load_per_request_workload(per_req)
    series_by_key = load_measured_tpot_series(sim_preds)
    key = ("swebench-multiturn-synth", 80)
    if key not in workloads or key not in series_by_key:
        return
    params = RooflineParams()
    pred = predict_jump_turn(workloads[key], 80, params)
    detect = detect_jump_turn(series_by_key[key])
    assert pred.predicted_jump_turn is not None, "expected physics jump on swe c=80"
    assert detect.detected_jump_turn is not None, "expected measured jump on swe c=80"
    assert abs(pred.predicted_jump_turn - detect.detected_jump_turn) <= 3, (
        f"predicted={pred.predicted_jump_turn} detected={detect.detected_jump_turn}"
    )
