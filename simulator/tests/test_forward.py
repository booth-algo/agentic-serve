"""Tests for the forward predictor (simulator/forward.py).

Run: PYTHONPATH=/root/agentic-serve python -m pytest simulator/tests/test_forward.py
(Do NOT set RAMP_TPOT_REQUIRE_POOLS for pytest — pooled fallback is expected here.)
"""

from dataclasses import asdict

import pytest

import profiling.process.build_simulator_rows as B
from configs.loader import all_deployments, compose_roofline, load_deployment, _PIN_KEYS, CONFIGS_DIR
from simulator.cohort_scale import cohort_scale_mean
from simulator.ttft_queue_sim import _prefill_floor_for
from simulator import forward


def test_compose_roofline_reproduces_load_deployment():
    """The extracted compose_roofline is byte-identical to load_deployment's inline merge for every
    existing deployment (the loader refactor introduced no behavior change)."""
    import json
    deps = list((CONFIGS_DIR / "deployments").glob("*.json"))
    assert deps, "no deployment configs found"
    for p in deps:
        d = json.loads(p.read_text())
        pins = {}
        if "max_num_batched_tokens" in d:
            pins["max_num_batched_tokens"] = int(d["max_num_batched_tokens"])
        for k in _PIN_KEYS:
            if d.get(k) is not None:
                pins[k] = float(d[k])
        composed = compose_roofline(d["gpu"], d["model"], int(d["tp"]), int(d["available_kv_blocks"]), pins)
        assert asdict(composed) == asdict(load_deployment(p).roofline), p.name


def _first_calibrated_bench_cell():
    """A calibrated deployment + a (profile, conc) that has a measured bench file, or None."""
    for cfg in all_deployments():
        if cfg.decode_grid is None and cfg.saturated_ceiling is None:
            continue
        root = B.BENCH_BASE / cfg.bench_dir
        if not root.exists():
            continue
        for profile in B.PROFILES:
            for conc in B.CONCURRENCIES:
                if (root / f"{profile}_conc{conc}.json").exists():
                    return cfg, profile, conc, root
    return None


def test_forward_self_consistency_vs_build_row():
    """Fed build_row's EXACT inputs (turns, qbar, sched, cohort=None marginal path, floor, profile),
    forward.predict_turns reproduces build_row's per-turn TTFT/TPOT — proving the driver wires the
    predictors faithfully AND that the default-off `cohort=None` seam is byte-identical."""
    picked = _first_calibrated_bench_cell()
    if picked is None:
        pytest.skip("no calibrated cell with a measured bench file on this host")
    cfg, profile, conc, root = picked

    with forward.active_hardware(cfg.decode_grid, cfg.saturated_ceiling):
        row = B.build_row(profile, conc, cfg.roofline, cfg, root)
    assert row is not None
    ref = row["multiturn_turn_predictions"]
    ref_tpot = [t["tpot_pred"] for t in ref]
    ref_ttft = [t["ttft_pred"] for t in ref]

    turns, sp = B.build_turns(root / f"{profile}_conc{conc}.json")
    key = cfg.gpu_key if cfg.model == "Llama-3.1-8B" else None
    tpot, ttft, _ = forward.predict_turns(
        turns, cfg.roofline, conc,
        decode_grid=cfg.decode_grid, saturated_ceiling=cfg.saturated_ceiling,
        qbar=cohort_scale_mean(profile, float(conc), key),
        shared_prefix_tokens=sp,
        sched=forward._sched_for(cfg.max_model_len, cfg.max_num_seqs, cfg.roofline),
        prefill_floor_ms=_prefill_floor_for(key),
        cohort=None, gpu_key=key, profile=profile,
    )
    assert max(abs(a - b) for a, b in zip(tpot, ref_tpot)) < 1e-3
    assert max(abs(a - b) for a, b in zip(ttft, ref_ttft)) < 1e-3


def test_predict_forward_smoke_calibrated():
    """predict_forward on a calibrated cell returns finite, positive, ordered metrics + a coarse
    calibration_status. Uses a degenerate single-point ISL:OSL distribution."""
    picked = _first_calibrated_bench_cell()
    if picked is None:
        pytest.skip("no calibrated cell with a measured bench file on this host")
    cfg, profile, conc, root = picked
    res = forward.predict_forward(
        gpu=cfg.gpu, model=cfg.model, tp=cfg.tp, engine=cfg.engine,
        concurrency=max(1, conc), isl_osl_samples=[(2000, 200)] * 16,
    )
    assert res.ttft_ms > 0 and res.tpot_ms > 0
    assert res.e2el_ms >= res.ttft_ms  # e2el = ttft + osl*tpot
    assert res.calibration_status in ("measured", "partial", "extrapolated")


def test_predict_forward_rejects_empty():
    with pytest.raises(ValueError):
        forward.predict_forward(gpu="A100", model="Llama-3.1-8B", tp=1, concurrency=8, isl_osl_samples=[])
    with pytest.raises(ValueError):  # no workload form at all
        forward.predict_forward(gpu="A100", model="Llama-3.1-8B", tp=1, concurrency=8)


def test_predict_forward_single_turn_percentiles():
    """A varied single-turn distribution yields per-request percentiles, ordered p50<=p90<=p99."""
    samples = [(1000, 100), (2000, 150), (3000, 200), (1500, 300), (2500, 80)] * 6
    res = forward.predict_forward(gpu="A100", model="Llama-3.1-8B", tp=1, concurrency=30,
                                  isl_osl_samples=samples)
    assert res.ttft_pcts and res.e2el_pcts
    assert 0 < res.ttft_pcts["p50"] <= res.ttft_pcts["p90"] <= res.ttft_pcts["p99"]
    assert 0 < res.e2el_pcts["p50"] <= res.e2el_pcts["p90"] <= res.e2el_pcts["p99"]


def test_predict_forward_multi_turn():
    """Multi-turn trajectories produce one headline per turn index + finite metrics."""
    trajs = [
        [(0, 1000, 100), (1100, 200, 120)],
        [(0, 1500, 80)],
        [(0, 2000, 150), (2150, 300, 90), (2540, 100, 60)],
    ]
    res = forward.predict_forward(gpu="A100", model="Llama-3.1-8B", tp=1, concurrency=9,
                                  trajectories=trajs)
    assert len(res.per_turn["ttft"]) == 3  # max turn count across sessions
    assert res.ttft_ms > 0 and res.e2el_ms >= res.ttft_ms


def test_predict_forward_quantiles():
    """A quantile-summary workload runs and yields finite metrics + percentiles."""
    q = {"isl": {0.5: 1800, 0.9: 4000, 0.99: 8000}, "osl": {0.5: 150, 0.9: 400, 0.99: 900}}
    res = forward.predict_forward(gpu="A100", model="Llama-3.1-8B", tp=1, concurrency=16, quantiles=q)
    assert res.ttft_ms > 0 and res.tpot_ms > 0 and res.ttft_pcts
