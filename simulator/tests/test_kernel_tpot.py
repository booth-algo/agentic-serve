"""Tests for the kernel-composition per-turn TPOT predictor.

State under test (2026-06-10 ramp restructure): the saturation weight is the COMPUTED
eviction-recompute duty cycle ``_overflow_weight`` — zero tuned constants. The former
tuned ramp band (P_LO=0.88 / P_HI_SHORT=1.22 / P_HI_LONG=2.0 + the OUT_KNEE
interpolation) is GONE from the module; its history stays pinned via the ramp_knees
artifacts (still-valid measured record) below.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

from simulator.closed_form_tpot import RooflineParams
from simulator.cohort_scale import cohort_scale_mean, trapezoid_mean
from simulator.kernel_tpot import (
    OUT_KNEE_HI,
    OUT_KNEE_LO,
    KernelTurnInput,
    _overflow_weight,
    _smoothstep,
    predict_cell_tpot,
    predict_turn_tpot,
    saturated_ceiling_ms,
)

KERNELS_DIR = Path(__file__).resolve().parents[2] / "profile_data/kernels"


def _turn(cached: float, new: float, out: float, sched: float,
          qbar: float = 1.0) -> KernelTurnInput:
    return KernelTurnInput(
        cached_context_tokens=cached,
        new_prefill_tokens=new,
        output_tokens=out,
        scheduled_requests=sched,
        cohort_scale_mean=qbar,
    )


# ------------------------------------------------------------ smoothstep
def test_smoothstep_endpoints_and_midpoint() -> None:
    assert _smoothstep(0.0, 9.0, 24.0) == 0.0
    assert _smoothstep(100.0, 9.0, 24.0) == 1.0
    mid = _smoothstep((9.0 + 24.0) / 2, 9.0, 24.0)
    assert abs(mid - 0.5) < 1e-9


# ------------------------------------------------------------ tuned knees: GONE
def test_tuned_ramp_knee_attributes_are_deleted() -> None:
    """The 2026-06-10 restructure ELIMINATED the tuned ramp band. Re-introducing any
    of these names as module constants must be a conscious decision, not drift."""
    import simulator.kernel_tpot as K

    for name in ("P_LO", "P_HI_SHORT", "P_HI_LONG"):
        assert not hasattr(K, name), f"tuned knob {name} reappeared in kernel_tpot"
    # OUT_KNEE_LO/HI survive ONLY as the measured ceiling-cluster output labels
    # (28/86 tok = the saturated-ceiling artifact anchor outputs), not as ramp knees.
    art = json.loads((KERNELS_DIR / "saturated_ceiling_H100_llama31_8b.json").read_text())
    anchor_outs = sorted(float(a["output_tokens"]) for a in art["anchors"])
    assert OUT_KNEE_LO == anchor_outs[0] == 28.0
    assert OUT_KNEE_HI == anchor_outs[-1] == 86.0


# ------------------------------------------------------------ overflow weight (computed)
# _overflow_weight(pressure, qbar, out_steps, ctx_tokens, sched, b_eff, p, developed)


def test_overflow_weight_onset_requires_pool_physically_full() -> None:
    """Round-2 onset gate (P0b): w = 0 unless BOTH the pool is physically full
    (pressure >= 1: vLLM v1 preempts only on allocation failure) AND the
    distribution-integrated demand overflows (z = pressure·qbar > 1)."""
    p = RooflineParams()
    # pool not full -> no eviction, regardless of the spread tail
    assert _overflow_weight(0.5, 1.0, 28, 5000, 40, 40, p) == 0.0
    assert _overflow_weight(0.96, 1.3463, 36, 7600, 17, 17, p) == 0.0  # A100 term@20 case
    # pool full but spread-integrated demand below pool (osworld qbar < 1)
    assert _overflow_weight(1.01, 0.9834, 88, 10000, 50, 40, p) == 0.0  # z = 0.993
    # both conditions met -> fires
    assert _overflow_weight(1.05, 1.0, 28, 5000, 160, 53, p) > 0.0


def test_overflow_weight_is_chunk_quantized_drain_fraction() -> None:
    """w = n_evicted · ceil(ctx·qbar/budget) / out for single-chunk victims:
    the evicted cohort's once-per-turn re-prefill occupies that fraction of the
    turn's decode steps. Exact arithmetic identity, no constants."""
    p = RooflineParams()
    pressure, qbar, out, ctx, sched, b_eff = 1.4, 1.0, 80.0, 5000.0, 100.0, 53.0
    z = pressure * qbar
    budget = p.max_num_batched_tokens - b_eff
    chunks = math.ceil(ctx * qbar / budget)
    assert chunks == 1
    expected = (1.0 - 1.0 / z) * sched * chunks / out
    assert abs(_overflow_weight(pressure, qbar, out, ctx, sched, b_eff, p) - expected) < 1e-12
    # clamped at 1 (every decode step's budget recompute-filled = the regime
    # where the measured saturated ceiling was anchored)
    assert _overflow_weight(6.0, 1.0, 22.0, 5000.0, 300.0, 53.0, p) == 1.0


def test_overflow_weight_rotation_boost_multichunk_only() -> None:
    """Multi-chunk victims (ctx·qbar > budget) gain the ×z rotation amplification —
    the standing overflow rotates the victim queue ~z×/turn; single-chunk victims
    complete within one step and the chain de-synchronizes."""
    p = RooflineParams()
    pressure, out, sched, b_eff = 1.5, 88.0, 55.0, 38.0
    budget = p.max_num_batched_tokens - b_eff
    ctx_multi = budget * 1.4   # 2 chunks
    z = pressure * 1.0
    nev = (1.0 - 1.0 / z) * sched
    w_multi = _overflow_weight(pressure, 1.0, out, ctx_multi, sched, b_eff, p)
    assert abs(w_multi - min(1.0, nev * 2 / out * z)) < 1e-12  # ×z applied
    ctx_single = budget * 0.5  # 1 chunk
    w_single = _overflow_weight(pressure, 1.0, out, ctx_single, sched, b_eff, p)
    assert abs(w_single - min(1.0, nev * 1 / out)) < 1e-12     # no ×z


def test_overflow_weight_fresh_crossing_growth_damping() -> None:
    """developed=False (a cell's first overflow turn): single-chunk victims' boundary
    wave lands in the admission burst (TTFT side) — ITL-visible evictions are capped
    by decode growth: w <= sched·chunks/ctx. Multi-chunk waves spill into decode and
    are NOT damped (small budgets cannot swallow them)."""
    p = RooflineParams()
    pressure, out, sched, b_eff = 1.43, 20.0, 314.0, 296.0
    ctx = 1471.0  # H100 term@320 turn-2 shape: chunks = 1
    w_dev = _overflow_weight(pressure, 1.0, out, ctx, sched, b_eff, p, developed=True)
    w_fresh = _overflow_weight(pressure, 1.0, out, ctx, sched, b_eff, p, developed=False)
    assert w_fresh < w_dev
    assert abs(w_fresh - sched * 1 / ctx) < 1e-12  # growth-forced evictions only
    # multi-chunk: no damping
    ctx_big = 12000.0
    assert _overflow_weight(1.43, 1.0, 88, ctx_big, 55, 38, p, developed=False) == \
        _overflow_weight(1.43, 1.0, 88, ctx_big, 55, 38, p, developed=True)


def test_overflow_band_widens_with_output_emergent() -> None:
    """The long-output widening the retired OUT_KNEE interpolation hand-binned
    EMERGES from the turn's own decode-step count in the drain-fraction
    denominator: more steps amortize the same eviction drain -> smaller w."""
    p = RooflineParams()
    w_short = _overflow_weight(1.3, 1.0, 28.0, 5000.0, 160.0, 53.0, p)
    w_long = _overflow_weight(1.3, 1.0, 88.0, 5000.0, 160.0, 53.0, p)
    assert 0.0 < w_long < w_short


def test_overflow_weight_scales_with_engine_budget() -> None:
    """A smaller per-step token budget (A100's resolved max_num_batched_tokens=2048 vs
    H100's 8192 — vLLM device rule) means more re-prefill chunk-steps per victim ->
    saturates FASTER at the same overflow."""
    h100 = RooflineParams()
    a100ish = RooflineParams(max_num_batched_tokens=2048)
    args = (1.2, 1.0, 28.0, 3000.0, 100.0, 40.0)
    assert _overflow_weight(*args, a100ish) > _overflow_weight(*args, h100)


def test_qbar_sizes_mass_not_onset() -> None:
    """The measured cohort spread (qbar) sizes the overflow mass once the pool is
    full — it does NOT move the onset below pool-full (the round-1 spread-onset
    over-fired pools that were not physically full)."""
    p = RooflineParams()
    # below pool-full: no fire even with the largest measured spread (term 1.3463)
    assert _overflow_weight(0.9, 1.3463, 28, 7600, 17, 17, p) == 0.0
    # at pool-full: a bigger spread -> bigger z -> more evicted -> bigger w
    w_lo = _overflow_weight(1.1, 1.0003, 28, 5000, 100, 53, p)
    w_hi = _overflow_weight(1.1, 1.3463, 28, 5000, 100, 53, p)
    assert 0.0 < w_lo < w_hi


def test_development_state_sequencing_in_cell_path() -> None:
    """predict_cell_tpot damps a cell's FIRST overflow turn (developed=False) and
    releases the damping once a prior turn overflowed with sustainable output."""
    p = RooflineParams()
    # construct a single-chunk overflow turn shape (H100 term@320-t2-like)
    hot = _turn(cached=1300, new=160, out=20, sched=314, qbar=1.3463)
    cold = _turn(cached=200, new=100, out=20, sched=10, qbar=1.3463)
    # standalone (developed defaults True) > first-overflow turn in a cell
    standalone = predict_turn_tpot(hot, p)
    in_cell = predict_cell_tpot([cold, hot, hot], p)
    assert in_cell[1] < standalone           # turn 1 = first overflow -> damped
    assert abs(in_cell[2] - standalone) < 1e-9  # turn 2 = developed -> steady law


# ------------------------------------------------------------ firing-gate hysteresis (round 3)
def _swe40_like(pressure_frac: float, p: RooflineParams,
                qbar: float = 1.1269) -> KernelTurnInput:
    """A swe-like turn whose median-session pressure is ``pressure_frac`` of pool-full."""
    cached, new, out = 6000.0, 100.0, 28.0
    psb = math.ceil((cached + new + 0.5 * out) / p.cache_block_size)
    sched = pressure_frac * p.available_kv_blocks / psb
    return _turn(cached, new, out, sched, qbar=qbar)


def test_firing_gate_hysteresis_holds_through_pressure_flicker() -> None:
    """Round-3 fix (H100 swe@40 t20–29): once a turn physically fills the pool
    (pressure >= 1, z > 1), dip turns whose block-quantized pressure flickers a few
    percent below 1 while z stays > 1 KEEP firing at the pool-full effective pressure
    — the eviction backlog does not vanish with a 3% median-pressure dip. The raw
    per-turn gate (standalone call) still refuses the dip turn."""
    p = RooflineParams()
    full = _swe40_like(1.05, p)   # arms: pressure 1.05, z = 1.18
    dip = _swe40_like(0.97, p)    # raw gate refuses: pressure 0.97, but z = 1.09 > 1
    floor = predict_turn_tpot(dip, p)                 # un-armed: kernel floor
    armed = predict_turn_tpot(dip, p, armed=True)     # armed: fires at max(p,1)=1.0
    assert armed > floor + 5.0
    # cell path: the dip turn AFTER an arming turn is the armed prediction
    in_cell = predict_cell_tpot([full, full, dip], p)
    med_out = 28.0  # all turns share output -> ceiling_output == own output
    assert abs(in_cell[2] - predict_turn_tpot(dip, p, ceiling_output=med_out,
                                              developed=True, armed=True)) < 1e-9
    assert in_cell[2] > floor + 5.0  # no 27<->132 ms oscillation at the dip


def test_hysteresis_disarms_when_demand_underflows() -> None:
    """Disarm condition: z <= 1 (demand back under the pool drains the backlog).
    A later sub-pool-full turn does NOT fire just because the cell once armed."""
    p = RooflineParams()
    full = _swe40_like(1.05, p)
    cold = _swe40_like(0.30, p)   # z = 0.34 <= 1 -> disarm
    dip = _swe40_like(0.97, p)    # raw gate refuses; must stay refused (disarmed)
    in_cell = predict_cell_tpot([full, cold, dip], p)
    assert abs(in_cell[2] - predict_turn_tpot(dip, p, ceiling_output=28.0,
                                              developed=False)) < 1e-9
    assert in_cell[2] < 40.0      # kernel-floor territory, not the armed branch


def test_never_full_cell_never_arms_a100_term20_protection() -> None:
    """The gate's protected twin (A100 term@20: pressure peaks 0.965, z up to 1.30,
    measured CLEAN): a cell whose pressure NEVER reaches 1 never arms — hysteresis
    must not regress the round-2 protection."""
    p = RooflineParams()
    turns = [_swe40_like(f, p, qbar=1.3463) for f in (0.87, 0.93, 0.965, 0.94)]
    in_cell = predict_cell_tpot(turns, p)
    for t, pred in zip(turns, in_cell):
        assert abs(pred - predict_turn_tpot(t, p, ceiling_output=28.0,
                                            developed=False)) < 1e-9
        assert pred < 40.0  # all stay on the kernel floor: never armed, never fired


# ------------------------------------------------------------ qbar (measured artifact)
def test_cohort_scale_mean_pins_measured_quantile_artifacts() -> None:
    """qbar = trapezoid mean of the MEASURED context_scale_quantiles (pooled
    *_realized.json artifacts) — the per-profile saturation onsets in median-pressure
    units are 1/qbar, fully computed. Changing the artifacts must be reviewed here."""
    expected = {  # measured 2026-06-10 (pooled artifacts, n=1212 sessions total)
        "swebench-multiturn-synth": 1.1269,
        "terminalbench-multiturn-synth": 1.3463,
        "osworld-multiturn-synth": 0.9834,
        "chat-multiturn-synth": 1.0003,
    }
    for prof, qbar in expected.items():
        assert abs(cohort_scale_mean(prof) - qbar) < 5e-4, prof
    # computed onsets (1/qbar): swe 0.887, term 0.743, osworld 1.017, chat 1.000
    assert abs(1.0 / cohort_scale_mean("swebench-multiturn-synth") - 0.887) < 1e-3
    assert abs(1.0 / cohort_scale_mean("terminalbench-multiturn-synth") - 0.743) < 1e-3
    assert abs(1.0 / cohort_scale_mean("osworld-multiturn-synth") - 1.017) < 1e-3
    assert abs(1.0 / cohort_scale_mean("chat-multiturn-synth") - 1.000) < 1e-3
    # profile without a measured spec -> 1.0 (median-session pressure, onset pool-full)
    assert cohort_scale_mean("no-such-profile") == 1.0


def test_trapezoid_mean_math() -> None:
    assert trapezoid_mean([]) == 1.0
    assert trapezoid_mean([2.5]) == 2.5
    assert trapezoid_mean([1.0, 1.0, 1.0]) == 1.0
    # uniform[0,1] via its quantiles -> mean 0.5
    q = [i / 100 for i in range(101)]
    assert abs(trapezoid_mean(q) - 0.5) < 1e-12


def test_qbar_default_keeps_constructors_working_and_onset_at_pool_full() -> None:
    """KernelTurnInput without cohort_scale_mean (all existing constructors) defaults
    to 1.0 -> median-session pressure with onset exactly at pool overflow."""
    t = KernelTurnInput(cached_context_tokens=1000, new_prefill_tokens=100,
                        output_tokens=60, scheduled_requests=40)
    assert t.cohort_scale_mean == 1.0
    assert predict_turn_tpot(t) > 0


def test_qbar_scales_saturation_at_pool_full_not_below() -> None:
    """Round 2: the measured spread sizes the MASS once the pool is physically
    full; below pool-full it does not fire (the round-1 spread-onset over-fired
    not-yet-full pools — evaluator mode 2)."""
    p = RooflineParams()
    cached, new, out = 6000.0, 100.0, 28.0
    psb = math.ceil((cached + new + 0.5 * out) / p.cache_block_size)
    # below pool-full: spread does NOT pull the onset earlier
    sched_low = 0.93 * p.available_kv_blocks / psb
    base = predict_turn_tpot(_turn(cached, new, out, sched_low, qbar=1.0), p)
    spread = predict_turn_tpot(_turn(cached, new, out, sched_low, qbar=1.3463), p)
    assert abs(spread - base) < 1e-9  # both gated: pool not full
    # at pool-full: the spread overflow fires while the median-only view does not
    sched_full = 1.02 * p.available_kv_blocks / psb
    base_full = predict_turn_tpot(_turn(cached, new, out, sched_full, qbar=1.0), p)
    spread_full = predict_turn_tpot(_turn(cached, new, out, sched_full, qbar=1.3463), p)
    assert spread_full > base_full  # z = 1.37 fires harder than z = 1.02


# ------------------------------------------------------------ saturated ceiling
def test_saturated_ceiling_inverse_in_output() -> None:
    """Short-output turns saturate higher than long-output turns."""
    short = saturated_ceiling_ms(27)   # swe/terminal-like
    long = saturated_ceiling_ms(87)    # osworld-like
    assert short > long
    # matches the measured anchors within a few ms
    assert 220 < short < 260
    assert 120 < long < 160


def test_saturated_ceiling_clamps_to_short_anchor() -> None:
    # No 1/output term to explode anymore: a tiny output clamps to the measured
    # short-output plateau anchor and stays bounded by the max measured plateau.
    tiny = saturated_ceiling_ms(1)
    assert tiny == saturated_ceiling_ms(28)  # clamps to the shortest measured anchor
    assert 220.0 < tiny < 261.0              # a measured plateau, not unbounded


def test_ceiling_is_swappable_per_config(tmp_path) -> None:
    """saturated_ceiling_ms reads the ACTIVE ceiling artifact; swapping it per-config (as the
    generator does for A100) changes the result, and restoring returns the H100 default."""
    import json as _json

    from simulator import kernel_tpot as K

    h100 = saturated_ceiling_ms(28)  # default H100 short-output anchor (~243)
    alt = tmp_path / "ceiling_alt.json"
    alt.write_text(_json.dumps({"anchors": [{"output_tokens": 28, "plateau_ms": 175.0},
                                            {"output_tokens": 86, "plateau_ms": 126.0}]}))
    orig = K._active_ceiling_json
    try:
        K._active_ceiling_json = alt
        assert abs(saturated_ceiling_ms(28) - 175.0) < 1e-6  # reads the swapped artifact
        assert abs(saturated_ceiling_ms(86) - 126.0) < 1e-6
    finally:
        K._active_ceiling_json = orig
    assert abs(saturated_ceiling_ms(28) - h100) < 1e-6       # restored to H100
    assert h100 != 175.0


# ------------------------------------------------------------ regime behavior
def test_flat_regime_returns_kernel_step() -> None:
    """Low pressure (small ctx, low concurrency) => prediction ≈ kernel step,
    well below the saturated ceiling.
    """
    out = predict_turn_tpot(_turn(cached=200, new=100, out=80, sched=5))
    assert out < 20.0  # decode-step floor territory


def test_tiny_output_turn_does_not_saturate() -> None:
    """A high-pressure turn producing very few output tokens can't sustain
    saturation — it predicts near the (low) kernel step, not the ceiling. The
    same workload with enough output tokens does saturate.
    """
    tiny = predict_turn_tpot(_turn(cached=1400, new=100, out=9, sched=320))
    sustained = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=320))
    assert tiny < 40.0, tiny
    assert sustained > 150.0, sustained
    assert tiny < sustained


def test_saturating_turn_approaches_output_ceiling() -> None:
    """High overflow (big ctx, large cohort) => prediction near T_upper(output); the
    short-output turn caps higher AND fills its step budgets in fewer decode steps."""
    hi = predict_turn_tpot(_turn(cached=9000, new=100, out=27, sched=160))
    lo = predict_turn_tpot(_turn(cached=9000, new=100, out=87, sched=160))
    assert hi > lo
    assert hi > 150.0  # well into the amplified regime
    assert lo < hi


def test_short_output_predicts_above_long_under_same_cohort() -> None:
    """Same high-pressure cohort: a short-output turn predicts above a long-output one
    (higher measured ceiling + larger recompute duty per step) — the behavior the
    retired output-binned knee hand-coded, now emergent."""
    short = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=160))
    long = predict_turn_tpot(_turn(cached=8000, new=100, out=200, sched=160))
    assert short > long


def test_prediction_monotone_in_pressure() -> None:
    """Holding workload fixed, more concurrent sessions => higher TPOT."""
    base = predict_turn_tpot(_turn(cached=5000, new=100, out=40, sched=20))
    more = predict_turn_tpot(_turn(cached=5000, new=100, out=40, sched=200))
    assert more > base


def test_prediction_bounded_by_kernel_step_and_ceiling() -> None:
    t = _turn(cached=6000, new=120, out=50, sched=120)
    pred = predict_turn_tpot(t)
    assert pred >= predict_turn_tpot(_turn(6000, 120, 50, 1)) - 1e-6  # >= floor
    assert pred <= saturated_ceiling_ms(50) + 1e-6                    # <= ceiling


def test_predict_cell_returns_per_turn_list() -> None:
    turns = [_turn(100 * i, 100, 60, 40) for i in range(5)]
    out = predict_cell_tpot(turns)
    assert len(out) == 5
    assert all(x > 0 for x in out)


def test_params_passthrough_does_not_crash() -> None:
    p = RooflineParams()
    assert predict_turn_tpot(_turn(1000, 100, 60, 40), p) > 0


def test_max_num_batched_tokens_is_engine_config() -> None:
    """The step budget is the vLLM resolved default (8192 H100-class / 2048 A100 via
    the deployment JSONs), an engine config — pin the dataclass default + the A100
    deployment composition so neither silently drifts."""
    assert RooflineParams().max_num_batched_tokens == 8192
    from configs.loader import all_deployments
    by_key = {d.gpu_key: d for d in all_deployments()
              if d.model == "Llama-3.1-8B" and d.engine == "vllm"}
    assert by_key["H100"].roofline.max_num_batched_tokens == 8192
    assert by_key["H100x2"].roofline.max_num_batched_tokens == 8192
    assert by_key["A100"].roofline.max_num_batched_tokens == 2048  # vLLM A100 device rule


# ------------------------------------------------------------ ramp-knee provenance (history)
def test_ramp_knees_measured_band_remains_pinned_history() -> None:
    """Provenance lock for the ramp-knee MEASUREMENT artifacts (2026-06-09 de-fit
    attempt + corrected-floor follow-up; see the De-fit log in
    profiling/docs/prediction_construction.md and ramp_knee_adoption_plan.md).

    The tuned knees themselves were ELIMINATED on 2026-06-10 (the distribution-overflow
    recompute-duty restructure, fitted_constants_audit.md item 6) — but the artifacts
    remain the valid measured record of the old band and of the per-GPU floor-excess
    stop-point, and they are the empirical target the restructure was validated
    against. Pinned so neither the builder rule nor the GT can drift silently:
      * the measured (uncorrected) band values stay reproducible;
      * current_literals keeps the HISTORICAL record of the tuned values that were in
        production when the artifact was built (0.88 / 1.22 / 2.0 — now removed);
      * per-GPU floor-excess D stays below the 0.5 ms Phase-0 stop-point;
      * H100 knees_corrected == knees (D = 0 there).
    """
    art = json.loads((KERNELS_DIR / "ramp_knees_h100_llama31_8b.json").read_text())
    knees = art["knees"]
    assert knees["P_LO"]["value"] == 0.4456          # measured onset (n=8, 3 profiles)
    assert knees["P_HI_SHORT"]["value"] == 1.6866    # measured short knee (n=6, 2 profiles)
    assert not knees["P_HI_LONG"]["adoptable"]       # data-starved: 1 profile, 2 cells
    # Historical record: the tuned literals in production at artifact build time —
    # REMOVED from kernel_tpot on 2026-06-10 (a rebuilt artifact records None + note).
    assert art["current_literals"] == {"P_LO": 0.88, "P_HI_SHORT": 1.22, "P_HI_LONG": 2.0}

    expected_d = {"h100": 0.0, "a100": 0.1246, "h100x2": 0.0}
    for gpu, d in expected_d.items():
        a = json.loads((KERNELS_DIR / f"ramp_knees_{gpu}_llama31_8b.json").read_text())
        assert a["floor_excess_ms"]["value"] == d
        assert a["floor_excess_ms"]["value"] < 0.5   # the Phase-0 stop-point rule
    corr = art["knees_corrected"]
    assert corr["P_LO"]["value"] == knees["P_LO"]["value"]
    assert corr["P_HI_SHORT"]["value"] == knees["P_HI_SHORT"]["value"]


def test_measured_onsets_collapse_in_z_units() -> None:
    """The core physical claim of the restructure: the per-GPU measured onset medians
    DISAGREE in raw pressure (0.4456 / 0.8540 / 0.6048 — spread 0.41, why no global
    P_LO exists) but COLLAPSE around the predicted z = 1 once multiplied by the
    profile's measured qbar (z = p_low·qbar). Recomputed from the artifacts + the
    quantile artifacts every run — a builder/GT change that breaks the collapse must
    be reviewed, not absorbed."""
    expected = {"h100": 0.964, "h100x2": 1.188, "a100": 0.963}  # measured 2026-06-10
    onsets = {}
    for gpu in expected:
        art = json.loads((KERNELS_DIR / f"ramp_knees_{gpu}_llama31_8b.json").read_text())
        zs = [c["p_low"] * cohort_scale_mean(c["profile"], float(c["conc"]))
              for c in art["cells"]]
        onsets[gpu] = statistics.median(zs)
        assert abs(onsets[gpu] - expected[gpu]) < 1e-3, (gpu, onsets[gpu])
    spread = max(onsets.values()) - min(onsets.values())
    assert spread < 0.23   # vs 0.4084 in raw pressure units (the ~2x collapse)


BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")


@pytest.mark.skipif(not BENCH_BASE.exists(), reason="bench data mount not available")
def test_predicted_saturation_crossing_matches_narrow_band_cells() -> None:
    """Closed-form w=0.9 crossing vs the measured p_high for the NARROW-band H100x2
    cells (band width < 0.1 pressure — the cleanest measured transitions). The
    crossing is found numerically on the round-2 law (chunk-quantized drain
    fraction + rotation boost, developed steady state) for the cell's median
    workload and must land within 10% of the artifact's measured p_high — with
    ZERO tuned constants. Measured 2026-06-10: swe c120 3.6%, swe c160 2.4%,
    term c200 5.6% (its crossing clamps at the pressure>=1 physical gate vs the
    measured 0.9468 — the documented sub-pool-full residual)."""
    from configs.loader import all_deployments
    from profiling.process.build_simulator_rows import build_turns

    dep = next(d for d in all_deployments()
               if d.gpu_key == "H100x2" and d.model == "Llama-3.1-8B" and d.engine == "vllm")
    p = dep.roofline
    art = json.loads((KERNELS_DIR / "ramp_knees_h100x2_llama31_8b.json").read_text())
    narrow = [c for c in art["cells"] if c["p_high"] - c["p_low"] < 0.1]
    assert len(narrow) >= 3   # term c200, swe c120, swe c160

    def w_at(pressure: float, qbar: float, med_ctx: float, med_out: float) -> float:
        psb = math.ceil(med_ctx / p.cache_block_size)
        sched = pressure * p.available_kv_blocks / psb
        b_eff = max(1.0, min(sched, p.available_kv_blocks / psb))
        return _overflow_weight(pressure, qbar, med_out, med_ctx, sched, b_eff, p)

    for c in narrow:
        bench = BENCH_BASE / dep.bench_dir / f"{c['profile']}_conc{c['conc']}.json"
        turns, _ = build_turns(bench)
        qbar = cohort_scale_mean(c["profile"], float(c["conc"]), "H100x2")
        med_out = statistics.median(max(1.0, float(t["output_tokens"])) for t in turns)
        med_ctx = statistics.median(
            float(t["cached_context_tokens"]) + float(t["new_prefill_tokens"])
            + 0.5 * max(1.0, float(t["output_tokens"])) for t in turns)
        lo, hi = 1.0, 5.0
        for _ in range(60):  # bisect the w = 0.9 crossing in pressure units
            mid = (lo + hi) / 2.0
            if w_at(mid, qbar, med_ctx, med_out) < 0.9:
                lo = mid
            else:
                hi = mid
        pred_p09 = (lo + hi) / 2.0
        rel_err = abs(pred_p09 - c["p_high"]) / c["p_high"]
        assert rel_err < 0.10, (c["profile"], c["conc"], pred_p09, c["p_high"])
