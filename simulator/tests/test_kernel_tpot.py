"""Tests for the kernel-composition per-turn TPOT predictor."""

from __future__ import annotations

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_tpot import (
    OUT_KNEE_HI,
    OUT_KNEE_LO,
    P_HI_LONG,
    P_HI_SHORT,
    P_LO,
    KernelTurnInput,
    _smoothstep,
    predict_cell_tpot,
    predict_turn_tpot,
    saturated_ceiling_ms,
)


def _turn(cached: float, new: float, out: float, sched: float) -> KernelTurnInput:
    return KernelTurnInput(
        cached_context_tokens=cached,
        new_prefill_tokens=new,
        output_tokens=out,
        scheduled_requests=sched,
    )


# ------------------------------------------------------------ smoothstep
def test_smoothstep_endpoints_and_midpoint() -> None:
    assert _smoothstep(0.0, P_LO, P_HI_LONG) == 0.0
    assert _smoothstep(10.0, P_LO, P_HI_LONG) == 1.0
    mid = _smoothstep((P_LO + P_HI_LONG) / 2, P_LO, P_HI_LONG)
    assert abs(mid - 0.5) < 1e-9


def test_output_gated_step_is_steeper_for_short_output() -> None:
    """Short-output (swe/terminal) workloads reach the saturation ceiling at a
    lower pressure (steeper step) than long-output (chat/osworld) ones, and
    cap higher — so under the same high-pressure cohort a short-output turn
    predicts above a long-output one.
    """
    assert P_HI_SHORT < P_HI_LONG
    assert OUT_KNEE_LO < OUT_KNEE_HI
    short = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=160))
    long = predict_turn_tpot(_turn(cached=8000, new=100, out=200, sched=160))
    assert short > long


def test_tiny_output_turn_does_not_saturate() -> None:
    """A high-pressure turn producing very few output tokens can't sustain
    saturation — it predicts near the (low) kernel step, not the ceiling. The
    same workload with enough output tokens does saturate.
    """
    # tiny output (≈9 tok): full cohort, but the turn finishes before the
    # eviction queue builds → near kernel step, far below the ceiling.
    tiny = predict_turn_tpot(_turn(cached=1400, new=100, out=9, sched=320))
    # enough output (≥ sustain HI) at the same cohort + bigger context → saturates.
    sustained = predict_turn_tpot(_turn(cached=8000, new=100, out=28, sched=320))
    assert tiny < 40.0, tiny
    assert sustained > 150.0, sustained
    assert tiny < sustained


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


def test_saturating_turn_approaches_output_ceiling() -> None:
    """High pressure (big ctx, large cohort) => prediction near T_upper(output)."""
    # short output => high ceiling
    hi = predict_turn_tpot(_turn(cached=9000, new=100, out=27, sched=160))
    # long output => lower ceiling, same pressure regime
    lo = predict_turn_tpot(_turn(cached=9000, new=100, out=87, sched=160))
    assert hi > lo
    assert hi > 150.0  # well into the amplified regime
    assert lo < hi


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


# ------------------------------------------------------------ ramp-knee provenance
def test_ramp_knees_tuned_values_and_measured_band_both_pinned() -> None:
    """Provenance lock for the ramp knees (2026-06-09 de-fit attempt + the same-day
    corrected-floor follow-up — see the De-fit log in
    profiling/docs/prediction_construction.md and
    profiling/docs/ramp_knee_adoption_plan.md "Execution record").

    The knees are TUNED KNOBS: the reproducible jump-band measurement
    (build_ramp_knees -> ramp_knees_h100_llama31_8b.json) DISAGREES with them
    (measured band ~[0.45, 1.69] vs tuned [0.88, 1.22]) and adopting the measured
    values failed the no-regression gates (H100 TPOT 15.4->23.3%). The follow-up
    corrected-floor round (ramp_knee_adoption_plan.md) measured the
    pressure-independent decode floor-excess D per GPU and hit the pre-registered
    Phase-0 stop-point: D < 0.5 ms everywhere (H100 0.0, A100 0.1246, H100x2 0.0)
    -> floor misattribution unsupported, NOTHING adopted, knees stay tuned, no
    production wiring of D. This pins ALL of it so none of it drifts silently:
      * the tuned literals stay at their documented values — retuning them requires
        consciously updating this test + the honest tuned-knob comments;
      * the artifact's measured band stays reproducible — a change to the builder's
        pre-registered detection rule (or to the GT) that moves the measurement
        must be reviewed, not absorbed;
      * the per-GPU D stays below the 0.5 ms stop-point threshold — if a data or
        rule change pushes it over, the adoption question must be re-opened
        deliberately, not silently.
    """
    import json
    from pathlib import Path

    assert P_LO == 0.88
    assert P_HI_SHORT == 1.22
    assert P_HI_LONG == 2.0

    kernels_dir = Path(__file__).resolve().parents[2] / "profile_data/kernels"
    art = json.loads((kernels_dir / "ramp_knees_h100_llama31_8b.json").read_text())
    knees = art["knees"]
    assert knees["P_LO"]["value"] == 0.4456          # measured onset (n=8, 3 profiles)
    assert knees["P_HI_SHORT"]["value"] == 1.6866    # measured short knee (n=6, 2 profiles)
    assert not knees["P_HI_LONG"]["adoptable"]       # data-starved: 1 profile, 2 cells
    # The artifact records which literals were in production when it was built.
    assert art["current_literals"] == {"P_LO": P_LO, "P_HI_SHORT": P_HI_SHORT,
                                       "P_HI_LONG": P_HI_LONG}

    # Corrected-floor round (gate run 2026-06-09, ADOPTED = none): D negligible on
    # every GPU per the pre-registered Phase-0 stop-point, so D is NOT wired into
    # production and knees_corrected is documentation only.
    expected_d = {"h100": 0.0, "a100": 0.1246, "h100x2": 0.0}
    for gpu, d in expected_d.items():
        a = json.loads((kernels_dir / f"ramp_knees_{gpu}_llama31_8b.json").read_text())
        assert a["floor_excess_ms"]["value"] == d
        assert a["floor_excess_ms"]["value"] < 0.5   # the stop-point rule that gated adoption
    # With D=0 on H100 the corrected band is identical to the uncorrected one.
    corr = art["knees_corrected"]
    assert corr["P_LO"]["value"] == knees["P_LO"]["value"]
    assert corr["P_HI_SHORT"]["value"] == knees["P_HI_SHORT"]["value"]
