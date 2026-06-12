"""Tests for the forward 3D-roofline eviction-deficit ramp TPOT predictor."""

from __future__ import annotations

import json

import pytest

import simulator.ramp_tpot as ramp_tpot
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.ramp_tpot import (
    DEF_LO,
    DIST_DIR,
    PROFILE_DIST,
    forward_survival,
    predict_cell_tpot_ramp,
    predict_turn_ramp,
    sched_hat,
)


def _turn(idx: int, cached: float, new: float, out: float, sched: float | None = None) -> dict:
    d = {
        "turn_index": idx,
        "cached_context_tokens": cached,
        "new_prefill_tokens": new,
        "output_tokens": out,
    }
    if sched is not None:
        d["scheduled_requests"] = sched
    return d


# ----------------------------------------------------------- forward survival
def test_forward_survival_monotone_and_normalized() -> None:
    for prof, f in PROFILE_DIST.items():
        s = forward_survival(f)
        assert s[0] == 1.0, prof
        assert all(0.0 <= x <= 1.0 for x in s), prof
        assert all(b <= a + 1e-12 for a, b in zip(s, s[1:])), f"{prof} not non-increasing"


def test_sched_hat_decreases_and_matches_round() -> None:
    # swebench survival is ~flat; osworld drains steeply.
    swe0 = sched_hat("swebench-multiturn-synth", 100, 0)
    assert swe0 == 100  # S(0)=1
    osw_early = sched_hat("osworld-multiturn-synth", 160, 1)
    osw_late = sched_hat("osworld-multiturn-synth", 160, 12)
    assert osw_late < osw_early < 160  # steep drain


# ----------------------------- per-GPU / per-concurrency cohort resolver ---------------------------
from simulator.ramp_tpot import (  # noqa: E402
    _gpu_slug,
    _select_conc,
    context_scale_quantiles,
    survival_for,
)


def test_survival_for_default_byte_identical_to_pooled() -> None:
    # Default args (no concurrency, no gpu_key) MUST reproduce the legacy pooled curve exactly,
    # and so must a gpu_key with no per-GPU file present (falls back to pooled).
    for prof in PROFILE_DIST:
        pooled = forward_survival(PROFILE_DIST[prof])
        assert survival_for(prof) == pooled, prof
        assert survival_for(prof, 320, "no-such-gpu-xyz") == pooled, prof
        assert context_scale_quantiles(prof) == context_scale_quantiles(prof, 320, "no-such-gpu-xyz"), prof


def test_gpu_slug_deterministic() -> None:
    assert _gpu_slug("A100") == "a100"
    assert _gpu_slug("A100 (sglang)") == "a100sglang"
    assert _gpu_slug("H100x2") == "h100x2"
    assert _gpu_slug(None) == "" and _gpu_slug("") == ""


def test_select_conc_nearest_rule() -> None:
    fake = {"by_concurrency": {"10": {}, "80": {}, "200": {}}}
    assert _select_conc(fake, 80) == "80"           # exact
    assert _select_conc(fake, 90) == "80"           # nearest
    assert _select_conc(fake, 1) == "10"            # below range
    assert _select_conc(fake, 999) == "200"         # above range
    assert _select_conc(fake, 45) == "10"           # tie (|45-10|=35==|45-80|) -> smaller conc
    assert _select_conc({}, 80) is None             # no by_concurrency -> pooled
    assert _select_conc(None, 80) is None
    assert _select_conc(fake, None) is None          # concurrency None -> pooled


# ------------------------- S13: committed replay pools + loud missing-pool path -------------------
# The per-GPU realized files carry the TTFT trajectory-replay pools. They were gitignored
# (~105MB pretty-printed) and fresh checkouts silently fell back to the pooled cohort, which
# flipped two gate verdicts on 2026-06-09/10. Decision rule executed 2026-06-10: minified, the
# full Llama-3.1-8B ground-truth set is 19.9MB <= 25MB, so the files are COMMITTED. These tests
# make a missing pool impossible to miss: presence is pinned for the gate gpu_keys, and the
# resolver path warns loudly / hard-fails for anything absent.

GATE_GPU_SLUGS = ("h100", "a100", "h100x2")  # gate_scoped_rows DEFAULT_GPU_KEYS, slugged


def _per_gpu_path(profile: str, slug: str):
    return DIST_DIR / PROFILE_DIST[profile].replace(".json", f"_{slug}.json")


def test_committed_pools_present_for_gate_gpu_keys() -> None:
    """A fresh checkout MUST carry the replay pools for every gate gpu_key x profile —
    if this fails, the replay-ON gate baseline cannot be trusted (S13)."""
    for prof in PROFILE_DIST:
        for slug in GATE_GPU_SLUGS:
            f = _per_gpu_path(prof, slug)
            assert f.exists(), (
                f"missing committed per-GPU replay pool {f} — gates would silently run "
                f"replay-OFF; regenerate via build_realized_session_distributions"
            )
            d = json.loads(f.read_text())
            bc = d.get("by_concurrency") or {}
            assert bc, f
            # the fields production reads (ramp_tpot readers) must be present
            assert any(blk.get("trajectory_pool") for blk in bc.values()), f
            assert any(blk.get("turn_count") for blk in bc.values()), f
            assert any(blk.get("context_scale_quantiles") for blk in bc.values()), f


def test_committed_pools_carry_regeneration_note() -> None:
    files = sorted(DIST_DIR.glob("*_realized_*.json"))
    assert len(files) >= len(PROFILE_DIST) * len(GATE_GPU_SLUGS)
    for f in files:
        d = json.loads(f.read_text())
        note = d.get("_committed_note", "")
        assert "build_realized_session_distributions" in note, f


def test_per_gpu_resolver_prefers_committed_file() -> None:
    p = ramp_tpot._resolve_dist_path("swebench-multiturn-synth", "H100")
    assert p == _per_gpu_path("swebench-multiturn-synth", "h100")
    pool = ramp_tpot.trajectory_pool("swebench-multiturn-synth", 80, "H100")
    assert pool and all(len(t) == 3 for s in pool[:3] for t in s)


def test_missing_per_gpu_pool_warns_loudly_once(capsys) -> None:
    """Absent per-GPU file -> pooled fallback PLUS one unmissable stderr warning per
    (gpu_key, profile) — never silent, never a flood."""
    ramp_tpot._MISSING_POOL_WARNED.clear()
    prof = "chat-multiturn-synth"
    p1 = ramp_tpot._resolve_dist_path(prof, "ghost-gpu-9000")
    p2 = ramp_tpot._resolve_dist_path(prof, "ghost-gpu-9000")
    assert p1 == p2 == DIST_DIR / PROFILE_DIST[prof]  # pooled fallback unchanged
    err = capsys.readouterr().err
    assert err.count("PER-GPU REPLAY POOL MISSING") == 1
    assert "ghostgpu9000" in err and prof in err


def test_no_warning_without_gpu_key(capsys) -> None:
    ramp_tpot._MISSING_POOL_WARNED.clear()
    ramp_tpot._resolve_dist_path("chat-multiturn-synth", None)
    ramp_tpot._resolve_dist_path("chat-multiturn-synth", "")
    assert "PER-GPU REPLAY POOL MISSING" not in capsys.readouterr().err


def test_require_pools_env_escalates_to_hard_error(monkeypatch) -> None:
    """RAMP_TPOT_REQUIRE_POOLS=1 (gate runs): a requested-but-absent per-GPU pool is FATAL,
    on every occurrence; present committed pools still resolve."""
    monkeypatch.setenv("RAMP_TPOT_REQUIRE_POOLS", "1")
    with pytest.raises(FileNotFoundError, match="REPLAY POOL MISSING"):
        ramp_tpot._resolve_dist_path("chat-multiturn-synth", "ghost-gpu-9001")
    with pytest.raises(FileNotFoundError):  # raises again — no warn-once swallowing
        ramp_tpot._resolve_dist_path("chat-multiturn-synth", "ghost-gpu-9001")
    with pytest.raises(FileNotFoundError):  # and through the public reader
        ramp_tpot.trajectory_pool("osworld-multiturn-synth", 80, "ghost-gpu-9001")
    # committed files keep working under the strict env
    assert ramp_tpot.trajectory_pool("swebench-multiturn-synth", 80, "H100")


def test_unknown_profile_falls_back_to_concurrency() -> None:
    # No spec -> sched_hat returns the full concurrency (no drain), predictor still runs.
    assert sched_hat("nonexistent-profile", 64, 5) == 64.0
    out = predict_cell_tpot_ramp([_turn(0, 1000, 100, 30)], "nonexistent-profile", 64)
    assert len(out) == 1 and out[0] > 0


# ------------------------------------------------------------ per-turn ramp
def test_low_deficit_turn_is_the_bandwidth_floor() -> None:
    """Tiny cohort + small context => defcap << DEF_LO => frac=0 => pred == kernel floor."""
    p = RooflineParams()
    pred = predict_turn_ramp(cached=200, new_prefill=100, output=80, sched=2, ceiling_output=80, params=p)
    ctx_mid = 200 + 100 + 0.5 * 80
    floor = decode_step_ms(min(2.0, p.available_kv_blocks / (ctx_mid / 16)), ctx_mid, p)
    assert abs(pred - floor) < 1e-6, (pred, floor)


def test_prediction_monotone_in_cohort() -> None:
    """More concurrent sessions (higher deficit) => higher predicted ITL."""
    lo = predict_turn_ramp(8000, 100, 28, sched=20, ceiling_output=28)
    hi = predict_turn_ramp(8000, 100, 28, sched=200, ceiling_output=28)
    assert hi > lo


def test_short_output_saturates_higher_than_long_output() -> None:
    """At the same high deficit, a short-output (coding) turn reaches a higher ceiling
    than a long-output (osworld-like) turn — output-keyed + drain-aware ceiling."""
    short = predict_turn_ramp(9000, 100, 27, sched=200, ceiling_output=27)
    long = predict_turn_ramp(9000, 100, 90, sched=200, ceiling_output=90)
    assert short > long
    assert short > 150.0  # well into the saturated regime


def test_ramp_fires_above_the_watermark() -> None:
    """A turn whose deficit clears DEF_LO predicts above the floor; one well below it
    sits at the floor."""
    p = RooflineParams()
    # tune cohort so defcap is comfortably above / below DEF_LO at fixed context.
    below = predict_turn_ramp(3000, 100, 28, sched=5, ceiling_output=28, params=p)
    above = predict_turn_ramp(3000, 100, 28, sched=200, ceiling_output=28, params=p)
    floor_below = decode_step_ms(min(5.0, p.available_kv_blocks / ((3000 + 100 + 14) / 16)), 3000 + 100 + 14, p)
    assert abs(below - floor_below) < 1e-6      # below watermark -> floor
    assert above > below + 50.0                 # above watermark -> lifted


# --------------------------------------------------------- forward vs oracle
def test_forward_path_ignores_measured_scheduled() -> None:
    """The forward path must NOT read scheduled_requests — identical predictions
    whether or not the (absurd) measured cohort is present."""
    turns_with = [_turn(i, 2000 + 1000 * i, 100, 28, sched=999) for i in range(8)]
    turns_without = [_turn(i, 2000 + 1000 * i, 100, 28) for i in range(8)]
    fwd_with = predict_cell_tpot_ramp(turns_with, "swebench-multiturn-synth", 160)
    fwd_without = predict_cell_tpot_ramp(turns_without, "swebench-multiturn-synth", 160)
    assert fwd_with == fwd_without


def test_oracle_path_uses_measured_scheduled() -> None:
    """The oracle baseline DOES depend on the measured cohort (drain-isolation only)."""
    turns_hi = [_turn(i, 2000 + 1000 * i, 100, 28, sched=320) for i in range(8)]
    turns_lo = [_turn(i, 2000 + 1000 * i, 100, 28, sched=20) for i in range(8)]
    a = predict_cell_tpot_ramp(turns_hi, "swebench-multiturn-synth", 320, oracle=True)
    b = predict_cell_tpot_ramp(turns_lo, "swebench-multiturn-synth", 320, oracle=True)
    assert a != b


def test_empty_cell_returns_empty() -> None:
    assert predict_cell_tpot_ramp([], "swebench-multiturn-synth", 160) == []


# ----------------------------------------------- forward watermark-recovery cap
def _uncapped_cell(turns: list[dict], profile: str, C: float) -> list[float]:
    """The per-turn ramp WITHOUT the cell-level cap (for comparison)."""
    import statistics as _st

    co = _st.median([max(1.0, float(t["output_tokens"])) for t in turns])
    return [
        predict_turn_ramp(
            float(t["cached_context_tokens"]),
            float(t["new_prefill_tokens"]),
            float(t["output_tokens"]),
            sched_hat(profile, C, int(t["turn_index"])),
            co,
        )
        for t in turns
    ]


def _osworld_like_turns() -> list[dict]:
    # Context grows then PLATEAUS (~8650 cached) while the steep osworld survival
    # drains the cohort — so forecast KV demand load=sched_hat*blk peaks then FALLS,
    # which is exactly when the recovery cap engages.
    return [_turn(i, min(200 + 2000 * i, 8650), 1100, 85) for i in range(14)]


def test_cap_only_lowers_never_raises() -> None:
    """The forward cap is one-sided: capped prediction <= uncapped ramp everywhere."""
    turns = _osworld_like_turns()
    capped = predict_cell_tpot_ramp(turns, "osworld-multiturn-synth", 160)
    uncapped = _uncapped_cell(turns, "osworld-multiturn-synth", 160)
    assert all(c <= u + 1e-9 for c, u in zip(capped, uncapped))


def test_cap_is_noop_for_flat_survival_terminalbench() -> None:
    """Flat-survival short-output cells (terminalbench) have monotone-rising KV
    demand => rel==1 => the cap is an exact no-op (the swe/terminal wins are kept)."""
    turns = [_turn(i, 1000 + 400 * i, 130, 27) for i in range(18)]
    capped = predict_cell_tpot_ramp(turns, "terminalbench-multiturn-synth", 160)
    uncapped = _uncapped_cell(turns, "terminalbench-multiturn-synth", 160)
    assert capped == uncapped


def test_cap_bites_and_recovers_for_draining_osworld() -> None:
    """Steep-survival long-output cells (osworld) drain => KV demand peaks then
    falls => the cap pulls the saturated tail DOWN (recovery), so some turn is
    capped below its uncapped ramp and the late turn is lowered."""
    turns = _osworld_like_turns()
    capped = predict_cell_tpot_ramp(turns, "osworld-multiturn-synth", 160)
    uncapped = _uncapped_cell(turns, "osworld-multiturn-synth", 160)
    assert any(c < u - 1.0 for c, u in zip(capped, uncapped))  # cap bites somewhere
    assert capped[-1] < uncapped[-1] - 1.0  # the recovery tail is pulled down
