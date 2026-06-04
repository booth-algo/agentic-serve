"""Tests for the forward closed-loop event-driven TTFT queue sim.

Covers: empty cell -> []; return length/alignment; determinism (no RNG / no
wall-clock); single-session queue-free floor; monotonicity in concurrency at
saturation; closed-loop arrival == prior completion; KV gate head-of-line
blocking; emergent saturate-ramp (long-session cell) + recover (draining cell);
forward path ignores measured scheduled_requests; e2el composition uses kernel
TPOT; unknown-profile fallback; max_events guard; aggregation == median contract.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_DASH = REPO_ROOT / "inference-benchmark/dashboard/public/simulator-predictions.json"


def _real_cell(profile: str, concurrency: int) -> list[dict] | None:
    """Pull a real (profile, concurrency) cell's per-turn dicts from the dashboard
    JSON, or None when the artifact is unavailable (keeps the suite hermetic)."""
    if not _DASH.exists():
        return None
    payload = json.loads(_DASH.read_text())
    for row in payload.get("H100", []):
        if row.get("profile") == profile and row.get("concurrency") == concurrency:
            return row.get("multiturn_turn_predictions") or None
    return None

from simulator.closed_form_tpot import RooflineParams
from simulator.ttft_queue_sim import (
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    _build_cohort,
    _draw_turn_count,
    _run_sim,
    predict_cell_e2el_qsim,
    predict_cell_ttft_qsim,
)

SWE = "swebench-multiturn-synth"
OSW = "osworld-multiturn-synth"
CHAT = "chat-multiturn-synth"


def _mk_turns(n: int, *, cached=0.0, new=1400.0, output=30.0, start=0) -> list[dict]:
    return [
        {
            "turn_index": start + i,
            "cached_context_tokens": cached + 200.0 * i,
            "new_prefill_tokens": new,
            "output_tokens": output,
        }
        for i in range(n)
    ]


def test_empty_turns_returns_empty():
    assert predict_cell_ttft_qsim([], SWE, 40) == []


def test_return_length_and_alignment():
    turns = _mk_turns(12)
    out = predict_cell_ttft_qsim(turns, SWE, 40)
    assert len(out) == len(turns)
    for v in out:
        assert isinstance(v, float)
        assert v > 0.0
        assert v == v  # not NaN
        assert v < float("inf")


def test_determinism_reproducible():
    turns = _mk_turns(15)
    a = predict_cell_ttft_qsim(turns, SWE, 80)
    b = predict_cell_ttft_qsim(turns, SWE, 80)
    assert a == b  # byte-identical: no RNG, no wall-clock


def test_single_session_low_concurrency_ttft_floor():
    # concurrency=1, one short turn => queue-free: prefill + 1 decode + overhead,
    # ZERO queue wait. Should be a small floor, not the oversubscription regime.
    turns = _mk_turns(1, new=256.0, output=8.0)
    out = predict_cell_ttft_qsim(turns, SWE, 1)
    assert len(out) == 1
    # A single session never waits in the FIFO; first-token latency is one prefill
    # pass + one decode step + scheduler overhead — well under a second.
    assert 0.0 < out[0] < 500.0


def test_ttft_monotone_in_concurrency_at_saturation():
    # Once the cohort KV firmly exceeds the pool, median TTFT grows with cohort (backlog
    # grows). Compare at turn 5 — by then c200's cumulative KV (200x(800+1600)=480k tokens)
    # firmly exceeds the 436k-token pool, while c40 stays at the queue-free floor. (Earlier
    # turns sit right at the saturation knee where the chunked-budget de-serialization can
    # put hi ~= lo at the floor; turn 5 is unambiguously in the oversubscribed regime.)
    turns = _mk_turns(8, new=1600.0, output=40.0)
    lo = predict_cell_ttft_qsim(turns, SWE, 40)
    hi = predict_cell_ttft_qsim(turns, SWE, 200)
    assert hi[5] > lo[5]


def test_barrier_round_robin_synchronizes_turns():
    # The harness dispatches all sessions' turn-N requests together and asyncio.gather()s
    # before turn N+1. So (a) every session's turn-(t+1) request arrives at the SAME epoch
    # (a synchronized herd), and (b) that epoch equals the LAST turn-t completion (barrier).
    turns = _mk_turns(6, new=512.0, output=20.0)
    sessions = _build_cohort(turns, SWE, 8)
    from simulator.ttft_queue_sim import (
        _ServerState, PrefixLRUCache, _release_herd, _on_arrival, _on_step,
        _on_first_token, _on_depart, _ARRIVAL, _STEP, _FIRST_TOKEN, _DEPART,
    )
    import heapq

    p = RooflineParams()
    cache = PrefixLRUCache(p.available_kv_blocks, p.cache_block_size)
    state = _ServerState(params=p, cache=cache, sessions=sessions)
    _release_herd(state, 0)
    events = 0
    while state.heap and events < 1_000_000:
        epoch, _seq, kind, payload = heapq.heappop(state.heap)
        state.clock = epoch
        events += 1
        if kind == _ARRIVAL:
            _on_arrival(state, payload[0], payload[1])
        elif kind == _STEP:
            _on_step(state)
        elif kind == _FIRST_TOKEN:
            _on_first_token(state, payload)
        elif kind == _DEPART:
            _on_depart(state, payload[0], payload[1])

    max_turns = max(s.turn_count for s in sessions)
    checked = 0
    for t in range(max_turns - 1):
        arrivals = [
            state.results[(sid, t + 1)]["arrival_epoch"]
            for sid in range(len(sessions))
            if sessions[sid].turn_count > t + 1 and (sid, t + 1) in state.results
        ]
        completions = [
            state.results[(sid, t)]["completion_epoch"]
            for sid in range(len(sessions))
            if sessions[sid].turn_count > t and (sid, t) in state.results
            and "completion_epoch" in state.results[(sid, t)]
        ]
        if len(arrivals) < 2 or not completions:
            continue
        # (a) all turn-(t+1) arrivals are simultaneous (the herd)
        assert max(arrivals) == pytest.approx(min(arrivals))
        # (b) the herd releases at the barrier = last turn-t completion
        assert arrivals[0] == pytest.approx(max(completions))
        checked += 1
    assert checked > 0


def test_kv_gate_blocks_admission():
    # Shrink KV so capacity < concurrency: some reqs MUST wait (queue_wait > 0) and
    # the FIFO head blocks. We assert that turn-0 TTFT under the tiny pool is much
    # larger than the single-session floor (head-of-line blocking realized).
    # new=1000 < long_prefill_token_threshold (1310) so the single-session floor is ONE
    # un-chunked prefill pass — isolating the blocking signal from the chunk-cap (a
    # >threshold prefill would itself span 2 chunks and inflate the floor).
    tiny = RooflineParams(available_kv_blocks=400)  # ~4 sessions fit at ~90 blocks each
    turns = _mk_turns(5, new=1000.0, output=30.0)
    out = predict_cell_ttft_qsim(turns, SWE, 60, tiny)
    floor = predict_cell_ttft_qsim(turns, SWE, 1, tiny)
    # With 60 sessions and room for ~4, the backlog wait dominates.
    assert out[0] > floor[0] * 3


def test_saturate_ramp_long_session_cell():
    # A long-session (flat-survival) swebench cell: per-turn TTFT RISES across turns
    # as the persistent cohort keeps KV pressure up. Late plateau > early steady state.
    turns = _real_cell(SWE, 200)
    if turns is None:
        # Fallback synthetic long cell when the artifact is unavailable.
        turns = _mk_turns(20, new=1500.0, output=28.0)
    out = predict_cell_ttft_qsim(turns, SWE, 200)
    early = statistics.median(out[1:5])      # skip the turn-0 herd
    late = statistics.median(out[12:18])
    assert late > early  # emergent ramp


def test_recover_on_draining_cell():
    # osworld has a steep survival drain -> the per-turn TTFT curve rises then FALLS
    # (recover) as the cohort drains. Uses the REAL osworld c200 cell (the faithful
    # saturate-ramp-recover shape lives in the measured token trajectory, not a
    # flat-token synthetic). Interior peak, last < peak.
    turns = _real_cell(OSW, 200)
    if turns is None:
        pytest.skip("dashboard JSON unavailable")
    out = predict_cell_ttft_qsim(turns, OSW, 200)
    body = out[1:]  # skip the turn-0 cohort-dispatch herd
    peak_idx = max(range(len(body)), key=lambda i: body[i])
    assert 0 < peak_idx < len(body) - 1  # interior peak (rises then recedes)
    assert body[-1] < body[peak_idx]      # recovers below the peak


def test_high_conc_does_not_silently_fall_back():
    # REGRESSION: at high concurrency the cohort's persistent KV fills the pool; before the
    # two-tier preemption fix every next-turn head was eviction-protected -> nothing admitted
    # -> no step scheduled -> the sim drained after turn 0 and _aggregate SILENTLY filled the
    # rest with the static fallback. The real sim MUST simulate deep into the climb.
    turns = _real_cell(SWE, 200)
    if turns is None:
        turns = _mk_turns(30, new=1500.0, output=28.0)
    sessions = _build_cohort(turns, SWE, 200)
    deep = max(s.turn_count for s in sessions) - 1  # deepest turn the cohort reaches
    if deep < 10:
        pytest.skip("cohort too short to exercise the deadlock regime")
    ttfts = _run_sim(sessions, RooflineParams(), 4_000_000)
    sim_turns = {ti for (_sid, ti) in ttfts}  # turns with a REAL first-token (not fallback)
    assert max(sim_turns) >= min(deep, 20)  # sim ran past the turn-0 herd, no silent fallback


def test_context_scale_spread_applied():
    # The cohort applies a measured per-session context SCALE so sessions do NOT all carry the
    # identical median trajectory (the spread is what keeps the median session a hit near the
    # cliff). Skips cleanly if the realized scale artifact is unavailable.
    from simulator.ramp_tpot import context_scale_quantiles

    if not context_scale_quantiles(SWE):
        pytest.skip("context-scale quantiles unavailable")
    turns = _real_cell(SWE, 200) or _mk_turns(20, new=1500.0, output=28.0)
    sessions = _build_cohort(turns, SWE, 200)
    ctx5 = {round(s.turns[5].cached_context_tokens, 1) for s in sessions if s.turn_count > 5}
    assert len(ctx5) > 1  # per-session scale spread, not median-for-all


def test_forward_path_never_reads_scheduled_requests():
    turns = _mk_turns(10, new=1200.0, output=30.0)
    bogus = [dict(t, scheduled_requests=99999) for t in turns]
    a = predict_cell_ttft_qsim(turns, SWE, 80)
    b = predict_cell_ttft_qsim(bogus, SWE, 80)
    assert a == b  # forward path ignores measured cohort


def test_oracle_path_runs():
    # oracle=True should not crash; returns a finite per-turn list (it falls back to
    # the forward cohort when measured timelines are unavailable in this env).
    turns = _mk_turns(6, new=512.0, output=20.0)
    out = predict_cell_ttft_qsim(turns, SWE, 8, oracle=True)
    assert len(out) == len(turns)
    assert all(v > 0 and v < float("inf") for v in out)


def test_e2el_composition_uses_kernel_tpot():
    turns = _mk_turns(5, new=800.0, output=24.0)
    ttft = predict_cell_ttft_qsim(turns, SWE, 40)
    tpot_preds = [11.0, 12.0, 13.0, 14.0, 15.0]
    e2el = predict_cell_e2el_qsim(turns, SWE, 40, ttft, tpot_preds=tpot_preds)
    for t, tt, tp, e in zip(turns, ttft, tpot_preds, e2el):
        assert e == pytest.approx(tt + float(t["output_tokens"]) * tp)


def test_unknown_profile_fallback():
    turns = _mk_turns(8, new=1000.0, output=30.0)
    out = predict_cell_ttft_qsim(turns, "nonexistent-profile", 40)
    assert len(out) == len(turns)
    assert all(v > 0 and v < float("inf") for v in out)


def test_max_events_guard_terminates():
    # A pathological cell with a tight event budget returns partial medians (does not
    # hang) — the guard trips gracefully.
    turns = _mk_turns(30, new=2000.0, output=200.0)
    out = predict_cell_ttft_qsim(turns, SWE, 200, max_events=500)
    assert len(out) == len(turns)
    assert all(v == v for v in out)  # no NaN


def test_aggregation_matches_meas_contract():
    # Hand-built tiny cell: the per-turn aggregate is the MEDIAN over sessions reaching
    # that turn_index — byte-identical to ttft_meas grouping. Verify the median rule by
    # re-aggregating the raw (session, turn) TTFTs from _run_sim.
    turns = _mk_turns(4, new=512.0, output=16.0)
    sessions = _build_cohort(turns, SWE, 3)
    p = RooflineParams()
    ttfts = _run_sim(sessions, p, 1_000_000)
    by_idx: dict[int, list[float]] = {}
    for (_sid, ti), v in ttfts.items():
        if v > 0:
            by_idx.setdefault(ti, []).append(v)
    cell = predict_cell_ttft_qsim(turns, SWE, 3)
    for t, agg in zip(turns, cell):
        ti = int(t["turn_index"])
        if ti in by_idx:
            assert agg == pytest.approx(statistics.median(by_idx[ti]))


def test_draw_turn_count_inverse_survival():
    # Monotone: a lower quantile (closer to 0) reaches at least as many turns as a
    # higher quantile (closer to 1). Survival is non-increasing.
    surv = [1.0, 0.8, 0.5, 0.3, 0.1]
    assert _draw_turn_count(surv, 0.05) >= _draw_turn_count(surv, 0.95)
    assert _draw_turn_count(surv, 0.05) == 5  # reaches all turns
    assert _draw_turn_count([], 0.5) == 1     # empty survival -> 1 turn


def test_no_fitted_constants():
    # Every module-level numeric constant is a vLLM serving default or config-derived —
    # NONE is fitted to the TTFT target.
    import simulator.ttft_queue_sim as mod

    # vLLM EngineArgs H100 + OPENAI_API_SERVER resolved defaults (arg_utils._set_default_args).
    assert mod.MAX_NUM_SEQS == 1024
    assert mod.MAX_NUM_BATCHED_TOKENS == 8192
    # Per-step prefill chunk cap: SchedulerConfig sets long_prefill_token_threshold to
    # int(max_model_len * 0.04) when chunked prefill is on and the flag is unset. The
    # benchmark ran max_model_len=32768 (server metadata) -> 1310. Config-derived, NOT a fit.
    assert mod.MAX_MODEL_LEN == 32768
    assert mod.LONG_PREFILL_TOKEN_THRESHOLD == int(32768 * 0.04) == 1310
    # Prefill cost = measured-serving anchors + pipeline FA3 kernel (all held out from the
    # multi-turn data we report). Asserted to value so a silent retune is caught.
    assert mod.PREFILL_FLOOR_MS == 26.0  # DE-FITTED 2026-06-03: measured min pure-prefill TTFT (c1 turn-0, cached≈0 ≈26.07 ms), replaces the fitted regression intercept 22.5
    assert mod.PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN == 0.00602  # off-GPU dispatch remainder
    # NEW = DERIVED tp-aware GEMM roofline + residual; on tp1 their sum reproduces the retired
    # fitted 0.0310 rate to the residual's 5-digit rounding (~2e-6 ms/tok; TTFT/E2EL gates unchanged).
    assert abs(mod._prefill_gemm_per_tok(mod.RooflineParams())
               + mod.PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN - 0.0310) < 5e-5
    assert mod.PREFILL_FA3_MS_PER_TOKEN2 == 8.31e-7        # pipeline FA3 kernel (fa3_prefill grid)
    # Host serving-stack cached split (live-measured: live_split_probe.py concurrency sweep on the real server).
    assert mod.PREFILL_HOST_SHARED_MS_PER_TOKEN == 0.0030515  # DE-FITTED 2026-06-03: live-server split 50/50 (was 0.003485)
    assert mod.PREFILL_HOST_PERREQ_MS_PER_TOKEN == 0.0030515  # 0.50×6.103e-3 (was 0.002618)
    # Public uppercase numeric module globals: the four config-derived vLLM values + the
    # three measured prefill-law coefficients. Private (underscore-prefixed) names — the
    # event-kind enum ints and _GRID_U_MAX=1024 — are physics/structure, excluded.
    public_numeric = {
        k: v
        for k, v in vars(mod).items()
        if k.isupper()
        and not k.startswith("_")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }
    assert set(public_numeric) == {
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "MAX_MODEL_LEN",
        "LONG_PREFILL_TOKEN_THRESHOLD",
        "PREFILL_FLOOR_MS",
        "PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN",
        "PREFILL_FA3_MS_PER_TOKEN2",
        "PREFILL_HOST_SHARED_MS_PER_TOKEN",
        "PREFILL_HOST_PERREQ_MS_PER_TOKEN",
    }
    # And the private numeric constants are only the grid edge + the 4 event-kind ints.
    private_numeric = {
        k
        for k, v in vars(mod).items()
        if k.startswith("_")
        and k.isupper()
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }
    assert private_numeric == {"_GRID_U_MAX", "_ARRIVAL", "_STEP", "_FIRST_TOKEN", "_DEPART"}
