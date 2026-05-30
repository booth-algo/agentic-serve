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
    # Turn-3 median TTFT grows with cohort under fixed KV (backlog grows).
    turns = _mk_turns(8, new=1600.0, output=40.0)
    lo = predict_cell_ttft_qsim(turns, SWE, 40)
    hi = predict_cell_ttft_qsim(turns, SWE, 200)
    assert hi[3] > lo[3]


def test_closed_loop_arrival_equals_prior_completion():
    # Instrument the raw sim: turn t+1 arrival == turn t completion for a session.
    turns = _mk_turns(6, new=512.0, output=20.0)
    sessions = _build_cohort(turns, SWE, 4)
    # Re-run with result epochs exposed by patching: use the internal _run_sim plus a
    # fresh state walk. Easiest: rebuild and read the results dict via a thin re-impl.
    from simulator.ttft_queue_sim import _ServerState, _on_arrival, _on_step, _on_first_token, _on_depart, _ARRIVAL, _STEP, _FIRST_TOKEN, _DEPART
    from simulator._legacy.vllm_block_pool import BlockPool
    import heapq

    p = RooflineParams()
    pool = BlockPool(p.available_kv_blocks, p.cache_block_size)
    state = _ServerState(params=p, pool=pool, sessions=sessions)
    for s in sessions:
        state.push(0.0, _ARRIVAL, (s.session_id, 0))
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

    # For session 0, check arrival[t+1] == completion[t] (think-time == 0).
    checked = 0
    for sid in range(len(sessions)):
        tc = sessions[sid].turn_count
        for t in range(tc - 1):
            cur = state.results.get((sid, t))
            nxt = state.results.get((sid, t + 1))
            if cur and nxt and "completion_epoch" in cur and "arrival_epoch" in nxt:
                assert nxt["arrival_epoch"] == pytest.approx(cur["completion_epoch"])
                checked += 1
    assert checked > 0


def test_kv_gate_blocks_admission():
    # Shrink KV so capacity < concurrency: some reqs MUST wait (queue_wait > 0) and
    # the FIFO head blocks. We assert that mid-turn TTFT under the tiny pool is much
    # larger than the single-session floor (head-of-line blocking realized).
    tiny = RooflineParams(available_kv_blocks=400)  # ~4 sessions fit at ~90 blocks each
    turns = _mk_turns(5, new=1400.0, output=30.0)
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
    # The ONLY new module-level numeric constants are the two vLLM serving defaults.
    import simulator.ttft_queue_sim as mod

    assert mod.MAX_NUM_SEQS == 512
    assert mod.MAX_NUM_BATCHED_TOKENS == 8192
    # Module-level numeric globals (uppercase) must be exactly these two + the grid
    # edge (1024, reused from ttft_predict, not a fit) + event-kind enum ints.
    # Public (non-underscore) uppercase numeric module globals: exactly the two vLLM
    # serving defaults. Private (underscore-prefixed) names — the event-kind enum ints
    # and _GRID_U_MAX=1024 (the cached-prefill grid edge, reused from ttft_predict, NOT
    # a fit) — are physics/structure, not headline knobs, and are excluded.
    public_numeric = {
        k: v
        for k, v in vars(mod).items()
        if k.isupper()
        and not k.startswith("_")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }
    assert set(public_numeric) == {"MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS"}
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
