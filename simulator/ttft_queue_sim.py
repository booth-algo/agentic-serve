"""Forward, closed-loop, event-driven multi-turn TTFT **queue** simulator with
**session-persistent KV + RECOMPUTE eviction** (vLLM v1 model).

TTFT is not a quasi-static per-turn quantity like TPOT — it is the wall-clock **queue
wait** a request sees before its first token (client_queue_wait ~= 0, so all server-side
queueing is folded into the measured ``ttft_ms``). The per-turn TTFT is a
*saturate-ramp-RECOVER* curve that emerges from the cohort's trajectory.

THE CLIMB MECHANISM (measured + traced against /root/vllm v1 scheduler): a multi-turn
session's KV **persists across its turns** (prefix reuse), growing each turn. Under load
the cohort's cumulative KV vastly exceeds the pool, so the scheduler **evicts** whole
sessions (RECOMPUTE — frees their blocks); an evicted session's next turn is a cache
**MISS** and must **re-prefill its entire context** (cached+new), not just the new tokens.
That full re-prefill congests the chunked-prefill token budget and head-of-line-blocks new
arrivals' first token. Because cached context grows every turn, the re-prefill cost
compounds — TTFT climbs unboundedly for full-staying cohorts (swebench/terminal, flat
survival) and saturates/recovers as the cohort drains (osworld). A turn whose session KV
is still **resident** is a cache HIT and prefills only its new tokens (cheap) — so the
hit/miss fraction self-adjusts to KV pressure and sets the magnitude.

MODEL (fit-free — NO new fitted constants):

* Cohort: ``C = round(concurrency)`` sessions; each session's turn-count drawn FORWARD from
  the profile turn-count survival histogram (``ramp_tpot.forward_survival`` / ``PROFILE_DIST``)
  DETERMINISTICALLY by quantile ``q_k=(k+0.5)/C`` (reproducible; no RNG, no wall-clock reads).
* Shared GPU continuous-batching steps; admission gated by KV blocks
  (``BlockPool(available_kv_blocks=27250, 16)``) + ``max_num_seqs`` + ``MAX_NUM_BATCHED_TOKENS``
  (vLLM serving defaults, documented config — not MAPE knobs).
* **KV blocks are owned per SESSION and persist across turns**; the session keeps its KV
  after a turn departs (resident) until it ENDS or is EVICTED. Eviction (LRU over resident,
  non-in-flight sessions) frees a session's blocks when the pool can't fit a needed
  allocation; the evicted session's next turn is a MISS (full re-prefill).
* Closed loop: turn ``t+1`` ARRIVES at turn ``t`` COMPLETION (think-time == 0). HIT (session
  resident, KV covers cached) → prefill only ``new``; MISS → prefill ``cached+new``.
* Per-step wall-time from MEASURED kernels: decode = ``decode_step_ms(batch, ctx)``; one fused
  prefill pass = ``cached_prefill_step_ms`` (roofline above the grid edge). ``TTFT[turn] =
  first_token_epoch - arrival_epoch``; aggregate per (profile, concurrency, turn_index) -> MEDIAN.

Output is the ADDITIVE column ``ttft_pred_qsim`` (+ ``e2el_pred_qsim``). The ``ttft_pred`` /
``tpot_pred`` / ``tpot_err`` / ``tpot_pred_kernel`` columns are never repointed (M0 + kernel
headline stay byte-identical).
"""

from __future__ import annotations

import heapq
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from simulator._legacy.vllm_block_pool import BlockPool
from simulator.cached_prefill_lookup import cached_prefill_step_ms
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator.ramp_tpot import PROFILE_DIST, forward_survival
from simulator.ttft_predict import _prefill_per_token_ms, predict_turn_ttft

__all__ = ["predict_cell_ttft_qsim", "predict_cell_e2el_qsim", "PROFILE_DIST"]

# --- vLLM serving defaults (documented runtime config, NOT fitted) ------------
MAX_NUM_SEQS = 512          # vLLM --max-num-seqs default (freeWorkers pool size)
MAX_NUM_BATCHED_TOKENS = 8192  # vLLM --max-num-batched-tokens default (per-step token budget)

# Event kinds; ordering is (epoch, seq, kind) — deterministic FIFO at equal epochs.
_ARRIVAL = 0
_STEP = 1
_FIRST_TOKEN = 2
_DEPART = 3

# Largest new-prefill the measured cached-prefill grid covers; above it the prefill-compute
# roofline (continuous with the grid edge) prices a fused multi-session pass. No new constant.
_GRID_U_MAX = 1024.0


def _prefill_pass_ms(total_chunk_tokens: float, mean_prefix: float, params: RooflineParams) -> float:
    """Wall-time of ONE fused prefill forward pass over ``total_chunk_tokens`` tokens at
    ``mean_prefix`` cached prefix — measured cached-prefill grid up to its U edge, the
    prefill-compute roofline above it (continuous, the large-batch extrapolant)."""
    u = max(1.0, float(total_chunk_tokens))
    if u <= _GRID_U_MAX:
        return cached_prefill_step_ms(u, max(1.0, float(mean_prefix)))
    return u * _prefill_per_token_ms(params) + params.scheduler_overhead_ms_per_step


# ---------------------------------------------------------------- session model


@dataclass
class TurnSpec:
    turn_index: int
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float


@dataclass
class Session:
    session_id: int
    turn_count: int
    turns: list[TurnSpec]
    next_turn_idx: int = 0
    # --- session-persistent KV residency state ---
    resident: bool = False    # currently holds its KV blocks in the pool (keyed by session_id)
    inflight: bool = False     # has a turn currently prefilling or running (protected from eviction)
    held_tokens: float = 0.0   # resident KV length (cumulative context), 0 if evicted/never-run
    last_seq: int = -1         # last-activity order; LRU eviction evicts the lowest


# ------------------------------------------------------------------ in-flight reqs


@dataclass
class _Req:
    """One (session, turn) request as it moves waiting -> prefilling -> running."""

    rid: int
    session_id: int
    turn_index: int
    arrival_epoch: float
    cached: float
    new_prefill: float
    output: float
    remaining_prefill: float
    output_left: int
    kv_tokens: float           # resident KV after this turn's prefill (cached+new), grows with decode
    is_miss: bool = False      # cache miss (session was evicted) -> re-prefilled full context


@dataclass
class _ServerState:
    params: RooflineParams
    pool: BlockPool
    sessions: list[Session]
    clock: float = 0.0
    seq: int = 0  # monotone tiebreak + LRU counter
    heap: list[tuple[float, int, int, Any]] = field(default_factory=list)
    waiting: list[_Req] = field(default_factory=list)        # FIFO
    prefilling: dict[int, _Req] = field(default_factory=dict)
    running: dict[int, _Req] = field(default_factory=dict)
    results: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    step_scheduled: bool = False
    evictions: int = 0  # diagnostic

    def push(self, epoch: float, kind: int, payload: Any) -> None:
        heapq.heappush(self.heap, (epoch, self.seq, kind, payload))
        self.seq += 1


def _encode_rid(session_id: int, turn_index: int) -> int:
    """Stable per-(session, turn) request id. 4096 turns max per session — ample."""
    return session_id * 4096 + turn_index


# ----------------------------------------------------------------- cohort builder


def _draw_turn_count(survival: list[float], quantile: float) -> int:
    """Inverse-survival, deterministic. ``survival[t]=S(t)`` = fraction alive AT turn t."""
    if not survival:
        return 1
    reached = 0
    for s in survival:
        if s >= quantile:
            reached += 1
        else:
            break
    return max(1, reached)


def _build_cohort(
    turns: list[dict[str, Any]], profile: str, concurrency: float
) -> list[Session]:
    """Deterministic survival-quantile cohort (forward; same source ``sched_hat`` uses)."""
    c = max(1, int(round(float(concurrency))))
    survival = forward_survival(PROFILE_DIST[profile]) if profile in PROFILE_DIST else None

    spec_by_idx: dict[int, TurnSpec] = {}
    max_turn_idx = 0
    for t in turns:
        ti = int(t.get("turn_index", 0))
        spec_by_idx[ti] = TurnSpec(
            turn_index=ti,
            cached_context_tokens=float(t.get("cached_context_tokens") or 0.0),
            new_prefill_tokens=float(t.get("new_prefill_tokens") or 0.0),
            output_tokens=max(1.0, float(t.get("output_tokens") or 1.0)),
        )
        max_turn_idx = max(max_turn_idx, ti)
    n_turn_slots = max_turn_idx + 1

    def spec_for(idx: int) -> TurnSpec:
        if idx in spec_by_idx:
            return spec_by_idx[idx]
        lower = [i for i in spec_by_idx if i <= idx]
        if lower:
            return spec_by_idx[max(lower)]
        return spec_by_idx[min(spec_by_idx)] if spec_by_idx else TurnSpec(idx, 0.0, 1.0, 1.0)

    sessions: list[Session] = []
    for k in range(c):
        q = (k + 0.5) / c
        if survival:
            tc = min(_draw_turn_count(survival, q), n_turn_slots) if n_turn_slots > 0 else _draw_turn_count(survival, q)
        else:
            tc = n_turn_slots if n_turn_slots > 0 else 1
        tc = max(1, tc)
        sessions.append(Session(session_id=k, turn_count=tc, turns=[spec_for(i) for i in range(tc)]))
    return sessions


# ------------------------------------------------------------------- the sim core


def _running_ctx_mean(state: _ServerState) -> float:
    if not state.running:
        return 1.0
    return statistics.fmean(r.kv_tokens for r in state.running.values())


def _ensure_free(state: _ServerState, need_blocks: int, protect_sid: int) -> bool:
    """Make ``need_blocks`` free by evicting KV **block-granularly** (vLLM LRU): trim the
    TAIL blocks of least-recently-active resident, non-in-flight sessions, keeping their
    shared PREFIX resident. A partially-trimmed session stays resident with a shorter
    ``held_tokens`` → its next turn is a partial cache hit (re-prefill only the evicted
    tail), not a full miss. Returns True once enough is free, False if no victims remain."""
    if need_blocks <= 0:
        return True
    bsz = state.pool.block_size
    while state.pool.get_num_free_blocks() < need_blocks:
        victim: Session | None = None
        for s in state.sessions:
            if s.resident and not s.inflight and s.session_id != protect_sid:
                if state.pool.num_blocks_owned(s.session_id) <= 0:
                    continue
                if victim is None or s.last_seq < victim.last_seq:
                    victim = s
        if victim is None:
            return state.pool.get_num_free_blocks() >= need_blocks
        deficit = need_blocks - state.pool.get_num_free_blocks()
        owned = state.pool.num_blocks_owned(victim.session_id)
        freed = state.pool.free_partial(victim.session_id, min(owned, deficit))
        remaining = owned - freed
        if remaining > 0:
            victim.held_tokens = float(remaining * bsz)  # surviving prefix stays resident
        else:
            victim.resident = False
            victim.held_tokens = 0.0
        state.evictions += 1
    return True


def _schedule(state: _ServerState) -> None:
    """FIFO admission. The hit/miss decision is made HERE (residency at admission time): a
    session still holding its KV is a HIT (prefill new only); an evicted session is a MISS
    (re-prefill its full cached+new context). Allocate the per-session block delta, evicting
    LRU resident sessions if the pool is full; stop on the first unservable head (HOL block —
    where the backlog, hence the TTFT climb, emerges)."""
    decode_slots = len(state.running)
    budget = MAX_NUM_BATCHED_TOKENS - decode_slots
    for r in state.prefilling.values():
        budget -= min(r.remaining_prefill, MAX_NUM_BATCHED_TOKENS)
    while state.waiting:
        if len(state.running) + len(state.prefilling) >= MAX_NUM_SEQS:
            return
        if budget <= 0:
            return
        head = state.waiting[0]
        sid = head.session_id
        sess = state.sessions[sid]
        # Partial-prefix hit (block-granular): the session's resident KV covers
        # ``held_tokens`` of this turn's cached prefix; only the EVICTED tail
        # (cached - held_tokens) must be re-prefilled, plus the new tokens. Full hit when
        # held_tokens >= cached (re-prefill new only); full miss when fully evicted.
        resident_prefix = min(head.cached, sess.held_tokens if sess.resident else 0.0)
        reprefill_cached = max(0.0, head.cached - resident_prefix)
        head.is_miss = reprefill_cached > 0.0
        head.remaining_prefill = reprefill_cached + head.new_prefill
        # The turn occupies blocks for its full resident context (cached+new == kv_tokens).
        target_blocks = state.pool.tokens_to_blocks(int(math.ceil(head.kv_tokens)))
        delta = target_blocks - state.pool.num_blocks_owned(sid)
        if delta > 0:
            if not _ensure_free(state, delta, protect_sid=sid):
                return  # can't free enough KV -> head-of-line block
            if not state.pool.allocate(sid, delta):
                return
        state.waiting.pop(0)
        sess.resident = True
        sess.inflight = True
        sess.held_tokens = head.kv_tokens
        sess.last_seq = state.seq
        state.seq += 1
        state.prefilling[head.rid] = head
        budget -= min(head.remaining_prefill, MAX_NUM_BATCHED_TOKENS)


def _on_arrival(state: _ServerState, session_id: int, turn_index: int) -> None:
    sess = state.sessions[session_id]
    spec = sess.turns[turn_index]
    cached = spec.cached_context_tokens
    new = spec.new_prefill_tokens
    req = _Req(
        rid=_encode_rid(session_id, turn_index),
        session_id=session_id,
        turn_index=turn_index,
        arrival_epoch=state.clock,
        cached=cached,
        new_prefill=new,
        output=spec.output_tokens,
        remaining_prefill=new,  # provisional; _schedule sets hit/miss at admission
        output_left=max(1, int(round(spec.output_tokens))),
        kv_tokens=cached + new,
    )
    state.waiting.append(req)
    state.results[(session_id, turn_index)] = {"arrival_epoch": state.clock}
    _schedule(state)
    _ensure_step(state)


def _ensure_step(state: _ServerState) -> None:
    if state.step_scheduled:
        return
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _price_step(state: _ServerState) -> float:
    """One mixed prefill+decode step from measured kernels: ``max(decode_ms, prefill_ms) +
    scheduler_overhead``. Prefilling reqs share ONE fused pass over the per-step token budget."""
    p = state.params
    decode_batch = len(state.running)
    decode_ms = decode_step_ms(decode_batch, _running_ctx_mean(state), p) if decode_batch > 0 else 0.0

    prefill_ms = 0.0
    budget = max(0, MAX_NUM_BATCHED_TOKENS - decode_batch)
    n_pref = len(state.prefilling)
    if n_pref > 0 and budget > 0:
        share = max(1, budget // n_pref)
        total_chunk = 0.0
        prefixes: list[float] = []
        for r in state.prefilling.values():
            chunk = max(1.0, min(r.remaining_prefill, float(share)))
            r._chunk = chunk  # type: ignore[attr-defined]
            total_chunk += chunk
            prefixes.append(max(1.0, r.cached))
        total_chunk = min(total_chunk, float(MAX_NUM_BATCHED_TOKENS))
        mean_prefix = statistics.fmean(prefixes) if prefixes else 1.0
        prefill_ms = _prefill_pass_ms(total_chunk, mean_prefix, p)
    else:
        for r in state.prefilling.values():
            r._chunk = 0.0  # type: ignore[attr-defined]

    return max(decode_ms, prefill_ms) + p.scheduler_overhead_ms_per_step


def _on_step(state: _ServerState) -> None:
    state.step_scheduled = False
    if not state.running and not state.prefilling:
        return

    step_ms = _price_step(state)
    state.clock += step_ms

    # --- prefill bookkeeping: decrement chunks, emit FIRST_TOKEN on completion ---
    finished_prefill: list[int] = []
    for rid, r in state.prefilling.items():
        r.remaining_prefill -= getattr(r, "_chunk", 0.0)
        if r.remaining_prefill <= 0:
            finished_prefill.append(rid)
    for rid in finished_prefill:
        state.push(state.clock, _FIRST_TOKEN, rid)

    # --- decode bookkeeping: one token per running req; the SESSION's KV grows by one token,
    #     allocating a fresh block at block boundaries (evicting LRU resident sessions if the
    #     pool is full). ---
    finished_decode: list[int] = []
    for rid in list(state.running.keys()):
        r = state.running.get(rid)
        if r is None:
            continue
        r.kv_tokens += 1.0
        sess = state.sessions[r.session_id]
        sess.held_tokens = r.kv_tokens
        need = state.pool.tokens_to_blocks(int(math.ceil(r.kv_tokens)))
        grow = need - state.pool.num_blocks_owned(r.session_id)
        if grow > 0 and _ensure_free(state, grow, protect_sid=r.session_id):
            state.pool.allocate(r.session_id, grow)
        r.output_left -= 1
        if r.output_left <= 0:
            finished_decode.append(rid)
    for rid in finished_decode:
        r = state.running.get(rid)
        if r is not None:
            state.push(state.clock, _DEPART, (r.session_id, r.turn_index))

    _schedule(state)
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _on_first_token(state: _ServerState, rid: int) -> None:
    r = state.prefilling.pop(rid, None)
    if r is None:
        return
    # Every turn records its first token on prefill completion. For a MISS turn this is
    # AFTER re-prefilling the full context, so its (higher) TTFT correctly reflects the
    # recompute cost — that is the climb.
    state.results[(r.session_id, r.turn_index)]["first_token_epoch"] = state.clock
    state.running[rid] = r
    _ensure_step(state)


def _on_depart(state: _ServerState, session_id: int, turn_index: int) -> None:
    rid = _encode_rid(session_id, turn_index)
    r = state.running.pop(rid, None)
    sess = state.sessions[session_id]
    sess.inflight = False
    sess.last_seq = state.seq  # recently active -> last to be evicted (LRU)
    state.seq += 1
    if r is not None:
        sess.held_tokens = r.kv_tokens  # session KEEPS its grown KV (resident) for the next turn

    rec = state.results.get((session_id, turn_index))
    if rec is not None:
        rec["completion_epoch"] = state.clock
        rec.setdefault("first_token_epoch", state.clock)

    sess.next_turn_idx = max(sess.next_turn_idx, turn_index + 1)
    if sess.next_turn_idx < sess.turn_count:
        # Closed loop: next turn arrives now (think-time == 0). Session stays resident (holds
        # KV) but is now evictable until that turn is admitted.
        state.push(state.clock, _ARRIVAL, (session_id, sess.next_turn_idx))
    else:
        # Session finished -> free its KV.
        state.pool.free_request(session_id)
        sess.resident = False
        sess.held_tokens = 0.0
    _ensure_step(state)


def _run_sim(
    sessions: list[Session], params: RooflineParams, max_events: int
) -> dict[tuple[int, int], float]:
    pool = BlockPool(params.available_kv_blocks, params.cache_block_size)
    state = _ServerState(params=params, pool=pool, sessions=sessions)

    for s in sessions:
        state.push(0.0, _ARRIVAL, (s.session_id, 0))

    events = 0
    while state.heap and events < max_events:
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

    ttfts: dict[tuple[int, int], float] = {}
    for key, rec in state.results.items():
        if "first_token_epoch" in rec and "arrival_epoch" in rec:
            ttfts[key] = rec["first_token_epoch"] - rec["arrival_epoch"]
    return ttfts


def _aggregate(
    ttfts: dict[tuple[int, int], float],
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams,
) -> list[float]:
    by_idx: dict[int, list[float]] = {}
    for (_sid, ti), v in ttfts.items():
        if v > 0:
            by_idx.setdefault(ti, []).append(v)
    out: list[float] = []
    for t in turns:
        ti = int(t.get("turn_index", 0))
        vals = by_idx.get(ti)
        out.append(statistics.median(vals) if vals else _fallback_ttft(t, profile, concurrency, params))
    return out


def _fallback_ttft(
    turn: dict[str, Any], profile: str, concurrency: float, params: RooflineParams
) -> float:
    """Forward static-formula fallback for a turn no session reached (keeps list length)."""
    from simulator.ramp_tpot import sched_hat

    ti = int(turn.get("turn_index", 0))
    cached = float(turn.get("cached_context_tokens") or 0.0)
    new = float(turn.get("new_prefill_tokens") or 0.0)
    out = float(turn.get("output_tokens") or 1.0)
    sched = sched_hat(profile, float(concurrency), ti) if profile in PROFILE_DIST else float(concurrency)
    tpot = predict_cell_tpot([KernelTurnInput(cached, new, out, sched)], params)[0]
    return predict_turn_ttft(cached, new, out, sched, tpot, params)


# --------------------------------------------------------- oracle (validation only)


def _build_cohort_oracle(
    turns: list[dict[str, Any]], profile: str, concurrency: float
) -> list[Session] | None:
    """Validation-only: build the cohort from measured session_timelines (per-session turn
    lists) instead of the survival quantile. Off the forward path; None if unavailable."""
    try:
        from pathlib import Path

        from profiling.process.extractors.extract_benchmark_per_request import (
            collect_session_timelines,
        )
    except Exception:
        return None
    bench_root = Path(
        "/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm"
    )
    if not bench_root.exists():
        return None
    try:
        timelines = collect_session_timelines(bench_root)
    except Exception:
        return None
    cell = timelines.get(f"{profile}__{int(round(float(concurrency)))}")
    if not cell or not cell.get("sessions"):
        return None
    sessions: list[Session] = []
    for sid, sess_turns in enumerate(cell["sessions"]):
        specs = [
            TurnSpec(
                turn_index=int(tt["turn_index"]),
                cached_context_tokens=float(tt.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(tt.get("new_prefill_tokens") or 0.0),
                output_tokens=max(1.0, float(tt.get("output_tokens") or 1.0)),
            )
            for tt in sess_turns
        ]
        if specs:
            sessions.append(Session(session_id=sid, turn_count=len(specs), turns=specs))
    return sessions or None


# ------------------------------------------------------------------- public API


def predict_cell_ttft_qsim(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams | None = None,
    *,
    oracle: bool = False,
    max_events: int = 4_000_000,
) -> list[float]:
    """Per-turn TTFT (ms) for a (profile, concurrency) cell, emergent from a forward
    closed-loop event-driven queue sim with session-persistent KV + RECOMPUTE eviction.

    Returns one TTFT per input turn (median over sessions reaching that turn_index), aligned
    to ``turns`` order; ``[]`` for empty; a turn_index reached by no session falls back to the
    forward static predictor. Forward by default (cohort from ``forward_survival``);
    ``oracle=True`` overlays measured ``session_timelines`` (validation only)."""
    if not turns:
        return []
    p = params or RooflineParams()

    sessions: list[Session] | None = None
    if oracle:
        sessions = _build_cohort_oracle(turns, profile, float(concurrency))
    if sessions is None:
        sessions = _build_cohort(turns, profile, float(concurrency))

    ttfts = _run_sim(sessions, p, max_events)
    return _aggregate(ttfts, turns, profile, float(concurrency), p)


def predict_cell_e2el_qsim(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    ttft_qsim: list[float],
    tpot_preds: list[float] | None = None,
    params: RooflineParams | None = None,
) -> list[float]:
    """E2EL composition: ``e2el_qsim[t] = ttft_qsim[t] + output_tokens[t] * tpot[t]`` (tpot
    defaults to the kernel TPOT the emitter passes — byte-identical to the ``e2el_pred`` line)."""
    if not turns:
        return []
    p = params or RooflineParams()
    if tpot_preds is None:
        inputs = [
            KernelTurnInput(
                cached_context_tokens=float(t.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(t.get("new_prefill_tokens") or 0.0),
                output_tokens=float(t.get("output_tokens") or 0.0),
                scheduled_requests=float(concurrency),
            )
            for t in turns
        ]
        tpot_preds = predict_cell_tpot(inputs, p)
    out: list[float] = []
    for t, ttft, tpot in zip(turns, ttft_qsim, tpot_preds):
        out.append(float(ttft) + float(t.get("output_tokens") or 0.0) * float(tpot))
    return out
