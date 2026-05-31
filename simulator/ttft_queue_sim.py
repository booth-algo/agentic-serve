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
  (``PrefixLRUCache(available_kv_blocks=27250, 16)``) + ``max_num_seqs`` + ``MAX_NUM_BATCHED_TOKENS``
  (vLLM serving defaults, documented config — not MAPE knobs).
* **Block-level prefix cache (``PrefixLRUCache``): a session's cached PREFIX persists across
  turns AND across eviction.** Making room reuses the globally-LRU-oldest cached blocks,
  trimming a victim's prefix from its TAIL; the victim reclaims whatever survived on its next
  turn (HIT) and re-prefills only the trimmed tail — NOT its whole context unless every block
  was reused. A dead session is never freed; its blocks are LRU-oldest so they evict first
  (the buffer that shields active sessions). This retention is what makes a draining cohort's
  re-prefills cheap → the measured TTFT RECOVERY; heavy over-subscription churns all blocks →
  full re-prefills → the PEAK.
* Barrier round-robin (matches the harness ``run_multi_turn_benchmark``: all sessions' turn-N
  requests dispatched together, ``asyncio.gather`` between turns): turn ``t``'s ENTIRE herd of
  surviving sessions arrives at the SAME epoch; turn ``t+1`` is released only after EVERY
  turn-``t`` request departs. Per-turn TTFT is the queue wait of C contemporaneous arrivals.
  HIT (session resident, KV covers cached) → prefill only ``new``; MISS → prefill ``cached+new``.
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

from simulator.cached_prefill_lookup import cached_prefill_step_ms
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator.ramp_tpot import PROFILE_DIST, forward_survival
from simulator.ttft_predict import _prefill_per_token_ms, predict_turn_ttft

__all__ = ["predict_cell_ttft_qsim", "predict_cell_e2el_qsim", "PROFILE_DIST"]

# --- vLLM serving defaults (documented runtime config, NOT fitted) ------------
# Resolved by vLLM EngineArgs for H100 + OPENAI_API_SERVER (arg_utils._set_default_args,
# device_memory>=70GiB & not-A100): max_num_batched_tokens=8192, max_num_seqs=1024. The
# benchmark launched with these unset (server metadata: both null) so these resolved
# defaults are what actually ran.
MAX_NUM_SEQS = 1024         # vLLM H100 OPENAI_API_SERVER resolved default (running-set cap)
MAX_NUM_BATCHED_TOKENS = 8192  # vLLM H100 OPENAI_API_SERVER resolved default (per-step token budget)
# vLLM v1 caps EACH prefill (fresh OR resumed RECOMPUTE re-prefill) at
# ``long_prefill_token_threshold`` tokens per step when chunked prefill is on; with the
# threshold unset, SchedulerConfig sets it to ``int(max_model_len * 0.04)``
# (vllm/config/scheduler.py). The benchmark ran max_model_len=32768 (recorded in server
# metadata) -> 1310. So many prefills advance CONCURRENTLY (~budget/threshold of them),
# each by a bounded chunk, rather than one big re-prefill monopolizing the budget. This is
# the de-serialization that keeps a long re-prefill from head-of-line-blocking cheap turns.
# Config-derived (max_model_len x vLLM's 0.04), NOT a fitted constant.
MAX_MODEL_LEN = 32768
LONG_PREFILL_TOKEN_THRESHOLD = int(MAX_MODEL_LEN * 0.04)  # = 1310

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


# ------------------------------------------------------- block-level prefix cache


class PrefixLRUCache:
    """Block-level KV prefix cache with global LRU eviction (vLLM v1 BlockPool + APC,
    modeled at session granularity).

    Each session's cached PREFIX (a number of contiguous blocks from the start of its
    context) persists across turns AND across eviction. Making room for a new allocation
    reuses the globally-LRU-OLDEST cached blocks, trimming a victim session's prefix from its
    TAIL (the most-recent, least-shared end). A session reclaims whatever prefix SURVIVED on
    its next turn (a cache HIT) and re-prefills only the trimmed tail — never its whole
    context unless every block was physically reused.

    This is what makes the measured saturate-ramp-RECOVER emerge with the right magnitude: a
    MILDLY over-subscribed cohort (the drained tail) reuses few blocks per turn, so sessions
    keep almost all their prefix → cheap hits → TTFT recovers; a HEAVILY over-subscribed one
    (the peak) churns the whole pool → full re-prefills → TTFT peaks. Replacing the previous
    free-on-evict pool (which reset a victim to 0 blocks → full-re-prefill cascade) with this
    retention model is the fix for the osworld over-prediction. Capacity and block size are
    vLLM config (``available_kv_blocks`` / ``cache_block_size``); NO fitted constants."""

    def __init__(self, capacity_blocks: int, block_size: int = 16) -> None:
        self.capacity = int(capacity_blocks)
        self.block_size = int(block_size)
        self.cached: dict[int, int] = {}   # session_id -> resident prefix blocks
        self.recency: dict[int, int] = {}  # session_id -> last-touch tick (LRU key)
        self._tick = 0
        self.evictions = 0

    def tokens_to_blocks(self, num_tokens: float) -> int:
        return int(math.ceil(max(0.0, float(num_tokens)) / self.block_size))

    def cached_blocks(self, sid: int) -> int:
        return self.cached.get(sid, 0)

    def used(self) -> int:
        return sum(self.cached.values())

    def free(self) -> int:
        return self.capacity - self.used()

    def touch(self, sid: int) -> None:
        """Mark ``sid`` most-recently-used (so it is evicted LAST)."""
        self.recency[sid] = self._tick
        self._tick += 1

    def _evict(self, need: int, protect: set[int]) -> bool:
        """Free ``need`` physical blocks by trimming the TAIL blocks of LRU-oldest sessions
        (skipping ``protect`` = in-flight, whose blocks are in active use). Returns True once
        enough is free, False if no evictable blocks remain (-> head-of-line block)."""
        if need <= self.free():
            return True
        victims = sorted(
            (s for s in self.cached if s not in protect and self.cached[s] > 0),
            key=lambda s: (self.recency.get(s, -1), s),  # oldest first; sid tiebreak (determinism)
        )
        for v in victims:
            if self.free() >= need:
                break
            trim = min(self.cached[v], need - self.free())
            self.cached[v] -= trim
            self.evictions += 1
            if self.cached[v] <= 0:
                del self.cached[v]
        return self.free() >= need

    def grow_to(self, sid: int, target_blocks: int, protect: set[int]) -> bool:
        """Make ``sid`` resident up to ``target_blocks``, RECLAIMING its surviving prefix and
        allocating only the delta (evicting LRU others if needed). Touches ``sid`` (MRU).
        Returns False (HOL block) if the delta cannot be freed. Context only grows, so a
        target below the current residency just keeps the larger residency."""
        cur = self.cached.get(sid, 0)
        if target_blocks <= cur:
            self.touch(sid)
            return True
        delta = target_blocks - cur
        if not self._evict(delta, protect | {sid}):
            return False
        self.cached[sid] = target_blocks
        self.touch(sid)
        return True


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
    # KV residency now lives in the shared PrefixLRUCache (keyed by session_id); eviction
    # protection for the current herd lives in _ServerState.herd_pending.


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
    cache: PrefixLRUCache
    sessions: list[Session]
    clock: float = 0.0
    seq: int = 0  # monotone tiebreak for the event heap
    heap: list[tuple[float, int, int, Any]] = field(default_factory=list)
    waiting: list[_Req] = field(default_factory=list)        # FIFO
    prefilling: dict[int, _Req] = field(default_factory=dict)
    running: dict[int, _Req] = field(default_factory=dict)
    herd_pending: set[int] = field(default_factory=set)      # current-herd sessions not yet departed (evict-protected)
    results: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    step_scheduled: bool = False
    # --- barrier round-robin state (matches the benchmark harness, see _release_herd) ---
    current_turn: int = 0       # turn_index of the herd currently in flight
    herd_remaining: int = 0     # requests of the current herd not yet departed; 0 -> barrier

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


def _schedule(state: _ServerState) -> None:
    """Admission for the current herd. The hit/miss decision is made from the block-level
    prefix cache: the session's SURVIVING resident prefix is a HIT (re-prefill only the
    evicted tail + new tokens); a fully-evicted session is a full MISS. Reserve the full
    context blocks, RECLAIMING the surviving prefix and evicting ONLY non-herd cache — dead
    sessions or sessions that already completed THIS round — never a herd member still
    awaiting its turn (a hit-to-be), which is what kept the cohort from cascading to 100%
    miss. A head that can't get blocks yet is DEFERRED (skipped, stays waiting) and retried
    once a completion frees blocks — so hits run while misses wait, and the resident set
    ROTATES (the saturate-ramp-RECOVER). Also gated by the per-step token budget + max_seqs."""
    if not state.waiting:
        return
    decode_slots = len(state.running)
    budget = MAX_NUM_BATCHED_TOKENS - decode_slots
    for r in state.prefilling.values():
        budget -= min(r.remaining_prefill, LONG_PREFILL_TOKEN_THRESHOLD)
    cache = state.cache
    deferred: list[_Req] = []
    for head in state.waiting:
        if budget <= 0 or len(state.running) + len(state.prefilling) >= MAX_NUM_SEQS:
            deferred.append(head)
            continue
        sid = head.session_id
        # Block-level prefix-cache hit: the surviving resident prefix covers up to
        # ``cached_blocks * block_size`` tokens of this turn's cached context; only the
        # EVICTED tail of that prefix plus the new tokens must be (re-)prefilled.
        resident_prefix = min(head.cached, cache.cached_blocks(sid) * cache.block_size)
        reprefill_cached = max(0.0, head.cached - resident_prefix)
        head.is_miss = reprefill_cached > 0.0
        head.remaining_prefill = reprefill_cached + head.new_prefill
        target_blocks = cache.tokens_to_blocks(head.kv_tokens)
        # Reserve the full context, reclaiming the surviving prefix; evict only NON-herd cache
        # (``herd_pending`` is protected). If the delta can't be freed yet, DEFER this head and
        # retry once a completion frees blocks (rotation) — do NOT block the whole queue.
        if not cache.grow_to(sid, target_blocks, state.herd_pending):
            deferred.append(head)
            continue
        state.prefilling[head.rid] = head
        budget -= min(head.remaining_prefill, LONG_PREFILL_TOKEN_THRESHOLD)
    state.waiting = deferred


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

    # vLLM v1 consumes the per-step token budget GREEDILY in FIFO order (running/admission
    # order), each prefill taking up to ``long_prefill_token_threshold`` tokens, until the
    # budget is exhausted; later prefills stall this step (chunk 0). Decodes (1 token each)
    # consume budget first. So several prefills advance concurrently by a bounded chunk —
    # NOT one big re-prefill monopolizing the pass (which over-serialized and over-charged).
    prefill_ms = 0.0
    budget = max(0, MAX_NUM_BATCHED_TOKENS - decode_batch)
    total_chunk = 0.0
    prefixes: list[float] = []
    for r in state.prefilling.values():  # dict insertion order == FIFO admission order
        chunk = min(r.remaining_prefill, float(LONG_PREFILL_TOKEN_THRESHOLD), float(budget))
        if chunk <= 0:
            r._chunk = 0.0  # type: ignore[attr-defined]
            continue
        r._chunk = chunk  # type: ignore[attr-defined]
        budget -= chunk
        total_chunk += chunk
        prefixes.append(max(1.0, r.cached))
    if total_chunk > 0:
        mean_prefix = statistics.fmean(prefixes)
        prefill_ms = _prefill_pass_ms(total_chunk, mean_prefix, p)

    return max(decode_ms, prefill_ms) + p.scheduler_overhead_ms_per_step


def _on_step(state: _ServerState) -> None:
    state.step_scheduled = False
    if not state.running and not state.prefilling:
        return

    step_ms = _price_step(state)
    state.clock += step_ms

    # --- prefill bookkeeping: decrement chunks (blocks were reserved at admission),
    #     emit FIRST_TOKEN on completion. ---
    finished_prefill: list[int] = []
    for rid, r in state.prefilling.items():
        r.remaining_prefill -= getattr(r, "_chunk", 0.0)
        if r.remaining_prefill <= 1e-9:
            finished_prefill.append(rid)
    for rid in finished_prefill:
        state.push(state.clock, _FIRST_TOKEN, rid)

    # --- decode bookkeeping: one token per running req; the SESSION's KV grows by one token,
    #     reserving a fresh block at block boundaries (evicting only non-herd cache). ---
    finished_decode: list[int] = []
    for rid in list(state.running.keys()):
        r = state.running.get(rid)
        if r is None:
            continue
        r.kv_tokens += 1.0
        need = state.cache.tokens_to_blocks(r.kv_tokens)
        if need > state.cache.cached_blocks(r.session_id):
            state.cache.grow_to(r.session_id, need, state.herd_pending)  # best-effort
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
    state.running.pop(rid, None)
    sess = state.sessions[session_id]
    state.herd_pending.discard(session_id)  # completed this round -> now evictable (rotation)
    state.cache.touch(session_id)  # just finished a turn -> MRU (evicted last); KV persists

    rec = state.results.get((session_id, turn_index))
    if rec is not None:
        rec["completion_epoch"] = state.clock
        rec.setdefault("first_token_epoch", state.clock)

    sess.next_turn_idx = max(sess.next_turn_idx, turn_index + 1)
    # Barrier round-robin (matches the harness: asyncio.gather() between turns, see
    # _release_herd). The next turn's herd is released only after EVERY request in the
    # current turn has departed. The session keeps its KV resident across the barrier.
    state.herd_remaining -= 1
    if state.herd_remaining <= 0:
        _advance_herd(state)
    _ensure_step(state)


def _release_herd(state: _ServerState, turn_idx: int) -> None:
    """Release turn ``turn_idx``'s synchronized **herd**: EVERY surviving session (one with
    ``turn_count > turn_idx``) arrives at the SAME epoch (the barrier-release time).

    This mirrors the benchmark harness exactly (``run_multi_turn_benchmark``): interleaved
    round-robin — all sessions' turn-N requests are dispatched together and ``asyncio.gather``
    waits for the whole turn before turn N+1. So per-turn TTFT is dominated by the queue wait
    of C contemporaneous arrivals, not by per-session spacing — the missing low-turn climb."""
    herd = [s for s in state.sessions if s.turn_count > turn_idx]
    state.current_turn = turn_idx
    state.herd_remaining = len(herd)
    # Every herd member is evict-PROTECTED until it departs: a re-prefilling miss must not
    # trim a herd member still awaiting its turn (that member is a hit-to-be). Only dead /
    # already-completed-this-round sessions are evictable.
    state.herd_pending = {s.session_id for s in herd}
    for s in herd:
        state.push(state.clock, _ARRIVAL, (s.session_id, turn_idx))


def _advance_herd(state: _ServerState) -> None:
    """Barrier reached (all of the current turn departed): release the next turn's herd.

    A session whose conversation has ENDED is NOT freed — vLLM does not proactively release a
    finished request's KV blocks; with prefix caching on they stay cache-resident under LRU
    and are reclaimed only when another allocation needs the space. Because a dead session is
    never touched again, its blocks are the LRU-oldest, so the cache evicts THEM first — they
    are the eviction buffer that shields still-active sessions' prefixes. (With the
    retention-correct PrefixLRUCache this finally works; the old free-on-evict pool would have
    cascaded to full re-prefills here regardless.)"""
    next_turn = state.current_turn + 1
    if any(s.turn_count > next_turn for s in state.sessions):
        _release_herd(state, next_turn)


def _run_sim(
    sessions: list[Session], params: RooflineParams, max_events: int
) -> dict[tuple[int, int], float]:
    cache = PrefixLRUCache(params.available_kv_blocks, params.cache_block_size)
    state = _ServerState(params=params, cache=cache, sessions=sessions)

    _release_herd(state, 0)  # turn-0 herd: all sessions arrive at epoch 0

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
