# simulator_v2/engine/queue_sim.py

"""TTFT via an event-driven queue sim: replays a cohort through vLLM's scheduler
(chunked prefill, decode priority) over a session-persistent KV prefix cache,
timing each request's first token. Step cost = hw.fused_step_ms (no fitted rates).

Under load the cohort's KV overflows the pool -> LRU tail-trim eviction -> evicted
sessions re-prefill next turn (recompute), congesting the prefill budget and driving
the high-concurrency TTFT climb. SHARED across modes; only the cohort source differs.

The API-server frontend (`serving_frontend.py`) delays each ARRIVAL to its drain
epoch at herd release; TTFT stays clocked from the release (= client dispatch).
"""

from __future__ import annotations

import heapq
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from simulator_v2.core.mode import Mode, mode
from simulator_v2.core.types import Hardware, SchedulerSettings, Turn
from simulator_v2.engine.serving_frontend import herd_arrival_epochs

# Event kinds on the heap (ordered by epoch, then insertion seq).
_ARRIVAL, _STEP, _FIRST_TOKEN, _DEPART = 0, 1, 2, 3

_TURNS_PER_SESSION = 4096  # rid = session_id * this + turn_index (ample headroom)
_DEFAULT_MAX_BATCHED = 8192
_DEFAULT_MAX_SEQS = 256
_EVENT_GUARD = 5_000_000  # hard cap on processed events (runaway protection)


# ----------------------------------------------------------------- entities


@mode(Mode.SHARED)
@dataclass
class Session:
    """One session's turn trajectory; the sim replays `concurrency` of these."""
    session_id: int
    turns: list[Turn]
    next_turn_idx: int = 0


@mode(Mode.SHARED)
@dataclass
class _Req:
    """One (session, turn) request as it moves waiting -> prefilling -> running."""
    rid: int
    session_id: int
    turn_index: int
    arrival_epoch: float
    cached: float            # resident prefix it can attend (a cache hit)
    new_prefill: float       # new tokens this turn must prefill
    output: float            # tokens it will generate
    remaining_prefill: float # prefill tokens left (decrements as chunks run)
    output_left: int         # decode tokens left
    kv_tokens: float         # resident KV after prefill (cached + new), grows in decode
    chunk: float = 0.0       # tokens this request prefills in the current step
    is_miss: bool = False        # own prefix was evicted -> re-prefilled cached+new
    resident_prefix: float = 0.0 # cached tokens still resident (a hit); set at admission
    prefill_total: float = 0.0   # GPU prefill work this turn (set at admission); spreads host cost


@mode(Mode.SHARED)
@dataclass
class _ServerState:
    """Mutable simulator state for one run: clock, event heap, the three request
    queues, the barrier-herd bookkeeping, and the per-request results."""
    hw: Hardware
    sched: SchedulerSettings
    sessions: list[Session]
    cache: "PrefixLRUCache"
    by_id: dict[int, Session] = field(default_factory=dict)
    clock: float = 0.0
    seq: int = 0  # monotone tiebreak for the event heap
    heap: list[tuple[float, int, int, Any]] = field(default_factory=list)
    waiting: list[_Req] = field(default_factory=list)          # FIFO admission queue
    prefilling: dict[int, _Req] = field(default_factory=dict)  # rid -> req mid-prefill
    running: dict[int, _Req] = field(default_factory=dict)     # rid -> req decoding
    results: dict[tuple[int, int], float] = field(default_factory=dict)  # (sid, turn) -> ttft_ms
    step_scheduled: bool = False
    current_turn: int = 0     # turn_index of the herd in flight
    herd_remaining: int = 0   # requests of the current herd not yet departed; 0 -> barrier
    herd_pending: set[int] = field(default_factory=set)  # current-herd sessions not yet departed (evict-protected)
    recompute_tokens: float = 0.0  # evicted-prefix tokens re-prefilled (KV-pressure signal)
    # Cross-session APC prefix (deduped): one session pays it, the rest credit it once primed.
    shared_prefix_tokens: float = 0.0
    shared_primed: bool = False

    def push(self, epoch: float, kind: int, payload: Any) -> None:
        """Schedule an event on the heap (clock-ordered, stable by insertion)."""
        heapq.heappush(self.heap, (epoch, self.seq, kind, payload))
        self.seq += 1


@mode(Mode.SHARED)
def _encode_rid(session_id: int, turn_index: int) -> int:
    """Stable per-(session, turn) request id."""
    return session_id * _TURNS_PER_SESSION + turn_index


# ----------------------------------------------------------------- cohort


@mode(Mode.SHARED)
def _build_cohort(
    turns: list[Turn], concurrency: int,
    trajectories: list[list[Turn]] | None = None,
) -> list[Session]:
    """Build `concurrency` Sessions to replay. With real per-session trajectories,
    cycle them to fill the cohort (their varied sizes/turn counts de-synchronize
    the herd). Without, fall back to replicating the cell's median turn sequence."""
    c = max(1, int(concurrency))
    if trajectories:
        return [Session(session_id=i, turns=list(trajectories[i % len(trajectories)]))
                for i in range(c)]
    return [Session(session_id=i, turns=list(turns)) for i in range(c)]


# ----------------------------------------------------------------- scheduler limits


@mode(Mode.SHARED)
def _max_batched(sched: SchedulerSettings) -> float:
    return float(sched.max_num_batched_tokens or _DEFAULT_MAX_BATCHED)


@mode(Mode.SHARED)
def _long_prefill(sched: SchedulerSettings) -> float:
    return float(sched.long_prefill_token_threshold or _max_batched(sched))


@mode(Mode.SHARED)
def _max_seqs(sched: SchedulerSettings) -> int:
    return int(sched.max_num_seqs or _DEFAULT_MAX_SEQS)


# ----------------------------------------------------------------- KV prefix cache


@mode(Mode.SHARED)
class PrefixLRUCache:
    """Session-granular KV prefix cache with global LRU eviction (vLLM block pool +
    APC). Each session's prefix persists across turns; making room trims the
    LRU-oldest sessions' tails, so a victim keeps what survived and re-prefills only
    the lost tail next turn. No fitted constants."""

    def __init__(self, capacity_blocks: int, block_size: int) -> None:
        self.capacity = int(capacity_blocks)
        self.block_size = max(1, int(block_size))
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
        """Mark `sid` most-recently-used (so it is evicted last)."""
        self.recency[sid] = self._tick
        self._tick += 1

    def _trim_tail(self, victims: list[int], need: int) -> None:
        """Partially trim each victim's tail, in order, until `need` blocks are free."""
        for v in victims:
            if self.free() >= need:
                break
            trim = min(self.cached[v], need - self.free())
            self.cached[v] -= trim
            self.evictions += 1
            if self.cached[v] <= 0:
                del self.cached[v]

    def _evict(self, need: int, hard_protect: set[int], soft_protect: set[int]) -> bool:
        """Free `need` blocks LRU-oldest, two tiers: (1) reclaim dead/departed residents
        (rotation buffer), then (2) trim idle herd residents (`soft_protect`) under
        over-subscription. In-flight (`hard_protect`) is never touched."""
        if need <= self.free():
            return True
        free_residents = sorted(
            (s for s in self.cached
             if s not in hard_protect and s not in soft_protect and self.cached[s] > 0),
            key=lambda s: (self.recency.get(s, -1), s),  # oldest first; sid tiebreak
        )
        self._trim_tail(free_residents, need)
        if self.free() >= need:
            return True
        herd_residents = sorted(
            (s for s in self.cached
             if s in soft_protect and s not in hard_protect and self.cached[s] > 0),
            key=lambda s: (self.recency.get(s, -1), s),
        )
        self._trim_tail(herd_residents, need)
        return self.free() >= need

    def grow_to(
        self, sid: int, target_blocks: int, hard_protect: set[int], soft_protect: set[int]
    ) -> bool:
        """Make `sid` resident up to `target_blocks`, reusing its surviving prefix
        and allocating only the delta (evicting LRU-oldest tails to fit). Touches
        `sid` (MRU). Returns False only if the delta can't be freed even after
        eviction (a context larger than the free pool behind pinned in-flight KV)."""
        cur = self.cached.get(sid, 0)
        if target_blocks <= cur:
            self.touch(sid)
            return True
        delta = target_blocks - cur
        if not self._evict(delta, hard_protect | {sid}, soft_protect - {sid}):
            return False
        self.cached[sid] = target_blocks
        self.touch(sid)
        return True


# ----------------------------------------------------------------- scheduling + pricing


@mode(Mode.SHARED)
def _schedule(state: _ServerState) -> None:
    """Admit waiting requests (FIFO): decide hit/miss from the live resident prefix,
    reserve the context blocks (evicting LRU tails to fit), and defer any that don't
    fit yet. Gated by max_num_seqs."""
    if not state.waiting:
        return
    cache = state.cache
    block = cache.block_size
    in_flight = {r.session_id for r in state.prefilling.values()}
    in_flight |= {r.session_id for r in state.running.values()}
    max_seqs = _max_seqs(state.sched)
    deferred: list[_Req] = []
    for head in state.waiting:
        if len(state.running) + len(state.prefilling) >= max_seqs:
            deferred.append(head)
            continue
        sid = head.session_id
        # Resident (a hit): own surviving prefix, or the shared APC prefix once primed.
        # Session storage is NET of the shared span (stored once globally, see below),
        # so the shared span is added back when crediting residency.
        shared = state.shared_prefix_tokens if state.shared_primed else 0.0
        own_resident = min(head.cached, float(cache.cached_blocks(sid) * block) + shared)
        shared_resident = (
            min(state.shared_prefix_tokens, head.cached + head.new_prefill)
            if state.shared_prefix_tokens > 0.0 and state.shared_primed else 0.0
        )
        resident_credit = max(own_resident, shared_resident)
        cached_hit = min(head.cached, resident_credit)
        new_hit = min(head.new_prefill, max(0.0, resident_credit - head.cached))
        reprefill_cached = head.cached - cached_hit
        reprefill_new = head.new_prefill - new_hit
        head.is_miss = reprefill_cached > 0.0
        head.resident_prefix = resident_credit
        head.remaining_prefill = reprefill_cached + reprefill_new
        head.prefill_total = max(1.0, head.remaining_prefill)
        # Reserve the context; evict to fit. NET of the shared prefix: vLLM's APC
        # stores the cross-session span ONCE (block content hash), so charging every
        # session its own copy is phantom demand (concurrency x shared tokens) that
        # fires the eviction cascade turns early. The single shared copy (~a few
        # blocks) is treated as free.
        target_blocks = cache.tokens_to_blocks(
            max(0.0, head.kv_tokens - state.shared_prefix_tokens))
        if not cache.grow_to(sid, target_blocks, in_flight | {sid}, state.herd_pending - {sid}):
            deferred.append(head)
            continue
        state.prefilling[head.rid] = head
        in_flight.add(sid)
        state.recompute_tokens += reprefill_cached  # evicted prefix re-prefilled (the climb)
        if state.shared_prefix_tokens > 0.0:
            state.shared_primed = True  # first admission pays the shared prefix; peers credit it
    state.waiting = deferred


@mode(Mode.SHARED)
def _price_step(state: _ServerState) -> float:
    """Wall-time (ms) of one mixed step: decode takes the token budget first, the rest
    is chunked prefill (capped at long_prefill). GPU cost = decode + chunk prefill +
    cross-context attention, additive (one fused forward pass; FLOPs add). Host cost
    pipelines with the GPU, so the step is max(gpu, host)."""
    decode_batch = len(state.running)
    decode_ctx = (
        statistics.mean(r.kv_tokens for r in state.running.values())
        if state.running else 0.0
    )
    budget = max(0.0, _max_batched(state.sched) - decode_batch)
    cap = _long_prefill(state.sched)
    shared_rate, perreq_rate, new_rate = state.hw.prefill_host_rates
    cross_rate = state.hw.cross_attn_ms_per_token_pair
    total_chunk = 0.0
    cross_ms = 0.0       # chunk-vs-resident-context attention (measured slope x U x P)
    cached_w_sum = 0.0   # Sum cached_i * frac_i (frac-weighted, for the once-per-step shared host)
    cached_w_n = 0
    perreq_host_ms = 0.0
    for req in state.prefilling.values():
        chunk = min(req.remaining_prefill, cap, budget)
        chunk = max(0.0, chunk)
        req.chunk = chunk
        budget -= chunk
        total_chunk += chunk
        if chunk > 0.0:
            # The chunk attends everything already resident for this request (its
            # hit prefix + previously completed chunks); the full-causal chunk grid
            # prices the chunk alone, so the cross term is added per request.
            resident = max(0.0, req.kv_tokens - req.remaining_prefill)
            cross_ms += cross_rate * chunk * resident
            frac = chunk / req.prefill_total if req.prefill_total > 0 else 1.0
            perreq_host_ms += perreq_rate * req.cached * frac
            cached_w_sum += req.cached * frac
            cached_w_n += 1
    if total_chunk <= 0.0:
        return state.hw.fused_step_ms(0, decode_batch, decode_ctx)
    # Host serving cost (re-tokenize + dispatch): CPU work that pipelines with the GPU,
    # so the step is max(gpu, host) -- host-bound for cheap hits, hidden under big
    # recomputes. 0 rates -> byte-identical.
    mean_cached = cached_w_sum / cached_w_n if cached_w_n else 0.0
    host_ms = shared_rate * mean_cached + perreq_host_ms + new_rate * total_chunk
    decode_ms = state.hw.decode_step_ms(decode_batch, decode_ctx)
    prefill_gpu_ms = state.hw.fused_step_ms(int(total_chunk), 0, 0) + cross_ms
    return max(decode_ms + prefill_gpu_ms, host_ms)


# ----------------------------------------------------------------- event handlers


@mode(Mode.SHARED)
def _ensure_step(state: _ServerState) -> None:
    """Make sure a STEP is queued at the current clock when work remains."""
    if not state.step_scheduled and (state.waiting or state.prefilling or state.running):
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


@mode(Mode.SHARED)
def _on_arrival(
    state: _ServerState, session_id: int, turn_index: int,
    release_epoch: float | None = None,
) -> None:
    """A request reaches the engine (post-frontend): build its _Req and enqueue it.
    TTFT clocks from `release_epoch`, not the frontend-drained engine arrival."""
    session = state.by_id[session_id]
    turn = session.turns[turn_index]
    new_prefill = max(0.0, float(turn.new_prefill_tokens))
    output = max(1, int(round(turn.osl_tokens)))
    arrival = state.clock if release_epoch is None else float(release_epoch)
    state.waiting.append(_Req(
        rid=_encode_rid(session_id, turn_index),
        session_id=session_id, turn_index=turn_index, arrival_epoch=arrival,
        cached=float(turn.cache_hit_tokens), new_prefill=new_prefill, output=float(output),
        remaining_prefill=new_prefill, output_left=output,
        kv_tokens=float(turn.cache_hit_tokens) + new_prefill,
    ))
    _ensure_step(state)


@mode(Mode.SHARED)
def _on_step(state: _ServerState) -> None:
    """Advance one engine step: admit, price it (advances the clock), decrement
    prefill chunks / decode tokens, emit FIRST_TOKEN on prefill completion and
    DEPART on the last output token, then reschedule."""
    state.step_scheduled = False
    _schedule(state)
    if not state.prefilling and not state.running:
        return
    state.clock += _price_step(state)

    for rid in list(state.prefilling):
        req = state.prefilling[rid]
        req.remaining_prefill -= req.chunk
        if req.remaining_prefill <= 1e-9:
            state.push(state.clock, _FIRST_TOKEN, rid)

    decode_in_flight = (
        {r.session_id for r in state.prefilling.values()}
        | {r.session_id for r in state.running.values()}
    )
    for req in state.running.values():
        req.output_left -= 1
        req.kv_tokens += 1.0
        # Decode grows KV; reserve blocks -- NET of the shared span, matching
        # _schedule's accounting (gross here would re-claim the shared copy per
        # session every turn, silently undoing the dedup).
        need = state.cache.tokens_to_blocks(
            max(0.0, req.kv_tokens - state.shared_prefix_tokens))
        if need > state.cache.cached_blocks(req.session_id):
            state.cache.grow_to(req.session_id, need, decode_in_flight, state.herd_pending)
        if req.output_left <= 0:
            state.push(state.clock, _DEPART, (req.session_id, req.turn_index))

    _ensure_step(state)


@mode(Mode.SHARED)
def _host_floor(state: _ServerState) -> float:
    """Per-request host overhead (ms) added to TTFT once at first token (tokenize +
    dispatch + detok + return) -- a measured config value, not composed."""
    return float(state.hw.request_overhead_ms)


@mode(Mode.SHARED)
def _on_first_token(state: _ServerState, rid: int) -> None:
    """A request finished prefill: record its TTFT and move it to `running`."""
    req = state.prefilling.pop(rid, None)
    if req is None:
        return
    ttft = state.clock - req.arrival_epoch + _host_floor(state)
    state.results[(req.session_id, req.turn_index)] = ttft
    state.running[rid] = req
    _ensure_step(state)


@mode(Mode.SHARED)
def _on_depart(state: _ServerState, session_id: int, turn_index: int) -> None:
    """A request finished its last output token: free it. When the whole herd has
    departed (barrier), release the next turn's herd."""
    state.running.pop(_encode_rid(session_id, turn_index), None)
    state.herd_pending.discard(session_id)  # done this turn -> evictable (rotation buffer)
    state.cache.touch(session_id)           # just finished -> MRU; KV persists across the barrier
    state.herd_remaining -= 1
    if state.herd_remaining <= 0:
        _release_next_herd(state)
    _ensure_step(state)


@mode(Mode.SHARED)
def _release_next_herd(state: _ServerState) -> None:
    """Barrier round-robin: release the next turn's herd at the current clock;
    ARRIVALs land at their frontend-drain epochs."""
    nxt = state.current_turn + 1
    arrivals = [s for s in state.sessions if len(s.turns) > nxt]
    if not arrivals:
        return
    state.current_turn = nxt
    state.herd_remaining = len(arrivals)
    state.herd_pending = {s.session_id for s in arrivals}
    epochs = herd_arrival_epochs(
        state.hw.frontend, [s.turns[nxt] for s in arrivals], state.clock)
    for s, epoch in zip(arrivals, epochs):
        state.push(epoch, _ARRIVAL, (s.session_id, nxt, state.clock))


# ----------------------------------------------------------------- driver


@mode(Mode.SHARED)
def _run_sim(state: _ServerState) -> dict[tuple[int, int], float]:
    """Pop events until the heap drains, dispatching each to its handler. Returns
    the TTFT (ms) per (session_id, turn_index)."""
    guard = 0
    while state.heap:
        epoch, _, kind, payload = heapq.heappop(state.heap)
        state.clock = max(state.clock, epoch)
        if kind == _STEP:
            _on_step(state)
        elif kind == _ARRIVAL:
            _on_arrival(state, *payload)
        elif kind == _FIRST_TOKEN:
            _on_first_token(state, payload)
        elif kind == _DEPART:
            _on_depart(state, *payload)
        guard += 1
        if guard > _EVENT_GUARD:
            break
    return state.results


@mode(Mode.SHARED)
def _aggregate(
    raw: dict[tuple[int, int], float], turns: list[Turn], cohort: list[Session]
) -> list[float]:
    """Reduce raw per-request TTFTs into a per-turn-index TTFT (ms), aligned to
    `turns` -- the median over the cohort's requests at each turn index."""
    out: list[float] = []
    for ti in range(len(turns)):
        vals = [raw[(s.session_id, ti)] for s in cohort if (s.session_id, ti) in raw]
        out.append(statistics.median(vals) if vals else 0.0)
    return out


@mode(Mode.SHARED)
def predict_ttft(
    hw: Hardware, turns: list[Turn], concurrency: int, *,
    trajectories: list[list[Turn]] | None = None,
    shared_prefix_tokens: float = 0.0,
    stats: dict | None = None,
) -> list[float]:
    """Per-turn TTFT (ms) for one cell: build the cohort, run the queue sim, aggregate
    to per-turn-index medians. `stats` (optional) is filled with {evictions,
    recompute_tokens} diagnostics."""
    if not turns:
        return []
    cohort = _build_cohort(turns, concurrency, trajectories)
    state = _ServerState(
        hw=hw, sched=hw.sched, sessions=cohort,
        cache=PrefixLRUCache(hw.kv_pool_blocks, hw.cache_block_size),
        by_id={s.session_id: s for s in cohort},
        shared_prefix_tokens=max(0.0, float(shared_prefix_tokens)),
    )
    # Release the turn-0 herd at t=0 (synchronized).
    arrivals = [s for s in cohort if s.turns]
    state.herd_remaining = len(arrivals)
    state.herd_pending = {s.session_id for s in arrivals}
    epochs = herd_arrival_epochs(hw.frontend, [s.turns[0] for s in arrivals], 0.0)
    for s, epoch in zip(arrivals, epochs):
        state.push(epoch, _ARRIVAL, (s.session_id, 0, 0.0))

    raw = _run_sim(state)
    if stats is not None:
        stats["evictions"] = state.cache.evictions
        stats["recompute_tokens"] = state.recompute_tokens
    return _aggregate(raw, turns, cohort)
