"""Forward, closed-loop, event-driven multi-turn TTFT **queue** simulator with
**session-persistent KV + RECOMPUTE preemption** (vLLM v1 model).

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
  the profile turn-count survival histogram (``ramp_tpot.forward_survival`` / ``PROFILE_DIST``,
  the REALIZED success-filtered distribution) DETERMINISTICALLY by quantile ``q_k=(k+0.5)/C``
  (reproducible; no RNG, no wall-clock reads). Each session ALSO gets a measured per-session
  context-size SCALE (``ramp_tpot.context_scale_quantiles``) applied to the median trajectory,
  so the cohort's KV working set has the real SPREAD — small sessions stay cache-resident
  (hits) while the large minority is evicted, keeping the MEDIAN session a hit near the pool
  cliff (the osworld saturate-RECOVER). Survival + scale are measured WORKLOAD properties.
* Shared GPU continuous-batching steps; admission gated by KV blocks
  (``PrefixLRUCache(available_kv_blocks=27250, 16)``) + ``max_num_seqs`` + ``MAX_NUM_BATCHED_TOKENS``
  (vLLM serving defaults, documented config — not MAPE knobs).
* **Block-level prefix cache (``PrefixLRUCache``) with two-tier eviction.** A session's cached
  PREFIX persists across turns AND across eviction. Tier 1 reclaims FREE residents (departed/dead
  sessions' blocks — the LRU buffer that shields active sessions). Tier 2, only under GENUINE
  over-subscription (the cohort KV exceeds the pool so tier 1 can't satisfy admission), PREEMPTS
  idle herd residents by ``preempt_policy`` (``'tail'`` = most-recently-used first, vLLM v1
  RECOMPUTE tail-preempt ≫ ``'lru'``), evicting WHOLE sessions so the overflow concentrates on a
  full-miss minority (median stays a hit). In-flight KV (a req prefilling/decoding THIS step) is
  never evicted. Without tier 2 the sim DEADLOCKS at high concurrency (whole herd protected → no
  admission → silent fallback to the static formula); without whole-session eviction the trim
  spreads thin and every session becomes a partial miss (osworld 2× over-count). A turn's hit/miss
  is FROZEN at barrier release (``resident_at_barrier``) so a peer's in-pass eviction can't cascade
  it into a spurious miss. Light over-subscription → cheap hits → TTFT RECOVERS; heavy → full
  re-prefills → the PEAK.
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
from simulator.ramp_tpot import PROFILE_DIST, context_scale_quantiles, forward_survival
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

# --- PREFILL COST: measured-serving anchors + pipeline FA3 kernel -------------
# TTFT prefill cost has three measured parts (the H100 HIT-vs-MISS profile settled the
# physics: a cache HIT SKIPS the GPU prefill of cached tokens, so the per-cached-token cost
# is HOST work — re-tokenize/hash the re-sent conversation — NOT a GPU kernel; the GPU
# prefills only the new/re-prefilled tokens):
#
# 1. NEW (serving per-(re)prefilled-token, 0.0310 ms/tok measured at c1): SPLIT into a DERIVED,
#    tensor-parallel-aware GEMM roofline (``_prefill_gemm_per_tok`` = 2·(params/tp)/tok at
#    util_flops=0.65 → 0.02498 ms/tok on tp1) PLUS a small per-token off-GPU dispatch residual
#    (0.00602 ms/tok). NEW is 1.24× the realistic roofline — NOT "7×" (the retired comment
#    compared to a util=1 / large-K kernel 0.0042; corrected by the de-fit audit). The residual
#    is framework dispatch (ATen/CUDA-library/launch, per TaxBreak ISPASS'26), a backed-out
#    remainder PENDING the host-vs-device stage-split microbench. The GEMM part is now fit-free
#    and tp-scales; see profiling/docs/prefill_law_defit_trace.md.
# 2. FA3 (pipeline attention kernel, 8.31e-7 ms/token^2): from fa3_prefill_H100.csv
#    (FA3(8192)=27.9ms / (8192²/2)). Adds the SUPER-LINEAR attention growth — negligible for a
#    HIT (Q=new small), the quadratic re-encode for a MISS. Extra physical grounding at ~no
#    accuracy cost (the serving re-prefill is ~linear; FA3 is small vs chunked+host overhead).
# 3. HOST (re-tokenize the re-sent cached context, 0.006103 ms/1k total): the dominant HIT
#    cost. Batch split measured on this H100 (cached_prefill_batch_ttft_H100.csv:
#    TTFT(16,P)/TTFT(1,P)≈6.6-7.5) → ~57% amortized once per step + ~43% per request.
# All three are MEASURED (c1 + controlled serving sweeps + the pipeline FA3 grid) — held out
# from the multi-turn data we report.
PREFILL_FLOOR_MS = 22.5                          # fixed per-request cost (schedule+first-token+detok+return); genuine intercept ~= min pure-prefill TTFT. PENDING microbench (new=1,cached=0).
# NEW = DERIVED tp-aware GEMM roofline (``_prefill_gemm_per_tok``, below) + this off-GPU dispatch
# residual. On tp1 their sum reproduces the retired fitted 0.0310 rate to 5-digit rounding
# (~2e-6 ms/tok; TTFT/E2EL gates unchanged). On tp2 the GEMM part halves → tp2 prefill is no
# longer tp1-anchored (TTFT 43.3→31.7% cell-MAPE).
PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN = 0.00602
PREFILL_FA3_MS_PER_TOKEN2 = 8.31e-7             # pipeline FA3 attention kernel, ms per token^2
PREFILL_HOST_SHARED_MS_PER_TOKEN = 0.003485     # host re-tokenize, amortized once per step (0.571×6.103e-3)
PREFILL_HOST_PERREQ_MS_PER_TOKEN = 0.002618     # host re-tokenize, per request, summed (0.429×6.103e-3)


def _prefill_gemm_per_tok(p: RooflineParams) -> float:
    """DERIVED compute-bound prefill GEMM time per (re)prefilled token: 2·(n_params/tp) FLOPs
    per token at ``peak_flops·util_flops``, tensor-parallel sharded. The fit-free dominant part
    of the serving NEW rate — 0.02498 ms/tok on tp1, halving per added TP rank. No fitted constant."""
    tp = max(1, int(getattr(p, "tensor_parallel", 1)))
    return 2.0 * (float(p.n_params) / tp) / (p.peak_flops_per_s * p.util_flops) * 1e3


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

    def _trim_tail(self, sids: list[int], need: int, whole: bool = False) -> None:
        """Free blocks by evicting ``sids`` (in the given order) until ``need`` blocks are free.
        ``whole=True`` evicts each victim's ENTIRE resident prefix (concentrating the overflow
        onto a minority of full-miss sessions, so the MEDIAN session stays a full hit — the
        measured osworld saturate-recover); ``whole=False`` trims only the marginal tail."""
        for v in sids:
            if self.free() >= need:
                break
            trim = self.cached[v] if whole else min(self.cached[v], need - self.free())
            self.cached[v] -= trim
            self.evictions += 1
            if self.cached[v] <= 0:
                del self.cached[v]

    def _evict(
        self, need: int, hard_protect: set[int], soft_protect: set[int], policy: str = "tail"
    ) -> bool:
        """Free ``need`` physical blocks in two tiers:

        1. **Reclaim free residents** — sessions in NEITHER protect set (departed/dead
           sessions' residual prefix), trimmed LRU-oldest-first. This is the rotation buffer.
        2. **Preempt under genuine over-subscription** — if the cohort's persistent KV fills
           the pool so tier 1 can't satisfy ``need``, evict ``soft_protect`` (herd members not
           in-flight = idle resident hits-to-be) by ``policy``: ``'tail'`` = most-recently-used
           first (vLLM v1 RECOMPUTE tail-preempt), ``'lru'`` = oldest-first (prefix-cache LRU).
           A preempted session re-prefills its trimmed tail on its turn (a MISS) — the climb.

        ``hard_protect`` (a req in-flight THIS step, KV pinned) is never evicted. Returns True
        once ``need`` is free, False only if even preempting every idle resident is not enough
        (a single over-large head behind pinned in-flight KV — deferred, retried on completion)."""
        if need <= self.free():
            return True
        free_residents = sorted(
            (s for s in self.cached
             if s not in hard_protect and s not in soft_protect and self.cached[s] > 0),
            key=lambda s: (self.recency.get(s, -1), s),  # oldest first; sid tiebreak (determinism)
        )
        self._trim_tail(free_residents, need)
        if self.free() >= need:
            return True
        # tier 2: genuine over-subscription -> preempt idle herd residents (vLLM RECOMPUTE),
        # evicting WHOLE sessions (concentrate the overflow on a full-miss minority) so the
        # median session stays a hit — without this the trim spreads thin and every session
        # becomes a partial miss (osworld ~2x over-count).
        soft = [
            s for s in self.cached
            if s in soft_protect and s not in hard_protect and self.cached[s] > 0
        ]
        soft.sort(key=lambda s: (self.recency.get(s, -1), s), reverse=(policy == "tail"))
        self._trim_tail(soft, need, whole=True)
        return self.free() >= need

    def grow_to(
        self,
        sid: int,
        target_blocks: int,
        hard_protect: set[int],
        soft_protect: set[int],
        policy: str = "tail",
    ) -> bool:
        """Make ``sid`` resident up to ``target_blocks``, RECLAIMING its surviving prefix and
        allocating only the delta (reclaiming free residents, then preempting idle herd
        residents under over-subscription — see ``_evict``). Touches ``sid`` (MRU). Returns
        False (HOL block) only if the delta cannot be freed even after preemption. Context only
        grows, so a target below the current residency just keeps the larger residency."""
        cur = self.cached.get(sid, 0)
        if target_blocks <= cur:
            self.touch(sid)
            return True
        delta = target_blocks - cur
        if not self._evict(delta, hard_protect | {sid}, soft_protect - {sid}, policy):
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
    resident_prefix: float = 0.0  # cached tokens that are a HIT (attended, not re-prefilled); set at admission
    prefill_total: float = 0.0    # total tokens to (re-)prefill this turn; set at admission (for chunk fraction)


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
    preempt_policy: str = "tail"  # over-subscription victim: 'tail' (vLLM RECOMPUTE) or 'lru'
    resident_at_barrier: dict[int, int] = field(default_factory=dict)  # sid -> resident blocks at herd release (hit/miss frozen here)

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

    # Per-session context-size SCALE (measured workload spread): each cohort session runs
    # systematically larger/smaller contexts than the median, so the KV working set has the
    # real spread — small sessions stay resident (hits) while the large minority is evicted,
    # keeping the MEDIAN session a hit near the pool cliff (the osworld saturate-RECOVER). The
    # per-(conc,turn) MEDIAN trajectory is preserved; only the per-session spread is added.
    scale_q = context_scale_quantiles(profile)

    def session_scale(qk: float) -> float:
        if not scale_q:
            return 1.0
        nq = len(scale_q)
        return scale_q[min(nq - 1, max(0, int(round(qk * (nq - 1)))))]

    def scaled_spec(idx: int, f: float) -> TurnSpec:
        s = spec_for(idx)
        if f == 1.0:
            return s
        return TurnSpec(
            turn_index=s.turn_index,
            cached_context_tokens=s.cached_context_tokens * f,
            new_prefill_tokens=s.new_prefill_tokens * f,
            output_tokens=s.output_tokens,
        )

    sessions: list[Session] = []
    for k in range(c):
        q = (k + 0.5) / c
        if survival:
            tc = min(_draw_turn_count(survival, q), n_turn_slots) if n_turn_slots > 0 else _draw_turn_count(survival, q)
        else:
            tc = n_turn_slots if n_turn_slots > 0 else 1
        tc = max(1, tc)
        f = session_scale(q)
        sessions.append(Session(session_id=k, turn_count=tc, turns=[scaled_spec(i, f) for i in range(tc)]))
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
    # Hard-protect = sessions with a req in-flight THIS step (KV pinned). Grows as we admit, so
    # a head admitted earlier in this pass is never preempted to make room for a later one.
    in_flight = {r.session_id for r in state.prefilling.values()}
    in_flight |= {r.session_id for r in state.running.values()}
    deferred: list[_Req] = []
    for head in state.waiting:
        if budget <= 0 or len(state.running) + len(state.prefilling) >= MAX_NUM_SEQS:
            deferred.append(head)
            continue
        sid = head.session_id
        # Block-level prefix-cache hit: the surviving resident prefix covers up to
        # ``cached_blocks * block_size`` tokens of this turn's cached context; only the
        # EVICTED tail of that prefix plus the new tokens must be (re-)prefilled. Resident
        # blocks are read from the BARRIER SNAPSHOT (frozen at herd release), not live, so a
        # peer's in-pass eviction can't cascade this session into a spurious MISS.
        snap_blocks = state.resident_at_barrier.get(sid, cache.cached_blocks(sid))
        resident_prefix = min(head.cached, snap_blocks * cache.block_size)
        reprefill_cached = max(0.0, head.cached - resident_prefix)
        head.is_miss = reprefill_cached > 0.0
        head.remaining_prefill = reprefill_cached + head.new_prefill
        head.resident_prefix = resident_prefix          # HIT prefix attended each prefill step
        head.prefill_total = max(1.0, head.remaining_prefill)  # to spread the cached-attn cost across chunks
        target_blocks = cache.tokens_to_blocks(head.kv_tokens)
        # Reserve the full context: reclaim the surviving prefix, then free residents, then (only
        # under genuine over-subscription) PREEMPT an idle herd resident (``herd_pending`` minus
        # in-flight) per ``preempt_policy`` — vLLM RECOMPUTE. The preempted session re-prefills
        # its trimmed tail on its turn (a MISS). In-flight KV is never evicted. If even that can't
        # free the delta (one over-large head behind pinned in-flight KV), DEFER and retry on a
        # completion — never a hard stall.
        if not cache.grow_to(sid, target_blocks, in_flight, state.herd_pending, state.preempt_policy):
            deferred.append(head)
            continue
        state.prefilling[head.rid] = head
        in_flight.add(sid)
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
    """One mixed prefill+decode step. Decode = measured kernel. Prefill = three measured terms:
      * NEW   : serving per-(re)prefilled-token rate (linear, batched ∝ total chunk tokens),
      * FA3   : pipeline attention kernel (super-linear; per request, ∝ M·(R + M/2) where
                M = tokens this req (re-)prefills this turn, R = its resident prefix — tiny for
                a HIT, the quadratic re-encode for a MISS),
      * HOST  : re-tokenize the re-sent cached context — the dominant HIT cost; split into a
                per-step SHARED part (amortized across the batch) + a per-request part. Both
                FA3 and HOST·perreq are charged ONCE per request, frac-spread across chunks.
    The vLLM v1 chunked budget (decode-first, <= long_prefill_token_threshold per req) governs
    which reqs advance each step."""
    p = state.params
    decode_batch = len(state.running)
    decode_ms = decode_step_ms(decode_batch, _running_ctx_mean(state), p) if decode_batch > 0 else 0.0

    budget = max(0, MAX_NUM_BATCHED_TOKENS - decode_batch)
    total_chunk = 0.0       # batched NEW-token rate scales with total tokens this step
    gpu_fa3_ms = 0.0        # pipeline FA3 attention (per request; super-linear for re-prefills)
    host_perreq_ms = 0.0    # per-request host re-tokenize (summed over concurrent prefills)
    cached_w_sum = 0.0      # frac-weighted cached, for the per-step SHARED host term
    cached_w_n = 0
    any_prefill = False
    for r in state.prefilling.values():  # dict insertion order == FIFO admission order
        chunk = min(r.remaining_prefill, float(LONG_PREFILL_TOKEN_THRESHOLD), float(budget))
        if chunk <= 0:
            r._chunk = 0.0  # type: ignore[attr-defined]
            continue
        r._chunk = chunk  # type: ignore[attr-defined]
        budget -= chunk
        any_prefill = True
        total_chunk += chunk
        frac = chunk / r.prefill_total if r.prefill_total > 0 else 1.0
        M = r.prefill_total          # tokens this turn (re-)prefills (reprefill_cached + new)
        R = r.resident_prefix        # resident prefix the (re-)prefill attends
        gpu_fa3_ms += PREFILL_FA3_MS_PER_TOKEN2 * M * (R + 0.5 * M) * frac
        host_perreq_ms += PREFILL_HOST_PERREQ_MS_PER_TOKEN * r.cached * frac
        cached_w_sum += r.cached * frac
        cached_w_n += 1
    if any_prefill:
        gpu_new_ms = (_prefill_gemm_per_tok(p) + PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN) * total_chunk
        mean_cached = cached_w_sum / cached_w_n if cached_w_n else 0.0
        host_shared_ms = PREFILL_HOST_SHARED_MS_PER_TOKEN * mean_cached  # amortized once/step
        prefill_ms = (
            PREFILL_FLOOR_MS + gpu_new_ms + gpu_fa3_ms + host_shared_ms + host_perreq_ms
        )
    else:
        prefill_ms = 0.0
    return max(decode_ms + p.scheduler_overhead_ms_per_step, prefill_ms)


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
    #     reserving a fresh block at block boundaries (reclaiming free residents, then
    #     preempting idle herd residents under over-subscription — in-flight KV pinned). ---
    decode_in_flight = {r.session_id for r in state.prefilling.values()}
    decode_in_flight |= {r.session_id for r in state.running.values()}
    finished_decode: list[int] = []
    for rid in list(state.running.keys()):
        r = state.running.get(rid)
        if r is None:
            continue
        r.kv_tokens += 1.0
        need = state.cache.tokens_to_blocks(r.kv_tokens)
        if need > state.cache.cached_blocks(r.session_id):
            state.cache.grow_to(
                r.session_id, need, decode_in_flight, state.herd_pending, state.preempt_policy
            )  # best-effort
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
    # Freeze each herd member's resident prefix AT release: this turn's hit/miss is decided
    # against what was cache-resident when the herd was scheduled, so admitting one member can
    # NOT retroactively turn a peer's resident hit into a MISS (the cascade that over-counted
    # osworld ~2x the physical working-set overflow). Physical eviction still runs live below.
    state.resident_at_barrier = {s.session_id: state.cache.cached_blocks(s.session_id) for s in herd}
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
    sessions: list[Session], params: RooflineParams, max_events: int,
    preempt_policy: str = "tail",
) -> dict[tuple[int, int], float]:
    cache = PrefixLRUCache(params.available_kv_blocks, params.cache_block_size)
    state = _ServerState(params=params, cache=cache, sessions=sessions, preempt_policy=preempt_policy)

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

        from profiling.process.extract_benchmark_per_request import (
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
    preempt_policy: str = "tail",
) -> list[float]:
    """Per-turn TTFT (ms) for a (profile, concurrency) cell, emergent from a forward
    closed-loop event-driven queue sim with session-persistent KV + RECOMPUTE preemption.

    Returns one TTFT per input turn (median over sessions reaching that turn_index), aligned
    to ``turns`` order; ``[]`` for empty; a turn_index reached by no session falls back to the
    forward static predictor. Forward by default (cohort from ``forward_survival``);
    ``oracle=True`` overlays measured ``session_timelines`` (validation only). ``preempt_policy``
    selects the over-subscription victim: ``'tail'`` (vLLM RECOMPUTE, MRU-first) or ``'lru'``."""
    if not turns:
        return []
    p = params or RooflineParams()

    sessions: list[Session] | None = None
    if oracle:
        sessions = _build_cohort_oracle(turns, profile, float(concurrency))
    if sessions is None:
        sessions = _build_cohort(turns, profile, float(concurrency))

    ttfts = _run_sim(sessions, p, max_events, preempt_policy=preempt_policy)
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
