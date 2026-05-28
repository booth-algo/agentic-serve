"""Shape-only vLLM scheduler-step simulator.

This module intentionally does not model latency.  It replays the scheduler
shape that matters before timing is attached: decode batch, chunked prefill,
queue sizes, and KV-block headroom.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from simulator._legacy.vllm_block_pool import BlockHash, BlockPool


PrefillMode = Literal["benchmark_cache", "synthetic_shared_prefix"]


@dataclass(frozen=True)
class VllmSchedulerConfig:
    max_num_batched_tokens: int = 16_384
    max_num_seqs: int = 512
    prefill_chunk_tokens: int = 16_384
    cache_block_size: int = 16
    available_gpu_kv_blocks: int = 27_651
    decode_counts_against_token_budget: bool = True
    decode_joins_next_step: bool = True
    shared_prefix_tokens: int = 0
    bos_token_offset: int = 0
    admit_by_kv_capacity: bool = True
    # 3D roofline bandwidth boundary.  When both hbm_bw_gbps and
    # kv_bytes_per_token are > 0, the simulator caps per-step decode batch
    # so that the running KV read fits within hbm_bw_gbps * step_budget_ms.
    # When the cap bites, excess decode requests roll over to the next step
    # (decode wave splitting).  All three fields must be set to enable;
    # the default (zeros) preserves the legacy 2D capacity-only model.
    hbm_bw_gbps: float = 0.0
    kv_bytes_per_token: float = 0.0
    step_budget_ms: float = 0.0
    bandwidth_caps_decode: bool = True
    bandwidth_caps_prefill: bool = True
    bandwidth_decode_priority: bool = True
    # Per-step staircase admission (mirrors vLLM v1 scheduler's progressive
    # admit loop gated by allocate_slots).  When True, admit waiting requests
    # per step gated by the simulated BlockPool's allocate() and the
    # remaining token budget.  Produces the mixed/intrusion steps that real
    # vLLM exhibits.  Default False preserves legacy shape.
    use_block_pool: bool = False
    # Per-step admission cap.  0 = unlimited (matches vLLM v1, which admits
    # as many waiting requests as fit in the token budget + KV pool).  Any
    # positive value enforces a hard ceiling for diagnostics/regression
    # tests.  The historical default 4 was a simulator artefact that under-
    # drained the waiting queue at high concurrency.
    max_admissions_per_step: int = 0
    # Cap on the number of NEW admissions per scheduler step.  vLLM v1's
    # ``max_num_partial_prefills`` defaults to 1 but that's specifically
    # for LONG chunked prefills (requests whose prefill spans multiple
    # steps).  For the workloads we model, prefills typically complete in
    # one step, so a literal "1 admission per step" cap massively
    # under-admits.  Default 0 (unlimited) is closer to observed vLLM
    # behavior for our short-prefill workloads.  Keep the knob for
    # diagnostic runs.
    max_num_partial_prefills: int = 0


@dataclass(frozen=True)
class VllmRequestShape:
    request_id: int
    total_context_tokens: int
    cached_context_tokens: int
    new_prefill_tokens: int
    output_tokens: int
    prefill_work_tokens: int | None = None
    shared_prefix_tokens: int = 0
    is_prefix_anchor: bool = False
    # Block-level cache lookup chain (used only when ``use_block_pool=True``).
    # Length should equal ``ceil(total_context_tokens / block_size)``.
    # Shared-prefix siblings carry identical hashes for the shared block range.
    block_hashes: tuple[BlockHash, ...] | None = None
    # Multi-turn continuity: when this request finishes its decode (output
    # tokens exhausted), the simulator spawns a "follow-up" request that
    # represents the session's NEXT turn arriving immediately afterwards.
    # The follow-up needs ``next_turn_new_prefill_tokens`` of prefill work
    # but doesn't itself spawn another follow-up.  This models the real-
    # vLLM behavior where session A finishes turn N during the same wall
    # window that session B is still decoding → turn-(N+1) prefill for A
    # creates a mixed step.  Set 0 (default) to disable (single-turn
    # simulation, the legacy behavior).
    next_turn_new_prefill_tokens: int = 0


@dataclass(frozen=True)
class VllmPrefillChunk:
    request_id: int
    scheduled_tokens: int
    prefix_tokens: int


@dataclass(frozen=True)
class VllmStepShape:
    step_id: int
    decode_batch: int
    decoded_request_ids: tuple[int, ...]
    prefill_seqs: int
    prefill_tokens: int
    prefill_chunks: tuple[VllmPrefillChunk, ...]
    waiting_queue: int
    running_queue: int
    free_kv_blocks: int
    completed_prefill_request_ids: tuple[int, ...]
    completed_request_ids: tuple[int, ...]
    used_kv_blocks: int = 0
    capacity_waiting_requests: int = 0
    admitted_request_ids: tuple[int, ...] = ()
    preemptions: int = 0
    preempted_request_ids: tuple[int, ...] = ()
    recompute_prefill_tokens: int = 0
    decode_kv_read_bytes: float = 0.0
    prefill_kv_read_bytes: float = 0.0
    bandwidth_waiting_decode_requests: int = 0
    bandwidth_capped: bool = False


@dataclass(frozen=True)
class VllmTurnShapeSummary:
    steps: int
    decode_only_steps: int
    prefill_only_steps: int
    mixed_decode_prefill_steps: int
    empty_or_unknown_steps: int
    total_decode_slots: int
    total_prefill_tokens: int
    total_scheduled_tokens: int
    max_decode_batch: int
    mean_decode_batch: float
    mean_decode_batch_ratio: float
    max_waiting_queue: int
    max_running_queue: int
    min_free_kv_blocks: int
    max_used_kv_blocks: int
    max_capacity_waiting_requests: int
    total_preemptions: int
    total_recompute_prefill_tokens: int
    prefill_intrusion_candidate: bool
    effective_microbatching_candidate: bool
    decode_wave_candidate: bool
    kv_pressure_candidate: bool
    bandwidth_capped_steps: int = 0
    max_decode_kv_read_bytes: float = 0.0
    total_decode_kv_read_bytes: float = 0.0
    max_bandwidth_waiting_decode_requests: int = 0


@dataclass(frozen=True)
class VllmTurnShapeResult:
    steps: tuple[VllmStepShape, ...]
    summary: VllmTurnShapeSummary


@dataclass
class _RequestState:
    shape: VllmRequestShape
    prefill_remaining: int
    output_remaining: int
    prefilled_tokens: int = 0
    decode_emitted: int = 0
    resident_cached_tokens: int = 0
    resident_decode_tokens: int = 0
    preemptions: int = 0
    # True when this state represents a turn-(N+1) follow-up spawned by an
    # original turn-N request finishing decode.  Followups create mixed
    # steps while admitting but don't gate the turn's completion.
    is_followup: bool = False

    @property
    def ready_to_decode(self) -> bool:
        return self.prefill_remaining == 0 and self.output_remaining > 0

    @property
    def finished(self) -> bool:
        return self.prefill_remaining == 0 and self.output_remaining == 0


@dataclass(frozen=True)
class _AllocationResult:
    allocated: bool
    used_kv_blocks: int
    preemptions: int = 0
    preempted_request_ids: tuple[int, ...] = ()
    recompute_prefill_tokens: int = 0


def _resolve_per_request(
    values: Sequence[int] | None,
    fallback: int,
    request_count: int,
) -> list[int]:
    """Map a per-request distribution onto exactly ``request_count`` slots.

    - ``None`` → uniform ``fallback`` for every slot.
    - len > request_count → truncate to first ``request_count`` entries.
    - len < request_count → pad by cycling (preserves distribution shape).
    """
    if values is None:
        return [fallback] * request_count
    sanitized = [int(v) for v in values if v is not None]
    if not sanitized:
        return [fallback] * request_count
    if len(sanitized) >= request_count:
        return sanitized[:request_count]
    out: list[int] = []
    for i in range(request_count):
        out.append(sanitized[i % len(sanitized)])
    return out


def benchmark_row_requests(
    *,
    request_count: int,
    context_len: int,
    output_tokens: int,
    new_prefill_tokens: int,
    cached_context_tokens: int,
    mode: PrefillMode = "benchmark_cache",
    shared_prefix_tokens: int = 0,
    bos_token_offset: int = 0,
    output_tokens_per_request: Sequence[int] | None = None,
    new_prefill_tokens_per_request: Sequence[int] | None = None,
    cached_context_tokens_per_request: Sequence[int] | None = None,
    next_turn_new_prefill_tokens: int = 0,
) -> list[VllmRequestShape]:
    """Expand one benchmark-turn aggregate row into per-request shapes.

    When the ``*_per_request`` lists are provided, each request gets its own
    value from the list (preserving per-request workload variance — e.g. the
    long-tail output length that drives real vLLM's decode-only step count).
    When omitted, the scalar fallback is used uniformly for every request.
    """
    request_count = max(1, int(request_count))
    context_len = max(1, int(context_len))
    output_tokens = max(1, int(output_tokens))
    new_prefill_tokens = max(0, min(int(new_prefill_tokens), context_len))
    cached_context_tokens = max(0, min(int(cached_context_tokens), context_len))
    bos = max(0, int(bos_token_offset))
    shared_prefix = max(0, min(int(shared_prefix_tokens), context_len + bos))

    osl_per = _resolve_per_request(
        output_tokens_per_request, output_tokens, request_count
    )
    new_prefill_per = _resolve_per_request(
        new_prefill_tokens_per_request, new_prefill_tokens, request_count
    )
    cached_per = _resolve_per_request(
        cached_context_tokens_per_request, cached_context_tokens, request_count
    )

    requests: list[VllmRequestShape] = []
    for request_id in range(request_count):
        is_anchor = request_id == 0
        per_osl = max(1, int(osl_per[request_id]))
        per_new = max(0, min(int(new_prefill_per[request_id]), context_len))
        per_cached = max(0, min(int(cached_per[request_id]), context_len))
        if mode == "synthetic_shared_prefix":
            if is_anchor:
                prefill_work = context_len + bos
                request_shared = 0
            else:
                prefill_work = max(0, context_len + bos - shared_prefix)
                request_shared = shared_prefix
            request_cached = 0
            req_new_prefill = new_prefill_tokens
        else:
            prefill_work = max(1, per_new)
            request_shared = 0
            request_cached = per_cached
            req_new_prefill = per_new
        requests.append(
            VllmRequestShape(
                request_id=request_id,
                total_context_tokens=context_len + bos,
                cached_context_tokens=request_cached,
                new_prefill_tokens=req_new_prefill,
                output_tokens=per_osl,
                prefill_work_tokens=max(1, prefill_work),
                shared_prefix_tokens=request_shared,
                is_prefix_anchor=is_anchor,
                next_turn_new_prefill_tokens=max(0, int(next_turn_new_prefill_tokens)),
            )
        )
    return requests


def _default_block_hashes(
    request: VllmRequestShape, block_size: int
) -> tuple[BlockHash, ...]:
    """Synthesize block_hashes when the caller didn't supply them.

    Two shared-prefix mechanisms run in priority order:

    1. ``shared_prefix_tokens`` (synthetic_shared_prefix mode) wins when set
       — sibling siblings get a common anchor key for that leading region.
    2. Otherwise (benchmark_cache mode), ``cached_context_tokens`` defines
       an intra-turn shared anchor.  Real vLLM's block hashes are
       Merkle-style (parent_hash + token_ids), so two requests with
       identical leading tokens produce identical hashes — once the first
       sibling caches its blocks, the rest hit naturally.  Modeling this
       lets the simulator reproduce vLLM's "first request prefills, the
       other 79 admit cheap" behavior at high concurrency.

    Past the shared region every request still gets a unique tail key.
    """
    if request.block_hashes is not None:
        return request.block_hashes
    total_blocks = max(0, int(request.total_context_tokens) // block_size)
    synthetic_shared_tokens = max(0, int(request.shared_prefix_tokens))
    if synthetic_shared_tokens > 0:
        shared_blocks = min(total_blocks, synthetic_shared_tokens // block_size)
        prefix_key: tuple = ("shared-prefix", synthetic_shared_tokens)
    else:
        cached_tokens = max(0, int(request.cached_context_tokens))
        shared_blocks = min(total_blocks, cached_tokens // block_size)
        prefix_key = ("cached-prefix", cached_tokens)
    hashes: list[BlockHash] = []
    for i in range(shared_blocks):
        hashes.append(BlockHash(prefix_key, i))
    tail_key = ("tail", request.request_id)
    for i in range(shared_blocks, total_blocks):
        hashes.append(BlockHash(tail_key, i))
    # Always include at least one trailing block so requests with token counts
    # smaller than block_size still get a unique cache identity.
    if not hashes:
        hashes.append(BlockHash(("tail", request.request_id), 0))
    return tuple(hashes)


def _simulate_with_block_pool_admission(
    requests: Sequence[VllmRequestShape],
    cfg: VllmSchedulerConfig,
) -> VllmTurnShapeResult:
    """Per-step staircase scheduler that mirrors vLLM v1's admit loop.

    Each step does:
      1. Run decode (1 token per running ready request) — deducts from budget.
      2. Run continued prefill for running, not-yet-ready requests — chunked.
      3. Admit up to ``max_admissions_per_step`` waiting requests, each
         gated by ``BlockPool.allocate`` + remaining token budget.  Stop on
         first failure to admit (creates the staircase).

    Cache blocks become visible to siblings via ``BlockPool.cache_blocks``
    when a request's prefill completes its current chunk.
    """
    max_tokens = max(1, int(cfg.max_num_batched_tokens))
    max_seqs = max(1, int(cfg.max_num_seqs))
    chunk_tokens = max(1, int(cfg.prefill_chunk_tokens))
    block_size = max(1, int(cfg.cache_block_size))
    total_blocks = max(1, int(cfg.available_gpu_kv_blocks))
    max_admissions = max(0, int(cfg.max_admissions_per_step))  # 0 = unlimited

    pool = BlockPool(total_blocks=total_blocks, block_size=block_size)

    # Pre-populate the pool with any blocks the caller advertised as already
    # cached (cross-turn prefix shared by all requests).  We do this by
    # finding the set of block_hashes that ALL requests share for their
    # ``cached_context_tokens`` prefix — those become pre-cached in the pool
    # before the turn starts.
    request_hashes: list[tuple[BlockHash, ...]] = [
        _default_block_hashes(r, block_size) for r in requests
    ]
    cached_prefix_len_blocks = 0
    if request_hashes:
        # Use the first request's cached blocks as the shared-prefix anchor;
        # only consider hashes also present in every other request.
        anchor = request_hashes[0]
        max_cached_anchor = min(
            anchor and len(anchor),
            max(0, int(requests[0].cached_context_tokens) // block_size),
        )
        for i in range(max_cached_anchor):
            h = anchor[i]
            if all(i < len(rh) and rh[i] == h for rh in request_hashes):
                cached_prefix_len_blocks = i + 1
            else:
                break
        if cached_prefix_len_blocks > 0:
            pool.cache_blocks(anchor[:cached_prefix_len_blocks])

    states: list[_RequestState] = []
    waiting: list[_RequestState] = []
    originals: list[_RequestState] = []
    for idx, request in enumerate(requests):
        state = _RequestState(
            shape=request,
            prefill_remaining=0,  # set on admission via cache hit lookup
            output_remaining=max(1, int(request.output_tokens)),
            is_followup=False,
        )
        state._block_hashes = request_hashes[idx]  # type: ignore[attr-defined]
        state._num_tokens = max(1, int(request.total_context_tokens))  # type: ignore[attr-defined]
        state._admitted = False  # type: ignore[attr-defined]
        state._cached_blocks_at_admit = 0  # type: ignore[attr-defined]
        state._owned_blocks = 0  # type: ignore[attr-defined]
        states.append(state)
        waiting.append(state)
        originals.append(state)

    running: list[_RequestState] = []
    steps: list[VllmStepShape] = []
    step_id = 0
    followup_counter = 0

    def _spawn_followup(parent: _RequestState) -> None:
        """Spawn a turn-(N+1) follow-up request for a finishing original.

        Models real-vLLM behavior where a session that finishes turn N
        immediately fires turn N+1's prompt — admission of that follow-up
        creates the mixed step we'd otherwise miss in single-turn sims.
        """
        nonlocal followup_counter
        next_new_prefill = max(0, int(parent.shape.next_turn_new_prefill_tokens))
        if next_new_prefill <= 0:
            return
        followup_counter += 1
        # Unique negative ID to namespace followups distinct from originals.
        followup_id = -(1000 + followup_counter)
        # Follow-up reuses the parent's context size + cache structure but
        # with a fresh tail (next turn's new content has its own hashes).
        followup_total = max(1, int(parent.shape.total_context_tokens))
        followup_shape = VllmRequestShape(
            request_id=followup_id,
            total_context_tokens=followup_total,
            cached_context_tokens=max(0, int(parent.shape.cached_context_tokens)),
            new_prefill_tokens=next_new_prefill,
            output_tokens=1,  # we don't simulate the followup's decode
            shared_prefix_tokens=0,
            next_turn_new_prefill_tokens=0,  # followups don't chain
        )
        followup_state = _RequestState(
            shape=followup_shape,
            prefill_remaining=0,
            output_remaining=1,
            is_followup=True,
        )
        followup_state._block_hashes = _default_block_hashes(  # type: ignore[attr-defined]
            followup_shape, block_size
        )
        followup_state._num_tokens = followup_total  # type: ignore[attr-defined]
        followup_state._admitted = False  # type: ignore[attr-defined]
        followup_state._cached_blocks_at_admit = 0  # type: ignore[attr-defined]
        followup_state._owned_blocks = 0  # type: ignore[attr-defined]
        states.append(followup_state)
        waiting.append(followup_state)

    def _admit(candidate: _RequestState, budget: int) -> tuple[int, int]:
        """Try to admit ``candidate``.  Returns (scheduled_tokens, new_budget).

        Returns (-1, budget) when admission failed.
        """
        block_hashes = candidate._block_hashes  # type: ignore[attr-defined]
        num_tokens = candidate._num_tokens  # type: ignore[attr-defined]
        # Block-aligned cache lookup excluding the last block (matches vLLM).
        max_hit_blocks = max(0, num_tokens // block_size - (1 if num_tokens % block_size == 0 else 0))
        cached_blocks = pool.find_longest_cache_hit(block_hashes, max_hit_blocks=max_hit_blocks)
        num_computed_tokens = cached_blocks * block_size
        new_tokens_needed = num_tokens - num_computed_tokens
        scheduled = min(new_tokens_needed, budget, chunk_tokens)
        if scheduled <= 0:
            return (-1, budget)
        # Allocate enough blocks for the request's full sequence (vLLM's
        # default), minus what's already cached.
        blocks_needed = pool.tokens_to_blocks(num_tokens) - cached_blocks
        if not pool.allocate(candidate.shape.request_id, blocks_needed):
            return (-1, budget)
        candidate._admitted = True  # type: ignore[attr-defined]
        candidate._cached_blocks_at_admit = cached_blocks  # type: ignore[attr-defined]
        candidate._owned_blocks = blocks_needed  # type: ignore[attr-defined]
        candidate.prefill_remaining = new_tokens_needed - scheduled
        candidate.prefilled_tokens = scheduled
        candidate.resident_cached_tokens = num_computed_tokens
        return (scheduled, budget - scheduled)

    # Loop until every ORIGINAL turn-N request has finished its decode.
    # Pending follow-ups (turn-N+1 prefills spawned mid-decode) keep
    # admitting during that window — they create the mixed steps real
    # vLLM exhibits when sessions cross turn boundaries — but they don't
    # gate completion.  When no originals remain, the simulator stops.
    def _originals_active() -> bool:
        return any(not state.finished for state in originals)

    while _originals_active() or waiting or running:
        if not _originals_active():
            # All originals done; remaining follow-ups don't extend the
            # turn-N wall window in real vLLM either.  Stop.
            break

        token_budget = max_tokens
        decode_request_ids: list[int] = []
        prefill_chunks: list[VllmPrefillChunk] = []
        prefill_seqs = 0
        prefill_tokens_step = 0
        admitted_ids: list[int] = []
        completed_request_ids: list[int] = []
        completed_prefill_ids: list[int] = []
        capacity_waiting = 0

        # Phase 1: continued decode for running ready requests.
        for state in list(running):
            if state.finished or token_budget <= 0:
                continue
            if state.prefill_remaining > 0:
                continue
            if state.output_remaining <= 0:
                continue
            state.output_remaining -= 1
            state.decode_emitted += 1
            state.resident_decode_tokens += 1
            decode_request_ids.append(state.shape.request_id)
            token_budget -= 1
            if state.finished:
                completed_request_ids.append(state.shape.request_id)
                if not state.is_followup:
                    _spawn_followup(state)

        # Phase 2: continued prefill for running, not-yet-ready requests.
        for state in list(running):
            if state.prefill_remaining <= 0 or token_budget <= 0:
                continue
            tokens = min(state.prefill_remaining, token_budget, chunk_tokens)
            state.prefill_remaining -= tokens
            state.prefilled_tokens += tokens
            state.resident_cached_tokens = state.resident_cached_tokens  # unchanged
            prefix_tokens = state.resident_cached_tokens + (state.prefilled_tokens - tokens)
            prefill_chunks.append(
                VllmPrefillChunk(
                    request_id=state.shape.request_id,
                    scheduled_tokens=tokens,
                    prefix_tokens=prefix_tokens,
                )
            )
            prefill_seqs += 1
            prefill_tokens_step += tokens
            token_budget -= tokens
            if state.prefill_remaining == 0:
                completed_prefill_ids.append(state.shape.request_id)
                # Cache the request's blocks so siblings can find them.
                hashes = state._block_hashes  # type: ignore[attr-defined]
                cached_at_admit = state._cached_blocks_at_admit  # type: ignore[attr-defined]
                pool.cache_blocks(hashes[:pool.tokens_to_blocks(state._num_tokens)])  # type: ignore[attr-defined]

        # Phase 3: admission staircase.
        admissions_this_step = 0
        # vLLM v1 caps NEW admissions per step at ``max_num_partial_prefills``
        # (default 1).  0 = unlimited.
        partial_prefill_cap = max(0, int(cfg.max_num_partial_prefills))
        while (
            waiting
            and (max_admissions == 0 or admissions_this_step < max_admissions)
            and len(running) < max_seqs
            and token_budget > 0
            and (partial_prefill_cap == 0 or admissions_this_step < partial_prefill_cap)
        ):
            candidate = waiting[0]
            scheduled, token_budget = _admit(candidate, token_budget)
            if scheduled < 0:
                capacity_waiting = max(capacity_waiting, len(waiting))
                break
            waiting.pop(0)
            running.append(candidate)
            admitted_ids.append(candidate.shape.request_id)
            prefix_tokens = candidate.resident_cached_tokens
            prefill_chunks.append(
                VllmPrefillChunk(
                    request_id=candidate.shape.request_id,
                    scheduled_tokens=scheduled,
                    prefix_tokens=prefix_tokens,
                )
            )
            prefill_seqs += 1
            prefill_tokens_step += scheduled
            admissions_this_step += 1
            if candidate.prefill_remaining == 0:
                completed_prefill_ids.append(candidate.shape.request_id)
                hashes = candidate._block_hashes  # type: ignore[attr-defined]
                pool.cache_blocks(hashes[:pool.tokens_to_blocks(candidate._num_tokens)])  # type: ignore[attr-defined]

        # Sanity: prevent infinite loop when nothing is schedulable.
        if not decode_request_ids and not prefill_chunks and not admitted_ids:
            if not waiting:
                # Nothing more to do.
                break
            raise RuntimeError(
                "block-pool scheduler made no progress with waiting work; "
                "block pool may be too small or request shapes invalid"
            )

        # Free finished requests' blocks back to the pool.
        for state in list(running):
            if state.finished:
                pool.free_request(state.shape.request_id)
                running.remove(state)

        used_blocks = pool.total_blocks - pool.get_num_free_blocks()
        steps.append(
            VllmStepShape(
                step_id=step_id,
                decode_batch=len(decode_request_ids),
                decoded_request_ids=tuple(decode_request_ids),
                prefill_seqs=prefill_seqs,
                prefill_tokens=prefill_tokens_step,
                prefill_chunks=tuple(prefill_chunks),
                waiting_queue=len(waiting),
                running_queue=len(running),
                free_kv_blocks=pool.get_num_free_blocks(),
                completed_prefill_request_ids=tuple(completed_prefill_ids),
                completed_request_ids=tuple(completed_request_ids),
                used_kv_blocks=used_blocks,
                capacity_waiting_requests=capacity_waiting,
                admitted_request_ids=tuple(admitted_ids),
                preemptions=0,
                preempted_request_ids=(),
                recompute_prefill_tokens=0,
            )
        )
        step_id += 1

    return VllmTurnShapeResult(
        steps=tuple(steps),
        summary=summarize_vllm_turn_shape(
            steps,
            scheduled_request_count=len(requests),
        ),
    )


def simulate_vllm_turn_shape(
    requests: Sequence[VllmRequestShape],
    config: VllmSchedulerConfig | None = None,
) -> VllmTurnShapeResult:
    """Replay one simultaneous-arrival vLLM-like scheduler turn."""
    cfg = config or VllmSchedulerConfig()
    if cfg.use_block_pool:
        return _simulate_with_block_pool_admission(requests, cfg)
    max_tokens = max(1, int(cfg.max_num_batched_tokens))
    max_seqs = max(1, int(cfg.max_num_seqs))
    chunk_tokens = max(1, int(cfg.prefill_chunk_tokens))
    states = [
        _RequestState(
            shape=request,
            prefill_remaining=max(0, int(
                request.prefill_work_tokens
                if request.prefill_work_tokens is not None
                else request.new_prefill_tokens
            )),
            output_remaining=max(1, int(request.output_tokens)),
        )
        for request in requests
    ]
    waiting = list(states)
    active: list[_RequestState] = []
    completed: list[_RequestState] = []
    completed_request_ids_seen: set[int] = set()
    steps: list[VllmStepShape] = []
    step_id = 0

    while waiting or any(not state.finished for state in active):
        active = [state for state in active if not state.finished]
        for state in states:
            if state.finished and state.shape.request_id not in completed_request_ids_seen:
                completed.append(state)
                completed_request_ids_seen.add(state.shape.request_id)
        admitted_ids: tuple[int, ...] = ()
        capacity_waiting_requests = 0

        if not active and not waiting:
            raise RuntimeError("vLLM shape scheduler has no active work")

        token_budget = max_tokens
        preemptions = 0
        preempted_request_ids: list[int] = []
        recompute_prefill_tokens = 0
        protected_request_ids: set[int] = set()
        decode_states: list[_RequestState] = []
        current_used_blocks = _used_kv_blocks(active, completed, cfg)
        ready = [state for state in active if state.ready_to_decode]
        active_request_ids = {state.shape.request_id for state in active}
        decode_budget = token_budget if cfg.decode_counts_against_token_budget else len(ready)
        bw_budget_bytes = _bandwidth_budget_bytes(cfg)
        bw_per_token = max(0.0, float(cfg.kv_bytes_per_token))
        decode_kv_bytes = 0.0
        bandwidth_waiting_decode = 0
        bandwidth_capped = False
        for state in ready:
            if len(decode_states) >= decode_budget:
                bandwidth_waiting_decode += 1
                continue
            if state.shape.request_id not in active_request_ids or not state.ready_to_decode:
                continue
            if (
                cfg.bandwidth_caps_decode
                and bw_budget_bytes > 0
                and bw_per_token > 0
            ):
                state_bytes = _resident_context_tokens(state) * bw_per_token
                if decode_kv_bytes + state_bytes > bw_budget_bytes and decode_states:
                    bandwidth_capped = True
                    bandwidth_waiting_decode += 1
                    continue
            allocation = _ensure_decode_slot(
                active,
                waiting,
                state,
                cfg,
                protected_request_ids,
                current_used_blocks,
            )
            current_used_blocks = allocation.used_kv_blocks
            preemptions += allocation.preemptions
            preempted_request_ids.extend(allocation.preempted_request_ids)
            recompute_prefill_tokens += allocation.recompute_prefill_tokens
            for request_id in allocation.preempted_request_ids:
                active_request_ids.discard(request_id)
            if not allocation.allocated:
                capacity_waiting_requests = max(capacity_waiting_requests, 1)
                continue
            old_state_blocks = _state_kv_blocks(state, cfg)
            state.output_remaining -= 1
            state.decode_emitted += 1
            state.resident_decode_tokens += 1
            current_used_blocks += _state_kv_blocks(state, cfg) - old_state_blocks
            decode_states.append(state)
            protected_request_ids.add(state.shape.request_id)
            if bw_per_token > 0:
                decode_kv_bytes += _resident_context_tokens(state) * bw_per_token
            if cfg.decode_counts_against_token_budget:
                token_budget = max(0, token_budget - 1)
                if token_budget <= 0:
                    break
        decode_batch = len(decode_states)

        prefilled_states: list[_RequestState] = []
        prefill_chunks: list[VllmPrefillChunk] = []
        prefill_tokens = 0
        prefill_seqs = 0
        prefill_kv_bytes = 0.0
        active_request_ids = {state.shape.request_id for state in active}
        for state in list(active):
            if token_budget <= 0:
                break
            if state.shape.request_id not in active_request_ids:
                continue
            if state.prefill_remaining <= 0:
                continue
            if (
                cfg.bandwidth_decode_priority
                and bandwidth_capped
                and not cfg.bandwidth_caps_prefill
            ):
                # When bandwidth-capped under decode priority, keep prefill at
                # the same scope as it would have without bw modeling — fall
                # through.
                pass
            prefix_tokens = _resident_context_tokens(state)
            requested_tokens = min(state.prefill_remaining, token_budget, chunk_tokens)
            tokens = min(
                requested_tokens,
                _max_allocatable_tokens_from_used(
                    state,
                    cfg,
                    current_used_blocks,
                ),
            )
            if tokens <= 0:
                capacity_waiting_requests = max(capacity_waiting_requests, len(waiting) + 1)
                continue
            if (
                cfg.bandwidth_caps_prefill
                and bw_budget_bytes > 0
                and bw_per_token > 0
            ):
                bw_remaining = bw_budget_bytes - decode_kv_bytes - prefill_kv_bytes
                if bw_remaining <= 0:
                    if cfg.bandwidth_decode_priority and decode_states:
                        # Decode already saturated bandwidth — defer prefill.
                        bandwidth_capped = True
                        break
                # Prefill streams the prefix KV once and writes q new tokens.
                # BW = (prefix + q) * kv_bytes_per_token per chunk.
                prefix_bytes = prefix_tokens * bw_per_token
                if bw_remaining - prefix_bytes <= 0:
                    if cfg.bandwidth_decode_priority and decode_states:
                        bandwidth_capped = True
                        break
                max_chunk_tokens = max(
                    1, int((bw_remaining - prefix_bytes) / bw_per_token)
                )
                if max_chunk_tokens < tokens:
                    bandwidth_capped = True
                    tokens = max_chunk_tokens
                if tokens <= 0:
                    if cfg.bandwidth_decode_priority and decode_states:
                        break
                    tokens = 1
            old_state_blocks = _state_kv_blocks(state, cfg)
            state.prefill_remaining -= tokens
            state.prefilled_tokens += tokens
            current_used_blocks += _state_kv_blocks(state, cfg) - old_state_blocks
            token_budget -= tokens
            prefill_tokens += tokens
            prefill_seqs += 1
            if bw_per_token > 0:
                prefill_kv_bytes += (prefix_tokens + tokens) * bw_per_token
            protected_request_ids.add(state.shape.request_id)
            prefill_chunks.append(
                VllmPrefillChunk(
                    request_id=state.shape.request_id,
                    scheduled_tokens=tokens,
                    prefix_tokens=prefix_tokens,
                )
            )
            if state.prefill_remaining == 0:
                prefilled_states.append(state)

        if preemptions == 0 and token_budget > 0:
            newly_admitted: list[int] = []
            while waiting and len(active) < max_seqs and token_budget > 0:
                candidate = waiting[0]
                old_cached = candidate.resident_cached_tokens
                old_prefilled = candidate.prefilled_tokens
                old_remaining = candidate.prefill_remaining
                _mark_admitted(candidate)
                requested_tokens = min(
                    candidate.prefill_remaining,
                    token_budget,
                    chunk_tokens,
                )
                if requested_tokens > 0:
                    prefix_tokens = _resident_context_tokens(candidate)
                    candidate.prefill_remaining -= requested_tokens
                    candidate.prefilled_tokens += requested_tokens
                else:
                    prefix_tokens = _resident_context_tokens(candidate)

                projected_used_blocks = current_used_blocks + _state_kv_blocks(
                    candidate,
                    cfg,
                )
                if (
                    cfg.admit_by_kv_capacity
                    and cfg.available_gpu_kv_blocks > 0
                    and projected_used_blocks > int(cfg.available_gpu_kv_blocks)
                ):
                    candidate.resident_cached_tokens = old_cached
                    candidate.prefilled_tokens = old_prefilled
                    candidate.prefill_remaining = old_remaining
                    capacity_waiting_requests = max(
                        capacity_waiting_requests,
                        len(waiting),
                    )
                    break

                active.append(waiting.pop(0))
                current_used_blocks = projected_used_blocks
                newly_admitted.append(candidate.shape.request_id)
                if requested_tokens <= 0:
                    continue
                token_budget -= requested_tokens
                prefill_tokens += requested_tokens
                prefill_seqs += 1
                prefill_chunks.append(
                    VllmPrefillChunk(
                        request_id=candidate.shape.request_id,
                        scheduled_tokens=requested_tokens,
                        prefix_tokens=prefix_tokens,
                    )
                )
                if candidate.prefill_remaining == 0:
                    prefilled_states.append(candidate)
            admitted_ids = tuple(newly_admitted)

        if decode_batch == 0 and prefill_tokens == 0:
            if admitted_ids:
                continue
            raise RuntimeError("vLLM shape scheduler made no progress")

        completed_request_ids: list[int] = []
        for state in decode_states:
            if state.finished:
                completed_request_ids.append(state.shape.request_id)

        used_blocks = current_used_blocks
        free_blocks = int(cfg.available_gpu_kv_blocks) - used_blocks
        steps.append(
            VllmStepShape(
                step_id=step_id,
                decode_batch=decode_batch,
                decoded_request_ids=tuple(
                    state.shape.request_id for state in decode_states
                ),
                prefill_seqs=prefill_seqs,
                prefill_tokens=prefill_tokens,
                prefill_chunks=tuple(prefill_chunks),
                waiting_queue=len(waiting),
                running_queue=len(active),
                free_kv_blocks=free_blocks,
                completed_prefill_request_ids=tuple(
                    state.shape.request_id for state in prefilled_states
                ),
                completed_request_ids=tuple(completed_request_ids),
                used_kv_blocks=used_blocks,
                capacity_waiting_requests=capacity_waiting_requests,
                admitted_request_ids=admitted_ids,
                preemptions=preemptions,
                preempted_request_ids=tuple(preempted_request_ids),
                recompute_prefill_tokens=recompute_prefill_tokens,
                decode_kv_read_bytes=decode_kv_bytes,
                prefill_kv_read_bytes=prefill_kv_bytes,
                bandwidth_waiting_decode_requests=bandwidth_waiting_decode,
                bandwidth_capped=bandwidth_capped,
            )
        )
        step_id += 1

    return VllmTurnShapeResult(
        steps=tuple(steps),
        summary=summarize_vllm_turn_shape(
            steps,
            scheduled_request_count=len(requests),
        ),
    )


def summarize_vllm_turn_shape(
    steps: Sequence[VllmStepShape],
    *,
    scheduled_request_count: int,
) -> VllmTurnShapeSummary:
    decode_batches = [step.decode_batch for step in steps]
    prefill_tokens = [step.prefill_tokens for step in steps]
    free_blocks = [step.free_kv_blocks for step in steps]
    used_blocks = [step.used_kv_blocks for step in steps]
    capacity_waiting = [step.capacity_waiting_requests for step in steps]
    preemptions = [step.preemptions for step in steps]
    recompute_prefill_tokens = [step.recompute_prefill_tokens for step in steps]
    waiting = [step.waiting_queue for step in steps]
    running = [step.running_queue for step in steps]
    decode_kv_bytes = [step.decode_kv_read_bytes for step in steps]
    bandwidth_waiting = [step.bandwidth_waiting_decode_requests for step in steps]
    decode_only = 0
    prefill_only = 0
    mixed = 0
    empty = 0
    for decode_batch, prefill in zip(decode_batches, prefill_tokens):
        if decode_batch > 0 and prefill > 0:
            mixed += 1
        elif decode_batch > 0:
            decode_only += 1
        elif prefill > 0:
            prefill_only += 1
        else:
            empty += 1
    max_decode_batch = max(decode_batches, default=0)
    mean_decode_batch = statistics.fmean(decode_batches) if decode_batches else 0.0
    mean_ratio = (
        mean_decode_batch / scheduled_request_count
        if scheduled_request_count > 0
        else 0.0
    )
    return VllmTurnShapeSummary(
        steps=len(steps),
        decode_only_steps=decode_only,
        prefill_only_steps=prefill_only,
        mixed_decode_prefill_steps=mixed,
        empty_or_unknown_steps=empty,
        total_decode_slots=sum(decode_batches),
        total_prefill_tokens=sum(prefill_tokens),
        total_scheduled_tokens=sum(
            step.decode_batch + step.prefill_tokens for step in steps
        ),
        max_decode_batch=max_decode_batch,
        mean_decode_batch=mean_decode_batch,
        mean_decode_batch_ratio=mean_ratio,
        max_waiting_queue=max(waiting, default=0),
        max_running_queue=max(running, default=0),
        min_free_kv_blocks=min(free_blocks, default=0),
        max_used_kv_blocks=max(used_blocks, default=0),
        max_capacity_waiting_requests=max(capacity_waiting, default=0),
        total_preemptions=sum(preemptions),
        total_recompute_prefill_tokens=sum(recompute_prefill_tokens),
        prefill_intrusion_candidate=mixed > 0,
        effective_microbatching_candidate=(
            scheduled_request_count > 0
            and max_decode_batch > 0
            and max_decode_batch < scheduled_request_count
        ),
        decode_wave_candidate=scheduled_request_count > 0 and 0 < mean_ratio < 0.75,
        kv_pressure_candidate=bool(
            (free_blocks and min(free_blocks) <= 0)
            or any(waiting_count > 0 for waiting_count in capacity_waiting)
        ),
        bandwidth_capped_steps=sum(1 for step in steps if step.bandwidth_capped),
        max_decode_kv_read_bytes=max(decode_kv_bytes, default=0.0),
        total_decode_kv_read_bytes=sum(decode_kv_bytes),
        max_bandwidth_waiting_decode_requests=max(bandwidth_waiting, default=0),
    )


def _admit_waiting(
    active: list[_RequestState],
    waiting: list[_RequestState],
    completed: list[_RequestState],
    cfg: VllmSchedulerConfig,
    max_seqs: int,
    max_tokens: int,
    chunk_tokens: int,
) -> tuple[tuple[int, ...], int]:
    del completed
    admitted: list[int] = []
    capacity_waiting_requests = 0
    while waiting and len(active) < max_seqs:
        candidate = waiting[0]
        if cfg.admit_by_kv_capacity and not _can_admit(
            active,
            candidate,
            cfg,
            max_tokens,
            chunk_tokens,
        ):
            capacity_waiting_requests = len(waiting)
            break
        _mark_admitted(candidate)
        active.append(waiting.pop(0))
        admitted.append(candidate.shape.request_id)
    return tuple(admitted), capacity_waiting_requests


def _can_admit(
    active: Sequence[_RequestState],
    candidate: _RequestState,
    cfg: VllmSchedulerConfig,
    max_tokens: int,
    chunk_tokens: int,
) -> bool:
    if cfg.available_gpu_kv_blocks <= 0:
        return True
    old_cached = candidate.resident_cached_tokens
    old_prefilled = candidate.prefilled_tokens
    candidate.resident_cached_tokens = _resident_cached_tokens_on_admit(candidate)
    if candidate.prefill_remaining > 0:
        candidate.prefilled_tokens += min(
            candidate.prefill_remaining,
            max(1, int(max_tokens)),
            max(1, int(chunk_tokens)),
        )
    projected = list(active) + [candidate]
    fits = _free_kv_blocks(projected, (), cfg) >= 0
    candidate.resident_cached_tokens = old_cached
    candidate.prefilled_tokens = old_prefilled
    return fits


def _mark_admitted(state: _RequestState) -> None:
    state.resident_cached_tokens = _resident_cached_tokens_on_admit(state)


def _resident_cached_tokens_on_admit(state: _RequestState) -> int:
    return max(0, int(state.shape.cached_context_tokens))


def _ensure_decode_slot(
    active: list[_RequestState],
    waiting: list[_RequestState],
    target: _RequestState,
    cfg: VllmSchedulerConfig,
    protected_request_ids: set[int],
    used_kv_blocks: int,
) -> _AllocationResult:
    if _max_allocatable_tokens_from_used(target, cfg, used_kv_blocks) >= 1:
        return _AllocationResult(allocated=True, used_kv_blocks=used_kv_blocks)

    preempted_ids: list[int] = []
    recompute_tokens = 0
    while _max_allocatable_tokens_from_used(target, cfg, used_kv_blocks) < 1:
        victim = _select_preemption_victim(active, target, protected_request_ids)
        if victim is None:
            return _AllocationResult(
                allocated=False,
                used_kv_blocks=used_kv_blocks,
                preemptions=len(preempted_ids),
                preempted_request_ids=tuple(preempted_ids),
                recompute_prefill_tokens=recompute_tokens,
            )
        preempted_ids.append(victim.shape.request_id)
        recompute_tokens += _preempt_request(active, waiting, victim)
        used_kv_blocks = _used_kv_blocks(active, (), cfg)

    return _AllocationResult(
        allocated=True,
        used_kv_blocks=used_kv_blocks,
        preemptions=len(preempted_ids),
        preempted_request_ids=tuple(preempted_ids),
        recompute_prefill_tokens=recompute_tokens,
    )


def _select_preemption_victim(
    active: Sequence[_RequestState],
    target: _RequestState,
    protected_request_ids: set[int],
) -> _RequestState | None:
    for state in reversed(active):
        if state is target:
            continue
        if state.shape.request_id in protected_request_ids:
            continue
        if state.finished:
            continue
        return state
    return None


def _preempt_request(
    active: list[_RequestState],
    waiting: list[_RequestState],
    victim: _RequestState,
) -> int:
    recompute_tokens = _recompute_tokens_after_preemption(victim)
    victim.prefill_remaining = recompute_tokens
    victim.prefilled_tokens = 0
    victim.resident_cached_tokens = 0
    victim.resident_decode_tokens = 0
    victim.preemptions += 1
    _remove_identity(active, victim)
    waiting.insert(0, victim)
    return recompute_tokens


def _recompute_tokens_after_preemption(state: _RequestState) -> int:
    prefill_work = (
        state.shape.prefill_work_tokens
        if state.shape.prefill_work_tokens is not None
        else state.shape.new_prefill_tokens
    )
    return max(0, int(prefill_work)) + max(0, int(state.decode_emitted))


def _computed_prefix_tokens(state: _RequestState) -> int:
    return (
        max(0, int(state.resident_cached_tokens))
        + max(0, int(state.shape.shared_prefix_tokens))
        + max(0, int(state.prefilled_tokens))
    )


def _resident_context_tokens(state: _RequestState) -> int:
    return _computed_prefix_tokens(state) + max(0, int(state.resident_decode_tokens))


def _resident_tokens_excluding_shared(state: _RequestState) -> int:
    return (
        max(0, int(state.resident_cached_tokens))
        + max(0, int(state.prefilled_tokens))
        + max(0, int(state.resident_decode_tokens))
    )


def _max_allocatable_tokens_without_preemption(
    active: Sequence[_RequestState],
    target: _RequestState,
    cfg: VllmSchedulerConfig,
) -> int:
    return _max_allocatable_tokens_from_used(
        target,
        cfg,
        _used_kv_blocks(active, (), cfg),
    )


def _max_allocatable_tokens_from_used(
    target: _RequestState,
    cfg: VllmSchedulerConfig,
    used_blocks: int,
) -> int:
    if cfg.available_gpu_kv_blocks <= 0:
        return 1 << 60
    block = max(1, int(cfg.cache_block_size))
    free_blocks = max(0, int(cfg.available_gpu_kv_blocks) - used_blocks)
    target_tokens = _resident_tokens_excluding_shared(target)
    target_blocks = _ceil_div(target_tokens, block)
    max_target_blocks = target_blocks + free_blocks
    max_target_tokens = max_target_blocks * block
    return max(0, max_target_tokens - target_tokens)


def _state_kv_blocks(state: _RequestState, cfg: VllmSchedulerConfig) -> int:
    return _ceil_div(
        max(0, _resident_tokens_excluding_shared(state)),
        max(1, int(cfg.cache_block_size)),
    )


def _remove_identity(states: list[_RequestState], target: _RequestState) -> None:
    for index, state in enumerate(states):
        if state is target:
            del states[index]
            return
    raise ValueError("state is not active")


def _free_kv_blocks(
    active: Sequence[_RequestState],
    completed: Sequence[_RequestState],
    cfg: VllmSchedulerConfig,
) -> int:
    used = _used_kv_blocks(active, completed, cfg)
    return int(cfg.available_gpu_kv_blocks) - used


def _used_kv_blocks(
    active: Sequence[_RequestState],
    completed: Sequence[_RequestState],
    cfg: VllmSchedulerConfig,
) -> int:
    del completed
    block = max(1, int(cfg.cache_block_size))
    shared_prefix = max(
        [state.shape.shared_prefix_tokens for state in active],
        default=max(0, int(cfg.shared_prefix_tokens)),
    )
    used = _ceil_div(shared_prefix, block) if shared_prefix > 0 and active else 0
    for state in active:
        request_tokens = _resident_tokens_excluding_shared(state)
        used += _ceil_div(max(0, request_tokens), block)
    return used


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / divisor))


def _bandwidth_budget_bytes(cfg: VllmSchedulerConfig) -> float:
    """Return per-step bandwidth budget in bytes, or 0.0 when disabled."""
    bw = max(0.0, float(cfg.hbm_bw_gbps))
    step_ms = max(0.0, float(cfg.step_budget_ms))
    if bw <= 0.0 or step_ms <= 0.0:
        return 0.0
    return bw * 1e9 * (step_ms / 1000.0)


# ---- Multi-turn wall-clock replay ---------------------------------------
#
# The single-turn simulator (``simulate_vllm_turn_shape``) treats each turn
# as a standalone arrival batch.  Real vLLM benchmarks run many sessions
# concurrently, each cycling through multiple turns separated by
# client-side think time.  At early turns sessions are roughly synchronized
# and turn boundaries don't overlap; at later turns OSL variance has
# accumulated and turn-(N+1) prefills routinely happen while turn-N
# decodes are still running — producing the mixed steps a single-turn sim
# can't reproduce.
#
# ``simulate_vllm_multi_turn_replay`` replays the actual arrival timeline
# captured in the paired bench JSONs (each session's per-turn
# dispatch_started_at_ms).  A step pricer advances the wall clock, gating
# when each session's next turn enters the waiting queue.  The result
# tracks per-turn step shapes, so the predictor can extract per-turn stats
# from a single replay over the whole workload.


@dataclass(frozen=True)
class ReplayTurnSpec:
    turn_index: int
    arrival_offset_ms: float
    completion_offset_ms: float
    new_prefill_tokens: int
    cached_context_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class MultiTurnReplayStep:
    """One scheduler step in the wall-clock replay.

    ``cum_wall_ms_end`` is the simulated wall clock at the END of this
    step.  ``cohort_turn_index`` mirrors the real-vLLM trace's per-step
    ``turn_index`` tagging: it's the MAXIMUM turn_index that has been
    admitted to the scheduler at or before this step (monotonically
    non-decreasing across the step list).  Real traces show
    ``turn_index`` advancing only on prefill steps where a new turn's
    first request enters — exactly what this tracking captures.
    """
    step: VllmStepShape
    cum_wall_ms_end: float
    decoded_turn_indices: tuple[int, ...]  # parallel to step.decoded_request_ids
    prefill_turn_indices: tuple[int, ...]  # parallel to step.prefill_chunks
    cohort_turn_index: int = -1


@dataclass(frozen=True)
class MultiTurnReplayResult:
    steps: tuple[MultiTurnReplayStep, ...]
    sessions: tuple[tuple[ReplayTurnSpec, ...], ...]


def _build_replay_turn_spec(turn: Mapping[str, Any]) -> ReplayTurnSpec:
    return ReplayTurnSpec(
        turn_index=int(turn.get("turn_index", 0)),
        arrival_offset_ms=float(turn.get("arrival_offset_ms", 0.0)),
        completion_offset_ms=float(turn.get("completion_offset_ms", 0.0)),
        new_prefill_tokens=int(turn.get("new_prefill_tokens", 0)),
        cached_context_tokens=int(turn.get("cached_context_tokens", 0)),
        output_tokens=int(turn.get("output_tokens", 1)),
    )


def simulate_vllm_multi_turn_replay(
    sessions: Sequence[Sequence[Mapping[str, Any]]],
    cfg: VllmSchedulerConfig,
    step_pricer: Callable[[VllmStepShape, Mapping[int, int]], float],
    *,
    max_steps: int | None = None,
) -> MultiTurnReplayResult:
    """Replay the multi-turn workload with wall-clock-gated arrivals.

    Each session is a list of turn dicts (``ReplayTurnSpec`` fields).
    Turns arrive when ``cum_wall_ms >= arrival_offset_ms``.  ``step_pricer``
    returns the step duration in ms; the simulator integrates that forward.

    Returns a ``MultiTurnReplayResult`` containing every step plus the
    per-step turn-index tags needed to filter steps to a target turn's
    wall window for comparator stats.
    """
    parsed_sessions: list[tuple[ReplayTurnSpec, ...]] = [
        tuple(_build_replay_turn_spec(t) for t in session) for session in sessions
    ]
    # Build a single ordered "pending" queue of (arrival, session_idx,
    # turn_within_session_idx) so the simulator can pop arrivals as wall
    # clock advances.  Sessions queue their NEXT turn only when the
    # previous turn for the same session has been admitted.
    next_turn_idx_per_session: list[int] = [0] * len(parsed_sessions)
    completed_turns: list[set[int]] = [set() for _ in parsed_sessions]

    block_size = max(1, int(cfg.cache_block_size))
    max_tokens = max(1, int(cfg.max_num_batched_tokens))
    max_seqs = max(1, int(cfg.max_num_seqs))
    chunk_tokens = max(1, int(cfg.prefill_chunk_tokens))
    total_blocks = max(1, int(cfg.available_gpu_kv_blocks))
    max_admissions = max(0, int(cfg.max_admissions_per_step))

    pool = BlockPool(total_blocks=total_blocks, block_size=block_size)

    # Helpers for building shape from a session turn.
    next_req_id = [0]

    def _new_request(session_idx: int, turn: ReplayTurnSpec) -> _RequestState:
        # Each session-turn becomes a unique request_id.
        rid = next_req_id[0]
        next_req_id[0] += 1
        # total_context_tokens = cached_context + new_prefill (approximation).
        total = max(1, turn.cached_context_tokens + turn.new_prefill_tokens)
        shape = VllmRequestShape(
            request_id=rid,
            total_context_tokens=total,
            cached_context_tokens=max(0, turn.cached_context_tokens),
            new_prefill_tokens=max(1, turn.new_prefill_tokens),
            output_tokens=max(1, turn.output_tokens),
            shared_prefix_tokens=0,
            next_turn_new_prefill_tokens=0,
        )
        block_hashes = _default_block_hashes(shape, block_size)
        state = _RequestState(
            shape=shape,
            prefill_remaining=0,
            output_remaining=max(1, turn.output_tokens),
            is_followup=False,
        )
        state._block_hashes = block_hashes  # type: ignore[attr-defined]
        state._num_tokens = total  # type: ignore[attr-defined]
        state._admitted = False  # type: ignore[attr-defined]
        state._cached_blocks_at_admit = 0  # type: ignore[attr-defined]
        state._owned_blocks = 0  # type: ignore[attr-defined]
        state._session_idx = session_idx  # type: ignore[attr-defined]
        state._turn_index = turn.turn_index  # type: ignore[attr-defined]
        return state

    def _admit(candidate: _RequestState, budget: int) -> tuple[int, int]:
        block_hashes = candidate._block_hashes  # type: ignore[attr-defined]
        num_tokens = candidate._num_tokens  # type: ignore[attr-defined]
        max_hit_blocks = max(0, num_tokens // block_size - (1 if num_tokens % block_size == 0 else 0))
        cached_blocks = pool.find_longest_cache_hit(block_hashes, max_hit_blocks=max_hit_blocks)
        num_computed_tokens = cached_blocks * block_size
        new_tokens_needed = num_tokens - num_computed_tokens
        scheduled = min(new_tokens_needed, budget, chunk_tokens)
        if scheduled <= 0:
            return (-1, budget)
        blocks_needed = pool.tokens_to_blocks(num_tokens) - cached_blocks
        if not pool.allocate(candidate.shape.request_id, blocks_needed):
            return (-1, budget)
        candidate._admitted = True  # type: ignore[attr-defined]
        candidate._cached_blocks_at_admit = cached_blocks  # type: ignore[attr-defined]
        candidate._owned_blocks = blocks_needed  # type: ignore[attr-defined]
        candidate.prefill_remaining = new_tokens_needed - scheduled
        candidate.prefilled_tokens = scheduled
        candidate.resident_cached_tokens = num_computed_tokens
        return (scheduled, budget - scheduled)

    waiting: list[_RequestState] = []
    running: list[_RequestState] = []
    steps: list[MultiTurnReplayStep] = []
    step_id = 0
    cum_wall_ms = 0.0
    # Tracks the max turn_index ever admitted to the scheduler — mirrors
    # the real vLLM trace's per-step ``turn_index`` tag, which advances
    # only on the first admission of a new turn's request.
    cohort_turn_so_far = -1

    def _enqueue_arrivals() -> None:
        # For each session, queue its next turn IFF:
        #   1. arrival_offset_ms has passed (we're at/past wall time T),
        #   2. The session has no turn already in waiting/running, and
        #   3. **Cohort barrier**: all OTHER sessions have at least caught
        #      up to (next_turn_idx >= this turn's index).  Real bench
        #      clients are cohort-synced at turn boundaries — they fire
        #      turn N+1 for every session only after ALL turn-N's
        #      complete.  Empirical: max(turn-N completion) ≈
        #      min(turn-N+1 arrival) within 1 ms in the paired bench JSON.
        sessions_in_waiting = {
            getattr(w, "_session_idx", -1) for w in waiting
        }
        sessions_in_running = {
            getattr(r, "_session_idx", -1) for r in running
        }
        for si, session in enumerate(parsed_sessions):
            idx = next_turn_idx_per_session[si]
            if idx >= len(session):
                continue
            turn = session[idx]
            if turn.arrival_offset_ms > cum_wall_ms:
                continue
            if si in sessions_in_waiting or si in sessions_in_running:
                continue
            # Cohort barrier: don't queue session si's turn idx if ANY
            # other session is still working on a turn earlier than idx
            # (i.e. has such a request currently in waiting or running).
            # Real bench fires turn N+1 only after every session has
            # completed turn N.
            peer_blocking = False
            for w in waiting:
                w_si = getattr(w, "_session_idx", -1)
                if w_si == si:
                    continue
                if getattr(w, "_turn_index", -1) < idx:
                    peer_blocking = True
                    break
            if not peer_blocking:
                for r in running:
                    r_si = getattr(r, "_session_idx", -1)
                    if r_si == si:
                        continue
                    if getattr(r, "_turn_index", -1) < idx:
                        peer_blocking = True
                        break
            if peer_blocking:
                continue
            waiting.append(_new_request(si, turn))
            next_turn_idx_per_session[si] = idx + 1

    _enqueue_arrivals()

    pending_remaining = lambda: any(
        next_turn_idx_per_session[i] < len(parsed_sessions[i])
        for i in range(len(parsed_sessions))
    )

    def _next_pending_arrival() -> float | None:
        out: float | None = None
        for si, session in enumerate(parsed_sessions):
            idx = next_turn_idx_per_session[si]
            if idx >= len(session):
                continue
            arr = session[idx].arrival_offset_ms
            if out is None or arr < out:
                out = arr
        return out

    steps_taken = 0
    while waiting or running or pending_remaining():
        if max_steps is not None and steps_taken >= max_steps:
            break
        steps_taken += 1
        # If nothing's currently runnable but a future arrival exists, fast-
        # forward the wall clock to that arrival (don't record an empty
        # step — the trace doesn't either).
        if not waiting and not running:
            next_arr = _next_pending_arrival()
            if next_arr is not None and next_arr > cum_wall_ms:
                cum_wall_ms = next_arr
                _enqueue_arrivals()
                continue

        token_budget = max_tokens
        decode_request_ids: list[int] = []
        decoded_turns: list[int] = []
        prefill_chunks: list[VllmPrefillChunk] = []
        prefill_turns: list[int] = []
        prefill_seqs = 0
        prefill_tokens_step = 0
        admitted_ids: list[int] = []
        completed_request_ids: list[int] = []
        completed_prefill_ids: list[int] = []
        capacity_waiting = 0

        # Phase 1: decode running ready requests.
        for state in list(running):
            if state.finished or token_budget <= 0:
                continue
            if state.prefill_remaining > 0:
                continue
            if state.output_remaining <= 0:
                continue
            state.output_remaining -= 1
            state.decode_emitted += 1
            state.resident_decode_tokens += 1
            decode_request_ids.append(state.shape.request_id)
            decoded_turns.append(state._turn_index)  # type: ignore[attr-defined]
            token_budget -= 1
            if state.finished:
                completed_request_ids.append(state.shape.request_id)
                si = state._session_idx  # type: ignore[attr-defined]
                completed_turns[si].add(state._turn_index)  # type: ignore[attr-defined]

        # Phase 2: continued prefill for running, not-yet-ready requests.
        for state in list(running):
            if state.prefill_remaining <= 0 or token_budget <= 0:
                continue
            tokens = min(state.prefill_remaining, token_budget, chunk_tokens)
            state.prefill_remaining -= tokens
            state.prefilled_tokens += tokens
            prefix_tokens = state.resident_cached_tokens + (state.prefilled_tokens - tokens)
            prefill_chunks.append(
                VllmPrefillChunk(
                    request_id=state.shape.request_id,
                    scheduled_tokens=tokens,
                    prefix_tokens=prefix_tokens,
                )
            )
            prefill_turns.append(state._turn_index)  # type: ignore[attr-defined]
            prefill_seqs += 1
            prefill_tokens_step += tokens
            token_budget -= tokens
            if state.prefill_remaining == 0:
                completed_prefill_ids.append(state.shape.request_id)
                hashes = state._block_hashes  # type: ignore[attr-defined]
                pool.cache_blocks(hashes[:pool.tokens_to_blocks(state._num_tokens)])  # type: ignore[attr-defined]

        # Phase 3: admit waiting requests, gated by max_admissions / pool /
        # budget / max_num_partial_prefills.
        admissions_this_step = 0
        partial_prefill_cap = max(0, int(cfg.max_num_partial_prefills))
        while (
            waiting
            and (max_admissions == 0 or admissions_this_step < max_admissions)
            and len(running) < max_seqs
            and token_budget > 0
            and (partial_prefill_cap == 0 or admissions_this_step < partial_prefill_cap)
        ):
            candidate = waiting[0]
            scheduled, token_budget = _admit(candidate, token_budget)
            if scheduled < 0:
                capacity_waiting = max(capacity_waiting, len(waiting))
                break
            waiting.pop(0)
            running.append(candidate)
            admitted_ids.append(candidate.shape.request_id)
            # Trace-tag bookkeeping: the trace flips turn_index when the
            # first admission of a new turn lands.  We do the same by
            # taking the max of any newly-admitted candidate's turn.
            candidate_turn = int(candidate._turn_index)  # type: ignore[attr-defined]
            if candidate_turn > cohort_turn_so_far:
                cohort_turn_so_far = candidate_turn
            prefix_tokens = candidate.resident_cached_tokens
            prefill_chunks.append(
                VllmPrefillChunk(
                    request_id=candidate.shape.request_id,
                    scheduled_tokens=scheduled,
                    prefix_tokens=prefix_tokens,
                )
            )
            prefill_turns.append(candidate._turn_index)  # type: ignore[attr-defined]
            prefill_seqs += 1
            prefill_tokens_step += scheduled
            admissions_this_step += 1
            if candidate.prefill_remaining == 0:
                completed_prefill_ids.append(candidate.shape.request_id)
                hashes = candidate._block_hashes  # type: ignore[attr-defined]
                pool.cache_blocks(hashes[:pool.tokens_to_blocks(candidate._num_tokens)])  # type: ignore[attr-defined]

        # Free finished requests' blocks.
        for state in list(running):
            if state.finished:
                pool.free_request(state.shape.request_id)
                running.remove(state)

        # Build the step shape, price it, advance wall clock.
        used_blocks = pool.total_blocks - pool.get_num_free_blocks()
        step_shape = VllmStepShape(
            step_id=step_id,
            decode_batch=len(decode_request_ids),
            decoded_request_ids=tuple(decode_request_ids),
            prefill_seqs=prefill_seqs,
            prefill_tokens=prefill_tokens_step,
            prefill_chunks=tuple(prefill_chunks),
            waiting_queue=len(waiting),
            running_queue=len(running),
            free_kv_blocks=pool.get_num_free_blocks(),
            completed_prefill_request_ids=tuple(completed_prefill_ids),
            completed_request_ids=tuple(completed_request_ids),
            used_kv_blocks=used_blocks,
            capacity_waiting_requests=capacity_waiting,
            admitted_request_ids=tuple(admitted_ids),
            preemptions=0,
            preempted_request_ids=(),
            recompute_prefill_tokens=0,
        )
        # Need a context_lens mapping for the pricer.
        context_lens = {
            rid: getattr(_state_by_id(running, rid), "_num_tokens", 1)  # type: ignore[attr-defined]
            for rid in decode_request_ids
        }
        priced_ms = max(0.0, float(step_pricer(step_shape, context_lens)))
        cum_wall_ms += priced_ms
        steps.append(
            MultiTurnReplayStep(
                step=step_shape,
                cum_wall_ms_end=cum_wall_ms,
                decoded_turn_indices=tuple(decoded_turns),
                prefill_turn_indices=tuple(prefill_turns),
                cohort_turn_index=cohort_turn_so_far,
            )
        )
        step_id += 1

        # Now advance time and enqueue any newly-arrived session turns.
        _enqueue_arrivals()

        # Sanity: prevent infinite loop when nothing progresses.
        if (
            not decode_request_ids
            and not prefill_chunks
            and not admitted_ids
            and not waiting
            and not pending_remaining()
        ):
            break

    return MultiTurnReplayResult(
        steps=tuple(steps),
        sessions=tuple(parsed_sessions),
    )


def _state_by_id(states: Sequence[_RequestState], request_id: int) -> _RequestState | None:
    for state in states:
        if state.shape.request_id == request_id:
            return state
    return None


def summarize_replay_for_turn(
    result: MultiTurnReplayResult,
    target_turn: int,
) -> VllmTurnShapeSummary:
    """Roll up replay steps to a per-(target_turn) summary.

    Bucketing uses ``step.cohort_turn_index`` — the max turn_index ever
    admitted to the scheduler at or before this step.  This mirrors the
    real-vLLM trace's per-step ``turn_index`` field, which empirically
    advances only on prefill steps where a new turn's first request
    enters (verified across 5 transitions in the terminal c=80 trace).
    Step shape categorization (decode_only / prefill_only / mixed)
    follows the existing trace aggregator: based on decode_batch +
    prefill_tokens for the WHOLE step.
    """
    # No window construction needed — cohort_turn_index is precomputed per step.
    decode_only = 0
    prefill_only = 0
    mixed = 0
    total_steps = 0
    total_decode_slots = 0
    total_prefill_tokens = 0
    max_decode_batch = 0
    decode_batch_sum = 0
    capacity_waiting_max = 0
    min_free_kv = None
    for s in result.steps:
        if s.cohort_turn_index != target_turn:
            continue
        total_steps += 1
        d = s.step.decode_batch
        p_tokens = s.step.prefill_tokens
        total_decode_slots += d
        total_prefill_tokens += p_tokens
        max_decode_batch = max(max_decode_batch, d)
        decode_batch_sum += d
        if min_free_kv is None or s.step.free_kv_blocks < min_free_kv:
            min_free_kv = s.step.free_kv_blocks
        capacity_waiting_max = max(capacity_waiting_max, s.step.capacity_waiting_requests)
        if d > 0 and p_tokens == 0:
            decode_only += 1
        elif d == 0 and p_tokens > 0:
            prefill_only += 1
        elif d > 0 and p_tokens > 0:
            mixed += 1
    mean_decode_batch = (
        decode_batch_sum / total_steps if total_steps > 0 else 0.0
    )
    scheduled_count = sum(
        1 for session in result.sessions for t in session if t.turn_index == target_turn
    )
    return VllmTurnShapeSummary(
        steps=total_steps,
        decode_only_steps=decode_only,
        prefill_only_steps=prefill_only,
        mixed_decode_prefill_steps=mixed,
        empty_or_unknown_steps=0,
        total_decode_slots=total_decode_slots,
        total_prefill_tokens=total_prefill_tokens,
        total_scheduled_tokens=total_decode_slots + total_prefill_tokens,
        max_decode_batch=max_decode_batch,
        mean_decode_batch=mean_decode_batch,
        mean_decode_batch_ratio=(
            mean_decode_batch / scheduled_count if scheduled_count > 0 else 0.0
        ),
        max_waiting_queue=0,
        max_running_queue=0,
        min_free_kv_blocks=min_free_kv or 0,
        max_used_kv_blocks=0,
        max_capacity_waiting_requests=capacity_waiting_max,
        total_preemptions=0,
        total_recompute_prefill_tokens=0,
        prefill_intrusion_candidate=mixed > 0,
        effective_microbatching_candidate=False,
        decode_wave_candidate=False,
        kv_pressure_candidate=(min_free_kv or 0) <= 0,
        bandwidth_capped_steps=0,
        max_decode_kv_read_bytes=0.0,
        total_decode_kv_read_bytes=0.0,
        max_bandwidth_waiting_decode_requests=0,
    )
