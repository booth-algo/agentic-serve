"""Forward, closed-loop, event-driven multi-turn TTFT **queue** simulator.

TTFT is not a quasi-static per-turn quantity like TPOT — it is the wall-clock
**queue wait** a request sees before its first token (client_queue_wait ~= 0, so all
server-side queueing is folded into the measured ``ttft_ms``). A static per-turn
formula provably can't capture it (the production static M0 sits at ~62% MAPE): the
per-turn TTFT is a *saturate-ramp-RECOVER* curve that emerges from the cohort's
trajectory — backlog builds while the cohort oversubscribes KV and drains as
sessions finish.

This module borrows **llm-d-inference-sim's EVENT STRUCTURE** (a freeWorkers pool of
size ``max_num_seqs`` + a FIFO ``waitingQueue`` whose entries carry an ``enqueueTime``;
``queue_wait = dequeue_epoch - enqueueTime`` — its ``reqQueueTime`` metric), NOT its
snapshot TTFT formula. The per-turn TTFT here is *emergent* from a forward, closed-loop
discrete-event simulation of continuous-batching steps.

PHYSICAL MODEL (fit-free — NO new fitted constants):

* Cohort: ``C = round(concurrency)`` sessions. Each session's turn-count is drawn
  FORWARD from the profile's turn-count survival histogram
  (``ramp_tpot.forward_survival`` / ``PROFILE_DIST`` — the same source ``sched_hat``
  uses) DETERMINISTICALLY by survival quantile ``q_k = (k+0.5)/C`` (reproducible; no
  stochastic draws, no wall-clock reads).
* Shared GPU server runs continuous-batching steps; admission is gated by KV blocks
  (``BlockPool(available_kv_blocks=27250, cache_block_size=16)``) + ``max_num_seqs``
  + ``MAX_NUM_BATCHED_TOKENS=8192`` (vLLM serving defaults, documented runtime config,
  NOT MAPE-tuned knobs — same discipline as ``available_kv_blocks``).
* Closed loop: a session's turn ``t+1`` ARRIVES at its turn ``t`` COMPLETION epoch
  (think-time == 0, client_queue_wait == 0 — the bench contract). On arrival the
  request joins the waiting FIFO; it is admitted when capacity frees; prefill is
  chunked across steps, intruding into the decode-priority token budget; the first
  token is emitted at prefill completion.
* Per-step wall-time from MEASURED kernels ONLY: a decode step costs
  ``kernel_step_cost.decode_step_ms(batch, ctx)``; a prefill chunk costs
  ``cached_prefill_lookup.cached_prefill_step_ms(new_tokens, prefix_tokens)``. Both are
  measured H100 / Llama-3.1-8B grids (decode 7.4% MAPE).
* ``TTFT[turn] = first_token_epoch - arrival_epoch`` (= queue_wait + prefill_service +
  one decode step). Aggregate per (profile, concurrency, turn_index) -> MEDIAN, the
  byte-identical grouping the measured ``ttft_meas`` uses.

The wave factor ``_decode_residency_wave_factor`` is intentionally NOT multiplied onto
the decode step: admission already caps the running set at the KV capacity batch, so on
the admitted batch it is ~= 1 by construction and applying it would double-count the KV
pressure the queue itself models (see DESIGN risks: WAVE-FACTOR DOUBLE-COUNTING). The
backlog — hence the TTFT ramp — must be carried by FIFO head-of-line blocking, not a
hidden amplifier.

Output is the ADDITIVE column ``ttft_pred_qsim`` (+ ``e2el_pred_qsim = ttft_pred_qsim +
output * tpot_pred_kernel``). The existing ``ttft_pred`` / ``tpot_pred`` / ``tpot_err``
/ ``tpot_pred_kernel`` columns are never repointed (M0 and the kernel headline stay
byte-identical).
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
# These are the only two module-level numeric constants beyond what RooflineParams
# already documents. They are vLLM's stock serving defaults, same discipline as
# RooflineParams.available_kv_blocks — NOT MAPE-tuned knobs.
MAX_NUM_SEQS = 512          # vLLM --max-num-seqs default (freeWorkers pool size)
MAX_NUM_BATCHED_TOKENS = 8192  # vLLM --max-num-batched-tokens default (per-step token budget)

# Event kinds (monotone ints; ordering is (epoch, seq_counter, kind) — no wall-clock,
# no RNG, fully deterministic FIFO at equal epochs).
_ARRIVAL = 0
_STEP = 1
_FIRST_TOKEN = 2
_DEPART = 3

# Largest new-prefill the measured cached-prefill grid covers; above it the grid
# clamps and under-counts, so a fused multi-session prefill pass beyond it is priced by
# the prefill-compute roofline (the same edge ttft_predict._baseline_prefill_ms uses).
_GRID_U_MAX = 1024.0


def _prefill_pass_ms(total_chunk_tokens: float, mean_prefix: float, params: RooflineParams) -> float:
    """Wall-time of ONE fused prefill forward pass over ``total_chunk_tokens`` tokens
    (summed across the step's prefilling reqs) at ``mean_prefix`` cached prefix.

    Measured cached-prefill grid up to the grid's U edge; the prefill-compute roofline
    (2·N·U/(peak_flops·util) + scheduler overhead) above it — continuous with the grid
    at the edge, and the physically-correct large-batch extrapolant. No new constant."""
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
    blocks: int
    remaining_prefill: float
    output_left: int
    # context (cached+new) used as the resident KV length for decode pricing.
    ctx: float


@dataclass
class _ServerState:
    params: RooflineParams
    pool: BlockPool
    sessions: list[Session]
    clock: float = 0.0
    seq: int = 0  # monotone tiebreak counter
    heap: list[tuple[float, int, int, Any]] = field(default_factory=list)
    waiting: list[_Req] = field(default_factory=list)        # FIFO
    prefilling: dict[int, _Req] = field(default_factory=dict)
    running: dict[int, _Req] = field(default_factory=dict)
    # (session_id, turn_index) -> {arrival_epoch, first_token_epoch, completion_epoch}
    results: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    step_scheduled: bool = False

    def push(self, epoch: float, kind: int, payload: Any) -> None:
        heapq.heappush(self.heap, (epoch, self.seq, kind, payload))
        self.seq += 1


def _encode_rid(session_id: int, turn_index: int) -> int:
    """Stable per-(session, turn) request id. 4096 turns max per session — ample."""
    return session_id * 4096 + turn_index


# ----------------------------------------------------------------- cohort builder


def _draw_turn_count(survival: list[float], quantile: float) -> int:
    """Inverse-survival, deterministic. Returns the turn_count (>=1) for a session at
    survival quantile ``quantile``.

    ``survival[t] = S(t)`` = fraction of sessions still alive AT turn ``t`` (S(0)==1).
    A session reaches turns ``0..T-1`` (T turns total) where T is the largest index
    with ``S(T-1) >= quantile`` ... in practice: count how many turns have survival
    >= quantile, that many turns are reached, +1 for the always-present turn 0 floor.
    """
    if not survival:
        return 1
    # Number of turn indices t whose survival S(t) >= quantile. S(0)==1 always counts.
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
    """Deterministic survival-quantile cohort. Each of ``C = round(concurrency)``
    sessions gets a turn_count from the inverse survival at quantile ``(k+0.5)/C`` and
    pulls each turn's TurnSpec from the cell's per-turn dicts (keyed by turn_index),
    reusing exactly the inputs M0 / kernel TPOT consume.
    """
    c = max(1, int(round(float(concurrency))))
    survival = forward_survival(PROFILE_DIST[profile]) if profile in PROFILE_DIST else None

    # Per-turn-index workload lookup (cached/new/output aggregates).
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
        # Fall back to the nearest available lower turn_index (then any) so a session
        # that the survival curve carries past the measured turns still has a workload.
        lower = [i for i in spec_by_idx if i <= idx]
        if lower:
            return spec_by_idx[max(lower)]
        return spec_by_idx[min(spec_by_idx)] if spec_by_idx else TurnSpec(idx, 0.0, 1.0, 1.0)

    sessions: list[Session] = []
    for k in range(c):
        q = (k + 0.5) / c
        if survival:
            tc = _draw_turn_count(survival, q)
            # Cap at the number of turn slots actually present in this cell so we never
            # invent turns past the measured workload.
            tc = min(tc, n_turn_slots) if n_turn_slots > 0 else tc
        else:
            # Unknown profile: no survival drain — every session runs all turn slots.
            tc = n_turn_slots if n_turn_slots > 0 else 1
        tc = max(1, tc)
        session_turns = [spec_for(i) for i in range(tc)]
        sessions.append(Session(session_id=k, turn_count=tc, turns=session_turns))
    return sessions


# ------------------------------------------------------------------- the sim core


def _blocks_for(state: _ServerState, ctx_plus_output: float) -> int:
    return max(1, state.pool.tokens_to_blocks(int(math.ceil(ctx_plus_output))))


def _running_ctx_mean(state: _ServerState) -> float:
    """Mean resident context of the running (decoding) batch — the single ctx the
    decode kernel grid is priced at (kernel_tpot's ctx_mid convention)."""
    if not state.running:
        return 1.0
    return statistics.fmean(r.ctx for r in state.running.values())


def _schedule(state: _ServerState) -> None:
    """FIFO admission: greedily admit head-of-queue waiting reqs while capacity allows.

    Gates (all physical / runtime config):
      (a) |running| + |prefilling| < MAX_NUM_SEQS          (freeWorkers pool)
      (b) pool.allocate(rid, blocks) succeeds              (KV admission, 27250 blocks)
      (c) the step prefill token budget has room           (8192 - decode-slot tokens)

    On the first gate failure (FIFO head-of-line blocking) we stop — that is where the
    queue backlog, hence the TTFT ramp, emerges.
    """
    # Token budget left for prefill this step: 8192 minus 1 token per decode slot.
    decode_slots = len(state.running)
    budget = MAX_NUM_BATCHED_TOKENS - decode_slots
    # Already-committed prefill chunks (this scheduling pass) consume the budget too.
    for r in state.prefilling.values():
        budget -= min(r.remaining_prefill, MAX_NUM_BATCHED_TOKENS)
    while state.waiting:
        if len(state.running) + len(state.prefilling) >= MAX_NUM_SEQS:
            return
        if budget <= 0:
            return
        head = state.waiting[0]
        if not state.pool.allocate(head.rid, head.blocks):
            return  # KV head-of-line block
        state.waiting.pop(0)
        head.remaining_prefill = head.new_prefill
        state.prefilling[head.rid] = head
        budget -= min(head.new_prefill, MAX_NUM_BATCHED_TOKENS)


def _on_arrival(state: _ServerState, session_id: int, turn_index: int) -> None:
    sess = state.sessions[session_id]
    spec = sess.turns[turn_index]
    rid = _encode_rid(session_id, turn_index)
    ctx = spec.cached_context_tokens + spec.new_prefill_tokens
    blocks = _blocks_for(state, ctx + spec.output_tokens)
    req = _Req(
        rid=rid,
        session_id=session_id,
        turn_index=turn_index,
        arrival_epoch=state.clock,
        cached=spec.cached_context_tokens,
        new_prefill=spec.new_prefill_tokens,
        output=spec.output_tokens,
        blocks=blocks,
        remaining_prefill=spec.new_prefill_tokens,
        output_left=max(1, int(round(spec.output_tokens))),
        ctx=ctx,
    )
    state.waiting.append(req)
    state.results[(session_id, turn_index)] = {"arrival_epoch": state.clock}
    _schedule(state)
    _ensure_step(state)


def _ensure_step(state: _ServerState) -> None:
    """Schedule the next engine STEP at the current clock if engine work exists and no
    step is already pending."""
    if state.step_scheduled:
        return
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _price_step(state: _ServerState) -> float:
    """Wall-time of one mixed prefill+decode step from measured kernels.

    The prefilling reqs share ONE fused prefill forward pass per step (vLLM batches all
    scheduled prefill chunks into a single kernel launch over up to the per-step token
    budget — it is NOT one kernel call per request). So the step's prefill cost is the
    measured ``cached_prefill_step_ms`` priced ONCE at the total scheduled prefill
    tokens this step (capped at the leftover budget) against the cohort-mean prefix:

        decode_ms  = decode_step_ms(|running|, mean_running_ctx)        (measured grid)
        prefill_ms = cached_prefill_step_ms(total_chunk_tokens, mean_prefix)
        step_ms    = max(decode_ms, prefill_ms) + scheduler_overhead_ms_per_step

    Prefill intrudes into the same step as decode (continuous batching), so the step is
    the max of the two services plus one scheduler-overhead anchor (5.7 ms, single
    anchor from RooflineParams — not a fit). Each prefilling req's per-step chunk is its
    fair share of the leftover budget; the chunks are stashed on the reqs for the
    bookkeeping pass.
    """
    p = state.params
    decode_batch = len(state.running)
    decode_ms = 0.0
    if decode_batch > 0:
        decode_ms = decode_step_ms(decode_batch, _running_ctx_mean(state), p)

    prefill_ms = 0.0
    # Per-step prefill token budget left after decode slots claim 1 token each.
    budget = max(0, MAX_NUM_BATCHED_TOKENS - decode_batch)
    n_pref = len(state.prefilling)
    if n_pref > 0 and budget > 0:
        share = max(1, budget // n_pref)
        total_chunk = 0.0
        prefixes: list[float] = []
        for r in state.prefilling.values():
            chunk = min(r.remaining_prefill, float(share))
            chunk = max(1.0, chunk)
            r._chunk = chunk  # type: ignore[attr-defined]
            total_chunk += chunk
            prefixes.append(max(1.0, r.cached))
        # One fused prefill pass over the total scheduled chunk tokens (capped at the
        # budget), priced at the cohort-mean cached prefix.
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
        chunk = getattr(r, "_chunk", 0.0)
        r.remaining_prefill -= chunk
        if r.remaining_prefill <= 0:
            finished_prefill.append(rid)
    for rid in finished_prefill:
        state.push(state.clock, _FIRST_TOKEN, rid)

    # --- decode bookkeeping: one token per running req per step ---
    finished_decode: list[int] = []
    for rid, r in state.running.items():
        r.output_left -= 1
        if r.output_left <= 0:
            finished_decode.append(rid)
    for rid in finished_decode:
        r = state.running[rid]
        state.push(state.clock, _DEPART, (r.session_id, r.turn_index))

    # Newly-freed token budget / slots may admit waiting FIFO head.
    _schedule(state)

    # Self-reschedule while engine has work.
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _on_first_token(state: _ServerState, rid: int) -> None:
    r = state.prefilling.pop(rid, None)
    if r is None:
        return
    key = (r.session_id, r.turn_index)
    state.results[key]["first_token_epoch"] = state.clock
    # Move to running (holds a decode slot for output_left steps).
    state.running[rid] = r
    _ensure_step(state)


def _on_depart(state: _ServerState, session_id: int, turn_index: int) -> None:
    rid = _encode_rid(session_id, turn_index)
    r = state.running.pop(rid, None)
    if r is not None:
        state.pool.free_request(rid)
        # Edge case: a 1-token output can DEPART the same step it FIRST_TOKENs; if it
        # never got a first_token_epoch (still prefilling), close it out here.
    key = (session_id, turn_index)
    rec = state.results.get(key)
    if rec is not None:
        rec["completion_epoch"] = state.clock
        rec.setdefault("first_token_epoch", state.clock)

    # Closed loop: next turn arrives at this completion epoch (think-time == 0).
    sess = state.sessions[session_id]
    sess.next_turn_idx = max(sess.next_turn_idx, turn_index + 1)
    if sess.next_turn_idx < sess.turn_count:
        nxt = sess.next_turn_idx
        state.push(state.clock, _ARRIVAL, (session_id, nxt))
    _ensure_step(state)


def _run_sim(
    sessions: list[Session], params: RooflineParams, max_events: int
) -> dict[tuple[int, int], float]:
    """Run the forward closed loop. Returns {(session_id, turn_index): TTFT_ms}."""
    pool = BlockPool(params.available_kv_blocks, params.cache_block_size)
    state = _ServerState(params=params, pool=pool, sessions=sessions)

    # Seed: all sessions' turn 0 arrive at epoch 0 (cohort dispatch t0).
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
    """Group emergent (session, turn) TTFTs by turn_index -> MEDIAN, aligned to the
    input ``turns`` order. A turn_index reached by no simulated session falls back to
    the forward static predictor (predict_turn_ttft) so the list always matches
    ``turns`` length."""
    by_idx: dict[int, list[float]] = {}
    for (_sid, ti), v in ttfts.items():
        if v > 0:
            by_idx.setdefault(ti, []).append(v)

    # Forward fallback inputs (kernel TPOT for the oversubscription backlog term).
    out: list[float] = []
    for t in turns:
        ti = int(t.get("turn_index", 0))
        vals = by_idx.get(ti)
        if vals:
            out.append(statistics.median(vals))
        else:
            out.append(_fallback_ttft(t, profile, concurrency, params))
    return out


def _fallback_ttft(
    turn: dict[str, Any], profile: str, concurrency: float, params: RooflineParams
) -> float:
    """Forward static-formula fallback for a turn no session reached (keeps the list
    length aligned). Uses the kernel-TPOT amplifier for its backlog term."""
    from simulator.ramp_tpot import sched_hat

    ti = int(turn.get("turn_index", 0))
    cached = float(turn.get("cached_context_tokens") or 0.0)
    new = float(turn.get("new_prefill_tokens") or 0.0)
    out = float(turn.get("output_tokens") or 1.0)
    sched = sched_hat(profile, float(concurrency), ti) if profile in PROFILE_DIST else float(concurrency)
    tpot = predict_cell_tpot(
        [KernelTurnInput(cached, new, out, sched)], params
    )[0]
    return predict_turn_ttft(cached, new, out, sched, tpot, params)


# --------------------------------------------------------- oracle (validation only)


def _build_cohort_oracle(
    turns: list[dict[str, Any]], profile: str, concurrency: float
) -> list[Session] | None:
    """Validation-only: overlay measured session_timelines (arrival/completion offsets)
    to build the cohort from the measured per-session turn lists rather than the
    survival quantile. Off the forward path — used only when oracle=True. Returns None
    if the measured timelines are unavailable (sim falls back to the forward cohort)."""
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
    slug = f"{profile}__{int(round(float(concurrency)))}"
    cell = timelines.get(slug)
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
        if not specs:
            continue
        sessions.append(
            Session(session_id=sid, turn_count=len(specs), turns=specs)
        )
    return sessions or None


# ------------------------------------------------------------------- public API


def predict_cell_ttft_qsim(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams | None = None,
    *,
    oracle: bool = False,
    max_events: int = 2_000_000,
) -> list[float]:
    """Per-turn TTFT (ms) for a (profile, concurrency) cell, emergent from a forward
    closed-loop event-driven queue sim.

    Returns one TTFT per input turn, aligned to ``turns`` order by turn_index (median
    over all sessions reaching that turn_index), mirroring
    ``simulator/ttft_predict.py::predict_cell_ttft``. Returns ``[]`` for empty turns;
    a turn_index reached by no simulated session yields a forward fallback
    (``predict_turn_ttft``) so the list length always matches ``turns``.

    Forward by default (cohort + turn-counts from ``forward_survival``; the resident
    cohort is EMERGENT, never the measured ``scheduled_requests``). ``oracle=True``
    overlays measured ``session_timelines`` for validation only (off the forward path).
    """
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
    """E2EL composition: ``e2el_qsim[t] = ttft_qsim[t] + output_tokens[t] * tpot[t]``.

    ``tpot`` defaults to the kernel TPOT (``tpot_pred_kernel``) the emitter passes — no
    re-fit, byte-identical composition with the existing ``e2el_pred`` line.
    """
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
        output = float(t.get("output_tokens") or 0.0)
        out.append(float(ttft) + output * float(tpot))
    return out
