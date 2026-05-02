"""Event-driven scheduler simulation for multi-turn TTFT/TPOT/E2EL.

Models vLLM/SGLang continuous batching: decode-first scheduling, prefill
chunked into remaining token budget, batch drains as outputs complete.

Assumptions:
  - Turns are sequential (empty server at turn start).
  - Within a turn, all requests arrive simultaneously.
  - max_num_batched_tokens = 8192 (vLLM default, adjustable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .composer import Composer
from .configs.model_configs import ModelConfig


MAX_NUM_BATCHED_TOKENS = 8192


@dataclass
class _SimRequest:
    req_id: int
    prefill_total: int
    prefill_remaining: int
    total_context: int
    output_remaining: int
    state: str = "prefill"  # prefill | first_decode | decoding | finished
    ttft: float = 0.0
    decode_wall_ms: float = 0.0
    decode_steps: int = 0


def _step_time_ms(composer: Composer, cfg: ModelConfig, tp: int,
                  num_decode: int, num_prefill_tokens: int,
                  max_context: int, ttft_floor_ms: float) -> float:
    """Wall-clock time for one scheduler iteration.

    num_decode    — D, number of active decode requests (1 token each)
    num_prefill_tokens — C, prefill tokens allocated this step
    max_context   — max KV length across active requests
    """
    # Decode: D tokens, each with Q=1, KV=context
    decode_us = composer.predict_decode_step_us(
        cfg, kv_len=max_context, bs=max(1, num_decode),
        tensor_parallel_size=tp,
    )
    # Prefill: C tokens, Q=C, KV=context (approximate as equivalent prefill work)
    prefill_us = 0.0
    if num_prefill_tokens > 0:
        prefill_us = composer.predict_ttft_us(
            cfg, num_prefill_tokens, bs=1, kv_len=max_context,
            tensor_parallel_size=tp,
        )
    # Step time: scheduler alternates, so take the sum of decode + prefill
    # overhead. In practice they run in the same forward pass, but the
    # composer doesn't model mixed batches, so sum is conservative.
    step_us = decode_us + prefill_us
    return step_us / 1000.0 + ttft_floor_ms


def _ttft_floor_ms(tp: int, n_layers: int) -> float:
    """Fixed per-forward-pass overhead."""
    tp_barrier_us = 5.0 * 5.0 * n_layers if tp > 1 else 0.0
    scheduler_overhead_us = 500.0
    return (tp_barrier_us + scheduler_overhead_us) / 1000.0


def simulate_turn(composer: Composer, cfg: ModelConfig, tp: int,
                  requests: list[dict[str, int]],
                  max_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
                  ) -> list[dict[str, float]]:
    """Simulate one turn with N simultaneous arrivals.

    Each request dict: prefill_tokens, total_context, output_tokens.

    Returns per-request: ttft_ms, tpot_ms, e2el_ms.
    """
    if not requests:
        return []

    n_layers = cfg.n_layers
    floor_ms = _ttft_floor_ms(tp, n_layers)
    sim_requests = [
        _SimRequest(
            req_id=i,
            prefill_total=r["prefill_tokens"],
            prefill_remaining=r["prefill_tokens"],
            total_context=r["total_context"],
            output_remaining=max(0, r["output_tokens"] - 1),  # TTFT counts first token
        )
        for i, r in enumerate(requests)
    ]

    wall_clock = 0.0
    step = 0
    active = [r for r in sim_requests if r.state == "prefill"]

    # Phase 1: prefill. Independent prompts can't share prefill batching —
    # each request's unique text requires separate forward passes.
    # Prefill one request per step, interleaved with decode for already-active.
    while any(r.prefill_remaining > 0 for r in active):
        num_decode = sum(1 for r in active if r.state == "decoding")
        max_ctx = max(r.total_context for r in active)

        # Pick one prefill request (first in line) and process its remaining tokens
        pending_prefill = [r for r in active if r.state == "prefill"]
        pref_tokens_this_step = 0
        if pending_prefill:
            r = pending_prefill[0]
            pref_tokens_this_step = r.prefill_remaining
            r.prefill_remaining = 0
            r.state = "first_decode"

        step_ms = _step_time_ms(composer, cfg, tp, num_decode, pref_tokens_this_step,
                                max_ctx, floor_ms)
        wall_clock += step_ms

        # Decode: advance tokens for decoding requests
        for r in active:
            if r.state == "decoding":
                r.output_remaining -= 1
                r.decode_wall_ms += step_ms
                r.decode_steps += 1
                if r.output_remaining <= 0:
                    r.state = "finished"
            elif r.state == "first_decode":
                r.ttft = wall_clock  # TTFT recorded at step after prefill completion
                r.state = "decoding"

        step += 1

        # Decode: advance tokens for decoding requests
        for r in active:
            if r.state == "decoding":
                r.output_remaining -= 1
                r.decode_wall_ms += step_ms
                r.decode_steps += 1
                if r.output_remaining <= 0:
                    r.state = "finished"
            elif r.state == "first_decode":
                # First decode: TTFT recorded at this step boundary
                r.ttft = wall_clock
                r.state = "decoding"

        step += 1

    # Phase 2: pure decode — all requests are now in decode state.
    # Process in batches: advance all requests together until at least one finishes.
    active_decode = [r for r in active if r.state == "decoding"]
    while active_decode:
        D = len(active_decode)
        max_ctx = max(r.total_context for r in active_decode)

        # How many steps until the first decode request finishes?
        min_remaining = min(r.output_remaining for r in active_decode)
        batch_steps = min_remaining  # advance all requests by min_remaining steps

        decode_step_ms = _step_time_ms(composer, cfg, tp, D, 0, max_ctx, floor_ms)
        wall_clock += decode_step_ms * batch_steps

        for r in active_decode:
            r.output_remaining -= batch_steps
            r.decode_wall_ms += decode_step_ms * batch_steps
            r.decode_steps += batch_steps
            if r.output_remaining <= 0:
                r.state = "finished"

        active_decode = [r for r in active_decode if r.state != "finished"]
        step += batch_steps

    # Aggregate
    return [
        {
            "ttft_ms": round(r.ttft, 2),
            "tpot_ms": round(r.decode_wall_ms / max(1, r.decode_steps), 2),
            "e2el_ms": round(r.ttft + r.decode_wall_ms, 2),
        }
        for r in sim_requests
    ]


def simulate_multiturn(
    composer: Composer,
    cfg: ModelConfig,
    gpu: str,
    tp: int,
    turns: list[dict[str, Any]],
    backend: str | None = None,
    backend_version: str | None = None,
    model_key: str | None = None,
    profile: str | None = None,
    max_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
) -> list[dict[str, Any]]:
    """Simulate a multi-turn session: sequential turns into empty server.

    Each turn dict: successful, new_prefill_tokens, total_context_tokens,
                    output_tokens, turn_index.

    Returns per-turn prediction rows (ttft_pred, tpot_pred, e2el_pred, etc.).
    """
    turn_results: list[dict[str, Any]] = []
    for turn in turns:
        n_reqs = max(1, int(turn.get("successful", 1)))
        prefill = max(1, int(turn.get("new_prefill_tokens", 1)))
        total_ctx = max(1, int(turn.get("total_context_tokens", prefill)))
        osl = int(turn.get("output_tokens", 1))

        req_templates = [
            {
                "prefill_tokens": prefill,
                "total_context": total_ctx,
                "output_tokens": osl,
            }
            for _ in range(n_reqs)
        ]

        sim_results = simulate_turn(composer, cfg, tp, req_templates, max_batched_tokens)

        # Weighted average across requests
        n = len(sim_results)
        ttft_ms = sum(r["ttft_ms"] for r in sim_results) / n
        tpot_ms = sum(r["tpot_ms"] for r in sim_results) / n
        e2el_ms = sum(r["e2el_ms"] for r in sim_results) / n

        turn_results.append({
            "turn_index": turn.get("turn_index", len(turn_results)),
            "ttft_pred": round(ttft_ms, 2),
            "tpot_pred": round(tpot_ms, 2),
            "e2el_pred": round(e2el_ms, 2),
            "successful": n_reqs,
        })

    return turn_results
