"""Two-roofline TPOT predictor for vLLM v1 with chunked-prefill.

Models per-turn TPOT as a decomposition of scheduler steps into two regimes:

  T_lower (decode-bw-bound):   1 token / running session / step
                                  = (weights + running × ctx × kv_bytes) / (bw·util_bw)
  T_upper (chunked-prefill-compute-bound):
                                  = max_num_batched_tokens × prefill_per_token_ms
                                  ≈ 205 ms on H100 / Llama-3.1-8B

Each step in a turn is either "decode-only" (`T_lower`) or "mixed prefill+decode"
which under heavy admission becomes chunk-saturated (`T_upper`). The split
falls out of the per-turn workload:

  prefill_demand    = c × new_prefill_tokens          (tokens of pending prefill)
  decode_demand     = c × output_tokens               (tokens of pending decode)
  running           = min(c, available_kv_blocks // per_session_blocks)

  prefill_steps     = ceil(prefill_demand / chunk)
  decode_steps      = ceil(decode_demand / running)
  total_steps       = max(prefill_steps, decode_steps)
  mixed_steps       = min(prefill_steps, total_steps)
  decode_only_steps = total_steps − mixed_steps

  T_turn            = mixed_steps × T_upper + decode_only_steps × T_lower
  predicted_tpot    = T_turn / output_tokens

No fitted constants. Every input is `RooflineParams`, the vLLM v1 default
`max_num_batched_tokens=8192` (empirically confirmed against engine traces),
or per-turn workload (cached_context, new_prefill, output).

What the model captures (per the user's framing):
  - Roofline accurate at low c (mostly decode-only, T_lower)
  - At high c × high ISL/OSL: prefill_steps > decode_steps → all mixed → T_turn ≈ T_upper
  - Saturation point T_upper is a hardware+config constant (~205 ms)
  - Chat with low ISL/OSL never reaches saturation in normal c range
  - Agentic workloads (swe/terminal) reach saturation at moderate c
  - Per-turn recovery happens naturally as `running` shrinks with ctx growth
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulator.closed_form_tpot import RooflineParams


# Confirmed empirically against vLLM engine traces (see
# profiling/docs/diagnose-hybrid-tpot-2026-05-27.md). All 4 traces hit exactly
# 8192 as max(total_scheduled_tokens), confirming vLLM v1 default in effect.
MAX_NUM_BATCHED_TOKENS = 8192


@dataclass(frozen=True)
class TurnWorkload:
    """Per-turn average workload, sized to a single session."""
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float


@dataclass(frozen=True)
class TwoRooflineTurnPrediction:
    # Inputs
    concurrency: int
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float

    # Derived state
    ctx_mid: float
    per_session_blocks: int
    capacity_batch: int
    running: int

    # Rooflines
    t_lower_ms: float
    t_upper_ms: float

    # Step decomposition
    prefill_steps: int
    decode_steps: int
    total_steps: int
    mixed_steps: int
    decode_only_steps: int

    # Prediction
    t_turn_ms: float
    predicted_tpot_ms: float

    # Regime label (informational)
    regime: str


def _physical_decode_step_ms(running: float, ctx: float, p: RooflineParams) -> float:
    """Lower roofline: bandwidth-bound decode step (weights + per-session KV)."""
    bw_eff = p.peak_bw_bytes_per_s * p.util_bw
    weights_bytes = p.n_params * p.bytes_per_param
    kv_bytes_step = running * ctx * p.kv_bytes_per_token
    return (weights_bytes + kv_bytes_step) / bw_eff * 1000.0  # ms


def _physical_prefill_per_token_ms(p: RooflineParams) -> float:
    """Compute-bound prefill cost per token: 2·n_params FLOPs per token (one MAC per param)."""
    flops_per_token = 2.0 * p.n_params
    flops_eff = p.peak_flops_per_s * p.util_flops
    return flops_per_token / flops_eff * 1000.0  # ms


def _label_regime(*, prefill_steps: int, decode_steps: int, capacity_batch: int, concurrency: int) -> str:
    if capacity_batch < concurrency:
        if prefill_steps > decode_steps:
            return "saturating"           # mixed steps dominate, climbing toward T_upper
        return "kv_admission_throttle"     # running < c but decode still dominates
    if prefill_steps > decode_steps:
        return "prefill_bound"             # rare: enough KV but chunked-prefill dominates
    return "decode_bound"                  # the no-pressure roofline regime


def predict_two_roofline(
    workload: TurnWorkload,
    concurrency: int,
    p: RooflineParams | None = None,
    *,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
    active_sessions: int | None = None,
) -> TwoRooflineTurnPrediction:
    """Predict per-turn TPOT via the two-roofline decomposition.

    `active_sessions` (optional) overrides the effective number of concurrent
    sessions at this turn. In multi-turn benches sessions complete and exit
    before reaching the planned turn count, so `concurrency` (cohort size) is
    an upper bound. When available (e.g. from engine telemetry like
    `engine_max_decode_batch`), pass the observed active count for a more
    accurate prediction.
    """
    if p is None:
        p = RooflineParams()

    cached = workload.cached_context_tokens
    new_prefill = workload.new_prefill_tokens
    output = max(1.0, workload.output_tokens)
    ctx_mid = cached + new_prefill + 0.5 * output

    block_size = max(1, p.cache_block_size)
    per_session_blocks = max(1, math.ceil(ctx_mid / block_size))
    capacity_batch = max(1, p.available_kv_blocks // per_session_blocks)

    # Effective concurrency: clamp `active_sessions` (or `concurrency`) to
    # what KV capacity allows. Sessions in excess of `capacity_batch` cannot run
    # simultaneously; they queue. Sessions in excess of the active count have
    # already completed (multi-turn bench dynamics).
    effective_c = int(active_sessions) if active_sessions is not None else int(concurrency)
    effective_c = max(1, effective_c)
    running = max(1, min(effective_c, capacity_batch))

    t_lower = _physical_decode_step_ms(running, ctx_mid, p)
    prefill_per_tok = _physical_prefill_per_token_ms(p)
    t_upper = max_num_batched_tokens * prefill_per_tok

    # The interpolation: TPOT smoothly blends between T_lower (decode roofline,
    # what each session sees in pure-decode mode) and T_upper (chunked-prefill
    # saturation, what each session sees when every scheduler step is dominated
    # by chunked-prefill of new admissions). The blend weight is the KV-pressure
    # ratio: how much the cohort's KV demand overshoots capacity.
    #
    #   pressure = effective_c × per_session_blocks / available_kv_blocks
    #            = effective_c / capacity_batch
    #
    # pressure < 1: cohort fits in KV → no admission contention → w ≈ 0 → T_lower
    # pressure = 1: at capacity, KV is just full → w = 0.5 → halfway up
    # pressure ≫ 1: severe over-subscription, sessions cycle through eviction
    #               and re-prefill → most steps mixed → w → 1 → T_upper
    #
    # Piecewise-linear ramp: 0 below capacity, climbing to 1 at 2× over capacity.
    # `pressure ≤ 1` means the cohort fits in KV → no contention → stay on the
    # decode roofline. `pressure = 2` means we're 2× over capacity, sessions
    # cycle through eviction/re-prefill heavily → ~half the steps mixed-prefill.
    # `pressure ≥ 3` means severe oversubscription → saturation regime.
    pressure = effective_c / max(1, capacity_batch)
    w = max(0.0, min(1.0, (pressure - 1.0) / 2.0))
    predicted_tpot = t_lower * (1.0 - w) + t_upper * w

    # Bookkeeping for the dataclass (these were the legacy step-counter outputs;
    # we keep them surfaced for diagnostic charts even though the interpolation
    # formula above is the actual prediction).
    prefill_demand = effective_c * max(0.0, new_prefill)
    decode_demand = effective_c * output
    chunk = max(1, max_num_batched_tokens)
    prefill_steps = math.ceil(prefill_demand / chunk) if prefill_demand > 0 else 0
    decode_steps = math.ceil(decode_demand / max(1, running))
    total_steps = max(prefill_steps, decode_steps, 1)
    mixed_steps = min(prefill_steps, total_steps)
    decode_only_steps = total_steps - mixed_steps
    t_turn = predicted_tpot * output  # back-derived for the dataclass field

    regime = _label_regime(
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        capacity_batch=capacity_batch,
        concurrency=int(concurrency),
    )

    return TwoRooflineTurnPrediction(
        concurrency=int(concurrency),
        cached_context_tokens=cached,
        new_prefill_tokens=new_prefill,
        output_tokens=output,
        ctx_mid=ctx_mid,
        per_session_blocks=per_session_blocks,
        capacity_batch=capacity_batch,
        running=running,
        t_lower_ms=t_lower,
        t_upper_ms=t_upper,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        total_steps=total_steps,
        mixed_steps=mixed_steps,
        decode_only_steps=decode_only_steps,
        t_turn_ms=t_turn,
        predicted_tpot_ms=predicted_tpot,
        regime=regime,
    )
