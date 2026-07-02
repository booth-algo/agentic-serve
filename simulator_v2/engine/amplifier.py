# simulator_v2/engine/amplifier.py

import math

from simulator_v2.core.mode import Mode, mode
from simulator_v2.core.types import Hardware, Turn

# Output-sustain gate
# A turn that emits too few tokens finishes before the eviction wall builds, so it can't
# reach the ceiling however high the instantaneous pressure is.
SAT_SUSTAIN_LO = 9.0
SAT_SUSTAIN_HI = 24.0

@mode(Mode.SHARED)
def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Smooth 0->1 ramp: 0 at/below `lo`, 1 at/above `hi`, a smooth (C¹) curve in
    between. Used here as the output-length sustain gate."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    u = (x - lo) / (hi - lo)
    return u * u * (3.0 - 2.0 * u)


@mode(Mode.SHARED)
def _ctx_tokens(turn: Turn) -> float:
    """Per-session resident KV size (tokens) at the midpoint of decoding: the
    cached prefix + this turn's new prefill + half the output (the average number
    of tokens generated so far over the turn). This is the `ctx` that both the
    floor lookup and the KV-pressure calc use."""
    return turn.cache_hit_tokens + turn.new_prefill_tokens + 0.5 * turn.osl_tokens


@mode(Mode.SHARED)
def _overflow_weight(
    pressure: float, z: float, sched: float, b_eff: float,
    ctx: float, out: float, context_spread: float, budget_cap: float,
) -> float:
    """The eviction-recompute duty cycle, in [0, 1]: the fraction of this turn's
    decode steps the engine spends re-prefilling evicted sessions instead of
    generating tokens. This is what drags TPOT up from the floor toward the ceiling.

    It is 0 until the KV pool is physically full (`pressure >= 1`) AND the cohort's
    total demand overruns it (`z = pressure·context_spread > 1`). Past that point,
    per turn:

      - a `(1 - 1/z)` fraction of sessions can't stay resident, so vLLM evicts them
        (LIFO) and later RECOMPUTES them from scratch (no swap path):
        `n_evicted = (1 - 1/z) · scheduled` victims.
      - each victim re-prefills its ~`ctx·context_spread` tokens in budget-sized chunks
        (`budget = max_num_batched_tokens - b_eff`), taking `chunk_steps` steps.
      - so recompute eats `n_evicted · chunk_steps` of the turn's `out` decode
        steps  ->  `w = n_evicted · chunk_steps / out`.
      - a victim needing >= 2 chunks monopolizes re-admission, so the cohort cycles
        through the pool ~z times per turn  ->  `w *= z`.

    Clamped to 1 (a step can't be more than fully spent on recompute).
    """
    if pressure < 1.0 or z <= 1.0:
        return 0.0
    budget = max(1.0, budget_cap - b_eff)
    n_evicted = (1.0 - 1.0 / z) * max(1.0, sched)
    chunk_steps = math.ceil(ctx * max(1e-9, context_spread) / budget)
    w = n_evicted * chunk_steps / max(1.0, out)
    if chunk_steps >= 2:
        w *= z  # rotation amplification: the standing overflow re-evicts ~z×/turn
    return min(1.0, w)


@mode(Mode.SHARED)
def tpot_ms(hw: Hardware, turn: Turn, *, context_spread: float = 1.0) -> float:
    """Per-turn TPOT (ms): the decode-step floor ramped toward the saturated-ITL
    ceiling in proportion to how overloaded the KV cache is.

        ITL = floor + (w · sustain) · (ceiling − floor)

      - floor   = decode-step time at the KV-throttled batch
                  `b_eff = min(scheduled, pool capacity)` -- the unsaturated cost.
      - ceiling = the measured plateau TPOT under heavy overload (only looked up
                  when `w > 0`).
      - w       = eviction-recompute duty cycle (`_overflow_weight`): 0 when the
                  pool isn't overloaded, -> 1 when every step is recompute.
      - sustain = output-length gate: a turn too short to build the eviction wall
                  stays near the floor however high the instantaneous pressure is.

    `context_spread` is the cohort's context-size spread (`z = pressure·
    context_spread`); default 1.0 means "median-session pressure,
    onset exactly at pool-full" (the no-spread fallback). Because the ceiling is
    read only when `w > 0`, forward hardware (which has no ceiling) still runs
    everywhere except a genuinely saturated turn.
    """
    out = max(1.0, float(turn.osl_tokens))
    sched = max(1.0, float(turn.scheduled_requests))
    ctx = _ctx_tokens(turn)

    per_session_blocks = max(1, math.ceil(ctx / max(1, hw.cache_block_size)))
    capacity_batch = max(1.0, hw.kv_pool_blocks / per_session_blocks)
    b_eff = max(1.0, min(sched, capacity_batch))
    pressure = sched * per_session_blocks / hw.kv_pool_blocks
    z = pressure * max(0.0, float(context_spread))

    floor = hw.decode_step_ms(b_eff, ctx)

    budget_cap = hw.sched.max_num_batched_tokens
    if not budget_cap:
        raise ValueError("amplifier needs hw.sched.max_num_batched_tokens to be set")
    w = _overflow_weight(pressure, z, sched, b_eff, ctx, out, context_spread, float(budget_cap))
    w *= _smoothstep(out, SAT_SUSTAIN_LO, SAT_SUSTAIN_HI)

    # Unsaturated: the ceiling is never reached, so don't even look it up. This is
    # what lets forward mode (no ceiling) run everywhere except genuine saturation.
    if w <= 0.0:
        return floor
    ceiling = max(floor, hw.saturated_ceiling_ms(out))
    return floor + w * (ceiling - floor)
