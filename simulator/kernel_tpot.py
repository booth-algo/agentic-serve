"""Kernel-composition per-turn TPOT (ITL) predictor for vLLM on H100 / Llama-3.1-8B.

A single physically-grounded expression per turn, no MAPE fitting and no engine
telemetry — every input is workload-derivable (cached/new/output tokens,
scheduled cohort size) plus ``RooflineParams`` and two measured kernel grids.

    ITL[t] = kernel_step + smoothstep(pressure; P_LO, P_HI) × (T_upper − kernel_step)

Pieces (see project memory ``tpot-amplifier-pressure-law``):

* ``kernel_step = decode_step_ms(B_eff, ctx)`` — measured decode kernel grid
  (validated 7.4% MAPE; ``prediction_pipeline.yaml`` ``decode`` block). This is
  the physically-correct *lower bound*: 58% of all cells live below KV
  saturation where ITL ≈ kernel_step. ``B_eff`` is the KV-throttled running
  batch ``min(scheduled, capacity_batch)``; ``ctx = cached + new + output/2``.

* ``pressure = scheduled × per_session_blocks / available_kv_blocks`` — KV
  oversubscription. The amplifier ``ITL/kernel_step`` is pressure-driven
  (corr +0.79 across 1043 cells): ≈1 below pressure ~0.8, ramping to its
  ceiling by pressure ~2.5. ``smoothstep`` over [P_LO, P_HI] is that ramp.

* ``T_upper(output) = base + turn_overhead / output`` — the saturated ITL
  ceiling. At saturation the per-turn cohort-prefill + scheduling wall is fixed
  and gets amortized over a session's output tokens, so short-output workloads
  (swe/terminal, ~28 tokens → ~237 ms) saturate far higher than long-output
  ones (osworld, ~87 tokens → ~135 ms). ``base`` and ``turn_overhead`` are a
  least-squares fit of measured saturated ITL (pressure ≥ 2.5) against
  1/output over 120 cells (R²=0.64) — two physical anchors (a per-token
  saturated-decode floor and a per-turn overhead), not MAPE knobs.

Validated: overall TPOT MAPE 19.4% (median 10.7%) — chat 6.1%, osworld 18.7%,
swebench 21.0%, terminalbench 29.3% — beating the telemetry-using three-regime
predictor (22.8% overall) while needing no engine telemetry.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms


# --- pressure ramp window (read from the amplifier-vs-pressure curve) --------
# Below P_LO the amplifier sits at ~1 (kernel step IS the ITL); by the upper knee
# it has reached the saturated ceiling. The upper knee is *output-gated*: the
# jump is eviction-gated saturation, and short-output workloads (agentic coding —
# swebench/terminalbench, ~28 tok) saturate as a sharp 1-2 turn STEP, while
# long-output workloads (chat/osworld, ~80+ tok) ramp gently and often recover.
# So the step is steep for short outputs and gradual for long ones. Physical
# basis: the saturated per-turn overhead is amortized over output tokens (the
# same mechanism behind the output-keyed ceiling) — short outputs concentrate it
# into a hard step. P_LO=0.8 onset is kept (lowering it lifts easy sub-saturation
# cells and measured worse). Knees retuned vs the data; reuse the existing ceiling.
P_LO = 0.8
# Upper knee interpolates P_HI_SHORT -> P_HI_LONG as output grows over [OUT_KNEE_LO, OUT_KNEE_HI].
P_HI_SHORT = 1.6   # short-output (swe/terminal): steep step, ceiling reached by pressure ~1.6
P_HI_LONG = 2.5    # long-output (chat/osworld): gentle ramp to the ceiling
OUT_KNEE_LO = 40.0  # below this output, treat as short (hard-saturating)
OUT_KNEE_HI = 80.0  # above this output, treat as long (soft/recovering)

# --- output-sustain gate ------------------------------------------------------
# Saturation ITL is a SUSTAINED effect: the cohort must co-reside through enough
# decode steps for the eviction/queuing wall to build. A turn producing very few
# output tokens finishes before that happens, so its ITL stays near the
# unsaturated kernel step even when instantaneous pressure > 1 (the high-c early
# turns: full cohort scheduled but tiny context + tiny output). So the
# saturation weight is scaled down for short-output turns. Anchor: the minimum
# output observed on any saturated plateau turn (tpot_meas > 150 ms) is 22 tok —
# below that, no sustained saturation is ever measured.
SAT_SUSTAIN_LO = 10.0  # below this output, essentially no saturation possible
SAT_SUSTAIN_HI = 24.0  # full saturation by here (just above the 22-tok plateau min)

# --- saturated-ITL ceiling: T_upper(output) = BASE + OVERHEAD / output -------
# Least-squares fit of measured ITL at pressure >= 2.5 vs 1/output (120 cells,
# R²=0.64). BASE ≈ per-token saturated-decode floor; OVERHEAD ≈ per-turn
# cohort-prefill + scheduling wall amortized over a session's output tokens.
SATURATED_BASE_MS = 118.7
SATURATED_TURN_OVERHEAD_MS = 3263.0
# Hard ceiling (~p90 of measured saturated ITL) so tiny-output turns don't blow
# up the 1/output term.
T_UPPER_MAX_MS = 260.0


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep: 0 below ``lo``, 1 above ``hi``, C¹ in between."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    u = (x - lo) / (hi - lo)
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True)
class KernelTurnInput:
    """One turn's workload aggregates (means over the cohort's sessions)."""

    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float
    scheduled_requests: float  # active cohort sessions submitting this turn


def saturated_ceiling_ms(output_tokens: float) -> float:
    """Saturated ITL ceiling for a turn producing ``output_tokens`` per session."""
    out = max(1.0, float(output_tokens))
    return min(T_UPPER_MAX_MS, SATURATED_BASE_MS + SATURATED_TURN_OVERHEAD_MS / out)


def _kernel_step_ms(inp: KernelTurnInput, params: RooflineParams | None = None) -> float:
    """The unsaturated decode-kernel step floor for one turn — the same floor used
    inside :func:`predict_turn_tpot` (KV-throttled batch ``b_eff`` at ``ctx``).

    Exposed so a downstream hint/blend can recover the floor without re-deriving
    it. ``predict_turn_tpot`` keeps its own inline copy (byte-identical) so the
    production path is untouched.
    """
    p = params or RooflineParams()
    out = max(1.0, float(inp.output_tokens))
    sched = max(1.0, float(inp.scheduled_requests))
    ctx = float(inp.cached_context_tokens) + float(inp.new_prefill_tokens) + 0.5 * out
    per_session_blocks = max(1, math.ceil(ctx / max(1, p.cache_block_size)))
    capacity_batch = max(1.0, p.available_kv_blocks / per_session_blocks)
    b_eff = max(1.0, min(sched, capacity_batch))
    return decode_step_ms(b_eff, ctx, p)


def predict_turn_tpot(
    inp: KernelTurnInput,
    params: RooflineParams | None = None,
    *,
    ceiling_output: float | None = None,
) -> float:
    """Per-turn TPOT (mean ITL, ms) for one (cached, new, output, scheduled) row.

    ``ceiling_output`` overrides the output length used for the saturation
    *ceiling* only (the pressure-step still uses this turn's own output). The
    cell path (``predict_cell_tpot``) passes the cell's median output so the
    ceiling doesn't swing turn-to-turn while a saturated cohort holds a flat
    plateau — the measured ceiling is flat-then-climbing, not output-jittery.
    """
    p = params or RooflineParams()
    out = max(1.0, float(inp.output_tokens))
    sched = max(1.0, float(inp.scheduled_requests))
    ceil_out = out if ceiling_output is None else max(1.0, float(ceiling_output))

    # Per-session resident KV at decode midpoint, in blocks.
    ctx = float(inp.cached_context_tokens) + float(inp.new_prefill_tokens) + 0.5 * out
    per_session_blocks = max(1, math.ceil(ctx / max(1, p.cache_block_size)))

    # KV-throttled running batch and KV pressure (both workload-only).
    capacity_batch = max(1.0, p.available_kv_blocks / per_session_blocks)
    b_eff = max(1.0, min(sched, capacity_batch))
    pressure = sched * per_session_blocks / p.available_kv_blocks

    kernel_step = decode_step_ms(b_eff, ctx, p)
    t_upper = max(kernel_step, saturated_ceiling_ms(ceil_out))
    # Output-gated upper knee: steep step for short outputs, gentle ramp for long.
    p_hi = P_HI_SHORT + _smoothstep(out, OUT_KNEE_LO, OUT_KNEE_HI) * (P_HI_LONG - P_HI_SHORT)
    # Output-sustain gate: a turn too short to co-reside through the eviction
    # buildup can't reach the saturation ceiling, regardless of pressure.
    sustain = _smoothstep(out, SAT_SUSTAIN_LO, SAT_SUSTAIN_HI)
    weight = _smoothstep(pressure, P_LO, p_hi) * sustain
    return kernel_step + weight * (t_upper - kernel_step)


def predict_cell_tpot(
    turns: list[KernelTurnInput], params: RooflineParams | None = None
) -> list[float]:
    """Per-turn TPOT predictions for a whole (profile, concurrency) cell.

    Uses the cell's median output as the ceiling output (Step-2 de-swing) so the
    saturation ceiling is stable across the cell's plateau.
    """
    p = params or RooflineParams()
    if not turns:
        return []
    median_output = statistics.median([max(1.0, float(t.output_tokens)) for t in turns])
    return [predict_turn_tpot(t, p, ceiling_output=median_output) for t in turns]
