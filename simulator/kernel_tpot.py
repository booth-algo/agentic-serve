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

* ``T_upper(output)`` — the saturated ITL ceiling, read from MEASURED anchors
  (the median benchmark ITL at pressure ≥ 2.5, one per output-length cluster:
  short-output swe/terminal ~28 tok → ~243 ms, long-output osworld ~86 tok →
  ~135 ms) and linearly interpolated in output. Fit-free — measured medians +
  interpolation, the same pattern as the decode grid; replaces the retired
  least-squares ceiling ``118.7 + 3263/output``. Artifact:
  ``profile_data/kernels/saturated_ceiling_H100_llama31_8b.json``.

Validated: overall TPOT MAPE 19.4% (median 10.7%) — chat 6.1%, osworld 18.7%,
swebench 21.0%, terminalbench 29.3% — beating the telemetry-using three-regime
predictor (22.8% overall) while needing no engine telemetry.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms


# --- pressure ramp window (MEASURED eviction-watermark crossing) -------------
# DE-FITTED 2026-06-03: the four hand-tuned knees were re-anchored to MEASURED
# quantities (the eviction-watermark jump-pressure cluster + the saturated-ceiling
# output clusters), eliminating the "knees retuned vs the data" tuning. Gate:
# overall TPOT 15.89→15.42% (improves), swe-plateau 8.64→8.65% (holds), osworld
# plateau 19.1→11.8, terminalbench plateau 18.4→15.9, chat flat — no profile
# regresses. The amplifier rises from the watermark onset to the saturated ceiling;
# the upper knee is output-gated (short outputs saturate as a sharp step, long
# ones ramp gently / recover).
#
# P_LO / P_HI_SHORT are the measured eviction-watermark band — the jump-pressure
# cluster across all real-jump cells fires at pressure ≈ 0.88–1.22 (pool ~88–92%
# committed). These are ramp_tpot's DEF_LO/DEF_HI (= defcap −0.12/+0.22) read off
# the measured cluster floor/max, NOT a MAPE fit. P_HI_LONG is the two-roofline
# wave-factor knee at 2× pool commit (ramp_tpot DEF_SAT=1.0 → pressure 2.0).
P_LO = 0.88        # eviction-watermark onset (pressure 0.88, pool ~88% committed) — measured cluster floor
# Upper knee interpolates P_HI_SHORT -> P_HI_LONG as output grows over [OUT_KNEE_LO, OUT_KNEE_HI].
P_HI_SHORT = 1.22  # short-output (swe/terminal): sharp step, full recompute by pressure ~1.22 (measured cluster max)
P_HI_LONG = 2.0    # long-output (chat/osworld): gentle ramp to the ceiling by 2× pool commit (pressure 2.0)
# OUT_KNEE_LO/HI = the MEASURED saturated-ceiling output clusters (short ~28 tok,
# long ~86 tok in saturated_ceiling_H100_llama31_8b.json) — replaces the hand-picked
# [40,80] window with the same measured cluster outputs that key the ceiling. Keep
# in sync if the ceiling artifact is regenerated.
OUT_KNEE_LO = 28.0  # short-output ceiling cluster (measured)
OUT_KNEE_HI = 86.0  # long-output ceiling cluster (measured)

# --- output-sustain gate ------------------------------------------------------
# Saturation ITL is a SUSTAINED effect: the cohort must co-reside through enough
# decode steps for the eviction/queuing wall to build. A turn producing very few
# output tokens finishes before that happens, so its ITL stays near the
# unsaturated kernel step even when instantaneous pressure > 1 (the high-c early
# turns: full cohort scheduled but tiny context + tiny output). So the
# saturation weight is scaled down for short-output turns. Anchors (MEASURED): the
# p5 output of saturated turns (tpot_meas > 100 ms across the H100 run) is 9 tok —
# below it sustained saturation is essentially never measured; SAT_SUSTAIN_HI=24
# is just above the 22-tok min turn-median plateau output.
SAT_SUSTAIN_LO = 9.0   # measured p5 output of saturated turns — below it, ~no saturation
SAT_SUSTAIN_HI = 24.0  # full saturation by here (just above the 22-tok plateau min)

# --- saturated-ITL ceiling: measured anchors, interpolated --------------------
# The ceiling the amplifier pulls toward at saturation is read from MEASURED
# anchors — the median benchmark ITL at KV pressure >= 2.5 (the "C=300+"
# asymptote), one anchor per output-length cluster — and linearly interpolated
# in output. Fit-free (measured medians + interpolation, the same pattern as the
# decode kernel grid). REPLACES the retired least-squares ceiling
# 118.7 + 3263/output. Regenerate the artifact with
# `python3 -m profiling.process.build_saturated_ceiling`. See
# profiling/docs/fitted_constants_audit.md.
_CEILING_JSON = Path("profile_data/kernels/saturated_ceiling_H100_llama31_8b.json")

# Active ceiling artifact — swappable PER-CONFIG (e.g. build_simulator_rows sets this to an A100
# artifact before predicting the A100 cells, then restores), mirroring kernel_step_cost._default_grid.
# Defaults to the H100 anchors so imports / tests / ramp_tpot / kernel_tpot_hint are unaffected.
# _ceiling_anchors is @cache'd BY PATH, so swapping is safe with no cache_clear.
_active_ceiling_json = _CEILING_JSON


@cache
def _ceiling_anchors(path: Path = _CEILING_JSON) -> tuple[tuple[float, float], ...]:
    """Measured (output_tokens, plateau_ms) anchors, sorted ascending by output."""
    data = json.loads(path.read_text())
    anchors = sorted(
        (float(a["output_tokens"]), float(a["plateau_ms"])) for a in data["anchors"]
    )
    if not anchors:
        raise RuntimeError(f"no saturated-ceiling anchors in {path}")
    return tuple(anchors)


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
    """Saturated ITL ceiling for a turn producing ``output_tokens`` per session.

    Linear interpolation between the measured saturated-plateau anchors, clamped
    to the nearest anchor outside the measured output range (monotone: short
    output saturates higher). Fit-free — the anchors are measured medians.
    """
    out = max(1.0, float(output_tokens))
    anchors = _ceiling_anchors(_active_ceiling_json)
    if out <= anchors[0][0]:
        return anchors[0][1]
    if out >= anchors[-1][0]:
        return anchors[-1][1]
    for (o0, p0), (o1, p1) in zip(anchors, anchors[1:]):
        if o0 <= out <= o1:
            return p0 + (out - o0) / (o1 - o0) * (p1 - p0)
    return anchors[-1][1]


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
