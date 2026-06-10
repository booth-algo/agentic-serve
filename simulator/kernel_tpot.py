"""Kernel-composition per-turn TPOT (ITL) predictor for vLLM on H100 / Llama-3.1-8B.

A single physically-grounded expression per turn, no MAPE fitting and no engine
telemetry — every input is workload-derivable (cached/new/output tokens,
scheduled cohort size, the cohort's measured context-size spread) plus
``RooflineParams`` and the measured kernel grids.

    ITL[t] = kernel_step + w × sustain × (T_upper − kernel_step)

with ``w`` the **eviction-recompute duty cycle** (``_overflow_weight``) — fully
computed per turn, zero tuned constants (2026-06-10 ramp restructure; replaces the
tuned ``smoothstep(pressure; P_LO=0.88, p_hi(output))`` band — see the De-fit log
in ``profiling/docs/prediction_construction.md``).

Pieces (see project memory ``tpot-amplifier-pressure-law``):

* ``kernel_step = decode_step_ms(B_eff, ctx)`` — measured decode kernel grid
  (validated 7.4% MAPE; ``prediction_pipeline.yaml`` ``decode`` block). This is
  the physically-correct *lower bound*: 58% of all cells live below KV
  saturation where ITL ≈ kernel_step. ``B_eff`` is the KV-throttled running
  batch ``min(scheduled, capacity_batch)``; ``ctx = cached + new + output/2``.

* ``w = clamp(n_evicted · chunk_steps / out [· z if multi-chunk], 0, 1)`` — the
  chunk-quantized eviction-drain fraction of the turn's decode steps, where
  ``z = pressure × cohort_scale_mean`` is the cohort's distribution-integrated
  KV demand / pool (``pressure = scheduled × per_session_blocks /
  available_kv_blocks`` is the MEDIAN-session demand; ``cohort_scale_mean`` is
  the trapezoid mean of the MEASURED per-session context-scale quantiles),
  ``n_evicted = (1 − 1/z)·sched`` is the LIFO-evicted cohort and
  ``chunk_steps = ceil(ctx·qbar / (M − B_eff))`` is each victim's chunked
  re-prefill step count (``M = max_num_batched_tokens``, engine config).
  Saturation onset is the pool being PHYSICALLY full — ``pressure ≥ 1`` AND
  ``z > 1``, with HYSTERESIS on the cell path (once armed, the gate holds
  while z > 1: the backlog persists through block-quantization flicker of the
  raw pressure; round-3 2026-06-10) — multi-chunk victims gain the ×z rotation
  amplification, and a cell's first overflow turn is growth-damped
  (development + armed state tracked by ``predict_cell_tpot``). See
  ``_overflow_weight`` for the derivation.

* ``T_upper(output)`` — the saturated ITL ceiling, read from MEASURED anchors
  (the median benchmark ITL at pressure ≥ 2.5, one per output-length cluster:
  short-output swe/terminal ~28 tok → ~243 ms, long-output osworld ~86 tok →
  ~135 ms) and linearly interpolated in output. Fit-free — measured medians +
  interpolation, the same pattern as the decode grid; replaces the retired
  least-squares ceiling ``118.7 + 3263/output``. Artifact:
  ``profile_data/kernels/saturated_ceiling_H100_llama31_8b.json``. Physically
  consistent with w = 1: the ceiling was anchored in the regime where every
  step's budget is recompute-filled.

* ``sustain = smoothstep(output; 9, 24)`` — the measured output-sustain gate
  (a turn too short to co-reside through the eviction buildup cannot saturate).
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


# --- saturation transition: COMPUTED, no tuned ramp band (2026-06-10 restructure) ---
# The former tuned pressure-ramp knobs are GONE: P_LO=0.88 (onset), P_HI_SHORT=1.22 /
# P_HI_LONG=2.0 (output-binned upper knees) and the smoothstep-in-pressure ramp they
# parameterized. They were honest tuned-knobs (the reproducible jump-band measurement
# `build_ramp_knees` -> ramp_knees_*_llama31_8b.json disagreed with them: H100 onset
# 0.4456 vs tuned 0.88; cross-GPU onsets 0.45/0.85/0.60 — no global pressure constant
# exists) and compensating fits for the ramp SHAPE itself (2026-06-09 De-fit log).
# The restructured weight (`_overflow_weight`) computes the onset per cell from the
# MEASURED cohort context-size spread (z = pressure·cohort_scale_mean crossing 1 =
# pool overflow) and the transition shape from eviction-recompute duty over the
# engine's chunked-prefill step budget — every input measured or engine config. In
# z-units the measured per-GPU onset medians collapse to 0.964/1.188/0.963
# (H100/H100x2/A100; spread 0.225 vs 0.408 in raw pressure), clustered at the
# predicted z=1. The ramp_knees artifacts remain the measured record of the old
# band (pinned as history in test_kernel_tpot).

# --- measured output-cluster labels (NOT part of the TPOT formula) -------------
# The saturated-ceiling artifact anchors its plateau at two MEASURED output clusters
# (short ~28 tok = swe/terminal, long ~86 tok = osworld in
# saturated_ceiling_H100_llama31_8b.json). Kept ONLY as cluster labels for consumers
# that bin cells by output (ramp_tpot's diagnostic drain-aware lift, build_ramp_knees'
# short/mid/long cell labels). Since 2026-06-10 the production weight has NO
# output-binned knee — the output dependence of the saturation band emerges from
# ceil_out in the recompute duty-cycle denominator (see _overflow_weight). Keep in
# sync if the ceiling artifact is regenerated.
OUT_KNEE_LO = 28.0  # short-output ceiling cluster (measured anchor output)
OUT_KNEE_HI = 86.0  # long-output ceiling cluster (measured anchor output)

# --- output-sustain gate ------------------------------------------------------
# Saturation ITL is a SUSTAINED effect: the cohort must co-reside through enough
# decode steps for the eviction/queuing wall to build. A turn producing very few
# output tokens finishes before that happens, so its ITL stays near the
# unsaturated kernel step even when instantaneous pressure > 1 (the high-c early
# turns: full cohort scheduled but tiny context + tiny output). So the
# saturation weight is scaled down for short-output turns.
#
# Anchors (MEASURED; regenerable since 2026-06-10 by
# `python3 -m profiling.process.build_sat_sustain` ->
# profile_data/kernels/sat_sustain_H100_llama31_8b.json; pinned by
# test_kernel_tpot.test_sat_sustain_anchors_pinned_to_builder_artifact):
# saturated rows = measured tpot > 100 ms across the H100 headline run.
# HONEST CAVEAT (audit-v2 G1) — the two candidate populations DISAGREE:
#   * per-request rows (n=45450 saturated): p5 output = 9.0  -> the LO below
#   * turn-median rows (n=301 saturated):   p5 output = 24.0 -> equals the HI
# The population the predictor actually consumes is the TURN-MEDIAN one
# (build_simulator_rows.build_turns medians feed predict_cell_tpot; this
# smoothstep's `out` argument IS a turn median), so on the canonical population
# the [9, 24] band has no measured support below 21.5 and LO=9 stands only on
# the per-request read. Values kept unchanged this round (2026-06-10 parallel
# de-fit byte-identity contract); any retune starts from the artifact's
# turn-median numbers.
# SAT_SUSTAIN_HI (audit-v2 G2): the historical "+2" story (min turn-median
# plateau output 21.5 ≈ 22 tok, plus an underived +2 hand margin) is
# superseded by an exact derivation the builder found: 24.0 IS the p5 of
# turn-median plateau outputs — the same quantile as LO, on the canonical
# population. Both readings land on 24. sustain_mid = (LO+HI)/2 = 16.5 (the
# development clock in predict_cell_tpot) inherits these anchors.
SAT_SUSTAIN_LO = 9.0   # p5 output of saturated PER-REQUEST rows (turn-median p5 is 24.0 — see caveat)
SAT_SUSTAIN_HI = 24.0  # p5 output of saturated TURN-MEDIAN rows (== min plateau 21.5≈22 + the legacy +2)

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
    # MEASURED cohort context-size spread: the trapezoid mean of the profile's
    # per-session context-scale quantiles (``context_scale_quantiles`` in
    # inference-benchmark/data/distributions/*_realized*.json — per-session
    # median(total_context / per-(conc,turn)-median), success-filtered; resolver
    # simulator/ramp_tpot.context_scale_quantiles, helper
    # simulator/cohort_scale.cohort_scale_mean). ``pressure`` is computed from the
    # cell MEDIAN session; multiplying by this mean integrates the measured
    # session-size distribution, so ``z = pressure × cohort_scale_mean`` is the
    # cohort's SUMMED KV demand / pool — the quantity whose crossing of 1 starts
    # eviction. Default 1.0 (no spread artifact) = median-session pressure with
    # onset at pool-full; a measured artifact value, never a tuned knob.
    cohort_scale_mean: float = 1.0


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


def _overflow_weight(pressure: float, qbar: float, out_steps: float,
                     ctx_tokens: float, sched: float, b_eff: float,
                     p: RooflineParams, developed: bool = True) -> float:
    """Eviction-recompute duty cycle in [0, 1] — the computed saturation weight.

    Every input is a measured artifact value, an engine config value, or derived
    math; ZERO tuned numeric constants. Round-2 law (2026-06-10): the round-1
    once-per-turn linear recompute mass was falsified by the implied-duty
    extraction (measured duty 0.6–0.9 across z ∈ [1.1, 2.0] at pressure ≥ 1,
    2–3× the linear ramp), and the round-1 spread-onset over-fired cells whose
    pool was not physically full. Derivation (per turn):

    * ONSET GATE (physical, P0b): eviction requires the pool to be PHYSICALLY
      full — ``pressure ≥ 1`` (``pressure = sched·per_session_blocks /
      available_kv_blocks``, the median-session resident demand). Below it
      nothing can be evicted (vLLM v1 preempts only on decode allocation
      failure, /root/vllm/vllm/v1/core/sched/scheduler.py) and w = 0. The
      measured context-spread ``qbar`` (trapezoid mean of the per-session
      context-scale quantiles, *_realized.json artifacts) sizes the overflow
      MASS once overflow exists — it does NOT move the onset below pool-full.
      Round-3 refinement (firing-gate hysteresis, cell path): once a turn HAS
      filled the pool, dip turns whose block-quantized pressure flickers just
      below 1 while z stays > 1 keep firing at the pool-full effective
      pressure (H100 swe@40 t20–29, pressure 0.96–1.05 oscillating, measured
      ITL developing monotonically 28→219 ms). Cells that NEVER reach
      pressure = 1 still never fire: the observationally-twin cell A100
      term@20 (pressure peaking 0.965, z up to 1.30, measured CLEAN) requires
      the un-armed gate — aggregate inputs cannot split the pair, and the gate
      is the conservative choice. The remaining known cost: cells saturating
      strictly below pool-full (H100 term@80 t14–17, pressure 0.81–0.88) stay
      un-fired — the prefix-cache-thrash extension (documented, out of core
      scope) is the physical fix.
    * ``z = pressure · qbar`` — distribution-integrated KV demand / pool. The
      LIFO running.pop() preemption (size-independent, requeue to waiting head)
      makes the resident set a sticky prefix holding 1/z of the cohort:
      ``n_evicted = (1 − 1/z) · sched`` whole sessions are non-resident, and v1
      preemption is RECOMPUTE — each re-prefills its full context.
    * CHUNK-QUANTIZED DRAIN (the round-2 steepening that replaced the linear
      mass ramp): a victim's re-prefill of ``ctx·qbar`` tokens is scheduled in
      budget-sized chunks (vLLM chunked prefill; ``budget =
      max_num_batched_tokens − b_eff`` per step after resident decodes), and
      each chunk occupies one engine step's prefill budget:
      ``chunk_steps = ceil(ctx·qbar / budget)``. Re-prefilling the evicted
      cohort once therefore occupies ``n_evicted · chunk_steps`` of the turn's
      ``out`` decode steps (THIS turn's own output — P1a; the cell-median
      de-swing applies to the ceiling only):

          w = n_evicted · chunk_steps / out

      The quantization (ceil vs the fractional ``ctx·qbar/budget``) is the
      measured 2–4× steepening at small ctx/budget: it is what the implied-duty
      table demanded on H100 chat (ctx ≈ 2.4 k, budget ≈ 7.9 k → ×3.3) while
      leaving big-context cells (ctx ≳ budget) nearly linear.
    * ROTATION AMPLIFICATION for multi-chunk victims (``chunk_steps ≥ 2``):
      ``w ·= z``. A multi-step re-prefill monopolizes readmission, so the
      standing overflow keeps rotating the victim queue — the cohort cycles
      through the pool ``z`` times per residency span (z = demand/pool), so
      each victim re-prefills ~z× per turn. Single-chunk victims complete
      within one step and the chain de-synchronizes (measured: A100 chat
      single-chunk drains sit on the once-per-turn drain exactly).
    * FRESH-CROSSING DAMPING (``developed = False``, set by the cell path for
      the first overflow turn): the boundary overflow wave (the turn's
      admission re-prefills) lands in the turn-boundary admission burst — the
      TTFT side — when victims are single-chunk; only decode-phase KV growth
      forces ITL-visible evictions:
      ``w ≤ (sched·out/ctx) · chunk_steps / out = sched·chunk_steps/ctx``
      (growth-forced evictions = decode growth ``sched·out`` tokens / victim
      size ``ctx``; qbar cancels). Multi-chunk waves spill into the decode
      phase (small budgets cannot swallow them) and are NOT damped.

    w = 1 (every step's budget recompute-filled) is the regime where the
    measured saturated ceiling ``t_upper`` was anchored, so the weight still
    interpolates between two measured states. The output dependence the retired
    OUT_KNEE_LO/HI knee-interpolation hand-binned EMERGES from ``out`` in the
    drain-fraction denominator.
    """
    # P0b — physical onset: no eviction while the pool is not full.
    if pressure < 1.0:
        return 0.0
    z = pressure * max(0.0, float(qbar))
    if z <= 1.0:
        return 0.0
    budget = max(1.0, float(p.max_num_batched_tokens) - float(b_eff))
    out = max(1.0, float(out_steps))
    ctx = max(1.0, float(ctx_tokens))
    n_evicted = (1.0 - 1.0 / z) * max(1.0, float(sched))
    # Per-victim re-prefill: chunk-quantized engine steps (vLLM chunked prefill
    # schedules a resumed request's context in budget-sized chunks; each chunk
    # occupies one engine step's prefill budget).
    chunk_steps = math.ceil(ctx * max(1e-9, float(qbar)) / budget)
    w = n_evicted * chunk_steps / out
    if chunk_steps >= 2:
        w *= z  # rotation amplification: the standing overflow re-evicts ~z×/turn
    if not developed and chunk_steps == 1:
        # First overflow turn: boundary wave absorbed at admission (TTFT side);
        # only decode-growth-forced evictions are ITL-visible.
        w = min(w, float(sched) * chunk_steps / ctx)
    return min(1.0, w)


def predict_turn_tpot(
    inp: KernelTurnInput,
    params: RooflineParams | None = None,
    *,
    ceiling_output: float | None = None,
    developed: bool = True,
    armed: bool = False,
) -> float:
    """Per-turn TPOT (mean ITL, ms) for one (cached, new, output, scheduled) row.

    ``ceiling_output`` overrides the output length used for the saturation
    *ceiling* only (the pressure-step still uses this turn's own output). The
    cell path (``predict_cell_tpot``) passes the cell's median output so the
    ceiling doesn't swing turn-to-turn while a saturated cohort holds a flat
    plateau — the measured ceiling is flat-then-climbing, not output-jittery.

    ``developed`` is the eviction-development state (see ``_overflow_weight``):
    False on a cell's FIRST overflow turn (the boundary wave lands in admission,
    not decode), True when the previous turn both overflowed (z > 1) and could
    sustain eviction. The cell path computes the sequence; a standalone call
    defaults to the steady state (True).

    ``armed`` is the firing-gate HYSTERESIS state (round-3, 2026-06-10; cell
    path only): once a prior turn physically filled the pool (pressure ≥ 1 AND
    z > 1), the standing eviction backlog persists while the demand overflow
    lasts (z > 1) even when the median-session pressure flickers 1–4% below 1
    — ``per_session_blocks = ceil(ctx/block)`` quantizes pressure, so tiny
    ctx/sched jitter flips the raw P0b gate while the measured ITL develops
    monotonically (H100 swe@40 t20–29: gate flipped on/off turn-to-turn,
    prediction oscillated 27↔132 ms vs measured 28→219 ms). While armed the
    weight is computed at the effective pressure ``max(pressure, 1.0)`` — the
    pool cannot be less than full while evicted sessions still hold a
    re-prefill backlog. This is the SAME physical argument the development
    clock already codified (``_turn_overflows``: "the backlog persists through
    pressure-gate flicker") applied to the firing gate itself; the floor is
    the pool-full boundary (1.0), not a tuned number. Disarm is z ≤ 1 (demand
    back under the pool). A standalone call defaults to per-turn gating
    (False); never-full cells (A100 term@20, max pressure 0.965) never arm.
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
    # Output-sustain gate: a turn too short to co-reside through the eviction
    # buildup can't reach the saturation ceiling, regardless of pressure.
    sustain = _smoothstep(out, SAT_SUSTAIN_LO, SAT_SUSTAIN_HI)
    # Computed eviction-recompute duty cycle (replaces the tuned pressure
    # smoothstep band; see _overflow_weight). The recompute is amortized over THIS
    # turn's own decode-step count ``out`` (the cell-median de-swing ``ceil_out``
    # prices the ceiling only — using it as the amortizer inverted the sign of the
    # within-cell error on short/long-output drain turns, round-1 finding).
    # Hysteresis: while armed, the gate sees the pool-full effective pressure
    # (max(pressure, 1.0)) — see the ``armed`` docstring above.
    gate_pressure = max(pressure, 1.0) if armed else pressure
    weight = _overflow_weight(gate_pressure, inp.cohort_scale_mean, out, ctx, sched,
                              b_eff, p, developed) * sustain
    return kernel_step + weight * (t_upper - kernel_step)


def _turn_pressure_z(inp: KernelTurnInput, p: RooflineParams) -> tuple[float, float]:
    """(pressure, z) for one turn — the median-session resident demand / pool and
    the distribution-integrated demand / pool (z = pressure·qbar). The same
    arithmetic as :func:`predict_turn_tpot`'s inline copy; exposed for the cell
    path's state tracking (development clock + firing-gate hysteresis)."""
    out = max(1.0, float(inp.output_tokens))
    sched = max(1.0, float(inp.scheduled_requests))
    ctx = float(inp.cached_context_tokens) + float(inp.new_prefill_tokens) + 0.5 * out
    per_session_blocks = max(1, math.ceil(ctx / max(1, p.cache_block_size)))
    pressure = sched * per_session_blocks / p.available_kv_blocks
    return pressure, pressure * max(0.0, float(inp.cohort_scale_mean))


def _turn_overflows(inp: KernelTurnInput, p: RooflineParams) -> bool:
    """True when the turn's distribution-integrated demand overflows the pool
    (z = pressure·qbar > 1) — the development-clock condition (demand overflow,
    NOT the firing gate: the backlog persists through pressure-gate flicker)."""
    return _turn_pressure_z(inp, p)[1] > 1.0


def predict_cell_tpot(
    turns: list[KernelTurnInput], params: RooflineParams | None = None
) -> list[float]:
    """Per-turn TPOT predictions for a whole (profile, concurrency) cell.

    Uses the cell's median output as the ceiling output (Step-2 de-swing) so the
    saturation ceiling is stable across the cell's plateau, and tracks two
    pieces of eviction state turn to turn:

    * DEVELOPMENT: a turn is ``developed`` when the PREVIOUS turn both
      overflowed (z > 1 — demand overflow, robust to pressure-gate flicker) and
      could sustain eviction (its output clears the measured sustain band's
      midpoint, the Hermite half-point of the measured SAT_SUSTAIN_LO/HI
      anchors — no new constant).
    * FIRING-GATE HYSTERESIS (round-3, 2026-06-10): ``armed`` latches when a
      turn physically fills the pool (pressure ≥ 1 AND z > 1), stays latched
      while the demand overflow persists (z > 1), and releases when z ≤ 1.
      Armed turns are priced at the pool-full effective pressure (see
      ``predict_turn_tpot``); the raw per-turn P0b gate flickers with the
      block-quantized pressure while the physical backlog develops
      monotonically. Cells whose pressure never reaches 1 never arm (the
      A100 term@20 protection is preserved exactly).
    """
    p = params or RooflineParams()
    if not turns:
        return []
    median_output = statistics.median([max(1.0, float(t.output_tokens)) for t in turns])
    preds: list[float] = []
    developed = False  # turn 0 has no prior overflow turn
    armed = False      # firing-gate hysteresis: no turn has filled the pool yet
    sustain_mid = 0.5 * (SAT_SUSTAIN_LO + SAT_SUSTAIN_HI)
    for t in turns:
        pressure, z = _turn_pressure_z(t, p)
        if z <= 1.0:
            armed = False          # demand back under the pool: backlog drained
        elif pressure >= 1.0:
            armed = True           # pool physically full with demand overflow
        preds.append(predict_turn_tpot(t, p, ceiling_output=median_output,
                                       developed=developed, armed=armed))
        developed = (z > 1.0
                     and max(1.0, float(t.output_tokens)) >= sustain_mid)
    return preds
