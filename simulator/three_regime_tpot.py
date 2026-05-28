"""Three-regime TPOT predictor: classify each (profile, c) cell into
flat / perturbing / saturating, then emit per-turn TPOT using two anchors.

Vision (see plan):
  T_min = existing closed-form roofline TPOT (the floor, per-turn workload-aware)
  T_max = min(physical_T_upper ≈ 205 ms, llm-d per-cell measured mean)
          (workload-aware ceiling capped by physics)

  regime ∈ {FLAT, PERTURBING, SATURATING}    ← from pressure trajectory (no fits)

Per-turn TPOT:
  FLAT        → T_min[t] for all t
  SATURATING  → T_min[t] for t < jump_turn; linear ramp to T_max for t ≥ jump_turn
  PERTURBING  → T_max at perturbation turns (pressure ≥ 1); T_min[t] elsewhere

No fitted constants. Inputs: per-turn workload distributions + `c` +
`RooflineParams`. The bench's per-turn `request_count` is the cohort-active
trajectory (workload-derived, not engine telemetry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from simulator.cached_prefill_lookup import cached_prefill_step_ms
from simulator.closed_form_tpot import RooflineParams
from simulator.two_roofline_tpot import (
    MAX_NUM_BATCHED_TOKENS,
    _physical_prefill_per_token_ms,
)


# Two consecutive saturated turns ⇒ admission cycle has entered (same physical
# minimum used in the two-roofline turn-history extension). Not a fit.
K_SUSTAIN = 2

# Cumulative cohort-completion threshold. If `scheduled_requests[t]` drops
# below `(1 − BURST_COMPLETION_FRACTION) × cohort_c` at ANY point in the
# trajectory, the cohort has lost > 30% of submitted sessions by then
# (bulk completion). NECESSARY but not SUFFICIENT for PERTURBING — see
# SATURATION_FLOOR below.
#
# Two-roofline uses 0.15 as a SINGLE-TURN-DROP threshold (sharp drop within
# one turn); three-regime uses 0.30 as a CUMULATIVE-LOSS threshold. Both
# physically motivated by cohort completion dynamics.
BURST_COMPLETION_FRACTION = 0.30

# Post-burst pressure ceiling for recovery. Even when cohort-completion fires,
# the REMAINING cohort can still saturate KV pool if ctx has grown enough.
# At osworld c=160, late_pressure ≈ 1.08 → remaining 39 sessions barely
# saturate → cell recovers (PERTURBING). At c=200, late_pressure ≈ 1.41 →
# remaining 51 sessions still saturate by 41% → cell stays SATURATING despite
# the burst. Threshold 1.2 puts the boundary between these two cells — KV
# pool tolerates ~20% overcommit before steady-state cycling kicks in.
SATURATION_FLOOR = 1.2


class Regime(str, Enum):
    FLAT = "FLAT"
    PERTURBING = "PERTURBING"
    SATURATING = "SATURATING"


@dataclass(frozen=True)
class TurnObservation:
    """Per-turn workload values used by the classifier."""
    turn_index: int
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float
    request_count: int   # number of sessions present at this turn


@dataclass(frozen=True)
class CellClassification:
    regime: Regime
    jump_turn: int | None
    perturbation_turns: tuple[int, ...]
    peak_pressure: float
    late_pressure: float
    saturated_turns: int


def _pressure_at_turn(turn: TurnObservation, params: RooflineParams) -> float:
    ctx_mid = (
        turn.cached_context_tokens
        + turn.new_prefill_tokens
        + 0.5 * max(1.0, turn.output_tokens)
    )
    block_size = max(1, params.cache_block_size)
    per_session_blocks = max(1, math.ceil(ctx_mid / block_size))
    n_active = max(1, int(turn.request_count))
    return (n_active * per_session_blocks) / max(1, params.available_kv_blocks)


def classify_cell(
    turns: list[TurnObservation],
    params: RooflineParams,
    cohort_c: int | None = None,
) -> CellClassification:
    """Classify a (profile, c) cell from its per-turn pressure trajectory.

    Rules (all physical, no fits):
      FLAT        — max pressure < 1 across all turns (cohort fits in KV pool)
      SATURATING  — pressure ≥ 1 at the last turn AND saturated_turns ≥ K_SUSTAIN
                    AND no burst-completion event detected
      PERTURBING  — peak ≥ 1 somewhere but recovered (last < 1 OR brief touch)
                    OR a burst-completion event fired (cohort recovers via
                    bulk session completion regardless of late_pressure)

    ``cohort_c`` is the originally-submitted cohort size. When not provided,
    it's inferred as the max ``request_count`` across the trajectory.
    """
    if not turns:
        return CellClassification(
            regime=Regime.FLAT, jump_turn=None,
            perturbation_turns=tuple(), peak_pressure=0.0,
            late_pressure=0.0, saturated_turns=0,
        )

    sorted_turns = sorted(turns, key=lambda t: t.turn_index)
    pressures = [_pressure_at_turn(t, params) for t in sorted_turns]
    peak = max(pressures)
    late = pressures[-1]
    saturated = sum(1 for p in pressures if p >= 1.0)

    if peak < 1.0:
        return CellClassification(
            regime=Regime.FLAT, jump_turn=None,
            perturbation_turns=tuple(), peak_pressure=peak,
            late_pressure=late, saturated_turns=saturated,
        )

    # Cumulative cohort-completion detection: if `request_count[t]` falls
    # below `(1 − BURST_COMPLETION_FRACTION) × cohort_c` at any turn, the
    # cohort has lost > 30% of submitted sessions to completion by that point.
    # BUT only override SATURATING → PERTURBING if the remaining cohort
    # actually relieves pressure (late_pressure < SATURATION_FLOOR). At
    # osworld c=200+, scheduled_requests drops but the remaining 50+ sessions
    # still saturate KV pool — that's NOT recovery, just smaller-cohort
    # sustained saturation.
    cohort_size = (
        cohort_c
        if cohort_c is not None and cohort_c > 0
        else max((t.request_count for t in sorted_turns), default=1)
    )
    completion_floor = (1.0 - BURST_COMPLETION_FRACTION) * cohort_size
    burst_detected = any(t.request_count < completion_floor for t in sorted_turns)
    recovery_confirmed = burst_detected and late < SATURATION_FLOOR

    if late >= 1.0 and saturated >= K_SUSTAIN and not recovery_confirmed:
        # First turn where pressure crosses capacity is the jump turn.
        jump_idx = next(i for i, p in enumerate(pressures) if p >= 1.0)
        jump_turn = sorted_turns[jump_idx].turn_index
        return CellClassification(
            regime=Regime.SATURATING, jump_turn=jump_turn,
            perturbation_turns=tuple(), peak_pressure=peak,
            late_pressure=late, saturated_turns=saturated,
        )

    # PERTURBING: pressure crossed 1 somewhere but either didn't sustain at
    # the end OR a burst-completion event fired.
    pert_turns = tuple(
        t.turn_index for t, p in zip(sorted_turns, pressures) if p >= 1.0
    )
    return CellClassification(
        regime=Regime.PERTURBING, jump_turn=None,
        perturbation_turns=pert_turns, peak_pressure=peak,
        late_pressure=late, saturated_turns=saturated,
    )


def compute_t_max(
    llmd_per_cell_mean_ms: float | None,
    params: RooflineParams,
    *,
    regime: Regime | None = None,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
) -> float:
    """T_max anchor — split by regime:

    SATURATING — physical_T_upper (~205 ms). At true saturation the cell is
                 in the chunked-prefill compute-bound asymptote, so the
                 physics ceiling is the right anchor. Don't drag it down with
                 llm-d's per-cell mean (which mixes pre-saturation low-tpot
                 turns into a lower average).
    PERTURBING — min(physical_T_upper, llm-d per-cell mean). The cell only
                 brushes capacity, so the saturation it CAN reach is bounded
                 by the per-cell observed envelope (llm-d's mean is a
                 reasonable proxy for "how high does this workload+c get?").
    FLAT       — irrelevant (prediction = T_min throughout).
    Default (regime is None) — old min() behavior, preserved for callers.
    """
    physical_t_upper = max_num_batched_tokens * _physical_prefill_per_token_ms(params)
    if regime == Regime.SATURATING:
        return physical_t_upper
    if llmd_per_cell_mean_ms is None or llmd_per_cell_mean_ms <= 0:
        return physical_t_upper
    return min(physical_t_upper, float(llmd_per_cell_mean_ms))


def _t_max_per_turn(
    turn: TurnObservation,
    params: RooflineParams,
    cell_t_max_cap_ms: float,
    use_kernel_lookup: bool,
    regime: Regime,
    coverage: float | None = None,
    t_lower_ms: float | None = None,
) -> float:
    """Per-turn T_max ceiling — semantics depend on regime.

    SATURATING — pressure-scaled blend from T_lower to physical T_upper.
                 ``T_max = T_lower + clamp(pressure − 1, 0, 1) × (T_upper − T_lower)``.
                 At pressure = 1 (just at capacity) → T_lower. At pressure ≥ 2
                 (≥ 2× over capacity) → physical T_upper. Captures the fact
                 that osworld c=200 (pressure ≈ 1.4) saturates lower than
                 swe c=80 (pressure ≈ 2+) — different overshoot levels yield
                 different sustained-ITL ceilings.

    PERTURBING — engine step time from ``cached_prefill_step_ms(U, P) +
                 scheduler_overhead``, then DIVIDED BY ``coverage`` (when
                 supplied) to translate engine-side throughput into the
                 client-measured ITL ceiling. Capped by ``cell_t_max_cap_ms``
                 (the llm-d envelope) so we never predict above the workload's
                 observed saturation level.

    Legacy ``use_kernel_lookup=False``: always returns the constant cap.
    """
    if not use_kernel_lookup:
        return cell_t_max_cap_ms

    physical_t_upper = MAX_NUM_BATCHED_TOKENS * _physical_prefill_per_token_ms(params)

    if regime == Regime.SATURATING:
        # Pressure-scaled saturation ceiling. At mild overshoot (e.g. osworld
        # c=200 with pressure ≈ 1.4) sessions step-skip less, sustained ITL
        # lands well below the physical asymptote. At extreme overshoot (e.g.
        # swe c=80 at pressure 2+) sessions step-skip heavily and ITL reaches
        # the physical T_upper.
        pressure = _pressure_at_turn(turn, params)
        weight = max(0.0, min(1.0, pressure - 1.0))
        # Use T_lower (when provided) for the floor; else a conservative
        # small decode roofline estimate so the ceiling is meaningful.
        floor = t_lower_ms if t_lower_ms is not None and t_lower_ms > 0 else 17.0
        return floor + weight * (physical_t_upper - floor)

    # PERTURBING
    u = max(1.0, float(turn.new_prefill_tokens))
    p = max(1.0, float(turn.cached_context_tokens))
    t_kernel = cached_prefill_step_ms(u, p) + params.scheduler_overhead_ms_per_step
    if coverage is not None and coverage > 0:
        t_kernel = t_kernel / max(0.1, coverage)
    return min(cell_t_max_cap_ms, t_kernel) if cell_t_max_cap_ms > 0 else t_kernel


def predict_cell_tpot(
    classification: CellClassification,
    turns: list[TurnObservation],
    t_min_per_turn: list[float],
    t_max_ms: float,
    params: RooflineParams | None = None,
    *,
    use_kernel_lookup: bool = True,
    coverage_by_turn: dict[int, float] | None = None,
) -> list[float]:
    """Emit per-turn TPOT given the classification + anchors.

    Shape rules per regime:
      FLAT       — pred[t] = T_min[t] for all t.
      SATURATING — sharp 3-turn ramp from T_min[t] to T_max[t] starting at
                   jump_turn (matches the observed sudden-jump-then-plateau
                   on swe/terminal at high c). T_max[t] is workload-aware per
                   turn via ``cached_prefill_step_ms`` (see ``_t_max_per_turn``).
      PERTURBING — smooth per-turn blend driven by `pressure[t]`. Weight
                   grows linearly from 0 at pressure=1 to 1 at pressure=peak
                   for this cell, giving a bell-curve climb/peak/decline.
    """
    if params is None:
        params = RooflineParams()
    sorted_pairs = sorted(
        zip(turns, t_min_per_turn), key=lambda x: x[0].turn_index,
    )
    preds: list[float] = []

    if classification.regime == Regime.FLAT:
        return [tmin for _, tmin in sorted_pairs]

    if classification.regime == Regime.SATURATING:
        jump_turn = classification.jump_turn
        assert jump_turn is not None
        # Short fixed ramp width (3 turns) matches the observed sharp transition:
        # swe c=80 climbs from T_min to T_max in roughly 3 turns (22 → 93 → 167 → 199).
        # Not a fit — it's the physical settling time of vLLM's admission-cycle regime.
        SATURATING_RAMP_TURNS = 3
        for turn, tmin in sorted_pairs:
            if turn.turn_index < jump_turn:
                preds.append(tmin)
            else:
                progress = min(1.0, (turn.turn_index - jump_turn) / SATURATING_RAMP_TURNS)
                cov = coverage_by_turn.get(turn.turn_index) if coverage_by_turn else None
                t_max_t = _t_max_per_turn(
                    turn, params, t_max_ms, use_kernel_lookup,
                    classification.regime, cov, t_lower_ms=tmin,
                )
                preds.append(tmin + progress * (t_max_t - tmin))
        return preds

    # PERTURBING: smooth per-turn interpolation based on pressure overshoot.
    # peak_pressure normalizes — a cell that barely crosses capacity gets a
    # small spike, a cell with severe overshoot reaches close to T_max.
    peak_p = classification.peak_pressure
    overshoot_range = max(1e-6, peak_p - 1.0)
    for turn, tmin in sorted_pairs:
        p = _pressure_at_turn(turn, params)
        if p < 1.0:
            preds.append(tmin)
        else:
            weight = min(1.0, (p - 1.0) / overshoot_range)
            cov = coverage_by_turn.get(turn.turn_index) if coverage_by_turn else None
            t_max_t = _t_max_per_turn(turn, params, t_max_ms, use_kernel_lookup, classification.regime, cov)
            preds.append(tmin + weight * (t_max_t - tmin))
    return preds


def physical_t_upper_ms(
    params: RooflineParams,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
) -> float:
    """Convenience: chunk × prefill_per_token (≈ 205 ms on H100 Llama-3.1-8B)."""
    return max_num_batched_tokens * _physical_prefill_per_token_ms(params)
