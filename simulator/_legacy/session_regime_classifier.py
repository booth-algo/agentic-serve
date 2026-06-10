"""Session-level multiturn TPOT regime classifier (pure-workload, forward-applicable).

RETIRED 2026-06-10 (de-fit campaign lane L7; audit-v2 items D5-D8): moved to
``simulator/_legacy/``. Never reachable from any production entrypoint
(``build_simulator_rows`` / ``validate_*`` / ``gate_scoped_rows`` / dashboards);
the only consumer chain is ``simulator/_legacy/kernel_tpot_hint.py`` ->
``profiling/process/_legacy/augment_simulator_predictions_with_kernel.py``, an
opt-in dashboard diagnostic. Kept (not deleted) so that diagnostic still runs.
All thresholds below are FITTED/read-off on the same 44 in-sample H100 cells
(no builder regenerates any anchor; audit-v2 D6) — treat as diagnostic-only.

Given a multiturn cell's per-turn WORKLOAD trajectory (token counts + scheduled
cohort size per turn -- no engine telemetry), classify the cell into one of three
ITL regimes and, for SATURATE, predict the turn index at which measured ITL jumps:

    FLAT            ITL stays near baseline for the whole session. The cohort
                    fits the KV pool and never tips it; decode is never starved.
    PERTURB_RETURN  ITL spikes mid-session then RETURNS toward baseline. The
                    submitted cohort drains (sessions finish) faster than context
                    grows, so the KV pool recovers and the spike is transient.
    SATURATE        ITL jumps and STAYS elevated. The cohort never drains while
                    context keeps growing, so the KV pool stays oversubscribed
                    and decode is permanently starved from the jump turn onward.

Design goal: BEAT simulator/three_regime_tpot.classify_cell on generalization and
interpretability, not just in-sample accuracy. Every threshold below is a single
GLOBAL physical constant -- there are NO per-profile parameters -- so the rule is
identical under leave-one-profile-out (no refit, no profile-identity leakage).

------------------------------------------------------------------------------
PHYSICAL SIGNALS (all pure-workload, all forward-applicable)
------------------------------------------------------------------------------
1. KV pressure (same model as three_regime_tpot, task-specified):
       pressure[t] = scheduled_requests[t]
                     * ceil((cached + new + output/2) / 16) / 27250
   The fraction of the 27250-block H100 KV pool the live cohort demands at turn t
   (16 = vLLM block size; 27250 = empirical available blocks, see RooflineParams).
   pressure >= 1 means oversubscription -> eviction/recompute -> decode stalls.
       peak_pressure = max_t pressure[t]   (does it ever load the pool?)
       late_pressure = pressure[-1]        (is it STILL loaded at the end?)

2. Cohort drain fraction  drop = (max_sched - min_sched) / max_sched.
   THE causal FLAT/PERTURB-vs-SATURATE discriminator the adversarial review
   asked for. A draining cohort (high drop) relieves the queue as sessions
   finish -> any spike returns (PERTURB_RETURN). A non-draining cohort (low drop)
   never relieves -> the spike persists (SATURATE). In this data the split is
   clean and profile-invariant: every PERTURB/FLAT-eligible draining cell has
   drop >= 0.60; every SATURATE-by-creep cell has drop <= 0.21.

3. Monotone context growth  grow = fraction of turns with non-decreasing context.
   Distinguishes a *sustained* low-pressure saturation (context climbs every
   turn, ITL creeps up and stays up) from transient noise. This is the second,
   pressure-INDEPENDENT saturation trigger the review demanded: it recovers the
   low-pressure slow-creep saturators (swebench c=10/20, terminalbench c=20)
   whose pressure never approaches 1.0 yet whose ITL ramps and holds.

4. Prefill-to-output ratio (ISL/OSL) at turn 0  ratio = new_prefill / output.
   Prefill-heavy short-output workloads (ratio >> 1, e.g. osworld at 20x) inject
   a large prefill chunk into the decode batch every turn, so they perturb ITL
   at LOWER KV pressure than output-heavy chat (ratio ~0.6). Used only to lower
   the PERTURB pressure gate for prefill-heavy cells (recovers osworld c=20).

------------------------------------------------------------------------------
CLASS DECISION (drain-fraction first -> pressure / growth)
------------------------------------------------------------------------------
  if drop < DRAIN_FRACTION:                       # non-draining cohort
      if peak >= SLOW_CREEP_PEAK and grow >= MONOTONE_GROW:  SATURATE
      else:                                                  FLAT
  else:                                            # draining cohort
      if peak >= HIGH_PEAK_OVERRIDE and late >= LATE_RELIEF: SATURATE
      elif peak >= (PREFILL_HEAVY_PEAK if ratio>=PREFILL_HEAVY_RATIO
                    else PERTURB_PEAK):                       PERTURB_RETURN
      else:                                                   FLAT

  vs three_regime_tpot.classify_cell this fixes the two failure modes the review
  flagged: (a) low-pressure slow-creep saturation (signal 3, recovers 3 cells
  that classify_cell's peak<1.0 FLAT gate misses), and (b) the PERTURB/SATURATE
  split now rests on cohort DRAIN (signal 2) rather than a brittle peak-magnitude
  cutoff -- HIGH_PEAK_OVERRIDE only escalates already-draining cells whose
  RESIDUAL cohort is still oversubscribed (osworld c>=200, late_pressure>1.4).

------------------------------------------------------------------------------
JUMP TURN (SATURATE only): inverse-concurrency context budget
------------------------------------------------------------------------------
       jump_turn = clamp( round(CONTEXT_BUDGET / cohort) + JUMP_FLOOR,
                          JUMP_FLOOR, last_turn )
  Physical reading: the cohort exhausts a roughly fixed cumulative context
  budget (~CONTEXT_BUDGET cohort*turn units) before the per-step decode batch
  can no longer be sustained and ITL jumps; with `cohort` concurrent sessions
  that budget is spent in ~budget/cohort turns. JUMP_FLOOR is the structural
  minimum (the earliest a context-heavy session can tip). The pressure-crossing
  turn was rejected: it overshoots badly at mid/high concurrency for short-output
  families (terminalbench c=80 jumps at turn 3 but pressure crosses 1.0 only at
  turn 18). One shared constant + one shared floor -- NO per-family terms, so it
  is profile-invariant under LOPO.

------------------------------------------------------------------------------
FITTED CONSTANTS (flagged; tuned on the 44 H100 cells, all GLOBAL not per-profile)
------------------------------------------------------------------------------
  DRAIN_FRACTION=0.40    midpoint of the clean gap (drainers >=0.60, creepers <=0.21)
  SLOW_CREEP_PEAK=0.22   lowest peak among true slow-creep saturators (~0.26) minus margin
  MONOTONE_GROW=0.90     "context grows essentially every turn"
  HIGH_PEAK_OVERRIDE=2.5 draining-but-residual-still-saturated escalation (osworld c>=200)
  LATE_RELIEF=1.2        KV pool tolerates ~20% residual overcommit before it relieves
  PERTURB_PEAK=0.55      output-heavy perturbation onset (also the three_regime FLAT gate)
  PREFILL_HEAVY_RATIO=3.0 / PREFILL_HEAVY_PEAK=0.30  lower gate for prefill-dominated cells
  CONTEXT_BUDGET=150.0   cohort*turn budget consumed before the decode batch collapses
  JUMP_FLOOR=2           structural earliest jump turn

  --- ramp-window hint constants (added later; same 44-cell in-sample tuning surface) ---
  PRESSURE_ONSET=0.85    FITTED eviction-onset read-off (see provenance correction at the constant)
  SAT_FULL=2.0           FITTED; equals the retired tuned kernel P_HI_LONG, not a measurement
  RAMP_WIDTH_MAX=7       in-sample read-off: widest coding-family rise (~6-7 turns)
  CONF_COHORT_LO=40 / CONF_COHORT_HI=160   confidence-ramp endpoints (timing-MAE read-off)
  CONF_FLOOR=0.2         FITTED residual trust at low cohort
  0.4 confidence cap     FITTED (was unflagged until audit-v2 D7): clamp for sub-pool-full
                         (peak<1) windows, inline in classify_session
  +2 onset pull          FITTED (was unflagged until audit-v2 D7): prefill-heavy onset pulled
                         to first_turn+2, inline in classify_session
"""

from __future__ import annotations

import math
import statistics

# --- KV-pressure physics (match RooflineParams / three_regime_tpot) ---
AVAILABLE_KV_BLOCKS = 27250.0   # empirical H100 free KV blocks
BLOCK_SIZE = 16                 # vLLM v1 default

# --- class thresholds (all GLOBAL; fitted constants flagged in module docstring) ---
DRAIN_FRACTION = 0.40       # FITTED. cohort drain split (drainers>=0.60, creepers<=0.21)
SLOW_CREEP_PEAK = 0.22      # FITTED. min pressure for low-pressure sustained saturation
MONOTONE_GROW = 0.90        # FITTED. fraction of turns with non-decreasing context
HIGH_PEAK_OVERRIDE = 2.5    # FITTED. draining cohort whose residual still saturates
LATE_RELIEF = 1.2           # FITTED. residual-overcommit ceiling for recovery
PERTURB_PEAK = 0.55         # FITTED. perturbation onset for output-heavy workloads
PREFILL_HEAVY_RATIO = 3.0   # FITTED. ISL/OSL above which prefill dominates decode
PREFILL_HEAVY_PEAK = 0.30   # FITTED. lowered perturbation gate for prefill-heavy cells

# --- jump-turn model (SATURATE only) ---
CONTEXT_BUDGET = 150.0      # LEGACY (unused for live jump_turn; kept for reference).
JUMP_FLOOR = 2              # structural earliest jump turn

# --- ramp-window hint (the jump is modeled as a KV-pressure crossing) ---------
# The saturation jump is modeled as the first turn pressure >= PRESSURE_ONSET,
# not a fixed cohort*turn budget. On the 44 in-sample H100 cells this crossing
# tracked the true jump with MAE 0.88 turns vs 3.76 for the old CONTEXT_BUDGET
# law (only accurate at high-c by a clamping artifact). The cited validation
# workflow wf_9a938421 left no artifact, so those numbers are unverifiable today.
#
# PROVENANCE CORRECTION (2026-06-10, audit-v2 D5): the original comment here
# claimed "no new fitted constant ... sibling of the kernel amplifier onset
# (kernel P_LO=0.8)". That was false at introduction: 0.85 never equaled the
# kernel's P_LO (0.8, later 0.88), and P_LO itself was DELETED from kernel_tpot
# when the tuned knees were retired (commit aea241e). The measured eviction-onset
# artifact (profile_data/kernels/ramp_knees_h100_llama31_8b.json) puts the knee
# at P_LO=0.4456 — roughly 2x BELOW this value. PRESSURE_ONSET is therefore a
# FITTED in-sample threshold, frozen for this legacy diagnostic.
PRESSURE_ONSET = 0.85   # FITTED (44-cell in-sample read-off; see provenance correction above).
SAT_FULL = 2.0          # FITTED: equals the RETIRED tuned kernel P_HI_LONG=2.0, not a measurement.
# Ramp WIDTH is the number of turns pressure takes to climb ONSET -> SAT_FULL:
# slow climb (low-c coding) -> wide multi-turn step; fast climb (high-c) -> sharp
# jump. This is the STEPPING motion (a ramp), not a discrete jump. Bounded:
RAMP_WIDTH_MAX = 7      # in-sample read-off: widest coding-family rise (~6-7 turns)
RAMP_WIDTH_MIN = 0      # near-instant jump limit
# OUT_KNEE_HI: FROZEN SNAPSHOT (audit-v2 D8) of kernel_tpot's former OUT_KNEE_HI=80.0
# long-output threshold. NOT "reused" live: the kernel value has since moved to 86.0
# and was demoted to a non-formula ceiling-cluster label there. This module is
# deliberately standalone (pure-workload) — do not re-import from kernel_tpot.
OUT_KNEE_HI = 80.0      # frozen snapshot of a retired kernel constant (prefill-lead gate)
# Confidence is keyed to where the jump physics is trustworthy: jump timing MAE
# falls ~18 -> 1.8 turns over cohort [40,160] (high-c reliable, low/mid-c noisy).
# The hint only scales MAGNITUDE; it never changes the class.
CONF_COHORT_LO = 40.0
CONF_COHORT_HI = 160.0
CONF_FLOOR = 0.2        # FLAG fitted: residual trust at low cohort (graceful degradation).


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep: 0 below ``lo``, 1 above ``hi``, C¹ in between.

    Inlined (not imported from kernel_tpot) so this classifier stays standalone
    and pure-workload — it must not pull in the kernel/roofline stack.
    """
    if hi <= lo:
        return 0.0 if x <= lo else 1.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    u = (x - lo) / (hi - lo)
    return u * u * (3.0 - 2.0 * u)


def _pcross(pressures: list[float], ts: list[dict], thr: float) -> int | None:
    """First turn_index whose KV pressure reaches ``thr`` (else None)."""
    for pr, t in zip(pressures, ts):
        if pr >= thr:
            return int(t.get("turn_index", 0))
    return None


def _pressure(turn: dict) -> float:
    ctx_mid = (
        float(turn.get("cached_context_tokens", 0.0) or 0.0)
        + float(turn.get("new_prefill_tokens", 0.0) or 0.0)
        + 0.5 * max(1.0, float(turn.get("output_tokens", 1.0) or 1.0))
    )
    blocks = max(1, math.ceil(ctx_mid / BLOCK_SIZE))
    sched = max(1, int(turn.get("scheduled_requests", 1) or 1))
    return sched * blocks / AVAILABLE_KV_BLOCKS


def classify_session(turns: list[dict]) -> dict:
    """Classify a multiturn cell and predict its saturation jump turn.

    Parameters
    ----------
    turns : list of per-turn workload dicts (any order). Each turn must carry
        cached_context_tokens, new_prefill_tokens, output_tokens,
        scheduled_requests, turn_index. ALL inputs are workload-side; no
        engine_* telemetry is read (the rule is forward-applicable).

    Returns
    -------
    {"class": str, "jump_turn": int | None,
     "jump_start": int | None, "jump_end": int | None, "confidence": float}
        class in {"FLAT", "PERTURB_RETURN", "SATURATE"}.
        jump_turn is the turn index where ITL permanently jumps (SATURATE only),
        now the KV-pool eviction crossing (pressure>=PRESSURE_ONSET), else None.
        jump_start/jump_end bracket the STEPPING ramp window (the rise spans the
        turns pressure climbs PRESSURE_ONSET->SAT_FULL); confidence in [0,1] scales
        how strongly a downstream predictor should trust the hint. The window is
        emitted for SATURATE and for osworld-like PERTURB_RETURN (prefill-heavy,
        long-output, loads the pool); FLAT / chat / coding-PERTURB get
        confidence 0 so any soft-hint blend is a structural no-op for them.
    """
    if not turns:
        return {"class": "FLAT", "jump_turn": None}

    ts = sorted(turns, key=lambda t: int(t.get("turn_index", 0)))
    pressures = [_pressure(t) for t in ts]
    peak = max(pressures)
    late = pressures[-1]

    sched = [max(1, int(t.get("scheduled_requests", 1) or 1)) for t in ts]
    max_sched = max(sched)
    drop = (max_sched - min(sched)) / max_sched

    ctx = [
        float(t.get("cached_context_tokens", 0.0) or 0.0)
        + float(t.get("new_prefill_tokens", 0.0) or 0.0)
        for t in ts
    ]
    if len(ctx) > 1:
        grow = sum(1 for i in range(1, len(ctx)) if ctx[i] >= ctx[i - 1]) / (len(ctx) - 1)
    else:
        grow = 1.0

    out0 = max(1.0, float(ts[0].get("output_tokens", 1.0) or 1.0))
    ratio = float(ts[0].get("new_prefill_tokens", 0.0) or 0.0) / out0

    # ---- CLASS ----
    if drop < DRAIN_FRACTION:
        # Non-draining cohort: queue never relieves. Sustained context growth on
        # a meaningfully loaded pool => the spike persists => SATURATE, even when
        # absolute pressure stays well below 1.0 (slow-creep saturation).
        if peak >= SLOW_CREEP_PEAK and grow >= MONOTONE_GROW:
            label = "SATURATE"
        else:
            label = "FLAT"
    else:
        # Draining cohort: sessions finish and the pool recovers.
        if peak >= HIGH_PEAK_OVERRIDE and late >= LATE_RELIEF:
            # ...but the RESIDUAL cohort is still oversubscribed at the end.
            label = "SATURATE"
        else:
            gate = PREFILL_HEAVY_PEAK if ratio >= PREFILL_HEAVY_RATIO else PERTURB_PEAK
            label = "PERTURB_RETURN" if peak >= gate else "FLAT"

    last_turn = int(ts[-1].get("turn_index", len(ts) - 1))
    first_turn = int(ts[0].get("turn_index", 0))
    med_out = statistics.median(
        [max(1.0, float(t.get("output_tokens", 1.0) or 1.0)) for t in ts]
    )
    pc = _pcross(pressures, ts, PRESSURE_ONSET)

    # ---- JUMP TURN (SATURATE only): KV-pool eviction crossing ----
    # The permanent jump is the pressure>=PRESSURE_ONSET crossing (MAE 0.88 turns
    # vs true jump), replacing the old cohort*turn budget. None for non-SATURATE.
    if label == "SATURATE":
        jt = last_turn if pc is None else pc
        jump_turn: int | None = int(max(JUMP_FLOOR, min(jt, last_turn)))
    else:
        jump_turn = None

    # ---- RAMP-WINDOW HINT (the stepping motion: a ramp, not a discrete jump) ----
    # Emitted for SATURATE, and additionally for osworld-like PERTURB_RETURN
    # (prefill-heavy + long-output + genuinely loads the pool): those cells show a
    # real mid-session stepping plateau the soft hint should track. chat
    # (output-heavy, ratio<1, low peak) and coding PERTURB (short output) are
    # excluded, so they stay confidence 0 (hint no-op) — protecting them structurally.
    emit_window = label == "SATURATE" or (
        label == "PERTURB_RETURN"
        and ratio >= PREFILL_HEAVY_RATIO
        and med_out >= OUT_KNEE_HI
        and peak >= 1.0
    )
    jump_start: int | None = None
    jump_end: int | None = None
    confidence = 0.0
    if emit_window and pc is not None:
        onset = max(JUMP_FLOOR, min(pc, last_turn))
        # Pressure-slope width: turns to climb PRESSURE_ONSET -> SAT_FULL.
        pfull = _pcross(pressures, ts, SAT_FULL)
        if pfull is None:
            width = RAMP_WIDTH_MAX     # never fully saturates in-session -> widest step
        else:
            width = max(RAMP_WIDTH_MIN, min(RAMP_WIDTH_MAX, pfull - pc))
        # Prefill-heavy long-output cells (osworld) tip the pool ~turn 2, ahead of
        # the pressure crossing — pull the onset to the structural floor so the
        # early t2->t4 ramp is covered.
        # FITTED (audit-v2 D7): the +2 pull is a 44-cell in-sample read-off; it was
        # missing from this module's fitted inventory until 2026-06-10.
        if ratio >= PREFILL_HEAVY_RATIO and med_out >= OUT_KNEE_HI:
            onset = max(JUMP_FLOOR, min(onset, first_turn + 2))
        jump_start = int(onset)
        jump_end = int(min(last_turn, onset + int(round(width))))
        # Graded confidence: trust the jump timing more at high cohort (timing MAE
        # ~18->1.8 turns over [40,160]); a mistimed low/mid-c window only gets a
        # fractional pull (graceful degradation). MAGNITUDE only — never the class.
        confidence = CONF_FLOOR + (1.0 - CONF_FLOOR) * _smoothstep(
            float(max_sched), CONF_COHORT_LO, CONF_COHORT_HI
        )
        if peak < 1.0:
            # FITTED (audit-v2 D7): the 0.4 confidence cap for sub-pool-full windows
            # is a 44-cell in-sample read-off; it was missing from this module's
            # fitted inventory until 2026-06-10.
            confidence = min(confidence, 0.4)
    elif emit_window:
        # Loads-but-never-crosses (low-pressure slow-creep): no-op window.
        jump_start = jump_end = last_turn
        confidence = 0.0

    return {
        "class": label,
        "jump_turn": jump_turn,
        "jump_start": jump_start,
        "jump_end": jump_end,
        "confidence": confidence,
    }


# Convenience alias matching the candidate API (accepts a turns list or a cell dict).
def predict_cell(turns, profile=None, concurrency=None):
    if isinstance(turns, dict):
        turns = turns.get("multiturn_turn_predictions", [])
    r = classify_session(turns)
    return r["class"], r["jump_turn"]


def session_ramp_window(turns) -> dict:
    """Just the stepping-ramp hint for a cell: {jump_start, jump_end, confidence}.

    Thin standalone accessor so a downstream predictor can ask only for the hint
    without depending on the class/jump_turn fields.
    """
    if isinstance(turns, dict):
        turns = turns.get("multiturn_turn_predictions", [])
    r = classify_session(turns)
    return {k: r.get(k) for k in ("jump_start", "jump_end", "confidence")}


__all__ = ["classify_session", "predict_cell", "session_ramp_window"]
