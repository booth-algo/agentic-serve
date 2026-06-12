"""Kernel-composition TPOT with the standalone classifier's stepping-ramp HINT.

RETIRED 2026-06-10 (de-fit campaign lane L7; audit-v2 item D9): moved to
``simulator/_legacy/``. Not reachable from any production entrypoint
(``build_simulator_rows`` / ``validate_*`` / ``gate_scoped_rows`` / dashboards);
the only caller is ``profiling/process/_legacy/
augment_simulator_predictions_with_kernel.py``, an opt-in dashboard diagnostic
that injects a ``tpot_pred_kernel_hint`` comparison column on demand. Production
``build_simulator_rows`` does NOT emit that column (shipped predictions contain
zero such fields). Kept (not deleted) so the diagnostic still runs.

The production kernel predictor (``simulator/kernel_tpot.py``) is a pure
pressure-driven amplifier. It nails the plateau MAGNITUDE but its rise can lag
the measured saturation step by a few turns (the pressure ramp is gradual).
``session_regime_classifier`` independently predicts WHEN the step happens (the
KV-pool eviction crossing) and over how many turns it ramps (the pressure-slope
width). This module folds that timing hint into the kernel prediction as a
**one-sided soft pull**:

    pred[t] = pressure_path[t] + confidence * max(0, ramp_target[t] - pressure_path[t])
    ramp_target[t] = kstep[t] + smoothstep(t; jump_start, jump_end) * (ceiling - kstep[t])
    ramp_target[t] = min(ramp_target[t], max(pressure_path[t:]))   # forward-max recovery cap

Why this shape (claimed verified by workflow wf_9a938421 — unarchived; see the
HISTORICAL note below):

* ONE-SIDED (``max(0, ...)``): the hint can only pull a turn UP toward the
  output-keyed saturation ceiling, never push it down. So no cell can score worse
  than the pure-pressure baseline by construction.
* CONFIDENCE-SCALED & class-gated: the classifier emits confidence 0 for FLAT /
  chat / coding-PERTURB cells, so they are byte-identical to production. Only
  SATURATE and osworld-like PERTURB cells (which genuinely step) are touched, and
  the pull is scaled by how trustworthy the jump timing is at that cohort.
* FORWARD-MAX RECOVERY CAP: the pull may not exceed the highest value the
  pressure path itself reaches from this turn onward. For osworld (which recovers
  — its pressure path declines after the peak) this makes the hint follow the
  recovery DOWN instead of pinning to the static ceiling; for coding families
  whose pressure path is still climbing it is a no-op.

HISTORICAL — figures unverifiable today (audit-v2 D9): the numbers below were
read from workflow wf_9a938421, which left no artifact, and the underlying
kernel has changed twice since (tuned knees retired in ``aea241e``; firing-gate
hysteresis added 2026-06-10), so they no longer describe the current kernel.
The former claim that this is "an additive ``tpot_pred_kernel_hint`` column" is
stale: ``build_simulator_rows`` never emits it; only the _legacy augmenter
injects it on demand. As measured THEN, vs the then-production kernel (all 1043
cells): overall MAPE 16.48 -> 16.2, improving on every profile + plateau slice
(osworld-plateau 19.5 -> 16.2, swebench-plateau 9.2 -> 8.9, terminalbench-plateau
18.3 -> 17.7), with the production ``predict_cell_tpot`` untouched.
"""

from __future__ import annotations

import statistics

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_tpot import (
    KernelTurnInput,
    _kernel_step_ms,
    _smoothstep,
    predict_cell_tpot,
    saturated_ceiling_ms,
)
from simulator._legacy.session_regime_classifier import session_ramp_window


def predict_cell_tpot_hinted(
    turns: list[KernelTurnInput],
    turn_indices: list[int] | None = None,
    params: RooflineParams | None = None,
) -> list[float]:
    """Per-turn TPOT for a cell, with the classifier stepping-ramp soft hint.

    ``turn_indices`` is the per-turn ``turn_index`` (defaults to positional). It
    must be supplied when the stored turn order isn't 0..n-1 so the turn-space
    ramp lines up with the classifier's window (also turn-index space). Falls
    back to the production ``predict_cell_tpot`` for any cell the classifier gives
    confidence 0 (FLAT / chat / coding-PERTURB / slow-creep) — byte-identical.
    """
    p = params or RooflineParams()
    n = len(turns)
    if n == 0:
        return []
    if turn_indices is None:
        turn_indices = list(range(n))

    pressure_path = predict_cell_tpot(turns, p)

    win = session_ramp_window(
        [
            {
                "cached_context_tokens": float(t.cached_context_tokens),
                "new_prefill_tokens": float(t.new_prefill_tokens),
                "output_tokens": float(t.output_tokens),
                "scheduled_requests": float(t.scheduled_requests),
                "turn_index": int(ti),
            }
            for t, ti in zip(turns, turn_indices)
        ]
    )
    conf = win.get("confidence") or 0.0
    js, je = win.get("jump_start"), win.get("jump_end")
    if conf <= 0 or js is None:
        return pressure_path  # FLAT / chat / coding-PERTURB / slow-creep -> identical

    median_output = statistics.median([max(1.0, float(t.output_tokens)) for t in turns])
    ceiling = saturated_ceiling_ms(median_output)

    preds: list[float] = []
    for i, (inp, ti, base) in enumerate(zip(turns, turn_indices, pressure_path)):
        kstep = _kernel_step_ms(inp, p)
        ramp = _smoothstep(float(ti), float(js), float(je))
        ramp_target = kstep + ramp * (ceiling - kstep)
        # Forward-max recovery cap: never pull above the pressure path's own peak
        # from here on. Follows osworld's recovery down; no-op while still climbing.
        ramp_target = min(ramp_target, max(pressure_path[i:]))
        preds.append(base + conf * max(0.0, ramp_target - base))
    return preds


__all__ = ["predict_cell_tpot_hinted"]
