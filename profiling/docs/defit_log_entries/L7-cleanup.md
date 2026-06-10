# L7 — dead-path cleanup (audit-v2 D5–D9 + S7 misfile + supersession note)

**Date:** 2026-06-10 · **Lane:** L7 `deadpath-cleanup` · **Contract:** production predictions
BYTE-IDENTICAL (comments/docs/dead-path moves only — no behavior change anywhere).

## Reachability verdict (the D5–D9 gate question)

`simulator/session_regime_classifier.py` and `simulator/kernel_tpot_hint.py` are **provably dead
beyond `_legacy`**. Full-repo reference search (all file types, including dynamic/string forms
`classify_session` / `session_ramp_window` / `predict_cell_tpot_hinted` / `tpot_pred_kernel_hint`):

- `kernel_tpot_hint` imports: ONLY `profiling/process/_legacy/augment_simulator_predictions_with_kernel.py`
  (an opt-in dashboard diagnostic, itself already `_legacy`) and its own test.
- `session_regime_classifier` imports: ONLY `kernel_tpot_hint` and its own test.
- `build_simulator_rows.py`, `validate_tpot.py`, `validate_ttft.py`, `gate_scoped_rows.py`: zero
  references (direct or transitive).
- Dashboard: `ServingPredictionsPage.tsx` declares `tpot_pred_kernel_hint?: number` (OPTIONAL) and
  production `build_simulator_rows` never emits the column — it renders only if someone hand-runs
  the legacy augmenter. The remaining mentions in `ramp_tpot.py:43,343` and `kernel_tpot.py:121`
  are comments only (those files are L2/L5-owned; flagged here for integration, not edited).

## Action taken: RETIRE by moving to `simulator/_legacy/` (not delete) — why

The campaign offered "delete if `_legacy` already holds the only caller". The only caller IS
`_legacy` (`augment_simulator_predictions_with_kernel.py`), but that augmenter is not solely a
hint consumer — it also injects the still-live `tpot_pred_kernel` and `tpot_pred_ramp`
(L2-owned `ramp_tpot`) comparison columns, and the dashboard still declares the optional column.
Deleting the two modules would break the whole augmenter (or force deleting a file with live,
other-lane-owned function). Moving is therefore the lower-risk option: one-line import retarget
in the augmenter, code + corrected provenance preserved, byte-identical guaranteed.

Moves (git mv, history-preserving):
- `simulator/session_regime_classifier.py` → `simulator/_legacy/session_regime_classifier.py`
- `simulator/kernel_tpot_hint.py` → `simulator/_legacy/kernel_tpot_hint.py`
- their tests → `simulator/_legacy/tests/` (they test dead code; out of the production pytest
  scope `simulator/tests/ profiling/tests/`, but still green when run directly: 14/14 pass)
- import retargets: `kernel_tpot_hint` → `simulator._legacy.session_regime_classifier`;
  augmenter → `simulator._legacy.kernel_tpot_hint`. Tombstone headers added to both modules.

## Provenance corrections (false-provenance is a first-class bug)

- **D5** `PRESSURE_ONSET=0.85`: deleted the false "no new fitted constant … sibling of kernel
  P_LO=0.8" claim — P_LO (0.8, later 0.88) was deleted in `aea241e` and 0.85 never equaled it;
  the measured eviction-onset artifact (`ramp_knees_h100_llama31_8b.json`) puts the knee at
  0.4456 (~2× below). Relabeled FITTED (44-cell in-sample read-off); wf_9a938421 noted artifact-less.
- **D6** class thresholds: covered by the module tombstone (all flagged FITTED on the same
  44 in-sample H100 cells, no builder regenerates any anchor; module now diagnostic-only).
- **D7**: the 0.4 confidence cap and the +2 onset pull are now FLAGGED at their use sites and
  added to the module's own fitted-constant inventory (they were missing from it);
  `SAT_FULL=2.0` relabeled FITTED = retired tuned kernel P_HI_LONG, not a measurement.
- **D8** `OUT_KNEE_HI=80.0`: "REUSED from kernel_tpot" relabeled FROZEN SNAPSHOT — the kernel
  source moved to 86.0 and was demoted to a non-formula ceiling-cluster label; module stays
  deliberately standalone (no re-import).
- **D9** `kernel_tpot_hint` docstring: the "additive `tpot_pred_kernel_hint` column" and the
  16.48→16.2 wf_9a938421 validation marked HISTORICAL/unverifiable (column never emitted by
  production; workflow unarchived; kernel changed twice since).
- **S7 misfile** in `fitted_constants_audit.md`: `preempt_policy='tail'` moved out of
  VLLM-CONFIG into a new MODEL-CHOICE entry (engine tail-preempt is RUNNING-request semantics;
  idle sessions' blocks are evicted LRU-oldest from the free queue — audit-v2 S7).
- `fitted_constants_audit.md` header: dated note that audit-v2 supersedes its open items
  (also covers the v2 §3 item 10 stale `SAT_SUSTAIN_LO=10.0` row at the de-fit-needed section).

## Left for other lanes / integration (seen, not touched)

- `ramp_tpot.py` D1–D4 + its kernel_tpot_hint comment mentions — L2.
- `kernel_tpot.py:121` comment naming kernel_tpot_hint (now `_legacy`) — production module,
  one-word doc staleness; fix at integration if desired.
- `prediction_construction.md` items (v2 §3 #9) — integration phase only.

## Gate

Replay-ON `gate_scoped_rows` baseline captured BEFORE any edit (`/tmp/l7_base.*`); post-change
rerun (`/tmp/l7_after.*`) — predictions and metrics **byte-identical** (sha256 match), as the
contract requires. `pytest simulator/tests/ profiling/tests/ -q` green.
