# Direction from Aaron (2026-04-30 Meeting)

See also: `.codex-notes/from-aaron-meeting-2026-04-30.md` (canonical copy).

## Decision: Remove Empirical Correction Factors

The serving predictor should operate on first-principles physics (GEMM +
flash attention composition). The per-GPU alpha/beta framework correction
factors were absorbing gaps that should instead be modeled explicitly in the
kernel composition path.

**What was removed** (dirty worktree, not yet committed):
- `framework_corrections.py`: `FrameworkCorrection`, `CorrectionParams`,
  `_BACKEND_DEFAULTS`, `_CORRECTIONS`, `framework_correction()`,
  `get_correction_params()`, `get_correction_note()`
- `serving.py`: `framework_correction` import, `ttft_correction_applied`
  field. TTFT is now `raw_kernel * queue_factor` with no alpha/beta.
- `cache_aware.py`, `export_serving_predictions.py`, `validate.py`: same
  removal plus new cache tracking metadata.

**Why:**
1. The predictor should work from physics — if it doesn't, the gap IS the
   signal about what's missing (attention, KV overhead, scheduling).
2. The correction factors were fitted to the same data they corrected —
   not a holdout-validated model.
3. Patching with empirical factors hides real modeling gaps from reviewers.

**Effect on predictions:**
- Dense TPOT: already good without corrections (2-8% error)
- TTFT: now has a fixed ~8-18ms gap at short ISL, narrowing at long ISL.
  This gap is the next thing to model (flash attention, QK transpose, KV
  cache allocation).
- `coding-singleturn` and other prefix-cache rows without cache features
  are now flagged `unsupported` instead of generating 1600%+ error.

## Paper Direction

- Primary contribution: the physics-based predictor.
- Benchmark data is validation, not the product.
- Scope: single-node multi-GPU. Multi-node is future work.
- Missing cells in coverage grid are fine — the predictor fills them.
