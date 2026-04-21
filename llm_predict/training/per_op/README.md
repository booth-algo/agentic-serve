# training/per_op

Offline training pipeline for the per-op XGBoost predictor consumed by
`llm_predict/predictors/per_op/predictor.py`.

Productised from the ad-hoc script `experiment/train_perop_v4.py` —
same 26-feature analytical formulation, same XGB regression on
`log(duration_us)`, but structured to mirror
`llm_predict/training/per_kernel/` so the two training pipelines share
convention (held-out splits, LOMO CV, pkl payload, report layout).

## What lives here

| Module | Role |
|---|---|
| `feature_spec.py` | 26 analytical features + `OP_MAP` + leak-guard + `audit_features()` |
| `labeler.py` | raw CUDA-event CSV → `data/per_op_labeled.csv` |
| `trainer.py` | per-GPU XGBoost trainer (LOMO + held-out) → pkl + reports |
| `splits.py` | Thin wrapper over `per_kernel.splits` (same LOMO / held-out contract) |
| `tests/` | pytest smoke — feature count, leak-guard, flop formulas, labeler row shape |
| `reports/` | committed markdown + json training/labeling reports |
| `data/` | `per_op_labeled.csv` (the trainer input) |

## Input data

Raw CUDA-event CSVs from torch.profiler, one per (GPU × model), laid out as:

    {raw_root}/{GPU}/{model_dir}/tp1/phase1_bsseq_profiles.csv

`model_dir` must be one of the keys of
`llm_predict/training/per_kernel/model_specs._DIR_TO_SHORT` — the same
model registry the per-kernel pipeline uses.

Expected columns (either spelling accepted):

    layer_name | op_name    — one of OP_MAP's keys (self_attn / mlp / ...)
    bs                      — batch size
    seq                     — sequence length per batch
    n_tokens                — bs*seq during prefill; bs during decode
    kv_cache_len            — 0 for prefill, >0 for decode
    latency_ns | duration_us — measured op latency

Historical note: the legacy training data on gpu-4 at
`/data/LLMServingSim/llm_profile/perf_models/A100/*/tp1/phase1_bsseq_profiles.csv`
is NOT migrated by this session. It stays on gpu-4. Phase 5 of the
larger plan will produce fresh CUDA-event CSVs that feed into this
pipeline.

## Output artifacts

- `llm_predict/training/per_op/data/per_op_labeled.csv` — the combined
  labeled dataset (schema: see `labeler.OUT_COLS`).
- `llm_predict/profiling/data/{A100,RTX3090,RTX2080Ti}/trained/per_op/perop_v5_shape.pkl`
  — one pkl per GPU. Deliberately a new filename so it does not clobber
  the legacy `perop_analytical_v4.pkl` that the runtime loader at
  `llm_predict/predictors/per_op/predictor.py` still reads.
- `llm_predict/training/per_op/reports/{labeling,training}_report.{md,json}`
  — human-readable pipeline outputs.

### Pkl payload (stable contract)

```
{
    "model":           xgb.XGBRegressor,
    "feature_cols":    list[str],  # == feature_spec.PEROP_FEATURES (len 26)
    "target":          "log_duration_us",
    "gpu":             str,
    "n_training":      int,
    "heldout_mape":    float | None,
    "heldout_models":  list[str],
    "pool_models":     list[str],
    "version":         "per_op_v5",
}
```

## Reproduction

```bash
# Phase A1 — label
python -m llm_predict.training.per_op.labeler \
    --raw-root /path/to/profiling_data \
    --out llm_predict/training/per_op/data/per_op_labeled.csv

# Phase A2 + A3 — train
python -m llm_predict.training.per_op.trainer \
    --data llm_predict/training/per_op/data/per_op_labeled.csv \
    --out-dir llm_predict/profiling/data

# Smoke tests
python -m pytest llm_predict/training/per_op/tests/ -v
```

## Known duplication (to be resolved in Phase 5)

The 26-feature computation currently exists in three places:

1. `llm_predict/training/per_op/feature_spec.py:compute_features` (this module — canonical).
2. `llm_predict/predictors/per_op/features.py:compute_perop_features` (runtime).
3. `llm_predict/models/software/transformer.py:175:compute_perop_features` (duplicated).

Sites 2 and 3 should be refactored to import from site 1. We avoid that
refactor in this phase because the runtime loader in
`predictors/per_op/predictor.py` still points at the legacy
`perop_analytical_v4.pkl`, and moving the feature computation + the
runtime pkl path in the same commit would invalidate the deployed
predictor. The smoke test
`test_op_map_matches_runtime_features_module_keys` catches feature-length
drift between the two sites while they remain separate.

## Cut-over plan (Phase 5)

1. Collect fresh CUDA-event CSVs on each GPU at
   `{raw_root}/{gpu}/{model_dir}/tp1/phase1_bsseq_profiles.csv`.
2. Run the labeler + trainer above.
3. Once `perop_v5_shape.pkl` beats the legacy `perop_analytical_v4.pkl`
   on held-out MAPE, flip `predictors/per_op/predictor.py` default to
   the new filename.
4. Refactor `predictors/per_op/features.py` and `models/software/transformer.py`
   to `from llm_predict.training.per_op.feature_spec import compute_features`.

## Out of scope (this phase)

- Training an A100 pkl with the new pipeline (no fresh traces on this host).
- Modifying `predictors/per_op/predictor.py` (runtime loader stays on v4).
- Modifying `models/software/transformer.py` (still carries duplicated feature fn).
