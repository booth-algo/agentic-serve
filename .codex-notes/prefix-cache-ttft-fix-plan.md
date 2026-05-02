# Prefix-Cache TTFT Fix Plan

Date: 2026-05-02

## Goal

Make TTFT prediction distinguish full-prefill from prefix-cached prefill using
explicit cache features, and prevent prefix-cache affected rows from silently
falling back to full-prefill prediction.

MoE support and GPU-specific kernel-characteristic issues are explicitly out of
scope for this first slice.

## Current Failure Surface

- `llm_predict/serving.py` already accepts `new_prefill_tokens`,
  `cached_context_tokens`, and `cache_hit_rate`, but still calls
  `composer.predict_ttft_ms(...)` with a coarse prefill model.
- `llm_predict/cache_aware.py` derives cache features only from `perTurn`.
- `llm_predict/export_serving_predictions.py` only uses cache-aware prediction
  when `perTurn` exists or a prior exists; otherwise it falls back to normal
  full-prefill prediction.
- `llm_predict/composer.py` models TTFT as ordinary prefill kernel composition,
  not a separate cached-prefill regime.

## Implementation Plan

### 1. Define the Cache Feature Contract

Add explicit prediction metadata fields:

- `total_context_tokens`
- `new_prefill_tokens`
- `cached_context_tokens`
- `cache_hit_rate`
- `cache_feature_source`
- `cache_prediction_regime`

Regimes:

- `full_prefill`
- `prefix_cached_prefill`
- `unknown_prefix_cache`

### 2. Stop Silent Full-Prefill Fallback

In `llm_predict/export_serving_predictions.py`, if
`ttft_validation_scope == "prefix_cache_affected"` and neither `perTurn` nor a
valid prior exists, do not emit a normal full-prefill prediction as if it were
valid.

Instead:

- mark the row `cache_prediction_regime=unknown_prefix_cache`, or
- emit a conservative unsupported / low-confidence row, and
- exclude it from canonical TTFT accuracy aggregates until cache features exist.

This prevents the `coding-singleturn` 1000%+ TTFT errors from polluting model
quality metrics as if the full-prefill physics path were wrong.

### 3. Derive Prefix-Cache Priors for Single-Turn Trace Workloads

`coding-singleturn` has no `perTurn`, but it is still prefix-cache affected
because many requests likely share large prompt prefixes.

Add a trace-level prior builder that tokenizes or loads the benchmark prompts
and computes:

- common prefix length distribution,
- median new suffix tokens,
- median cached prefix tokens,
- cache hit rate estimate,
- source dataset / profile / model / backend key.

Store this as a reproducible artifact, not a hand-tuned calibration. The prior
should come from prompt structure, not measured latency.

### 4. Add an Explicit Cached-Prefill TTFT Model

Extend `llm_predict/composer.py` with a separate method shaped like:

```text
predict_cached_prefill_ms(
  new_prefill_tokens,
  total_context_tokens,
  cached_context_tokens,
  cache_hit_rate
)
```

It should model:

- GEMM/GEMV work for the new suffix,
- attention over total context,
- KV read/load for cached prefix,
- KV write for new tokens.

Keep full-prefill behavior unchanged for `cache_hit_rate == 0`.

### 5. Wire Cached-Prefill Through Serving

In `llm_predict/serving.py`, branch TTFT computation:

```text
if cache_aware:
    ttft_kernel = composer.predict_cached_prefill_ms(...)
else:
    ttft_kernel = composer.predict_ttft_ms(...)
```

Keep decode using total context. Prefix caching reduces TTFT prefill work, but
decode still attends over the full context.

### 6. Move Calibration Out of the Core Explanation

Keep framework correction metadata visible, but separate:

- raw physics prediction,
- cache-aware physics prediction,
- optional calibrated overlay.

The dashboard should be able to show that prefix caching changed the physics
inputs before any correction factor is applied.

### 7. Add Regression Tests

Add tests that lock the intended behavior:

- multi-turn `perTurn` rows use `new_prefill_tokens < total_context_tokens`,
- `coding-singleturn` does not fall back to full-prefill when marked
  prefix-cache affected,
- cached-prefill TTFT is lower than full-prefill for the same total context
  when cached tokens are high,
- decode cost still scales with total context,
- rows with unknown cache features are marked unsupported / low-confidence.

### 8. Local Rebuild and Validation

Rebuild `serving-predictions.json` locally and compare:

- `prefix_cache_affected && cache_aware_applied=false` should go to zero, or
  those rows should be excluded / flagged,
- `coding-singleturn` TTFT error should no longer dominate the current
  aggregate,
- dashboard should surface cache feature source / regime so bad rows are
  diagnosable.

### 9. GitHub Actions Path

Make the prior artifact generation part of the dashboard rebuild workflow before
exporting predictions.

CI should fail if:

- a prefix-cache affected canonical row falls back to full-prefill,
- required cache metadata is missing,
- serving prediction export produces unsupported rows without explicit dashboard
  labeling.

## Acceptance Criteria

- No prefix-cache affected row is silently treated as ordinary full-prefill.
- `coding-singleturn` gets trace-derived cache features or is explicitly marked
  unsupported.
- Cached-prefill TTFT uses a separate physics path from full-prefill.
- Local and GitHub Actions rebuilds produce the same cache metadata and
  prediction regimes.
- TTFT accuracy reporting separates prefix-cache, MoE, and ordinary
  full-prefill errors.
