# Cache-Aware Serving Predictor Note

Date: 2026-04-29

## Context

The current serving predictor mostly treats input length as a single scalar:

```text
TTFT prefill work ~= total_input_tokens
decode KV work ~= total_input_tokens + generated_tokens_so_far
```

That is acceptable for single-turn and synthetic full-prefill workloads, but it
breaks down for multi-turn agent workloads such as SWE-bench, TerminalBench,
and OSWorld. In these traces, the prompt seen by a turn is cumulative, while
the actual prefill work may be only the new suffix if prefix/KV cache reuse is
working.

## Key Insight

For multi-turn serving, `ISL` alone is not a sufficient predictor input.
The predictor needs to distinguish:

- `total_context_tokens`: full prompt/context length visible to the turn.
- `new_prefill_tokens`: tokens that were not already in the prefix/KV cache.
- `cached_context_tokens`: tokens reused from cache.
- `cache_hit_rate`: `cached_context_tokens / total_context_tokens`.
- `kv_scan_tokens`: tokens attended over during decode.
- `cache_residency`: whether cached prefixes survive under concurrency and memory pressure.

The serving cost decomposes more like:

```text
TTFT prefill compute ~= new_prefill_tokens
decode attention/KV cost ~= total_context_tokens + generated_tokens_so_far
capacity footprint ~= cached_context_tokens + active generated KV
```

The current model effectively uses:

```text
TTFT prefill compute ~= total_context_tokens
```

which overpredicts TTFT when prefix caching works. But using only
`new_prefill_tokens` would under-model decode and memory pressure, because the
decode phase still attends over the cached context.

## Implemented v1 Shape

Future serving prediction should accept cache-aware inputs:

```text
predict_serving(
  total_context_tokens,
  new_prefill_tokens,
  output_tokens,
  concurrency,
  cache_hit_rate,
  backend,
)
```

For the paper narrative, this is useful: agentic workloads are not just
"longer ISL." They introduce cache state as a first-class serving variable.

The v1 implementation uses aggregate `perTurn` summaries from `data.json`.
For each multi-turn row, it derives per-turn deltas, then uses a
successful-request-weighted representative turn for export/calibration speed.
This keeps TTFT tied to `new_prefill_tokens` and decode tied to full
`total_context_tokens`.

It also fits empirical prefix-cache contention factors keyed by
GPU/backend/model/profile/concurrency. These factors are applied only when the
base serving calibration is `medium_confidence` or `high_confidence`.

## Current Evidence

After the v1 cache-aware export:

- H100 full-prefill/stress rows: E2EL median error around 7.7%.
- H100 prefix-cache rows: E2EL median error around 5.5% in the fitted artifact.
- H100 SWE-bench multi-turn short: E2EL median error around 10.6%.
- H100 TerminalBench multi-turn medium: E2EL median error around 14.3%.
- H100 OSWorld multi-turn medium: E2EL median error around 3.0%.

Important caveat: the current contention factors are fitted on the same
profile/concurrency rows they correct. This is useful for making the dashboard
cache-aware and for explaining the missing variable, but it is not yet a
holdout-validated general cache-residency model.
