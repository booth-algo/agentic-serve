# Session Summary: Distributional Benchmark Refactor

Date: 2026-04-30

## Why This Happened

The benchmark suite was taking too long because the old agentic multi-turn profiles replayed real traces directly through vLLM/SGLang. The professor's intended method was distributional synthetic benchmarking:

- sample turn count from real agentic traces
- sample per-turn input growth / output token distributions
- generate synthetic growing histories with the same serving shape
- benchmark serving latency and throughput without replaying every real trace

Important clarification: synthetic traces do not magically reduce GPU time for the same token volume. The time savings come from collapsing short/medium/long legacy buckets into one canonical sampled profile per workload, and from choosing a smaller Monte Carlo sample size for sweep coverage.

## Merged Work

PR #50 was merged to `main`:

```text
https://github.com/booth-algo/agentic-serve/pull/50
merge commit: 0532de2
```

Direct follow-up commit on `main`:

```text
90496d7 Namespace current benchmark uploads
```

The local repo was clean after the direct commit.

## Core Code Changes

Distributional synthetic workloads now live under:

```text
inference-benchmark/src/workloads/distributional.py
inference-benchmark/src/workloads/trace_distributions.py
inference-benchmark/data/distributions/
```

Canonical active profiles:

```text
chat-singleturn
coding-singleturn
chat-multiturn
swebench-multiturn
terminalbench-multiturn
osworld-multiturn
```

Legacy short/medium/long and stress profiles remain runnable, but should be treated as archive/historical coverage.

## Prediction Observability Added

Future benchmark artifacts now record fields useful for cache-aware serving prediction:

- `session_id`
- `turn_index`
- `previous_context_tokens`
- `total_context_tokens`
- `new_prefill_tokens`
- `cached_context_tokens`
- `cache_hit_rate`
- block-aligned cache estimates using `prefix_cache_block_size`
- request timing metadata such as client queue wait and request wall time
- server launch metadata such as max model length, TP size, prefix caching state, chunked prefill state, GPU memory utilization

This is still logical cacheability, not engine-reported cache truth. Exact engine cache hits/misses would require vLLM/SGLang instrumentation.

## Context Safety Fix

Claude hit HTTP 400 overflows on A100 with 32K context, e.g. prompts at ~32752 tokens plus output exceeded the 32768 model limit.

Fix implemented:

- distributional generator reserves requested output tokens under `--max-context-tokens`
- also reserves a default `--context-safety-margin-tokens 256`

Effective guard:

```text
prompt_tokens + requested_output_tokens <= max_context_tokens - safety_margin
```

This avoids relying on `--max-model-len` being larger than the benchmark context cap.

## R2 Layout Fix

Raw current benchmark results now upload to:

```text
s3://agent-bench/results/current/${hardware}_${model}_tp${TP}_${backend}/
```

Old flat raw results remain in:

```text
s3://agent-bench/results/${hardware}_${model}_tp${TP}_${backend}/
```

The orchestrator has a fallback for jobs launched before the namespace patch: it can still find old remote `/tmp/results/${dir}` outputs and upload them into the new `results/current/...` R2 prefix.

## Dashboard / Coverage State

Dashboard build data recognizes:

- `dashboard_scope=current` for new active-profile artifacts
- missing/old scope as `archive`

Coverage should default to current canonical coverage. Archive view should show historical inventory and not count legacy missing cells against the current paper suite.

## Validation Already Run

Before PR #50 merge:

```text
cd inference-benchmark && python3 -m unittest discover tests
bash -n on patched benchmark scripts
cd inference-benchmark && python3 scripts/compile_sweep.py
cd inference-benchmark/dashboard && npm run lint
cd inference-benchmark/dashboard && npm run build
```

After R2 namespace patch:

```text
bash -n inference-benchmark/scripts/bench_orchestrator.sh
```

## Things To Remember

- Current canonical grid is vLLM-priority.
- SGLang multi-turn launcher exists now, but `sweep.yaml` does not yet emit SGLang multi-turn cells.
- H100x8 is intentionally not required for canonical coverage.
- The goal is not task correctness; it is serving shape, latency, throughput, OI/CF, and predictor data.
- Exact prefix cache residency still requires engine instrumentation.
