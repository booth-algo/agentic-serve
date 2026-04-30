# Distributional Synthetic Benchmark Refactor Plan

Date: 2026-04-30

## Goal

Refactor `inference-benchmark` to support fast distributional synthetic agentic workloads while preserving all legacy benchmark data, coverage tables, and historical profile names.

This replaces expensive replay of full TerminalBench/SWE-bench/OSWorld traces for coverage sweeps with synthetic sessions sampled from real trace distributions:

- sample number of turns from empirical traces
- sample per-turn new input token deltas from empirical traces
- sample per-turn output token counts from empirical traces
- generate synthetic text with growing histories so prefix caching is exercised

## Hard Guardrails

- Do not create a new top-level `inference-benchmark-v2`.
- Do not delete legacy profiles.
- Do not overwrite or remove historical coverage tables.
- Do not make legacy short/medium/long gaps count against canonical paper coverage.
- Do not normalize legacy multi-turn short/medium/long profiles into the new canonical profiles.
- Do not move or delete raw R2 `results/`.
- Do not make H100x8 required coverage.
- Only normalize `coding-agent -> coding-singleturn`, because that is a naming clarification for the same single-turn coding workload.

## Target Directory Shape

```text
inference-benchmark/
  src/workloads/
    profiles.py                  # canonical active profiles + legacy inactive profiles
    dataset.py                   # existing datasets
    distributional.py            # NEW: synthetic turn/ISL/OSL samplers
    trace_distributions.py       # NEW: load compact distribution JSONs

  scripts/
    build_trace_distributions.py # NEW: extract distributions from real traces

  data/distributions/
    chat_multiturn.json
    swebench_multiturn.json
    terminalbench_multiturn.json
    osworld_multiturn.json
```

## Canonical Active Profiles

These are the paper-facing profiles and should be the default denominator on the coverage page:

```text
chat-singleturn
coding-singleturn
chat-multiturn
swebench-multiturn
terminalbench-multiturn
osworld-multiturn
```

Compatibility alias:

```python
PROFILE_ALIASES = {
    "coding-agent": "coding-singleturn",
}
```

Dashboard/build-data should display historical `coding-agent` rows as `coding-singleturn`.

## Legacy / Historical Profiles

Keep these runnable and visible in a separate historical section when data exists:

```text
chat-short
chat-medium
chat-multiturn-short
chat-multiturn-medium
chat-multiturn-long
swebench-multiturn-short
swebench-multiturn-medium
swebench-multiturn-long
terminalbench-multiturn-short
terminalbench-multiturn-medium
terminalbench-multiturn-long
osworld-multiturn-short
osworld-multiturn-medium
osworld-multiturn-long
prefill-heavy
decode-heavy
random-1k
```

Coverage page should split:

```text
Canonical coverage: only the 6 paper profiles
Historical/legacy coverage: old short/medium/long + stress profiles
```

Default dashboard view should use canonical coverage. Legacy rows can still appear in benchmark tables if data exists.

## Distributional Sampling Details

Do not sample total ISL independently per turn.

Sample:

```text
turn_count ~ empirical trace distribution
new_input_tokens_t ~ empirical per-turn input growth
output_tokens_t ~ empirical per-turn output distribution
```

Build growing synthetic histories:

```text
turn 1: synthetic user chunk 1
turn 2: previous user + synthetic assistant output 1 + synthetic user chunk 2
turn 3: previous history + synthetic assistant output 2 + synthetic user chunk 3
...
```

For `t > 1`, synthetic user chunk length should be:

```text
max(1, sampled_new_prefill_tokens_t - previous_output_tokens)
```

API-reported prompt tokens remain source of truth for exported:

```text
new_prefill_tokens
cached_context_tokens
cache_hit_rate
```

Add optional runner argument:

```text
--max-context-tokens
```

Sweep scripts should pass server `$MAX_LEN`. Distributional datasets should cap or stop generated histories at this value so smaller GPUs do not fail while H100 can run longer.

## Concurrency Sets

Single-turn:

```text
1, 10, 20, 40, 80, 160, 256, 320
```

Multi-turn:

```text
5, 20, 40, 80, 160
```

## Hardware Plan

Use the lean paper set as required coverage. Run both vLLM and SGLang everywhere.

Required:

```text
H100x1        Llama-3.1-8B
H100x4        Llama-3.1-70B
A100-40GBx1   Llama-3.1-8B
RTX3090x1     Llama-3.1-8B
RTX2080Tix1   Llama-3.1-8B
```

Notes:

- H100x4 is enough for 70B/72B.
- H100x8 is optional scaling evidence only, not required for fit.
- H100x2 is too tight for 70B because weights leave little KV/CUDA-graph/activation headroom.

Optional appendix/scaling:

```text
H100x8        Llama-3.1-70B
A100-40GBx4   Llama-3.1-70B
Qwen/MoE diversity runs if cheap or already present
```

## Step-by-Step Implementation

### Phase 0: Stabilize Context

- Keep this note as source of truth before editing benchmark logic.
- Inspect current `inference-benchmark` profile, runner, sweep, uploader, and dashboard code.
- Record existing legacy profile lists before changing them.

### Phase 1: Distribution Artifacts Only

- Add `scripts/build_trace_distributions.py`.
- Add `src/workloads/trace_distributions.py`.
- Generate compact JSON distributions under `data/distributions/`.
- No runner behavior changes in this phase.

### Phase 2: Distributional Workload Module

- Add `src/workloads/distributional.py`.
- Implement turn count and per-turn delta sampling.
- Implement synthetic growing-history construction.
- Add small deterministic tests/fixtures for cache growth and max-context capping.

### Phase 3: Profiles Without Coverage Rewrite

- Add canonical active profiles in `profiles.py`.
- Mark legacy profiles inactive/historical, but keep them runnable.
- Add `coding-agent -> coding-singleturn` alias.
- Do not remove old short/medium/long definitions.

### Phase 4: Runner and Sweep Support

- Add runner support for distributional profiles.
- Add `--max-context-tokens`.
- Ensure per-turn exported rows include enough fields for future cache-aware prediction:
  - total prompt tokens
  - new prefill tokens when available
  - cached context tokens when derivable
  - output tokens
  - per-turn TTFT/TPOT/ITL/E2EL if measured by the runner
- Add missing generic SGLang multi-turn sweep script with the same interface as the vLLM multi-turn sweep script.

### Phase 5: Dashboard Coverage Split

- Keep one coverage page.
- Default to canonical coverage with denominator = 6 profiles.
- Add separate historical/legacy section.
- Preserve existing legacy coverage data and tables; move them into the historical section if needed.

### Phase 6: Validation

- Run Python compile checks on touched modules/scripts.
- Run distribution sampler unit checks.
- Run profile listing/dry-run checks.
- Run dashboard lint/build.
- Confirm no generated artifact loses legacy rows.

## Test / Acceptance Checklist

- `python3 -m py_compile` passes for touched Python modules/scripts.
- Distribution sampler tests pass on deterministic fixtures.
- `runner --list-profiles` defaults to canonical active profiles only, if that command exists.
- Legacy profiles are still runnable explicitly.
- Sweep dry-run shows canonical profiles for new active jobs.
- Dashboard `npm run lint` passes.
- Dashboard `npm run build` passes.
- Coverage default denominator is exactly 6 canonical profiles.
- Historical/legacy coverage remains visible and does not affect canonical completeness.
- Exported benchmark data includes `coding-singleturn` display name for old `coding-agent` rows.

