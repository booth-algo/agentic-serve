# Sweep Delegation: Current Canonical Distributional Runs

Date: 2026-04-30

## Goal

Run the current canonical benchmark grid without overwriting legacy R2 artifacts. Prioritize vLLM first. SGLang multi-turn support exists, but should be treated as lower priority unless explicitly enabled in `sweep.yaml`.

## Code State To Use

Use `main` at or after:

```text
90496d7 Namespace current benchmark uploads
```

This includes:

- distributional synthetic workloads
- cache/prefix prediction metadata
- context safety margin fix
- current R2 raw results namespace
- SGLang multi-turn launcher

## R2 Paths

New raw benchmark results should upload to:

```text
s3://agent-bench/results/current/
```

The orchestrator does this automatically via:

```text
inference-benchmark/scripts/bench_orchestrator.sh
```

Do not upload new canonical runs to the old flat prefix:

```text
s3://agent-bench/results/
```

## Current Canonical Profiles

Single-turn:

```text
chat-singleturn
coding-singleturn
```

Multi-turn:

```text
chat-multiturn
swebench-multiturn
terminalbench-multiturn
osworld-multiturn
```

## Current Concurrency Grid

Single-turn:

```text
1 10 20 40 80 160 256 320
```

Multi-turn:

```text
5 20 40 80 160
```

Rationale: multi-turn TPOT/throughput tends to plateau earlier, so high-concurrency tail points are less valuable than single-turn high-C coverage.

## First Smoke Sweep

Before launching the full grid, run one H100 smoke sweep for a small model across all 6 canonical profiles.

Suggested target:

```text
H100, Llama-3.1-8B, TP=1, vLLM
```

Profiles:

```text
chat-singleturn
coding-singleturn
chat-multiturn
swebench-multiturn
terminalbench-multiturn
osworld-multiturn
```

Use vLLM first. Make sure server launch and runner metadata agree:

```text
--enable-prefix-caching
--enable-chunked-prefill
--max-model-len <MAX_LEN>
--gpu-memory-utilization <GPU_MEM>
--tensor-parallel-size <TP>
```

Runner metadata should include:

```text
--prefix-caching-state on
--chunked-prefill on
--max-model-len <MAX_LEN>
--gpu-memory-utilization <GPU_MEM>
--tensor-parallel-size <TP>
```

For multi-turn also include:

```text
--max-context-tokens <MAX_LEN>
--context-safety-margin-tokens 256
```

## Orchestrator Path

The preferred path is:

```text
cd /root/agentic-serve/inference-benchmark
python3 scripts/compile_sweep.py
bash scripts/bench_orchestrator.sh
```

`bench_orchestrator.sh` reads:

```text
inference-benchmark/scripts/bench_jobs.txt
```

and uploads current raw results to:

```text
s3://agent-bench/results/current/${hardware}_${model}_tp${TP}_${backend}/
```

## Manual Launchers

Canonical vLLM:

```text
scripts/sweep_all_profiles.sh
scripts/sweep_multiturn_profiles.sh
```

Canonical SGLang single-turn:

```text
scripts/sweep_all_profiles_sglang.sh
```

SGLang multi-turn exists now:

```text
scripts/sweep_multiturn_profiles_sglang.sh
```

But it is not yet emitted by `sweep.yaml`; use only if explicitly requested.

Debug helpers now support metadata:

```text
scripts/smoke_test.sh
scripts/bench.sh
scripts/sweep.sh
scripts/run_one_bench.sh
```

## Context Overflow Warning

If a multi-turn smoke test fails with HTTP 400 context overflow:

1. Confirm branch includes the context safety fix.
2. Confirm runner receives:

```text
--max-context-tokens <MAX_LEN>
--context-safety-margin-tokens 256
```

3. Confirm server `--max-model-len` is at least `<MAX_LEN>`.

The generator now reserves output tokens plus safety margin, so overflows should be rare unless the script is stale or `MAX_LEN` differs between server and runner.

## Expected Runtime Shape

Mean turns in distributional traces:

| Workload | Mean Turns | Median | p90 | Max |
|---|---:|---:|---:|---:|
| `chat-multiturn` | 10.9 | 10 | 18 | 18 |
| `osworld-multiturn` | 13.1 | 8 | 30 | 30 |
| `swebench-multiturn` | 94.0 | 85 | 152 | 320 |
| `terminalbench-multiturn` | 75.1 | 61 | 130 | 876 |

SWE-bench and TerminalBench are deeper per sampled session, but canonical profiles use fewer sampled sessions than legacy replay. Synthetic traces reduce total sweep work mainly by replacing short/medium/long buckets with one sampled profile per workload.

## Priority Order

1. H100 vLLM smoke across the 6 canonical profiles.
2. H100 vLLM current canonical grid.
3. A100 / RTX3090 / RTX2080Ti current canonical grid where feasible.
4. Only then consider SGLang multi-turn cells.
5. Do not spend time filling archive/legacy cells unless explicitly requested.

## Do Not Do

- Do not run the full legacy short/medium/long archive grid.
- Do not upload new current runs to the old flat `results/` R2 prefix.
- Do not make H100x8 required.
- Do not treat exact engine prefix-cache hit telemetry as available; current fields are logical estimates.
