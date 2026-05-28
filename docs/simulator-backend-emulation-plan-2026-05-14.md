# Simulator Backend Emulation Plan - 2026-05-14

## Goal

Build the new standalone `simulator/` predictor into an accurate H100 +
Llama-3.1-8B serving simulator before expanding to other GPUs or models.

The guiding hypothesis is the professor's suggestion already validated in the
May 13 notes: isolated kernel measurements can predict serving latency when
they are composed with the correct serving schedule. The simulator should
therefore avoid direct aggregate MAPE fitting as the main mechanism. It should
compose GEMM, flash attention, KV cache, and runtime scheduling behavior into
TTFT, TPOT, and E2EL.

## Scope And Constraints

- Target only `H100` and `Llama-3.1-8B` until this path is reliable.
- Keep implementation in root-level `simulator/`, outside `llm_predict/`.
- Use existing benchmark/dashboard data for training and validation.
- Do not run H100 profiling unless explicitly needed.
- If any H100 profiling is added later, it must explicitly pin to GPU 6.
- Model vLLM and SGLang behavior only at the serving-scheduler level; do not
  attempt to clone either codebase.

## Current State

The first simulator slice exists:

- `simulator/kernel_models.py` trains GEMM and flash-attention XGBoost residual
  models over roofline baselines.
- `simulator/provider.py` composes kernel predictions into prefill and decode
  scheduler-step costs.
- `simulator/turns.py` adapts observed dashboard per-turn summaries into
  simulator turn inputs.
- `simulator/predictor.py` predicts per-turn and row-level TTFT, TPOT, and E2EL.
- `simulator/predict.py` exports
  `inference-benchmark/dashboard/public/simulator-predictions.json`.
- The dashboard has a Simulator page locked to H100 + Llama-3.1-8B.

This v1 is useful for inspecting failures, but it still needs backend runtime
emulation. The largest remaining gap is not just kernel latency; it is how vLLM
and SGLang schedule prefill/decode work and realize cache hits under pressure.

## Architecture Direction

The simulator should be layered as:

1. **Kernel layer**
   - Exact measured table when a shape exists.
   - XGBoost residual over roofline for nearby/unseen shapes.
   - Roofline fallback outside model coverage.
   - Output: latency for GEMM and flash-attention shapes.

2. **Model composer**
   - Convert Llama-3.1-8B prefill/decode work into per-layer kernel shapes.
   - Compose attention, MLP, elementwise, and step overhead into one scheduler
     step cost.
   - Keep this backend-neutral.

3. **Backend emulator**
   - A lightweight event-loop model for vLLM and SGLang.
   - Request states: waiting prefill, prefill-in-progress, decode-ready,
     first-token-emitted, decoding, done.
   - Scheduler states: active requests, KV cache residency, prefix-cache state,
     current prefill chunks, current decode batch.
   - Output: per-request TTFT, TPOT, E2EL, and per-step trace.

4. **Calibration layer**
   - Fit only small interpretable runtime parameters.
   - Avoid fitting final metrics directly unless used as a residual diagnostic.

5. **Dashboard/debug layer**
   - Export row-level metrics plus per-turn/per-step traces.
   - Show whether error came from kernel cost, queueing, cache replay, or decode
     batch efficiency.

## Backend Emulation Requirements

Yes, this plan involves emulating vLLM and SGLang "a bit." The required
emulation is narrow:

- continuous batching behavior,
- prefill/decode scheduling order,
- chunked prefill limits,
- maximum batched token limits,
- decode batch size over time,
- first-token timing,
- per-step scheduler overhead,
- KV block residency and eviction,
- prefix-cache/radix-cache realization,
- backend-specific decode efficiency.

This is not a full backend rewrite. It is a small scheduling model with enough
backend-specific knobs to explain TTFT and TPOT.

## BackendSpec

Add a `BackendSpec` object to describe vLLM/SGLang runtime behavior:

| Field | Meaning |
|---|---|
| `name` | `vllm` or `sglang` |
| `max_num_batched_tokens` | Max tokens processed in a scheduler step |
| `max_num_seqs` | Max active requests/sequences |
| `prefill_chunk_tokens` | Chunk size for large prefills |
| `prefill_policy` | Whether prefill is chunked, decode-first, or prefill-first |
| `decode_policy` | How decode-ready requests are batched |
| `scheduler_base_us` | Per-step fixed runtime overhead |
| `scheduler_per_request_us` | Per-step overhead per active request |
| `decode_efficiency_curve` | Effective multiplier vs decode batch/context |
| `prefill_efficiency_curve` | Effective multiplier vs prefill batch/context |
| `kv_block_tokens` | Backend KV block granularity |
| `kv_budget_tokens` | Effective KV cache capacity |
| `cache_mode` | none, vLLM APC/block cache, or SGLang radix cache |
| `cache_realization_rate` | Fraction of logical cached context realized in server |
| `eviction_policy` | Approximation for cache pressure and replay |

Initial values can come from known benchmark config and measured rows. After
that, fit only the continuous parameters.

## Event-Loop Algorithm

For each benchmark row and each observed turn:

1. Build one request per successful session at that turn.
2. For each request, initialize:
   - total context tokens,
   - logical new prefill tokens,
   - logical cached context tokens,
   - output token count.
3. Convert logical cache fields into realized prefill work:
   - apply backend cache mode,
   - apply cache realization rate,
   - apply KV budget and eviction/replay.
4. Run the scheduler loop:
   - admit waiting requests up to backend limits,
   - choose decode-ready requests for the decode batch,
   - choose prefill chunks when prefill work remains,
   - compose each step using the kernel composer,
   - add backend scheduler overhead,
   - update request state and timestamps.
5. Record:
   - TTFT as time until first decode token completes,
   - TPOT as average post-first-token decode interval,
   - E2EL as request completion time.
6. Aggregate request metrics into turn metrics and row metrics using the same
   aggregation semantics as the dashboard.

## vLLM-Specific Slice

Start with vLLM because the docs and prior debugging already identified key
behaviors:

- chunked prefill around `max_num_batched_tokens`,
- automatic prefix caching with block granularity,
- decode-first or mixed scheduling depending on active decode pressure,
- CUDA graph benefits for repeated decode shapes,
- KV budget pressure that can force replay of cached context.

Implementation target:

- one `VllmBackendSpec`,
- one deterministic event-loop scheduler,
- debug traces for a few known rows:
  - `chat-multiturn-synth`,
  - `swebench-multiturn-synth`,
  - `terminalbench-multiturn-synth`,
  - `osworld-multiturn-synth`.

## SGLang-Specific Slice

After vLLM is stable, add SGLang as a second `BackendSpec`:

- separate cache mode for radix/prefix cache behavior,
- separate scheduler overhead parameters,
- separate decode efficiency curve,
- separate prefill/decode scheduling policy if needed.

The first SGLang version should reuse the same event loop and override only
backend knobs. Only split code paths if measured traces prove the runtime
policy is materially different.

## Calibration Plan

Calibrate in this order:

1. **Kernel residuals**
   - Train GEMM and flash residual models from existing CSVs.
   - Validate train error and holdout error by shape family.

2. **Single-request sanity**
   - Use C=1 rows to calibrate base prefill/decode step behavior.
   - This isolates kernel composition from queueing.

3. **Decode batch efficiency**
   - Use single-turn rows across concurrency.
   - Fit decode efficiency vs batch/context without involving turn cache.

4. **Prefill queueing/chunking**
   - Use chat single-turn and early multi-turn rows.
   - Fit max batched token behavior and prefill efficiency.

5. **KV cache realization and eviction**
   - Use deep multi-turn rows.
   - Fit cache realization rate and effective KV budget.

6. **Backend-specific parameters**
   - Fit vLLM and SGLang separately.
   - Share kernel/model composer across both.

Calibration should produce a JSON artifact such as:

```text
simulator/artifacts/H100/backend_vllm_Llama-3.1-8B.json
simulator/artifacts/H100/backend_sglang_Llama-3.1-8B.json
```

## Validation Plan

Validation should report:

- row-level median absolute TTFT/TPOT/E2EL error,
- profile/backend/concurrency error tables,
- turn-bin errors, for example turns 0-4, 5-9, 10-19, 20+,
- per-step trace examples for high-error rows,
- whether each prediction used exact table, XGBoost, or roofline fallback.

Hold out at least one concurrency per profile/backend during calibration. A
model that only fits the seen grid is not good enough.

## Success Criteria

Initial success target for H100 + Llama-3.1-8B synthetic distributional rows:

- median E2EL error below 20% for each profile/backend,
- median TPOT error below 25% for each profile/backend,
- median TTFT error below 35% for each profile/backend,
- no profile/backend with obvious monotonic failure across concurrency,
- per-turn traces explain remaining misses in terms of queueing, cache, or
  kernel fallback.

These thresholds are intentionally stricter than the current v1 but still
realistic for a first backend-emulation pass.

## Implementation Steps

1. Add `BackendSpec` and backend config defaults.
2. Replace the current simplified turn prediction path with an event-loop
   scheduler behind a feature flag.
3. Implement vLLM scheduler semantics first.
4. Add trace output and tests for TTFT/TPOT/E2EL semantics.
5. Add backend parameter calibration CLI.
6. Export backend-specific diagnostics into `simulator-predictions.json`.
7. Wire dashboard controls or columns to show backend emulator trace summaries.
8. Add SGLang backend parameters and validate against SGLang rows.
9. Re-run simulator export and compare profile/backend medians.
10. Only after H100 + Llama-3.1-8B is stable, expand to other models/GPUs.

## Non-Goals

- Do not move this back into `llm_predict/`.
- Do not train a black-box model that directly predicts final TTFT/TPOT/E2EL
  from row features.
- Do not broaden to A100, 3090, 2080Ti, 70B, or MoE before H100 + 8B is
  explainable.
- Do not run broad profiling sweeps until the emulator identifies specific
  missing kernel shapes or runtime parameters.

