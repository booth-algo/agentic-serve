# MSE Validation Debug Process - 2026-05-05

## Summary

This session debugged why distributional MSE validation initially failed for dense agentic workloads, especially SWE-bench multi-turn. The final answer is that token-count matching alone is not enough. Faithful synthetic replay needs three properties:

1. model-tokenizer accounting,
2. code-agent morphology matching,
3. APC-aware shared-prefix structure.

After adding morphology-aware filler and a 1024-token shared synthetic prefix, the SWE-bench C=5 source-locked validation closed the deep-turn E2EL gap:

| Condition | Turn 10-19 E2EL delta |
|-----------|------------------------|
| APC on, no shared prefix | +31.1% |
| APC off | +11.7% |
| APC on, shared prefix | -3.2% |

This supports the paper framing that distributional replay must model the generative structure of agent sessions, not only marginal token lengths.

## Data Location

Canonical archived data is in R2:

```
s3://agent-bench/bench_mse/
  morphology-experiment-2026-05-05.md
  scripts/
  src/
  h100/results/
  h100-2/results/
```

R2 endpoint:

```
https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com
```

Bucket and prefix:

```
Bucket: agent-bench
Prefix: bench_mse/
```

Local pulled result roots used during this session:

```
inference-benchmark/results/mse_validation_paper/
inference-benchmark/results/mse_validation_option2/
inference-benchmark/results/mse_validation_source_locked/
inference-benchmark/results/mse_validation_source_locked_pair/
inference-benchmark/results/mse_validation_morphology/
inference-benchmark/results/mse_validation_highc/
inference-benchmark/results/mse_validation_apc_off/
inference-benchmark/results/mse_validation_prefix_aware/
```

## Initial Failure

The first MSE validation runs looked unusable for SWE-bench and TerminalBench. Errors reached 100-1000% for dense code/terminal workloads even when OSWorld looked reasonable.

The first root cause was token estimation. The old synthetic filler used labels like `s0_t0_user_0`, then estimated tokens using a natural-language `TOKEN_WORD_RATIO`. Those labels tokenize into many subword pieces because of underscores and digits. The generator produced far more real model tokens than intended.

Fix:

- rebuild distributions using captured vLLM input tokens or the actual tokenizer/chat template,
- remove the old label-token heuristic,
- use calibrated synthetic text measured by the model tokenizer.

This made short H100 paper runs plausible:

| Workload | C | Metric outcome |
|----------|---|----------------|
| SWE-bench | 5 | TPOT +3%, E2EL +3%, TTFT +28% |
| TerminalBench | 5 | TPOT +4%, E2EL +5%, TTFT +27% |
| SWE-bench | 20 | TPOT -1%, E2EL +4%, TTFT +15% |

TTFT remained noisy; TPOT and E2EL were the main validation metrics.

## Sampling Fixes

The next issue was population variance. Distributional sampling could duplicate source sessions and skip others, which inflated per-turn error when comparing to real trace replay.

Implemented and tested:

- no-replacement source-session sampling,
- source-locked replay through `SOURCE_SESSION_IDS_FILE`,
- source-ID metadata in request outputs.

This established two modes:

| Mode | Purpose |
|------|---------|
| No-replacement | honest generator behavior for paper validation |
| Source-locked | diagnostic ablation to isolate synthetic-content error |

Source-locking confirmed the SWE diagnosis was not caused by session mismatch. The overlap was 40/40 source sessions, Jaccard 1.000.

## SWE-Bench Residual Problem

Even after tokenizer accounting and source-locking, SWE-bench had a deep-turn residual gap. Shallow turns were nearly perfect, but turns 10-19 were off.

Source-locked English filler, C=5:

| Turn bin | E2EL delta |
|----------|------------|
| 00-04 | +3.2% |
| 05-09 | +2.2% |
| 10-19 | +45.5% |
| 20-29 | +15.8% |

This localized the issue to deeper agent context, not the whole workload.

## Morphology Experiment

The next hypothesis was text morphology. Synthetic English filler had about 2.2x more characters per token than real SWE-bench code/tool-output text.

Measured chars/token:

| Condition | Chars/token |
|-----------|-------------|
| English MSE | 8.39 |
| Real SWE trace | 3.80 |
| Code-morph MSE | 3.68 |

The code-morph generator hit the chars/token target, but did not fully close the latency gap.

| Turn bin | English E2EL delta | Code-morph E2EL delta |
|----------|---------------------|------------------------|
| 00-04 | +3.2% | +1.9% |
| 05-09 | +2.2% | +1.0% |
| 10-19 | +45.5% | +31.1% |
| 20-29 | +15.8% | +12.4% |

Conclusion: morphology was a real factor, but only a partial factor. Token count and chars/token parity were insufficient for deep SWE-bench replay.

Paper-safe wording from this stage:

> Token-count matching alone is sufficient for TerminalBench and shallow SWE turns, but not for deep SWE-bench traces. Deep code-agent workloads require morphology-aware synthetic replay; otherwise request preprocessing and prefix handling differ even when token counts match.

## High-Concurrency Check

C=100 runs were launched to see whether higher queue pressure made the mismatch worse.

Result summary:

| Condition | TTFT p50 MSE/REAL | TPOT p50 MSE/REAL | E2EL p50 MSE/REAL | Turn 10-19 E2EL delta |
|-----------|-------------------|-------------------|-------------------|------------------------|
| TerminalBench English | 2966 / 1825 ms | 136 / 60 ms | 7951 / 3964 ms | +29.4% |
| SWE code-morph | 8349 / 6140 ms | 204 / 181 ms | 14515 / 12248 ms | +16.3% |
| SWE English | 7591 / 6276 ms | 206 / 177 ms | 13597 / 12347 ms | +9.5% |

Interpretation:

- C=100 introduces broad saturation artifacts.
- Higher concurrency does not cleanly isolate the SWE-specific bug.
- C=5 source-locked runs remain the cleanest diagnostic setting.

## APC-Off Ablation

The key discriminating test was to turn off vLLM automatic prefix caching.

Sanity check:

| Condition | turn1/turn0 TTFT ratio |
|-----------|------------------------|
| APC on | 0.67x |
| APC off | 1.40x |

The ratio confirmed APC was behaviorally disabled.

APC-off result:

| Turn bin | APC on, no shared prefix | APC off |
|----------|---------------------------|---------|
| 00-04 | +1.9% | +28.6% |
| 05-09 | +1.0% | +16.9% |
| 10-19 | +31.1% | +11.7% |
| 20-29 | +6.9% | +0.1% |

Aggregate:

| Metric | APC on, no shared prefix | APC off |
|--------|---------------------------|---------|
| TTFT | +23.8% | +4.2% |
| TPOT | +14.7% | +10.6% |
| E2EL | +21.6% | +9.3% |

Conclusion: APC was amplifying the residual synthetic-vs-real mismatch. The issue was no longer simple token-count error.

## vLLM APC Check

The vLLM APC documentation confirms that automatic prefix caching reuses KV cache for new requests that share an exact token prefix with previous requests. APC is token-prefix based, not session-ID based.

Implications:

- Within-session growing transcripts are prefix-eligible if the rendered token prefix is exact.
- Cross-session shared harness/system prefixes are also prefix-eligible.
- Full cache behavior still depends on token-block hashing, exact chat-template rendering, eviction, and scheduler pressure.

Important caveat: runner metadata such as `cached_context_tokens` is a logical estimate from previous prompt length. It is not server-side APC telemetry. The ablations prove serving behavior changes under APC; they do not directly log vLLM internal cache-hit counters.

## Prefix-Aware Generator Fix

The final fix was to add a shared synthetic prefix across distributional sessions.

Implementation:

- `DISTRIBUTIONAL_PREFIX_AWARE=1`
- `DISTRIBUTIONAL_SHARED_PREFIX_TOKENS=1024`
- `DISTRIBUTIONAL_PREFIX_BLOCK_SIZE=16`
- `PREFIX_AWARE_SYNTHETIC=on` in `scripts/run_mse_validation.sh`

The shared prefix is inserted as a system message in MSE/distributional requests only. It is subtracted from the first-turn synthetic user payload so sampled total-context token targets remain matched.

Metadata in the pulled prefix-aware JSON confirms:

| Field | Value |
|-------|-------|
| `prefix_aware_synthetic` | `true` |
| `shared_prefix_requested_tokens` | `1024` |
| `shared_prefix_target_tokens` | `1024` |
| `shared_prefix_actual_tokens` | `1024` |
| `shared_prefix_block_size` | `16` |
| `shared_prefix_block_aligned` | `true` |
| `synthetic_filler_style` | `code` |
| `synthetic_target_chars_per_token` | `3.8` |

## Final Prefix-Aware Result

Request-level median recomputation from the pulled local JSONs:

| Turn bin | No shared prefix | APC off | Shared prefix |
|----------|------------------|---------|---------------|
| 00-04 | +1.9% | +28.6% | -0.4% |
| 05-09 | +1.0% | +16.9% | -2.0% |
| 10-19 | +31.1% | +11.7% | -3.2% |
| 20-29 | +6.9% | +0.1% | +1.7% |

The launcher handoff reported `+5.3%` for the shared-prefix 20-29 bin under its aggregation path; both computations keep the late bin inside the noise-scale band. The key diagnostic bin, turns 10-19, is stable at `-3.2%`.

Aggregate median deltas:

| Metric | No shared prefix | APC off | Shared prefix |
|--------|------------------|---------|---------------|
| TTFT | +23.8% | +4.2% | -6.1% |
| TPOT | +14.7% | +10.6% | +2.9% |
| E2EL | +21.6% | +9.3% | -2.4% |

## Final Interpretation

The best-supported conclusion is:

> Distributional replay for agentic serving must match both workload token statistics and the serving-visible prefix structure. Token-count-only replay fails for deep SWE-bench turns because it misses code morphology and shared-prefix APC structure. With morphology-aware filler and a shared APC-aligned harness prefix, distributional replay matches real trace replay within measurement noise across turn-depth bins.

Avoid overclaiming:

- Do not say we directly measured vLLM internal APC hit rates unless server telemetry is added.
- Do not say token count alone is sufficient.
- Do not frame the shared prefix as a paper trick; frame it as modeling the real agent harness/system prompt prefix that real sessions share.

Strong paper framing:

> Synthetic replay must preserve the generative process of agent sessions. For code-agent workloads, this means matching token lengths, code-like text morphology, and shared harness prefixes that serving systems cache. Our source-locked H100 ablation shows that adding APC-aware shared-prefix structure closes the deep-turn SWE-bench E2EL gap from `+31.1%` to `-3.2%`.

## Paper Table Candidate

Use this compact table for the methods or validation section:

| Replay variant | Morphology-aware | Shared prefix | APC | Aggregate E2EL delta | Turn 10-19 E2EL delta |
|----------------|------------------|---------------|-----|-----------------------|------------------------|
| Code-morph baseline | yes | no | on | +21.6% | +31.1% |
| Code-morph, APC off | yes | no | off | +9.3% | +11.7% |
| Prefix-aware replay | yes | yes, 1024 tokens | on | -2.4% | -3.2% |

Caption:

> Source-locked SWE-bench C=5 validation on H100/Llama-3.1-8B/vLLM. Distributional and real-trace replay use the same 40 source sessions. Errors are median synthetic-vs-real deltas.

## Code Touchpoints

Local implementation files:

- `inference-benchmark/src/workloads/distributional.py`
- `inference-benchmark/scripts/run_mse_validation.sh`
- `inference-benchmark/tests/test_distributional_workloads.py`

Relevant local docs:

- `docs/morphology-experiment-2026-05-05.md`
- `docs/mse-validation-sampling-plan.md`
- `docs/mse-validation-results.md`

Validation run locally after the prefix-aware implementation:

```
python3 -m unittest tests.test_distributional_workloads
bash -n scripts/run_mse_validation.sh
python3 -m py_compile inference-benchmark/src/workloads/distributional.py
```

## Next Work

1. Add a small plotting script/table extractor for the final paper figure.
2. If time permits, add vLLM server-side APC telemetry to confirm actual cache-hit counters.
3. Update the NeurIPS methodology text to describe APC-aware distributional replay.
4. Keep the negative result: token-count-only replay is insufficient for deep code-agent workloads.
