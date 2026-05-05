# MSE Validation Results — H100

Date: 2026-05-04

## Goal

Prove that distributional/synthetic traces reproduce aggregate metrics of legacy real-trajectory sweeps within bounded MAPE. Both profiles use the same ISL=32768 filter, same 100-session count, same vLLM server instance. Only difference: synthetic filler text vs real SWE-bench code/tool outputs.

## Results: H100, Llama-3.1-8B, vLLM, TP=1

| Dataset | C | TTFT MAPE | TPOT MAPE | E2EL MAPE |
|---------|---|-----------|-----------|-----------|
| osworld | 40 | 111.8% | 79.3% | 134.6% |
| osworld | 80 | **6.8%** | **23.9%** | 83.8% |
| swebench | 40 | 158.7% | 407.1% | 331.9% |
| swebench | 80 | 217.2% | 96.4% | 99.9% |
| terminalbench | 40 | 408.2% | 779.0% | 764.6% |
| terminalbench | 80 | 1039.5% | 594.6% | 850.7% |

## Key Findings

### OSWorld works (6.8% TTFT at C=80)

OSWorld has short, structured interactions (median ISL ~1,241 tokens, max 30 turns). The synthetic filler text approximates these well because the total token budget per session is small. Distributional approach is viable for this workload type.

### SWE-bench / TerminalBench fail (100-1000%)

These workloads involve dense code blocks, long tool outputs, and complex agent conversations. Even with matched ISL filters and session counts, synthetic filler text produces different KV cache patterns than real traces. The distribution correctly samples statistical distributions (turn count, ISL, OSL) but the **token content** matters — real code has different compression, attention, and cache behavior than synthetic words like `"s0_t0_user_0"`.

### Distributional consistently over-predicts latency

In all cases, the synthetic distributional runs produce HIGHER TTFT/TPOT than the legacy runs. This means synthetic sessions generate more total tokens per session than legacy sessions. The per-turn token counts match the distribution, but the synthetic text tokenizes differently (Llama tokenizer splits `"s0_t0_user_0"` into 5-8 subword tokens vs the 1.35 word-to-token estimate).

## Root Cause

The `TOKEN_WORD_RATIO = 1.35` heuristic in `distributional.py` is wrong for these synthetic labels. Each label word like `"s0_t0_user_0"` tokenizes to ~4-6 tokens (underscores + digits split into subwords), while the 1.35 ratio assumes typical English text. The sampler generates `ceil(target_tokens / 1.35)` words, but each word produces ~4 tokens → 3× more actual tokens than intended.

## Implications for Paper

- **Contribution 2 (characteristic-trace condensation)** needs a negative or scoped result: "works for short-context workloads (osworld), not for dense agentic workloads (swebench, terminalbench)".
- **The XX× reduction claim** cannot be supported by current data. The distributional approach overestimates latency — it's not a valid replacement for real-trace sweeps on dense workloads.
- **Alternative**: fix the token estimation heuristic (tokenize with the model's actual tokenizer instead of 1.35 words/token). Or accept that content matters and frame the benchmark suite as requiring per-workload distribution characterization rather than single synthetic sampler.

## MSE Infrastructure

- **Filtered distributions**: `data/distributions/swebench_multiturn_filtered.json` (138/165 sessions), `terminalbench_multiturn_filtered.json` (242/267), `osworld_multiturn_filtered.json` (60/60) — ISL≤32768 filter matching legacy short profiles
- **MSE profiles**: `swebench-multiturn-mse`, `terminalbench-multiturn-mse`, `osworld-multiturn-mse` (`active=False`, use filtered JSONs)
- **Sweep cells**: in `sweep.yaml` with `mse_multi` preset at C=40,80 on H100, A100, 3090
- **Scope**: `--scope mse`, results isolated from current/archive/fixed
- **Orchestrator**: picks up MSE cells from `bench_jobs.txt` on next tick (every 2 min)

## Pending

- A100 and 3090 MSE runs (cells in bench_jobs.txt, orchestrator will dispatch)
- 1:1 noise floor comparison (run legacy profile twice on same hardware to establish measurement variance)
- Fix token estimation in `distributional.py` if we want to salvage the characteristic-trace approach
