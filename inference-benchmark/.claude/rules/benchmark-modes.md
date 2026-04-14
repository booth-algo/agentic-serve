# Benchmark Mode Rules

Always follow these rules when running or implementing benchmarks.

## Mode selection

| Goal | Mode | Server | Client |
|------|------|--------|--------|
| Cross-validate with InferenceX | stress-test | No prefix cache | `--ignore-eos` |
| Realistic workload performance | single-turn | `--enable-prefix-caching` | none |
| KV cache reuse measurement | multi-turn | `--enable-prefix-caching` | none (not yet implemented) |

## Enforcement rules

1. **Single-turn mode MUST have prefix caching ON.** Always launch with `./scripts/launch_server.sh single-turn`. Never run single-turn profiles against a server started without `--enable-prefix-caching`.

2. **Stress-test mode MUST use `--ignore-eos` for FP8 models.** `--mode stress-test` in runner.py auto-enables this. Never report stress-test results from FP8 models without confirming OSL hit rate is >95%.

3. **Never compare TTFT between inference-benchmark and InferenceX.** Different arrival patterns (steady concurrency vs `--request-rate inf`) make TTFT incomparable. Always compare TPOT.

4. **Always annotate results with server flags.** When recording in `markdown/results.md`, include: prefix caching ON/OFF, --ignore-eos YES/NO, model precision (bfloat16/FP8), TP size.

5. **Multi-turn is not implemented.** Do not simulate multi-turn by concatenating messages manually. Implement `src/workloads/multi_turn.py` (ConversationSession) first.
