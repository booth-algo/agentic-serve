# Paper Reframing — Aaron Meeting 2026-05-02

## Contribution Restructure

**Drop**: Third contribution (roofline-based simulation / OICF). Not strong enough for NeurIPS.

**Split first contribution into two**:
1. Agentic workload benchmarking methodology (characteristic traces, multi-turn, saturation)
2. Measurement methodology (contrast different measurement methods)

**Claims to make** (in feature comparison table):
- First to consider **agentic workloads** (SWE-bench, TerminalBench, OSWorld)
- First to consider **multi-turn** serving benchmarks
- First to release **per-kernel NCU profiling** data
- ~~Delete: prediction-related claims~~ — predictor moved to EMNLP

## Characteristic Traces Validation

Aaron's ask: prove that running 20 synthetic/characteristic traces is representative of running the full dataset.

**To do**:
- Run one full benchmark pass on a dataset (e.g., SWE-bench multi-turn, 4K traces)
- Run 20 characteristic traces on same hardware
- Compare: MSE of aggregate metrics (TTFT, TPOT, E2EL distributions)
- Plot: characteristic-trace vs full-run distributions per metric
- Claim: "20 representative traces achieve X% MSE vs full dataset, requiring Y× fewer runs"

## Predictor Status

**Not for NeurIPS.** Move to EMNLP (deadline ~May 26) or next conference.

Requirements for EMNLP submission:
- Multi-node support (Aaron investigating AWS/Azure multi-node availability)
- Keep the physics-based predictor (GEMM + flash attention NCU) — it's the strength
- Add characteristic-trace validation: "predict serving latency for new hardware without running GPUs"

## Narrative Direction

**TTFT is less important for agentic workloads.** Aaron's logic:
- Agentic tasks are scheduled/recurring — no human waiting for first token
- Heartbeat/notification model: agent completes task, then notifies
- Focus paper on **TPOT and throughput** for agentic workloads
- TTFT matters for interactive/coding workloads, but those are different

**KV cache pressure on consumer GPUs**: real problem we observed (RTX3090 TTFT degradation). But multi-GPU pipelining (Aaron's architecture) multiplies effective bandwidth.

## Action Items

1. [ ] Rewrite intro: 2 contributions (agentic benchmark + measurement methodology), no predictor
2. [ ] Update feature comparison table: add first-to-claim rows, remove prediction rows
3. [ ] Build characteristic-trace vs full-run MSE comparison plots
4. [ ] Delete third contribution (roofline simulation) from paper body
5. [ ] George to apply for GPU grant under Rob's name
6. [ ] Aaron/Kevin to do cost-effectiveness calculation for 5090 serving (spreadsheet)
