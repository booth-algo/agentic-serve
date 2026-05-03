---
license: apache-2.0
task_categories:
  - tabular-regression
language:
  - en
tags:
  - llm-inference
  - benchmarking
  - gpu-profiling
  - vllm
  - sglang
  - agentic-workloads
size_categories:
  - 1K<n<10K
pretty_name: AgentPerfBench
configs:
  - config_name: trace_replay
    data_files:
      - split: summary
        path: trace_replay/summary.parquet
  - config_name: distributional
    data_files:
      - split: summary
        path: distributional/summary.parquet
  - config_name: roofline
    data_files:
      - split: kernel_profiles
        path: roofline/kernel_profiles.parquet
---

# AgentPerfBench

LLM inference benchmark dataset measuring serving performance (TTFT, TPOT, ITL, throughput) across 9 models, 14 GPU configurations, 2 serving engines, and 20+ workload profiles spanning single-turn chat, multi-turn agent sessions, and synthetic stress tests. Includes per-kernel CUDA profiling data for roofline analysis.

## Dataset Configurations

This dataset provides two benchmark configurations reflecting distinct data collection methodologies:

### trace_replay

Requests replay exact ISL/OSL sequences from recorded agent sessions (SWE-Bench, TerminalBench, OSWorld, ShareGPT). Input distributions are empirically grounded in real tool-use patterns, capturing realistic burstiness and turn-depth correlations.

Profiles: `chat-medium`, `chat-multiturn-long`, `chat-multiturn-medium`, `chat-multiturn-short`, `chat-short`, `chat-singleturn`, `coding-singleturn`, `decode-heavy`, `osworld-multiturn-long`, `osworld-multiturn-medium`, `osworld-multiturn-short`, `prefill-heavy`, `random-1k`, `swebench-multiturn-medium`, `swebench-multiturn-short`, `terminalbench-multiturn-medium`, `terminalbench-multiturn-short`

### distributional

Requests sample ISL/OSL from parameterized statistical distributions (e.g., lognormal) fitted to real workload statistics. Shorter to run than full trace replays, enabling faster characterization of profile-level serving behavior across the hardware matrix. MSE validation confirms distributional runs reproduce the latency and throughput characteristics of their trace-replay counterparts.

Profiles: `chat-multiturn`, `chat-singleturn`, `coding-singleturn`, `osworld-multiturn`, `swebench-multiturn`, `terminalbench-multiturn`

### Why two configurations?

**trace_replay** provides ecological validity — patterns are drawn from real agent sessions, grounding results in observed behavior. **distributional** enables efficient coverage — shorter run times allow systematic sweeps across the full model-hardware-concurrency matrix, while MSE validation against trace_replay confirms the results remain representative.

### Concurrency filtering

Rows where declared concurrency exceeds the session pool size have been excluded. This affects trace_replay data at concurrency > 100 (session pool was 100) and distributional/current data at concurrency > 10 (session pool was 10). Distributional data collected after the fix has no such limitation.

## Coverage

### Hardware

| GPU | VRAM | HBM Bandwidth | Peak BF16 TFLOPS |
|-----|------|---------------|------------------|
| NVIDIA H100 SXM | 80 GB | 3.35 TB/s | 989 |
| NVIDIA A100 SXM4 | 40 GB | 1.56 TB/s | 312 |
| NVIDIA RTX 3090 | 24 GB | 936 GB/s | 71 |
| NVIDIA RTX 2080 Ti | 11 GB | 616 GB/s | 27 |

Multi-GPU configurations: 1, 2, 4, 8 GPUs with tensor parallelism.

### Models

| Model | Family | Parameters | Architecture |
|-------|--------|-----------|--------------|
| Llama-3.1-8B-Instruct | Llama | 8B | Dense |
| Llama-3.1-70B-Instruct | Llama | 70B | Dense |
| Llama-3.3-70B-Instruct | Llama | 70B | Dense |
| Qwen2.5-72B-Instruct | Qwen | 72B | Dense |
| Qwen3.5-9B | Qwen | 9B | Dense |
| Qwen3.5-27B | Qwen | 27B | Dense |
| Mixtral-8x7B | Mixtral | 46.7B (12.9B active) | MoE |
| gpt-oss-20b | GPT-OSS | 21B (3.6B active) | MoE |
| gpt-oss-120b | GPT-OSS | 117B (5.1B active) | MoE |

### Engines

- vLLM 0.19.0
- SGLang 0.5.9

## Schema

Each row in `summary.parquet` (both configs) contains:

| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Deterministic hash of run parameters |
| model | string | Model short name |
| model_family | string | Model family (llama, qwen, gpt-oss, mixtral) |
| hardware | string | GPU configuration (e.g., H100x4) |
| engine | string | Serving engine (vllm, sglang) |
| tensor_parallelism | int | TP degree |
| profile | string | Workload profile name |
| concurrency | int | Concurrent request level |
| num_requests | int | Total requests in run |
| duration_s | float | Total run duration |
| successful_requests | int | Completed requests |
| failed_requests | int | Failed requests |
| request_throughput | float | Requests/second |
| input_token_throughput | float | Input tokens/second |
| output_token_throughput | float | Output tokens/second |
| total_token_throughput | float | Total tokens/second |
| mean/median/p90/p99_ttft_ms | float | Time to first token |
| mean/median/p90/p99_tpot_ms | float | Time per output token |
| mean/median/p90/p99_itl_ms | float | Inter-token latency |
| mean/median/p90/p99_e2el_ms | float | End-to-end latency |

## Benchmark Methodology

- **Concurrency model**: Closed-loop with semaphore control.
- **Concurrency sweep**: 1 to 320.
- **Requests per configuration**: 50-100, with 3-request warmup.
- **Metrics**: TTFT, TPOT, ITL, E2EL, request throughput, token throughput.
- **Percentiles**: mean, median, p90, p99.
- **Kernel profiling** (roofline config): PyTorch profiler on 2-layer forward passes, batch sizes [1, 4, 8, 32, 64].

## Future Releases

Per-request and multi-turn granularity data will be added when full result JSON files are available from the collection infrastructure.

## Intended Uses

- Comparing inference engine performance under controlled conditions.
- Capacity planning for agentic LLM deployments.
- Roofline analysis of GPU utilization under different workload regimes.
- Studying TTFT degradation under multi-turn context accumulation.

## Limitations

- Results are specific to tested hardware and software versions (vLLM 0.19.0, SGLang 0.5.9).
- Distributional profiles approximate but do not replicate exact production traffic patterns.
- No consumer GPUs beyond RTX 3090; no non-NVIDIA accelerators.
- Closed-loop concurrency only; open-loop (Poisson arrival) not included.
- No model quality metrics. This is a systems benchmark.

## Ethical Considerations

- No PII in the dataset.
- Synthetic profiles use random tokens. Trace-replay profiles derive from open benchmarks (SWE-Bench MIT, TerminalBench, OSWorld).
- Benchmark results should not be used as sole basis for hardware purchasing decisions.

## Source Datasets

- [SWE-Bench](https://github.com/princeton-nlp/SWE-bench) (MIT License)
- [TerminalBench](https://github.com/TerminalBench/TerminalBench)
- [ShareGPT (Aeala/ShareGPT_Vicuna_unfiltered)](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)
- [OSWorld](https://github.com/xlang-ai/OSWorld)

## Citation

```bibtex
@inproceedings{agentperfbench2026,
  title={AgentPerfBench: A Benchmarking and Evaluation Suite for Inference Performance of Agentic LLMs},
  author={Anonymous},
  booktitle={NeurIPS 2026 Evaluations and Datasets Track},
  year={2026}
}
```
