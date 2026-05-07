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
  - 100K<n<1M
pretty_name: AgentPerfBench
version: "1.0"
configs:
  - config_name: trace_replay
    data_files:
      - split: summary
        path: trace_replay/summary.parquet
  - config_name: synthetic_distributional
    data_files:
      - split: summary
        path: synthetic_distributional/summary.parquet
  - config_name: per_layer_kernel
    data_files:
      - split: summary
        path: per_layer_kernel/summary.parquet
  - config_name: kernels_labeled
    data_files:
      - split: train
        path: kernel_profiles/kernels_labeled.parquet
  - config_name: mse_validation
    data_files:
      - split: summary
        path: mse_validation/summary.parquet
dataset_info:
  - config_name: trace_replay
    features:
      - name: run_id
        dtype: string
      - name: model
        dtype: string
      - name: model_family
        dtype: string
      - name: hardware
        dtype: string
      - name: engine
        dtype: string
      - name: tensor_parallelism
        dtype: int64
      - name: profile
        dtype: string
      - name: concurrency
        dtype: int64
      - name: num_requests
        dtype: int64
      - name: duration_s
        dtype: float64
      - name: request_throughput
        dtype: float64
      - name: input_token_throughput
        dtype: float64
      - name: output_token_throughput
        dtype: float64
      - name: total_token_throughput
        dtype: float64
      - name: mean_ttft_ms
        dtype: float64
      - name: median_ttft_ms
        dtype: float64
      - name: p90_ttft_ms
        dtype: float64
      - name: p99_ttft_ms
        dtype: float64
      - name: mean_tpot_ms
        dtype: float64
      - name: median_tpot_ms
        dtype: float64
      - name: p90_tpot_ms
        dtype: float64
      - name: p99_tpot_ms
        dtype: float64
      - name: mean_itl_ms
        dtype: float64
      - name: median_itl_ms
        dtype: float64
      - name: p90_itl_ms
        dtype: float64
      - name: p99_itl_ms
        dtype: float64
      - name: mean_e2el_ms
        dtype: float64
      - name: median_e2el_ms
        dtype: float64
      - name: p90_e2el_ms
        dtype: float64
      - name: p99_e2el_ms
        dtype: float64
    splits:
      - name: summary
        num_examples: 2932
        num_bytes: 640182
  - config_name: synthetic_distributional
    features:
      - name: run_id
        dtype: string
      - name: model
        dtype: string
      - name: model_family
        dtype: string
      - name: hardware
        dtype: string
      - name: engine
        dtype: string
      - name: tensor_parallelism
        dtype: int64
      - name: profile
        dtype: string
      - name: concurrency
        dtype: int64
      - name: num_requests
        dtype: int64
      - name: duration_s
        dtype: float64
      - name: request_throughput
        dtype: float64
      - name: input_token_throughput
        dtype: float64
      - name: output_token_throughput
        dtype: float64
      - name: total_token_throughput
        dtype: float64
      - name: mean_ttft_ms
        dtype: float64
      - name: median_ttft_ms
        dtype: float64
      - name: p90_ttft_ms
        dtype: float64
      - name: p99_ttft_ms
        dtype: float64
      - name: mean_tpot_ms
        dtype: float64
      - name: median_tpot_ms
        dtype: float64
      - name: p90_tpot_ms
        dtype: float64
      - name: p99_tpot_ms
        dtype: float64
      - name: mean_itl_ms
        dtype: float64
      - name: median_itl_ms
        dtype: float64
      - name: p90_itl_ms
        dtype: float64
      - name: p99_itl_ms
        dtype: float64
      - name: mean_e2el_ms
        dtype: float64
      - name: median_e2el_ms
        dtype: float64
      - name: p90_e2el_ms
        dtype: float64
      - name: p99_e2el_ms
        dtype: float64
    splits:
      - name: summary
        num_examples: 1365
  - config_name: per_layer_kernel
    features:
      - name: record_type
        dtype: string
      - name: model
        dtype: string
      - name: hardware
        dtype: string
      - name: phase
        dtype: string
      - name: batch_size
        dtype: int64
      - name: sequence_length
        dtype: int64
      - name: component_name
        dtype: string
      - name: bound
        dtype: string
      - name: flops
        dtype: float64
      - name: bytes_accessed
        dtype: float64
      - name: operational_intensity
        dtype: float64
      - name: ridge_point
        dtype: float64
      - name: kernel_id
        dtype: int64
      - name: kernel_name
        dtype: string
      - name: block_size
        dtype: string
      - name: grid_size
        dtype: string
      - name: duration_us
        dtype: float64
      - name: compute_sm_throughput_pct
        dtype: float64
      - name: dram_throughput_pct
        dtype: float64
      - name: memory_throughput_pct
        dtype: float64
      - name: l1_tex_cache_throughput_pct
        dtype: float64
      - name: l2_cache_throughput_pct
        dtype: float64
      - name: sm_frequency_ghz
        dtype: float64
      - name: dram_frequency_ghz
        dtype: float64
    splits:
      - name: summary
        num_examples: 37
        num_bytes: 12000
  - config_name: kernels_labeled
    features:
      - name: source
        dtype: string
      - name: gpu
        dtype: string
      - name: model
        dtype: string
      - name: kernel_family
        dtype: string
      - name: kernel_name
        dtype: string
      - name: dtype
        dtype: string
      - name: held_out
        dtype: bool
      - name: M
        dtype: float64
      - name: N
        dtype: float64
      - name: K
        dtype: float64
      - name: bs
        dtype: float64
      - name: seq
        dtype: float64
      - name: n_heads
        dtype: float64
      - name: head_dim
        dtype: float64
      - name: kv_heads
        dtype: float64
      - name: numel
        dtype: float64
      - name: op_type
        dtype: string
      - name: gpu_time_duration_ms
        dtype: float64
      - name: launch_block_size
        dtype: float64
      - name: launch_grid_size
        dtype: float64
      - name: dram_bytes_sum
        dtype: float64
      - name: launch_registers_per_thread
        dtype: float64
    splits:
      - name: train
        num_examples: 148077
  - config_name: mse_validation
    features:
      - name: run_id
        dtype: string
      - name: model
        dtype: string
      - name: hardware
        dtype: string
      - name: engine
        dtype: string
      - name: profile
        dtype: string
      - name: concurrency
        dtype: int64
      - name: num_requests
        dtype: int64
      - name: successful_requests
        dtype: int64
      - name: failed_requests
        dtype: int64
      - name: duration_s
        dtype: float64
      - name: request_throughput
        dtype: float64
      - name: mean_ttft_ms
        dtype: float64
      - name: mean_tpot_ms
        dtype: float64
      - name: mean_e2el_ms
        dtype: float64
    splits:
      - name: summary
        num_examples: 28
---

# AgentPerfBench

LLM inference benchmark: 4,297 main sweep rows and 37 per-layer kernel validation rows, plus 148,077 per-kernel NCU profiles, across 9 models, 14 GPU configurations, and 2 serving engines (vLLM 0.19.0, SGLang 0.5.9). All models served in BF16 except gpt-oss, which uses mxfp4 for projection weights.

## Dataset configurations

### trace_replay (2,932 rows)

Replays exact ISL/OSL sequences from recorded agent sessions (SWE-Bench, TerminalBench, OSWorld, ShareGPT). 77 unique (model, hardware, engine) combinations, 17 profiles, 6 concurrency levels {1, 5, 10, 20, 40, 80}, 11.4% matrix fill.

17 profiles: `chat-medium`, `chat-multiturn-long`, `chat-multiturn-medium`, `chat-multiturn-short`, `chat-short`, `chat-singleturn`, `coding-singleturn`, `decode-heavy`, `osworld-multiturn-long`, `osworld-multiturn-medium`, `osworld-multiturn-short`, `prefill-heavy`, `random-1k`, `swebench-multiturn-medium`, `swebench-multiturn-short`, `terminalbench-multiturn-medium`, `terminalbench-multiturn-short`

### synthetic_distributional (1,365 rows)

ISL/OSL sampled from lognormal fits to real workload statistics. 49 unique (model, hardware, engine) combinations, 20 profiles, 11 concurrency levels {1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320}, 6.5% matrix fill.

20 profiles: `chat-medium`, `chat-multiturn`, `chat-multiturn-long`, `chat-multiturn-medium`, `chat-multiturn-short`, `chat-multiturn-synth`, `chat-short`, `chat-singleturn`, `chat-singleturn-synth`, `coding-singleturn`, `osworld-multiturn`, `osworld-multiturn-long`, `osworld-multiturn-medium`, `osworld-multiturn-short`, `osworld-multiturn-synth`, `swebench-multiturn`, `swebench-multiturn-synth`, `terminalbench-multiturn`, `terminalbench-multiturn-short`, `terminalbench-multiturn-synth`

### per_layer_kernel (37 rows)

Per-component operational intensity decomposition and Nsight Compute kernel profiles for Llama-3.1-8B on H100 (prefill phase). Analytical rows provide computed FLOPs, bytes, and OI per model component at batch sizes 1 and 80. NCU rows report measured SM and memory throughput per kernel from an 8-layer forward pass. Record types: `analytical_total`, `analytical_component`, `ncu_kernel`.

### kernels_labeled (148,077 rows)

Per-kernel Nsight Compute (ncu) profiles across 4 GPUs (A100, H100, RTX 3090, RTX 2080 Ti) and 13 model/sweep sources.

### mse_validation (28 rows)

Curated H100 / Llama-3.1-8B / vLLM validation table for the distributional synthetic replay generator. Paired synthetic and real trace replay runs; supplementary rows preserve no-replacement and high-concurrency debug runs. Raw JSON artifacts referenced through R2 URI columns. Per-run successful/failed request counts retained.

### Quality filtering

Concurrency levels: trace_replay {1, 5, 10, 20, 40, 80}, synthetic_distributional {1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320}. Configurations where fewer than 75% of requests completed successfully are excluded. Summary metrics are computed from successful requests only.

| Config | Rows |
|--------|------|
| trace_replay | 2,932 |
| synthetic_distributional | 1,365 |
| per_layer_kernel | 37 |
| kernels_labeled | 148,077 |
| mse_validation | 28 |

## Coverage

### Hardware

All benchmarks collected on PyTorch 2.10.0.

| GPU | VRAM | HBM bandwidth | Peak half-precision TFLOPS |
|-----|------|---------------|---------------------------|
| NVIDIA H100 SXM | 80 GB | 3.35 TB/s | 989 |
| NVIDIA A100 SXM4 | 40 GB | 1.56 TB/s | 312 |
| NVIDIA RTX 3090 | 24 GB | 936 GB/s | 71 |
| NVIDIA RTX 2080 Ti | 11 GB | 616 GB/s | 27 |

Multi-GPU configurations: 1, 2, 4, or 8 GPUs with tensor parallelism.

### Models

All models served in BF16 unless noted.

| Model | Family | Parameters | Architecture | Notes |
|-------|--------|-----------|--------------|-------|
| Llama-3.1-8B | Llama | 8B | Dense | |
| Llama-3.1-70B | Llama | 70B | Dense | |
| Llama-3.3-70B | Llama | 70B | Dense | |
| Qwen2.5-72B | Qwen | 72B | Dense | |
| Qwen3.5-9B | Qwen | 9B | Dense | |
| Qwen3.5-27B | Qwen | 27B | Dense | |
| Mixtral-8x7B | Mixtral | 46.7B (12.9B active) | MoE | |
| gpt-oss-20b | GPT-OSS | 21B (3.6B active) | MoE | mxfp4 projections |
| gpt-oss-120b | GPT-OSS | 117B (5.1B active) | MoE | mxfp4 projections |

### Engines

- vLLM 0.19.0
- SGLang 0.5.9

## Schema

Each row in `summary.parquet` (trace_replay and synthetic_distributional):

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
| request_throughput | float | Requests/second |
| input_token_throughput | float | Input tokens/second |
| output_token_throughput | float | Output tokens/second |
| total_token_throughput | float | Total tokens/second |
| mean/median/p90/p99_ttft_ms | float | Time to first token |
| mean/median/p90/p99_tpot_ms | float | Time per output token |
| mean/median/p90/p99_itl_ms | float | Inter-token latency |
| mean/median/p90/p99_e2el_ms | float | End-to-end latency |

## Loading

```python
from datasets import load_dataset

ds = load_dataset("agent-perf-bench/AgentPerfBench", "trace_replay")
# or "synthetic_distributional", "per_layer_kernel", "kernels_labeled", "mse_validation"
```

## Benchmark methodology

- Closed-loop concurrency with semaphore control.
- 3-request warmup before each configuration.
- Metrics: TTFT, TPOT, ITL, E2EL, request throughput, token throughput (mean, median, p90, p99).
- Metrics computed over successful requests only.
- Collection period: March 2026 onwards.

## Limitations

- Distributional profiles are fitted approximations, not direct production replays.
- Closed-loop concurrency only; no open-loop (Poisson) arrivals.

## Ethical considerations

No PII. Trace-replay profiles derive from open benchmarks (SWE-Bench MIT, TerminalBench, OSWorld). Synthetic profiles use random tokens.

## License

Benchmark data released under Apache-2.0. Source datasets retain their original licenses.

## Source datasets

- [SWE-Bench](https://github.com/princeton-nlp/SWE-bench) (MIT)
- [TerminalBench](https://github.com/TerminalBench/TerminalBench)
- [ShareGPT (Aeala/ShareGPT_Vicuna_unfiltered)](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)
- [OSWorld](https://github.com/xlang-ai/OSWorld)

## Future releases

- Additional hardware configurations and model families.
- Open-loop (Poisson) arrival mode.
- Additional per-kernel roofline profiles.

## Citation

```bibtex
@inproceedings{agentperfbench2026,
  title={AgentPerfBench: A Benchmarking and Evaluation Suite for Inference Performance of Agentic LLMs},
  author={Anonymous},
  booktitle={NeurIPS 2026 Evaluations and Datasets Track},
  year={2026}
}
```
