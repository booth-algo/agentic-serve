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
  - config_name: distributional
    data_files:
      - split: summary
        path: distributional/summary.parquet
  - config_name: kernels_labeled
    data_files:
      - split: train
        path: kernel_profiles/kernels_labeled.parquet
  - config_name: roofline_quadrant
    data_files:
      - split: train
        path: kernel_profiles/roofline_quadrant.parquet
  - config_name: predictions
    data_files:
      - split: train
        path: predictions/serving_predictions.parquet
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
      - name: successful_requests
        dtype: int64
      - name: failed_requests
        dtype: int64
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
        num_examples: 3147
        num_bytes: 694254
  - config_name: distributional
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
      - name: successful_requests
        dtype: int64
      - name: failed_requests
        dtype: int64
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
        num_examples: 245
        num_bytes: 70836
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
  - config_name: roofline_quadrant
    features:
      - name: model
        dtype: string
      - name: profile
        dtype: string
      - name: concurrency
        dtype: int64
      - name: engine
        dtype: string
      - name: hardware
        dtype: string
      - name: oi
        dtype: float64
      - name: cf_gb
        dtype: float64
      - name: output_tput
        dtype: float64
      - name: tpot_ms
        dtype: float64
      - name: ttft_ms
        dtype: float64
    splits:
      - name: train
        num_examples: 2163
  - config_name: predictions
    features:
      - name: hardware_config
        dtype: string
      - name: model
        dtype: string
      - name: backend
        dtype: string
      - name: profile
        dtype: string
      - name: data_scope
        dtype: string
      - name: concurrency
        dtype: int64
      - name: isl
        dtype: int64
      - name: osl
        dtype: int64
      - name: calibration_status
        dtype: string
      - name: ttft_validation_scope
        dtype: string
      - name: ttft_kernel_ms
        dtype: float64
      - name: ttft_base_ms
        dtype: float64
      - name: ttft_floor_ms
        dtype: float64
      - name: ttft_first_decode_ms
        dtype: float64
      - name: ttft_queue_ms
        dtype: float64
      - name: itl_meas
        dtype: float64
      - name: total_context_tokens
        dtype: int64
      - name: new_prefill_tokens
        dtype: int64
      - name: cached_context_tokens
        dtype: int64
      - name: cache_hit_rate
        dtype: float64
      - name: cache_aware_applied
        dtype: bool
      - name: cache_feature_source
        dtype: string
      - name: cache_prediction_regime
        dtype: string
      - name: ttft_prediction_supported
        dtype: bool
      - name: multiturn_prediction_mode
        dtype: string
      - name: predicted_turn_count
        dtype: float64
      - name: total_successful_turn_requests
        dtype: float64
      - name: mean_predicted_turn_ttft_ms
        dtype: float64
      - name: mean_predicted_turn_tpot_ms
        dtype: float64
      - name: multiturn_turn_predictions
        dtype: string
      - name: ttft_pred
        dtype: float64
      - name: ttft_meas
        dtype: float64
      - name: ttft_err
        dtype: float64
      - name: tpot_pred
        dtype: float64
      - name: tpot_meas
        dtype: float64
      - name: tpot_err
        dtype: float64
      - name: e2el_pred
        dtype: float64
      - name: e2el_meas
        dtype: float64
      - name: e2el_err
        dtype: float64
      - name: measurement_semantics_warning
        dtype: string
    splits:
      - name: train
        num_examples: 4715
---

# AgentPerfBench

LLM inference benchmark: 3,392 serving runs, 148,077 per-kernel NCU profiles, and 4,715 latency predictions across 9 models, 14 GPU configurations, and 2 serving engines (vLLM 0.19.0, SGLang 0.5.9). All models served in BF16 except gpt-oss, which uses mxfp4 for projection weights.

## Dataset configurations

### trace_replay (3,147 rows)

Replays exact ISL/OSL sequences from recorded agent sessions (SWE-Bench, TerminalBench, OSWorld, ShareGPT). 77 (model, hardware, engine) combinations, 17 profiles, 6 concurrency levels.

17 profiles: `chat-medium`, `chat-multiturn-long`, `chat-multiturn-medium`, `chat-multiturn-short`, `chat-short`, `chat-singleturn`, `coding-singleturn`, `decode-heavy`, `osworld-multiturn-long`, `osworld-multiturn-medium`, `osworld-multiturn-short`, `prefill-heavy`, `random-1k`, `swebench-multiturn-medium`, `swebench-multiturn-short`, `terminalbench-multiturn-medium`, `terminalbench-multiturn-short`

### distributional (245 rows)

ISL/OSL sampled from lognormal fits to real workload statistics. 42 combinations, 6 profiles, 7 concurrency levels.

6 profiles: `chat-multiturn`, `chat-singleturn`, `coding-singleturn`, `osworld-multiturn`, `swebench-multiturn`, `terminalbench-multiturn`

### kernels_labeled (148,077 rows)

Per-kernel Nsight Compute (ncu) profiles across 4 GPUs (A100, H100, RTX 3090, RTX 2080 Ti) and 13 model/sweep sources.

### roofline_quadrant (2,163 rows)

Operational intensity and achieved throughput per kernel, for roofline analysis. H100 reference hardware (989 peak TFLOPS, 3.35 TB/s HBM).

### predictions (4,715 rows)

Predicted vs. measured TTFT, TPOT, and E2EL for each serving configuration, with cache-aware prediction metadata. 14 hardware configs.

### Concurrency filtering

Concurrency is controlled by a fixed-size connection pool. Trace replay uses levels {1, 5, 10, 20, 40, 80}; distributional uses {1, 5, 10, 40, 80, 200, 320}.

Early runs used a session-pool size smaller than the declared concurrency (`num_sessions=100` for trace replay, `num_sessions=10` for distributional), capping actual load below the nominal value. Rows where declared concurrency exceeded the session pool were dropped.

| Config | Rows |
|--------|------|
| trace_replay | 3,147 |
| distributional | 245 |
| **Total** | **3,392** |

## Coverage

### Hardware

All benchmarks collected on PyTorch 2.10.0, CUDA 12.8.

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

Each row in `summary.parquet` (both configs):

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

## Loading

```python
from datasets import load_dataset

ds = load_dataset("agent-perf-bench/AgentPerfBench", "trace_replay")
# or "distributional", "kernels_labeled", "roofline_quadrant", "predictions"
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

## Citation

```bibtex
@inproceedings{agentperfbench2026,
  title={AgentPerfBench: A Benchmarking and Evaluation Suite for Inference Performance of Agentic LLMs},
  author={Anonymous},
  booktitle={NeurIPS 2026 Evaluations and Datasets Track},
  year={2026}
}
```
