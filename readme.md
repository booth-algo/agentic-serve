# AgentServe-Bench

Inference benchmarking and GPU performance prediction for agentic LLM workloads.

**Dashboard** (link withheld for anonymous review) · **Paper** (under review) · **Dataset** (released with paper)

## Overview

1. **Agentic inference benchmarks** — real traces from SWE-Bench and TerminalBench alongside chat baselines, with saturation analysis across concurrency levels
2. **Kernel-level roofline analysis** — Nsight Compute (ncu) hardware counter profiles showing actual GPU resource utilization per kernel
3. **Per-operator ML predictors** — XGBoost models trained on profiling data for latency estimation without GPU experiments

## Data

| Dataset | Size | Description |
|---------|------|-------------|
| Benchmark results | 2,400+ files | TTFT, TPOT, ITL, throughput across models × profiles × concurrency |
| ncu kernel profiles | 1,900+ kernels | Hardware counters for Llama-8B and Mixtral-8x7B forward passes |
| ncu GEMM roofline | 56 shapes | Isolated GEMMs at M=1 to 8192 for roofline plots |
| Per-op training data | 13K+ rows | CUDA event sub-module measurements across 6 models |

### Hardware

- NVIDIA A100-SXM4-40GB nodes
- NVIDIA H100-SXM5-80GB nodes
- NVIDIA RTX 3090 and RTX 2080 Ti nodes

### Models

| Model | Type | Params | TP |
|-------|------|-------:|---:|
| Llama-3.1-8B | Dense | 8B | 1, 2 |
| Llama-3.1-70B | Dense | 70B | 2, 4 |
| Qwen-2.5-72B | Dense | 72B | 2, 4 |
| Mixtral-8x7B | MoE (8 experts) | 47B | 4 |
| gpt-oss-20b | MoE (32 experts) | 20B | 1 |
| gpt-oss-120b | MoE (128 experts) | 120B | 2 |

### Workload Profiles

| Tier | Profile | ISL | OSL | Source |
|------|---------|----:|----:|--------|
| Chat | chat-singleturn | sampled | sampled | ShareGPT |
| Agentic | coding-singleturn | 17K | 800 | SWE-Bench |
| Chat | chat-multiturn | sampled | sampled | ShareGPT |
| Agentic | swebench-multiturn | sampled | sampled | SWE-Bench |
| Agentic | terminalbench-multiturn | sampled | sampled | TerminalBench |
| Agentic | osworld-multiturn | sampled | sampled | OSWorld |

## Repository Structure

```
agentic-serve/
├── inference-benchmark/     # Benchmark tool + results + dashboard
│   ├── src/                 # Async benchmark runner
│   ├── results/             # A100 + H100 benchmark JSONs
│   ├── dashboard/           # React dashboard (GitHub Pages)
│   └── configs/             # Benchmark configurations
├── llm_predict/             # Current cache-aware serving predictor
│   ├── kernels/             # GEMM, flash attention, elementwise predictors
│   ├── training/            # Current calibration/training scripts
│   ├── configs/             # GPU and model configs
│   └── data/                # Current predictor artifacts
├── llm_predict_legacy/      # Legacy GPU performance models and predictors
│   ├── models/              # Hardware, software, serving, and cost models
│   ├── predictors/          # Per-kernel and per-op predictor runtimes
│   ├── training/            # Predictor labeling, training, and validation
│   └── profiling/           # Predictor artifacts and profiling data
```

## Quick Start

### Run a benchmark

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key test

# Run benchmark
cd inference-benchmark
python -m src.benchmark.runner \
  --url http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --profile coding-singleturn \
  --concurrency 40 \
  --num-requests 50 \
  --api-key test \
  --output results/my_run.json
```

## Citation

Author list and venue withheld for anonymous review. Citation metadata
will be added on acceptance.

## License

[TODO]
