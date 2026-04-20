# AgentServe-Bench

Inference benchmarking and GPU performance prediction for agentic LLM workloads.

**Dashboard** (link withheld for anonymous review) · **Paper** (under review) · **Dataset** (released with paper)

## Overview

Current LLM inference benchmarks use single-turn chat workloads with short input sequences. Real agentic workloads — coding agents, terminal automation, tool-use systems — have fundamentally different characteristics: high input/output ratios (17K:800 tokens), multi-turn sessions spanning hundreds of steps, and growing context that shifts GPU utilization from memory-bound decode to compute-bound prefill.

AgentServe-Bench provides:

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
| Agentic | coding-agent | 17K | 800 | SWE-Bench |
| Agentic | swebench-multiturn | 65K | 2K | SWE-Bench |
| Agentic | terminalbench-multiturn | 65K | 2K | TerminalBench |
| Chat | chat-short / medium / long | 500–8K | 300–2K | ShareGPT |
| Synthetic | decode-heavy / prefill-heavy | 256–8K | 256–4K | Random |

## Repository Structure

```
agentic-serve/
├── inference-benchmark/     # Benchmark tool + results + dashboard
│   ├── src/                 # Async benchmark runner
│   ├── results/             # A100 + H100 benchmark JSONs
│   ├── dashboard/           # React dashboard (GitHub Pages)
│   └── configs/             # Benchmark configurations
├── llmcompass/              # GPU performance simulator
│   ├── software_model/      # Transformer block models (dense, MoE, MLA)
│   ├── hardware_model/      # GPU device models (A100, H100)
│   ├── profiler/            # ML predictor + kernel profiler
│   └── serving_model/       # Batch + queueing models
├── experiment/              # Profiling scripts + ncu data + training
│   ├── ncu_*.csv            # Kernel-level hardware counter data
│   ├── train_perop_v4.py    # XGBoost predictor training
│   ├── reprofile_v4.py      # CUDA events profiler
│   └── validate_model.py    # Model validation
├── device_configs/          # GPU hardware specifications
├── model_configs/           # LLM architecture configs
└── search/                  # TP/DP parallelism search
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
  --profile coding-agent \
  --concurrency 40 \
  --num-requests 50 \
  --api-key test \
  --output results/my_run.json
```

### Run ncu profiling

```bash
# Requires RmProfilingAdminOnly=0 (see experiment/ncu_setup.md)
ncu --set full --target-processes all \
  -o profile_output \
  python experiment/reprofile_v4.py cuda:0 /path/to/model 1 512
```

## Citation

Author list and venue withheld for anonymous review. Citation metadata
will be added on acceptance.

## License

[TODO]
