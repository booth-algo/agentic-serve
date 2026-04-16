# llm_predict — predictor library for AgentServe-Bench

> Part of [AgentServe-Bench](../readme.md). This directory contains the GPU
> performance prediction library that models LLM inference latency across
> shapes, hardware, and parallelism strategies.

## Scope

`llm_predict` predicts per-kernel, per-op, and end-to-end latency for
transformer inference. It is the analytical-composition + ML-hybrid
engine behind AgentServe-Bench's predicted-vs-measured story (paper
[dashboard](https://booth-algo.github.io/agentic-serve/)).

Derived from [LLMCompass (ISCA'24)](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf);
the hardware / cost / DSE simulator inherited from upstream is kept intact.
The analytical transformer composition, shape-aware predictors, and
calibration layer are AgentServe-Bench additions.

## Directory layout

```
llm_predict/
├── models/
│   ├── software/         # TransformerBlock{InitComputation,DecodeComputation,MLA}TP etc.
│   ├── hardware/         # System, Device, Memory, Interconnect models
│   ├── serving/          # request/batch-level serving models
│   └── cost/             # cost model (from upstream LLMCompass)
├── predictors/
│   ├── dispatch.py       # PredictorDispatch: tier-based routing
│   ├── per_op/           # PerOpPredictor: 4 ops/layer (XGBoost, 26 shape features)
│   ├── per_kernel/       # PerKernelPredictor: per-CUDA-kernel (XGBoost, ncu
│   │                     #   ground truth) — gemm/flash_attn/elementwise/misc
│   └── calibration/      # cross-cutting corrections:
│                         #   - moe.py: MoE dispatch overhead
│                         #   - allreduce.py: NCCL empirical table (per GPU)
│                         #   - framework.py: vLLM/SGLang framework overhead
├── profiling/
│   └── data/{A100,H100}/        # trained .pkl models (stored on R2)
├── search/                      # TP/DP search algorithms (in ../search at repo root)
├── dse/                         # design-space exploration
├── systolic_array_model/        # (upstream LLMCompass)
└── ae/                          # artifact-evaluation scripts for original LLMCompass paper
```

## Predictor tiers

| Tier | What it predicts | Features | Model | Training data |
|------|------------------|----------|-------|---------------|
| **per-op** | 4 ops/layer: attn, ffn, norm_pre, norm_post | 26 shape features (tok, d, h, kv, ffn, E, k, bs, seq + derived) | XGBoost | CUDA events on isolated layers, 13k rows, 6 models |
| **per-kernel** | individual CUDA kernel invocations (gemm, flash_attn, elementwise, misc) | shape-only (no runtime counters) | XGBoost per family | ncu ground truth on real model forward passes |
| **roofline (fallback)** | analytical upper bound | `(M,N,K)` + HW peak FLOPS/BW | closed-form | none |

## Quick usage

```python
from llm_predict.predictors.dispatch import PredictorDispatch

d = PredictorDispatch(gpu="A100")

# Per-kernel GEMM prediction (seconds)
t_gemm = d.predict_gemm(M=512, N=6144, K=4096)

# Attention prefill
t_attn = d.predict_attention_prefill(batch=1, seq_len=512, n_heads=32, head_dim=128)
```

For E2E composition examples (walk transformer layers, sum kernels × n_layers,
compare to measured TTFT), see the composers in `inference-benchmark/` and the
per-request validation at `/Users/kev/gated/step4_results/`.

## Profiling data

Per-GPU profile CSVs + trained pickles live on R2
(`s3://agent-bench/profiling-data/{A100,H100}/`), not in this repo. Fetch with:

```bash
aws --profile r2 s3 sync \
  s3://agent-bench/profiling-data/A100/ \
  llm_predict/profiling/data/A100/ \
  --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com
```

## Adding a new accelerator

See [docs/ADDING_NEW_ACCELERATOR.md](docs/ADDING_NEW_ACCELERATOR.md) and
[docs/ACCELERATOR_CODE_STRUCTURE.md](docs/ACCELERATOR_CODE_STRUCTURE.md).

## Running a simulation

See [docs/run.md](docs/run.md).

## Upstream attribution

Derived from LLMCompass (ISCA 2024):

[**LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference**](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf) —
*Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, David Wentzlaff*

```
@inproceedings{LLMCompass,
  author    = {Zhang, Hengrui and Ning, August and Prabhakar, Rohan Baskar and Wentzlaff, David},
  title     = {LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference},
  year      = {2024},
  booktitle = {Proceedings of the 51st Annual International Symposium on Computer Architecture},
}
```

The original LLMCompass artifact-evaluation harness is preserved under
`ae/` for reproducing Figures 5-12 of the upstream paper. AgentServe-Bench
adds: agentic-trace benchmarks, the analytical transformer composer,
per-op/per-kernel predictor hierarchy, the calibration layer, and
cross-GPU generalization validation.
