# A100 — Wall-clock TTFT Validation (vs inference-benchmark data.json)

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)
- Filter: concurrency=1, TP=1
- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.
- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.
- `median tpot (ms)` is shown for reference; per-kernel composer currently only predicts TTFT, so no TPOT prediction column.

## Per-row

| Model | arch | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % | median TPOT (ms) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | chat-singleturn | 258 | 1 | 15.90 | 33.67 | 52.78% | 60.34 | -79.2% | 4.94 |
| gpt-oss-20b | moe | vllm | chat-medium | 159 | 1 | 13.49 | 33.49 | 59.73% | 60.34 | -80.2% | 4.94 |
| gpt-oss-20b | moe | vllm | chat-short | 150 | 1 | 13.33 | 32.99 | 59.58% | 60.34 | -82.9% | 4.93 |
| Llama-8B | supported | vllm | chat-singleturn | 218 | 1 | 28.23 | 34.62 | 18.44% | 26.48 | 23.5% | 12.95 |
| Llama-8B | supported | vllm | chat-medium | 120 | 1 | 22.22 | 34.81 | 36.17% | 26.48 | 23.9% | 12.95 |
| Llama-8B | supported | vllm | chat-multiturn-short | 673 | 1 | 70.59 | 52.97 | 33.27% | 26.48 | 50.0% | 13.04 |
| Llama-8B | supported | vllm | chat-short | 127 | 1 | 22.29 | 35.18 | 36.62% | 26.48 | 24.7% | 12.89 |
| Qwen3.5-9B | hybrid_attn | vllm | chat-singleturn | 206 | 1 | 27.11 | 87.58 | 69.04% | 105.21 | -20.1% | 13.78 |
| Qwen3.5-9B | hybrid_attn | vllm | chat-medium | 103 | 1 | 20.61 | 89.55 | 76.99% | 105.21 | -17.5% | 13.78 |
| Qwen3.5-9B | hybrid_attn | vllm | chat-short | 95 | 1 | 20.61 | 89.43 | 76.95% | 105.21 | -17.6% | 13.76 |
| **supported MAPE** (4 rows) | | | | | | | | **31.13%** | | | |
| _out-of-scope_ | | | | | | | | _3 moe, 3 hybrid_attn — excluded from headline_ | | | |

**Mean overhead across 4 supported rows with ncu data:** 30.5%
