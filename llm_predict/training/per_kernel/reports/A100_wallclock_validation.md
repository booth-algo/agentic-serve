# A100 — Wall-clock TTFT Validation (vs inference-benchmark data.json)

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)
- Filter: concurrency=1, TP=1
- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.

## Per-row

| Model | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | vllm | chat-long | 258 | 1 | 15.90 | 33.67 | 52.78% | 60.34 | -79.2% |
| gpt-oss-20b | vllm | chat-medium | 159 | 1 | 13.49 | 33.49 | 59.73% | 60.34 | -80.2% |
| gpt-oss-20b | vllm | chat-short | 150 | 1 | 13.33 | 32.99 | 59.58% | 60.34 | -82.9% |
| Llama-8B | vllm | chat-long | 218 | 1 | 28.23 | 34.62 | 18.44% | 26.48 | 23.5% |
| Llama-8B | vllm | chat-medium | 120 | 1 | 22.22 | 34.81 | 36.17% | 26.48 | 23.9% |
| Llama-8B | vllm | chat-multiturn-short | 673 | 1 | 70.59 | 52.97 | 33.27% | 26.48 | 50.0% |
| Llama-8B | vllm | chat-short | 127 | 1 | 22.29 | 35.18 | 36.62% | 26.48 | 24.7% |
| Qwen3.5-9B | vllm | chat-long | 206 | 1 | 27.11 | 87.58 | 69.04% | 105.21 | -20.1% |
| Qwen3.5-9B | vllm | chat-medium | 103 | 1 | 20.61 | 89.55 | 76.99% | 105.21 | -17.5% |
| Qwen3.5-9B | vllm | chat-short | 95 | 1 | 20.61 | 89.43 | 76.95% | 105.21 | -17.6% |
| **MAPE over 10 rows** | | | | | | | **51.96%** | | |

**Mean overhead across 10 rows with ncu data:** -17.5%
