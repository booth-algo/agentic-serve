# A100 — Wall-clock TTFT Validation (vs inference-benchmark data.json)

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)
- Filter: concurrency=1, TP=1, avg_seq within ±16 of 128
- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.
- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.
- `median tpot (ms)` is shown for reference; per-kernel composer currently only predicts TTFT, so no TPOT prediction column.

## Per-row

| Model | arch | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % | median TPOT (ms) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | chat-medium | 120 | 1 | 22.22 | 34.81 | 36.17% | 26.48 | 23.9% | 12.95 |
| Llama-8B | supported | vllm | chat-short | 127 | 1 | 22.29 | 35.18 | 36.62% | 26.48 | 24.7% | 12.89 |
| **supported MAPE** (2 rows) | | | | | | | | **36.40%** | | | |

**Mean overhead across 2 supported rows with ncu data:** 24.3%
