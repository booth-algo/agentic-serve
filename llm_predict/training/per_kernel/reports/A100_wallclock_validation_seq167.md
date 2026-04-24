# A100 — Wall-clock TTFT Validation (vs inference-benchmark data.json)

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)
- Filter: concurrency=1, TP=1, avg_seq within ±1 of 167
- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.
- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.
- `median tpot (ms)` is shown for reference; per-kernel composer currently only predicts TTFT, so no TPOT prediction column.

## Per-row

| Model | arch | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % | median TPOT (ms) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | fixed-seq128 | 167 | 1 | 37.43 | 36.71 | 1.97% | 26.48 | 27.9% | 12.89 |
| gpt-oss-20b | moe | vllm | fixed-seq128 | 166 | 1 | 12.00 | 33.88 | 64.59% | 60.34 | -78.1% | 4.95 |
| **supported MAPE** (1 rows) | | | | | | | | **1.97%** | | | |
| _out-of-scope_ | | | | | | | | _1 moe — excluded from headline_ | | | |

**Mean overhead across 1 supported rows with ncu data:** 27.9%
