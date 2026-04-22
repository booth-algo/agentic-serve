# RTX2080Ti — Wall-clock TTFT Validation (vs inference-benchmark data.json)

- Predictor: RTX2080Ti pkls (['elementwise', 'gemm', 'misc'])
- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)
- Filter: concurrency=1, TP=1, avg_seq within ±1 of 167
- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.
- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.
- `median tpot (ms)` is shown for reference; per-kernel composer currently only predicts TTFT, so no TPOT prediction column.

## Per-row

| Model | arch | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % | median TPOT (ms) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | fixed-seq128 | 167 | 1 | 170.46 | 81.05 | 110.31% | 65.56 | 19.1% | 27.48 |
| **supported MAPE** (1 rows) | | | | | | | | **110.31%** | | | |

**Mean overhead across 1 supported rows with ncu data:** 19.1%
