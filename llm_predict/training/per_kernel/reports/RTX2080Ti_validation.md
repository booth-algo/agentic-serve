# RTX2080Ti — Per-Kernel Composition Validation

- Predictor: RTX2080Ti pkls (['elementwise', 'gemm', 'misc'])
- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv
- Input: bs=1, seq=128, tp=1
- Headline MAPE = supported architectures only (dense, non-hybrid-attn).
  MoE and hybrid-attn rows are known composer gaps and listed separately.

## Per-model

| Model | arch | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |
|---|---|---:|---:|---:|---:|
| Llama-8B | supported | 75.02 | 65.56 | 14.44% | 2037 |
| Qwen3.5-9B | hybrid_attn | 214.97 | 468.58 | 54.12% | 13061 |
| **supported aggregate** (1 rows) | | **75.02** | **65.56** | **Σ-err 14.44% · MAPE 14.44%** | |
| _out-of-scope_ | | | | _1 hybrid_attn — excluded from headline_ | |

## Measured time breakdown by family (ms)

| model      |   cast |   copy |   elementwise |     gemm |   reduce |   splitk_reduce |
|:-----------|-------:|-------:|--------------:|---------:|---------:|----------------:|
| Llama-8B   |  2.582 |  0.938 |         5.924 |   53.973 |    1.706 |           0.433 |
| Qwen3.5-9B | 22.078 |  0.329 |        21.291 |  409.234 |   15.651 |           0     |
| roofline   |  0     |  0     |         0     | 2116.46  |    0     |           0     |
