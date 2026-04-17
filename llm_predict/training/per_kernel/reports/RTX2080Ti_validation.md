# RTX2080Ti — Per-Kernel Composition Validation

- Predictor: RTX2080Ti pkls (['elementwise', 'gemm', 'misc'])
- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv
- Input: bs=1, seq=128, tp=1

## Per-model

| Model | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |
|---|---:|---:|---:|---:|
| Llama-8B | 577.17 | 65.56 | 780.42% | 2037 |
| Qwen3.5-9B | 236.23 | 468.58 | 49.59% | 13061 |
| **TOTAL** | **813.41** | **534.14** | **52.28%** | |

## Measured time breakdown by family (ms)

| model      |   cast |   copy |   elementwise |      gemm |   reduce |   splitk_reduce |
|:-----------|-------:|-------:|--------------:|----------:|---------:|----------------:|
| Llama-8B   |  2.582 |  0.938 |         5.924 |    53.973 |    1.706 |           0.433 |
| Qwen3.5-9B | 22.078 |  0.329 |        21.291 |   409.234 |   15.651 |           0     |
| roofline   |  0     |  0     |         0     | 32022.5   |    0     |           0     |
