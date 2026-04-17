# A100 — Per-Kernel Composition Validation

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv
- Input: bs=1, seq=128, tp=1

## Per-model

| Model | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |
|---|---:|---:|---:|---:|
| Llama-8B | 23.62 | 26.48 | 10.80% | 1461 |
| Llama-70B _(held-out)_ | 165.35 | 178.85 | 7.55% | 3381 |
| Llama-3.3-70B _(held-out)_ | 165.35 | 178.51 | 7.37% | 3381 |
| Qwen-72B _(held-out)_ | 169.65 | 181.39 | 6.47% | 3381 |
| Qwen3.5-9B | 20.90 | 105.21 | 80.14% | 13061 |
| Qwen3.5-27B | 56.87 | 233.93 | 75.69% | 25917 |
| gpt-oss-20b | 15.38 | 60.34 | 74.51% | 1723 |
| Mixtral-8x7B | 31.74 | 109.87 | 71.11% | 4665 |
| **TOTAL** | **648.86** | **1074.58** | **39.62%** | |

## Measured time breakdown by family (ms)

| model         |   cast |   copy |   elementwise |   flash_attn |     gemm |   reduce |   splitk_reduce |
|:--------------|-------:|-------:|--------------:|-------------:|---------:|---------:|----------------:|
| Llama-3.3-70B |  3.167 |  3.573 |        14.801 |        1.596 |  152.556 |    1.69  |           1.125 |
| Llama-70B     |  3.208 |  3.607 |        14.967 |        1.605 |  152.606 |    1.71  |           1.143 |
| Llama-8B      |  0.944 |  1.3   |         4.827 |        0.618 |   15.721 |    1.735 |           1.337 |
| Mixtral-8x7B  |  2.396 |  8.224 |         9.446 |        0.618 |   83.552 |    2.383 |           3.25  |
| Qwen-72B      |  3.204 |  3.611 |        15.032 |        1.601 |  155.235 |    1.705 |           1.004 |
| Qwen3.5-27B   | 62.383 |  0.994 |        56.144 |        0.476 |   69.694 |   43.36  |           0.883 |
| Qwen3.5-9B    | 30.159 |  0.491 |        26.831 |        0.206 |   25.178 |   21.062 |           1.281 |
| gpt-oss-20b   |  2.203 |  1.366 |        18.247 |        0     |   35.632 |    2.402 |           0.493 |
| misc_sweep    |  2.452 |  0.732 |         0     |        0     |    0     |    7.429 |           0     |
| roofline      |  0     |  0     |         0     |        0     | 6293.11  |    0     |           0     |
