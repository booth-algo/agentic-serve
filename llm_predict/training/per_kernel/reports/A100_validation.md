# A100 — Per-Kernel Composition Validation

- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv
- Input: bs=1, seq=128, tp=1
- Headline MAPE = supported architectures only (dense, non-hybrid-attn).
  MoE and hybrid-attn rows are known composer gaps and listed separately.

## Per-model

| Model | arch | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |
|---|---|---:|---:|---:|---:|
| Llama-8B | supported | 23.62 | 26.48 | 10.79% | 1461 |
| Llama-70B _(held-out)_ | supported | 165.36 | 178.85 | 7.54% | 3381 |
| Llama-3.3-70B _(held-out)_ | supported | 165.36 | 178.51 | 7.37% | 3381 |
| Qwen-72B _(held-out)_ | supported | 169.66 | 181.39 | 6.47% | 3381 |
| Qwen3.5-9B | hybrid_attn | 21.11 | 105.21 | 79.94% | 13061 |
| Qwen3.5-27B | hybrid_attn | 57.16 | 233.93 | 75.57% | 25917 |
| gpt-oss-20b | moe | 15.38 | 60.34 | 74.51% | 1723 |
| Mixtral-8x7B | moe | 31.74 | 109.87 | 71.11% | 4665 |
| **supported aggregate** (4 rows) | | **524.00** | **565.23** | **Σ-err 7.29% · MAPE 8.04%** | |
| _out-of-scope_ | | | | _2 moe, 2 hybrid_attn — excluded from headline_ | |

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
