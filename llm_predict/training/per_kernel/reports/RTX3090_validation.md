# RTX3090 — Per-Kernel Composition Validation

- Predictor: RTX3090 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv
- Input: bs=1, seq=128, tp=1
- Headline MAPE = supported architectures only (dense, non-hybrid-attn).
  MoE and hybrid-attn rows are known composer gaps and listed separately.

## Per-model

| Model | arch | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |
|---|---|---:|---:|---:|---:|
| Llama-8B | supported | 55.02 | 55.06 | 0.08% | 1365 |
| Llama-70B | supported | 385.33 | 393.35 | 2.04% | 3381 |
| Qwen-72B _(held-out)_ | supported | 390.90 | 399.80 | 2.23% | 3221 |
| Qwen3.5-9B | hybrid_attn | 46.87 | 108.94 | 56.97% | 12973 |
| Qwen3.5-27B | hybrid_attn | 128.69 | 269.15 | 52.19% | 25917 |
| gpt-oss-20b | moe | 24.12 | 128.27 | 81.19% | 1723 |
| Mixtral-8x7B | moe | 54.39 | 202.04 | 73.08% | 4376 |
| **supported aggregate** (3 rows) | | **831.25** | **848.22** | **Σ-err 2.00% · MAPE 1.45%** | |
| _out-of-scope_ | | | | _2 moe, 2 hybrid_attn — excluded from headline_ | |

## Measured time breakdown by family (ms)

| model        |   cast |   copy |   elementwise |   flash_attn |      gemm |   reduce |   splitk_reduce |
|:-------------|-------:|-------:|--------------:|-------------:|----------:|---------:|----------------:|
| Llama-70B    |  3.123 |  2.81  |        14.311 |        1.654 |   368.968 |    1.62  |           0.87  |
| Llama-8B     |  0.878 |  0.975 |         4.009 |        0.425 |    47.302 |    1.124 |           0.348 |
| Mixtral-8x7B |  1.867 |  5.889 |         7.297 |        0.425 |   182.84  |    1.613 |           2.113 |
| Qwen-72B     |  3.118 |  2.814 |        14.453 |        1.656 |   376.144 |    1.619 |           0     |
| Qwen3.5-27B  | 44.853 |  0.762 |        43.009 |        0.542 |   149.238 |   30.274 |           0.474 |
| Qwen3.5-9B   | 21.337 |  0.35  |        19.791 |        0.232 |    52.133 |   14.863 |           0.238 |
| flash_sweep  |  0     |  0     |         0     |       18.305 |     0     |    0     |           0     |
| gpt-oss-20b  |  2.008 |  1.075 |        22.444 |        0     |   100.198 |    2.221 |           0.324 |
| misc_sweep   |  2.93  |  0.508 |         0     |        0     |     0     |    8.716 |           0     |
| roofline     |  0     |  0     |         0     |        0     | 24674     |    0     |           0     |
