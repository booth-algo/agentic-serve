# Phase A3 Training Report (shape_v2)

- **gemm** features (12): `M`, `N`, `K`, `log_M`, `log_N`, `log_K`, `analytical_flops`, `analytical_bytes`, `analytical_ai`, `log_flops`, `log_bytes`, `dtype_onehot_bf16` ✓
- **flash_attn** features (8): `bs`, `seq`, `n_heads`, `head_dim`, `kv_heads`, `total_flops`, `log_seq`, `log_heads` ✓
- **elementwise** features (11): `numel`, `log_numel`, `op_type_onehot_rmsnorm`, `op_type_onehot_silu`, `op_type_onehot_mul`, `op_type_onehot_residual`, `op_type_onehot_rope`, `op_type_onehot_neg`, `op_type_onehot_fill`, `op_type_onehot_compare`, `op_type_onehot_other` ✓
- **misc** features (12): `numel_or_shape_total`, `log_numel_or_shape`, `size_m`, `size_n`, `size_k`, `log_size_m`, `log_size_n`, `log_size_k`, `kernel_family_onehot_reduce`, `kernel_family_onehot_splitk_reduce`, `kernel_family_onehot_cast`, `kernel_family_onehot_copy` ✓

## Headline: aggregate layer-total error per (GPU × held-out model)

| GPU | held-out model | sum(pred ms) | sum(meas ms) | abs err % |
|---|---|---:|---:|---:|
| A100 | Llama-3.3-70B | 165.19 | 178.50 | 7.46% |
| A100 | Llama-70B | 165.19 | 178.84 | 7.63% |
| A100 | Qwen-72B | 169.49 | 181.39 | 6.56% |
| RTX3090 | Llama-70B | 393.08 | 393.35 | 0.07% |
| RTX3090 | Qwen-72B | 397.93 | 399.80 | 0.47% |

## Held-out per-family MAPE (full-train)

| GPU | family | n_train | n_test | MAPE |
|---|---|---:|---:|---:|
| A100 | gemm | 4111 | 1683 | 7.10% |
| A100 | flash_attn | 64 | 240 | 3.51% |
| A100 | elementwise | 19007 | 5307 | 13.73% |
| A100 | misc | 25029 | 2910 | 15.19% |
| RTX3090 | gemm | 4088 | 1122 | 15.65% |
| RTX3090 | flash_attn | 64 | 160 | 35.77% |
| RTX3090 | elementwise | 18986 | 3538 | 18.99% |
| RTX3090 | misc | 24598 | 1780 | 18.48% |
| RTX2080Ti | gemm | 1937 | 0 | no-heldout (train only) |
| RTX2080Ti | flash_attn | 0 | 0 | skipped (n_train=0 < 5) |
| RTX2080Ti | elementwise | 6373 | 0 | no-heldout (train only) |
| RTX2080Ti | misc | 7857 | 0 | no-heldout (train only) |

## Leave-one-model-out CV

### A100

| held-out | gemm | flash_attn | elementwise | misc |
|---|---:|---:|---:|---:|
| Llama-8B | 16.8% (n=225) | 1.2% (n=32) | 8.6% (n=713) | 10.3% (n=490) |
| Mixtral-8x7B | 21.2% (n=914) | 1.1% (n=32) | 14.1% (n=1562) | 25.9% (n=2156) |
| Qwen3.5-27B | 64.1% (n=1122) | n/a | 9.4% (n=10508) | 12.0% (n=14255) |
| Qwen3.5-9B | 38.5% (n=562) | n/a | 9.0% (n=5260) | 10.3% (n=7223) |
| gpt-oss-20b | 110.1% (n=217) | n/a | 42.9% (n=964) | 30.8% (n=541) |
| misc_sweep | n/a | n/a | n/a | 55.9% (n=364) |
| roofline | 63.4% (n=1071) | n/a | n/a | n/a |

### RTX3090

| held-out | gemm | flash_attn | elementwise | misc |
|---|---:|---:|---:|---:|
| Llama-8B | 10.6% (n=225) | 0.3% (n=32) | 10.0% (n=713) | 18.2% (n=394) |
| Mixtral-8x7B | 15.8% (n=893) | 0.4% (n=32) | 14.0% (n=1541) | 23.6% (n=1909) |
| Qwen3.5-27B | 39.8% (n=1121) | n/a | 10.5% (n=10508) | 12.9% (n=14255) |
| Qwen3.5-9B | 51.7% (n=561) | n/a | 13.3% (n=5260) | 14.1% (n=7135) |
| gpt-oss-20b | 33.7% (n=217) | n/a | 42.4% (n=964) | 29.9% (n=541) |
| misc_sweep | n/a | n/a | n/a | 75.1% (n=364) |
| roofline | 68.4% (n=1071) | n/a | n/a | n/a |

### RTX2080Ti

| held-out | gemm | flash_attn | elementwise | misc |
|---|---:|---:|---:|---:|
| Llama-8B | 144.8% (n=289) | n/a | 13.9% (n=1033) | 21.9% (n=714) |
| Qwen3.5-9B | 75.7% (n=577) | n/a | 20.8% (n=5340) | 29.9% (n=7143) |
| roofline | 714.3% (n=1071) | n/a | n/a | n/a |

## Saved pkls
- **A100**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_kernel/perkernel_flash_attn_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_kernel/perkernel_elementwise_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_kernel/perkernel_misc_shape_v2.pkl`
- **RTX3090**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_flash_attn_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_elementwise_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_misc_shape_v2.pkl`
- **RTX2080Ti**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_kernel/perkernel_elementwise_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_kernel/perkernel_misc_shape_v2.pkl`
