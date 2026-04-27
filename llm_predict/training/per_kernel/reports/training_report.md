# Phase A3 Training Report (shape_v2)

- **gemm** features (12): `M`, `N`, `K`, `log_M`, `log_N`, `log_K`, `analytical_flops`, `analytical_bytes`, `analytical_ai`, `log_flops`, `log_bytes`, `dtype_onehot_bf16` ✓
- **flash_attn** features (8): `bs`, `seq`, `n_heads`, `head_dim`, `kv_heads`, `total_flops`, `log_seq`, `log_heads` ✓
- **elementwise** features (11): `numel`, `log_numel`, `op_type_onehot_rmsnorm`, `op_type_onehot_silu`, `op_type_onehot_mul`, `op_type_onehot_residual`, `op_type_onehot_rope`, `op_type_onehot_neg`, `op_type_onehot_fill`, `op_type_onehot_compare`, `op_type_onehot_other` ✓
- **misc** features (12): `numel_or_shape_total`, `log_numel_or_shape`, `size_m`, `size_n`, `size_k`, `log_size_m`, `log_size_n`, `log_size_k`, `kernel_family_onehot_reduce`, `kernel_family_onehot_splitk_reduce`, `kernel_family_onehot_cast`, `kernel_family_onehot_copy` ✓

## Headline: aggregate layer-total error per (GPU × held-out model)

| GPU | held-out model | sum(pred ms) | sum(meas ms) | abs err % |
|---|---|---:|---:|---:|
| RTX3090 | Llama-70B | 393.66 | 393.35 | 0.08% |
| RTX3090 | Qwen-72B | 398.50 | 399.80 | 0.33% |

## Held-out per-family MAPE (full-train)

| GPU | family | n_train | n_test | MAPE |
|---|---|---:|---:|---:|
| RTX3090 | gemm | 4088 | 1122 | 15.65% |
| RTX3090 | flash_attn | 181 | 160 | 1.11% |
| RTX3090 | elementwise | 18986 | 3538 | 18.99% |
| RTX3090 | misc | 24598 | 1780 | 18.48% |

## Leave-one-model-out CV

### RTX3090

| held-out | gemm | flash_attn | elementwise | misc |
|---|---:|---:|---:|---:|
| Llama-8B | 10.6% (n=225) | 0.5% (n=32) | 10.0% (n=713) | 18.2% (n=394) |
| Mixtral-8x7B | 15.8% (n=893) | 0.6% (n=32) | 14.0% (n=1541) | 23.6% (n=1909) |
| Qwen3.5-27B | 39.8% (n=1121) | n/a | 10.5% (n=10508) | 12.9% (n=14255) |
| Qwen3.5-9B | 51.7% (n=561) | n/a | 13.3% (n=5260) | 14.1% (n=7135) |
| flash_sweep | n/a | 111.3% (n=117) | n/a | n/a |
| gpt-oss-20b | 33.7% (n=217) | n/a | 42.4% (n=964) | 29.9% (n=541) |
| misc_sweep | n/a | n/a | n/a | 75.1% (n=364) |
| roofline | 68.4% (n=1071) | n/a | n/a | n/a |

## Saved pkls
- **RTX3090**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_flash_attn_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_elementwise_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_kernel/perkernel_misc_shape_v2.pkl`
