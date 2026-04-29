# Phase A3 Training Report (shape_v2)

- **gemm** features (12): `M`, `N`, `K`, `log_M`, `log_N`, `log_K`, `analytical_flops`, `analytical_bytes`, `analytical_ai`, `log_flops`, `log_bytes`, `dtype_onehot_bf16` ✓
- **flash_attn** features (8): `bs`, `seq`, `n_heads`, `head_dim`, `kv_heads`, `total_flops`, `log_seq`, `log_heads` ✓
- **elementwise** features (11): `numel`, `log_numel`, `op_type_onehot_rmsnorm`, `op_type_onehot_silu`, `op_type_onehot_mul`, `op_type_onehot_residual`, `op_type_onehot_rope`, `op_type_onehot_neg`, `op_type_onehot_fill`, `op_type_onehot_compare`, `op_type_onehot_other` ✓
- **misc** features (12): `numel_or_shape_total`, `log_numel_or_shape`, `size_m`, `size_n`, `size_k`, `log_size_m`, `log_size_n`, `log_size_k`, `kernel_family_onehot_reduce`, `kernel_family_onehot_splitk_reduce`, `kernel_family_onehot_cast`, `kernel_family_onehot_copy` ✓

## Headline: aggregate layer-total error per (GPU × held-out model)

| GPU | held-out model | sum(pred ms) | sum(meas ms) | abs err % |
|---|---|---:|---:|---:|

## Held-out per-family MAPE (full-train)

| GPU | family | n_train | n_test | MAPE |
|---|---|---:|---:|---:|
| H100 | gemm | 9092 | 0 | no-heldout (train only) |
| H100 | flash_attn | 234 | 0 | no-heldout (train only) |
| H100 | elementwise | 5530 | 0 | no-heldout (train only) |
| H100 | misc | 3291 | 0 | no-heldout (train only) |

## Leave-one-model-out CV

### H100

| held-out | gemm | flash_attn | elementwise | misc |
|---|---:|---:|---:|---:|
| Gemma-9B | 137.3% (n=528) | 51.5% (n=152) | 14.2% (n=2726) | 14.2% (n=1593) |
| Granite-8B | 110.2% (n=550) | 6.3% (n=78) | 10.9% (n=1892) | 7.1% (n=1026) |
| Llama-8B | 90.5% (n=32) | 5.1% (n=4) | 8.0% (n=108) | 7.6% (n=68) |
| gpt-oss-20b | 112.0% (n=152) | n/a | 26.1% (n=804) | 43.3% (n=604) |
| roofline | 137.1% (n=7830) | n/a | n/a | n/a |

## Saved pkls
- **H100**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_kernel/perkernel_flash_attn_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_kernel/perkernel_elementwise_shape_v2.pkl`
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_kernel/perkernel_misc_shape_v2.pkl`
