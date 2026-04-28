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
| H100 | gemm | 3570 | 0 | no-heldout (train only) |
| H100 | flash_attn | 0 | 0 | skipped (n_train=0 < 5) |
| H100 | elementwise | 0 | 0 | skipped (n_train=0 < 5) |
| H100 | misc | 0 | 0 | skipped (n_train=0 < 5) |

## Leave-one-model-out CV

### H100 — skipped (<2 training models)

## Saved pkls
- **H100**
  - `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_kernel/perkernel_gemm_shape_v2.pkl`
