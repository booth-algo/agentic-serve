# Per-Op Training Report (per_op_v5)

- features (26): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 285 | 228 | Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Llama-3.3-70B, Llama-70B, Qwen-72B | 31.79% |
| RTX3090 | 285 | 152 | Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Llama-70B, Qwen-72B | 32.84% |
| RTX2080Ti | 133 | 0 | Llama-8B, Qwen3.5-9B | — | — |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 31.56% |
| Llama-70B | 31.26% |
| Qwen-72B | 32.54% |

### RTX3090

| held-out model | MAPE |
|---|---:|
| Llama-70B | 32.11% |
| Qwen-72B | 33.58% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 209 | 76 | 15.55% |
| Mixtral | 209 | 76 | 19.32% |
| Qwen3.5-9B | 228 | 57 | 15.48% |
| gpt-oss-20b | 209 | 76 | 46.40% |

### RTX3090

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 209 | 76 | 15.86% |
| Mixtral | 209 | 76 | 21.27% |
| Qwen3.5-9B | 228 | 57 | 16.81% |
| gpt-oss-20b | 209 | 76 | 46.14% |

### RTX2080Ti

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 57 | 76 | 25.20% |
| Qwen3.5-9B | 76 | 57 | 13.04% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
- **RTX3090**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_op/perop_v5_shape.pkl`
- **RTX2080Ti**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_op/perop_v5_shape.pkl`
