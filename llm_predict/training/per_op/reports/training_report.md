# Per-Op Training Report (per_op_v5)

- features (26): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 6396 | 6228 | Llama-8B, Mixtral, gpt-oss-20b | Llama-3.3-70B, Llama-70B, Qwen-72B | 38.66% |
| RTX3090 | 285 | 152 | Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Llama-70B, Qwen-72B | 26.58% |
| RTX2080Ti | 133 | 0 | Llama-8B, Qwen3.5-9B | — | — |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 35.54% |
| Llama-70B | 36.17% |
| Qwen-72B | 44.05% |

### RTX3090

| held-out model | MAPE |
|---|---:|
| Llama-70B | 26.64% |
| Qwen-72B | 26.53% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 4264 | 2132 | 19.28% |
| Mixtral | 4264 | 2132 | 17.36% |
| gpt-oss-20b | 4264 | 2132 | 52.09% |

### RTX3090

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 209 | 76 | 21.11% |
| Mixtral | 209 | 76 | 21.78% |
| Qwen3.5-9B | 228 | 57 | 18.65% |
| gpt-oss-20b | 209 | 76 | 43.76% |

### RTX2080Ti

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 57 | 76 | 24.89% |
| Qwen3.5-9B | 76 | 57 | 15.36% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
- **RTX3090**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_op/perop_v5_shape.pkl`
- **RTX2080Ti**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_op/perop_v5_shape.pkl`
