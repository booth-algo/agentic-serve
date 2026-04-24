# Per-Op Training Report (per_op_v5)

- features (26): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 6472 | 4180 | Llama-70B, Llama-8B, Mixtral, gpt-oss-20b | Llama-3.3-70B, Qwen-72B | 30.68% |
| RTX3090 | 361 | 76 | Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Qwen-72B | 13.95% |
| RTX2080Ti | 133 | 76 | Llama-8B, Qwen3.5-9B | Qwen-7B | 17.15% |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 29.17% |
| Qwen-72B | 32.13% |

### RTX3090

| held-out model | MAPE |
|---|---:|
| Qwen-72B | 13.95% |

### RTX2080Ti

| held-out model | MAPE |
|---|---:|
| Qwen-7B | 17.15% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 6396 | 76 | 37.61% |
| Llama-8B | 4340 | 2132 | 17.96% |
| Mixtral | 4340 | 2132 | 17.33% |
| gpt-oss-20b | 4340 | 2132 | 42.75% |

### RTX3090

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 285 | 76 | 26.64% |
| Llama-8B | 285 | 76 | 17.60% |
| Mixtral | 285 | 76 | 19.30% |
| Qwen3.5-9B | 304 | 57 | 17.39% |
| gpt-oss-20b | 285 | 76 | 42.46% |

### RTX2080Ti

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 57 | 76 | 24.89% |
| Qwen3.5-9B | 76 | 57 | 15.36% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
- **RTX3090**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_op/perop_v5_shape.pkl`
- **RTX2080Ti**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_op/perop_v5_shape.pkl`
