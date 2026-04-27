# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 10465 | 152 | Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, Yi-34B, gpt-oss-20b | Llama-3.3-70B, Qwen-72B | 42.18% |
| RTX3090 | 14518 | 76 | Gemma-9B, Granite-8B, Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Qwen-72B | 40.11% |
| RTX2080Ti | 133 | 76 | Llama-8B, Qwen3.5-9B | Qwen-7B | 96.59% |
| H100 | 5932 | 0 | Llama-8B, Qwen3.5-9B, gpt-oss-20b | — | — |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 41.90% |
| Qwen-72B | 42.46% |

### RTX3090

| held-out model | MAPE |
|---|---:|
| Qwen-72B | 40.11% |

### RTX2080Ti

| held-out model | MAPE |
|---|---:|
| Qwen-7B | 96.59% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 10241 | 224 | 41.12% |
| Llama-8B | 8305 | 2160 | 24.87% |
| Mixtral | 8305 | 2160 | 19.60% |
| Qwen3.5-9B | 8848 | 1617 | 21.36% |
| Yi-34B | 8305 | 2160 | 45.96% |
| gpt-oss-20b | 8321 | 2144 | 55.36% |

### RTX3090

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Gemma-9B | 12370 | 2148 | 29.97% |
| Granite-8B | 12358 | 2160 | 18.46% |
| Llama-70B | 12362 | 2156 | 51.71% |
| Llama-8B | 12358 | 2160 | 16.85% |
| Mixtral | 12362 | 2156 | 19.57% |
| Qwen3.5-9B | 12916 | 1602 | 21.48% |
| gpt-oss-20b | 12382 | 2136 | 44.01% |

### RTX2080Ti

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 45 | 88 | 124.09% |
| Qwen3.5-9B | 88 | 45 | 79.80% |

### H100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 3772 | 2160 | 51.41% |
| Qwen3.5-9B | 4312 | 1620 | 34.92% |
| gpt-oss-20b | 3780 | 2152 | 44.54% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
- **RTX3090**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_op/perop_v5_shape.pkl`
- **RTX2080Ti**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_op/perop_v5_shape.pkl`
- **H100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_op/perop_v5_shape.pkl`
