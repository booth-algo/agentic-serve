# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 8305 | 152 | Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Llama-3.3-70B, Qwen-72B | 57.73% |
| RTX2080Ti | 133 | 76 | Llama-8B, Qwen3.5-9B | Qwen-7B | 96.59% |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 57.57% |
| Qwen-72B | 57.88% |

### RTX2080Ti

| held-out model | MAPE |
|---|---:|
| Qwen-7B | 96.59% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 8081 | 224 | 61.59% |
| Llama-8B | 6145 | 2160 | 22.41% |
| Mixtral | 6145 | 2160 | 14.31% |
| Qwen3.5-9B | 6688 | 1617 | 20.14% |
| gpt-oss-20b | 6161 | 2144 | 53.50% |

### RTX2080Ti

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 45 | 88 | 124.09% |
| Qwen3.5-9B | 88 | 45 | 79.80% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
- **RTX2080Ti**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX2080Ti/trained/per_op/perop_v5_shape.pkl`
