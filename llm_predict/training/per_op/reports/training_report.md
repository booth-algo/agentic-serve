# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| RTX3090 | 10210 | 76 | Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Qwen-72B | 42.27% |

## Per-heldout-model MAPE

### RTX3090

| held-out model | MAPE |
|---|---:|
| Qwen-72B | 42.27% |

## LOMO cross-validation

### RTX3090

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 8054 | 2156 | 49.39% |
| Llama-8B | 8050 | 2160 | 32.92% |
| Mixtral | 8054 | 2156 | 19.82% |
| Qwen3.5-9B | 8608 | 1602 | 23.75% |
| gpt-oss-20b | 8074 | 2136 | 45.21% |

## Saved pkls
- **RTX3090**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/RTX3090/trained/per_op/perop_v5_shape.pkl`
