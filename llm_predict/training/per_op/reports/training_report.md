# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| A100 | 8957 | 4180 | Llama-70B, Llama-8B, Mixtral, Qwen3.5-9B, gpt-oss-20b | Llama-3.3-70B, Qwen-72B | 26.64% |

## Per-heldout-model MAPE

### A100

| held-out model | MAPE |
|---|---:|
| Llama-3.3-70B | 15.14% |
| Qwen-72B | 37.68% |

## LOMO cross-validation

### A100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-70B | 6797 | 2160 | 42.82% |
| Llama-8B | 6713 | 2244 | 18.74% |
| Mixtral | 6713 | 2244 | 30.50% |
| Qwen3.5-9B | 8876 | 81 | 43.31% |
| gpt-oss-20b | 6729 | 2228 | 51.70% |

## Saved pkls
- **A100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/A100/trained/per_op/perop_v5_shape.pkl`
