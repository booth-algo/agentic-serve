# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| H100 | 5932 | 0 | Llama-8B, Qwen3.5-9B, gpt-oss-20b | — | — |

## Per-heldout-model MAPE

## LOMO cross-validation

### H100

| held-out | n_train | n_test | MAPE |
|---|---:|---:|---:|
| Llama-8B | 3772 | 2160 | 51.41% |
| Qwen3.5-9B | 4312 | 1620 | 34.92% |
| gpt-oss-20b | 3780 | 2152 | 44.54% |

## Saved pkls
- **H100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_op/perop_v5_shape.pkl`
