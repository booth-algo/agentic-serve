# Per-Op Training Report (per_op_v5)

- features (28): `tok`, `log_tok`, `flops`, `log_flops`, `wt_bytes`, `log_wt`, `ai`, `log_ai`, `is_attn`, `is_ffn`, `is_norm`, `d`, `ffn`, `E`, `k`, `d2`, `d_ffn`, `eff_tok`, `a_exp`, `tot_wt`, `bs`, `seq`, `log_bs`, `log_seq`, `attn_quad`, `log_attn_quad`, `kv_cache_len`, `log_kv_cache_len` ✓

## Headline: held-out MAPE per GPU

| GPU | n_train | n_heldout | pool | heldout | MAPE |
|---|---:|---:|---|---|---:|
| H100 | 2160 | 0 | Llama-8B | — | — |

## Per-heldout-model MAPE

## LOMO cross-validation

### H100 — skipped (<2 pool models)

## Saved pkls
- **H100**: `/home/kevinlau/agentic-serve/llm_predict/profiling/data/H100/trained/per_op/perop_v5_shape.pkl`
