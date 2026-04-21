# Per-Op Labeling Report

- total rows: **133**
- GPUs: ['RTX2080Ti']
- models: ['Llama-8B', 'Qwen3.5-9B']
- ops: ['attn', 'ffn', 'norm_post', 'norm_pre']

## Rows per (gpu × model × op)

| gpu | model | held_out | op | n_rows |
|---|---|---|---|---:|
| RTX2080Ti | Llama-8B | False | attn | 19 |
| RTX2080Ti | Llama-8B | False | ffn | 19 |
| RTX2080Ti | Llama-8B | False | norm_post | 19 |
| RTX2080Ti | Llama-8B | False | norm_pre | 19 |
| RTX2080Ti | Qwen3.5-9B | False | ffn | 19 |
| RTX2080Ti | Qwen3.5-9B | False | norm_post | 19 |
| RTX2080Ti | Qwen3.5-9B | False | norm_pre | 19 |