# Per-Op Labeling Report

- total rows: **570**
- GPUs: ['RTX2080Ti', 'RTX3090']
- models: ['Llama-70B', 'Llama-8B', 'Mixtral', 'Qwen-72B', 'Qwen3.5-9B', 'gpt-oss-20b']
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
| RTX3090 | Llama-70B | True | attn | 19 |
| RTX3090 | Llama-70B | True | ffn | 19 |
| RTX3090 | Llama-70B | True | norm_post | 19 |
| RTX3090 | Llama-70B | True | norm_pre | 19 |
| RTX3090 | Llama-8B | False | attn | 19 |
| RTX3090 | Llama-8B | False | ffn | 19 |
| RTX3090 | Llama-8B | False | norm_post | 19 |
| RTX3090 | Llama-8B | False | norm_pre | 19 |
| RTX3090 | Mixtral | False | attn | 19 |
| RTX3090 | Mixtral | False | ffn | 19 |
| RTX3090 | Mixtral | False | norm_post | 19 |
| RTX3090 | Mixtral | False | norm_pre | 19 |
| RTX3090 | Qwen-72B | True | attn | 19 |
| RTX3090 | Qwen-72B | True | ffn | 19 |
| RTX3090 | Qwen-72B | True | norm_post | 19 |
| RTX3090 | Qwen-72B | True | norm_pre | 19 |
| RTX3090 | Qwen3.5-9B | False | ffn | 19 |
| RTX3090 | Qwen3.5-9B | False | norm_post | 19 |
| RTX3090 | Qwen3.5-9B | False | norm_pre | 19 |
| RTX3090 | gpt-oss-20b | False | attn | 19 |
| RTX3090 | gpt-oss-20b | False | ffn | 19 |
| RTX3090 | gpt-oss-20b | False | norm_post | 19 |
| RTX3090 | gpt-oss-20b | False | norm_pre | 19 |