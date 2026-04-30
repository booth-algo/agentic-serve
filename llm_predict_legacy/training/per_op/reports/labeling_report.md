# Per-Op Labeling Report

- total rows: **3283**
- GPUs: ['A100', 'RTX2080Ti', 'RTX3090']
- models: ['Llama-3.3-70B', 'Llama-70B', 'Llama-8B', 'Mixtral', 'Qwen-72B', 'Qwen-7B', 'Qwen3.5-9B', 'gpt-oss-20b']
- ops: ['attn', 'ffn', 'norm_post', 'norm_pre']

## Rows per (gpu × model × op)

| gpu | model | held_out | op | n_rows |
|---|---|---|---|---:|
| A100 | Llama-3.3-70B | True | attn | 19 |
| A100 | Llama-3.3-70B | True | ffn | 19 |
| A100 | Llama-3.3-70B | True | norm_post | 19 |
| A100 | Llama-3.3-70B | True | norm_pre | 19 |
| A100 | Llama-70B | False | attn | 28 |
| A100 | Llama-70B | False | ffn | 28 |
| A100 | Llama-70B | False | norm_post | 28 |
| A100 | Llama-70B | False | norm_pre | 28 |
| A100 | Llama-8B | False | attn | 28 |
| A100 | Llama-8B | False | ffn | 28 |
| A100 | Llama-8B | False | norm_post | 28 |
| A100 | Llama-8B | False | norm_pre | 28 |
| A100 | Mixtral | False | attn | 28 |
| A100 | Mixtral | False | ffn | 28 |
| A100 | Mixtral | False | norm_post | 28 |
| A100 | Mixtral | False | norm_pre | 28 |
| A100 | Qwen-72B | True | attn | 19 |
| A100 | Qwen-72B | True | ffn | 19 |
| A100 | Qwen-72B | True | norm_post | 19 |
| A100 | Qwen-72B | True | norm_pre | 19 |
| A100 | Qwen3.5-9B | False | ffn | 27 |
| A100 | Qwen3.5-9B | False | norm_post | 27 |
| A100 | Qwen3.5-9B | False | norm_pre | 27 |
| A100 | gpt-oss-20b | False | attn | 24 |
| A100 | gpt-oss-20b | False | ffn | 24 |
| A100 | gpt-oss-20b | False | norm_post | 24 |
| A100 | gpt-oss-20b | False | norm_pre | 24 |
| RTX2080Ti | Llama-8B | False | attn | 22 |
| RTX2080Ti | Llama-8B | False | ffn | 22 |
| RTX2080Ti | Llama-8B | False | norm_post | 22 |
| RTX2080Ti | Llama-8B | False | norm_pre | 22 |
| RTX2080Ti | Qwen-7B | True | attn | 19 |
| RTX2080Ti | Qwen-7B | True | ffn | 19 |
| RTX2080Ti | Qwen-7B | True | norm_post | 19 |
| RTX2080Ti | Qwen-7B | True | norm_pre | 19 |
| RTX2080Ti | Qwen3.5-9B | False | ffn | 15 |
| RTX2080Ti | Qwen3.5-9B | False | norm_post | 15 |
| RTX2080Ti | Qwen3.5-9B | False | norm_pre | 15 |
| RTX3090 | Llama-70B | False | attn | 512 |
| RTX3090 | Llama-70B | False | ffn | 512 |
| RTX3090 | Llama-70B | False | norm_post | 512 |
| RTX3090 | Llama-70B | False | norm_pre | 512 |
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