# Per-Op Labeling Report

- total rows: **12624**
- GPUs: ['A100']
- models: ['Llama-3.3-70B', 'Llama-70B', 'Llama-8B', 'Mixtral', 'Qwen-72B', 'gpt-oss-20b']
- ops: ['attn', 'ffn', 'norm_post', 'norm_pre']

## Rows per (gpu × model × op)

| gpu | model | held_out | op | n_rows |
|---|---|---|---|---:|
| A100 | Llama-3.3-70B | True | attn | 512 |
| A100 | Llama-3.3-70B | True | ffn | 512 |
| A100 | Llama-3.3-70B | True | norm_post | 512 |
| A100 | Llama-3.3-70B | True | norm_pre | 512 |
| A100 | Llama-70B | True | attn | 512 |
| A100 | Llama-70B | True | ffn | 512 |
| A100 | Llama-70B | True | norm_post | 512 |
| A100 | Llama-70B | True | norm_pre | 512 |
| A100 | Llama-8B | False | attn | 533 |
| A100 | Llama-8B | False | ffn | 533 |
| A100 | Llama-8B | False | norm_post | 533 |
| A100 | Llama-8B | False | norm_pre | 533 |
| A100 | Mixtral | False | attn | 533 |
| A100 | Mixtral | False | ffn | 533 |
| A100 | Mixtral | False | norm_post | 533 |
| A100 | Mixtral | False | norm_pre | 533 |
| A100 | Qwen-72B | True | attn | 533 |
| A100 | Qwen-72B | True | ffn | 533 |
| A100 | Qwen-72B | True | norm_post | 533 |
| A100 | Qwen-72B | True | norm_pre | 533 |
| A100 | gpt-oss-20b | False | attn | 533 |
| A100 | gpt-oss-20b | False | ffn | 533 |
| A100 | gpt-oss-20b | False | norm_post | 533 |
| A100 | gpt-oss-20b | False | norm_pre | 533 |