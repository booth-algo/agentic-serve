# Phase A1 Shape Labeling Report

- Output: `/root/agentic-serve/llm_predict/training/per_kernel/data/kernels_labeled.csv`
- Total labeled rows: 125024
- GPUs: A100, RTX3090, RTX2080Ti
- Training rows (held_out=False): 108279
- Held-out rows (held_out=True): 16745

## Counts by GPU
{'A100': 56970, 'RTX2080Ti': 15098, 'RTX3090': 52956}

## Counts by kernel_family
{'cast': 41095, 'copy': 4706, 'elementwise': 53211, 'flash_attn': 624, 'gemm': 9743, 'reduce': 13365, 'splitk_reduce': 2280}

## Counts by GPU × family
kernel_family   cast  copy  elementwise  flash_attn  gemm  reduce  splitk_reduce
gpu                                                                             
A100           17814  2437        24314         352  4729    5857           1467
RTX2080Ti       5800   180         6373           0   868    1813             64
RTX3090        17481  2089        22524         272  4146    5695            749

## Per-CSV details
### A100 / Llama-3.1-70B-Instruct  (held_out=True)
- path: dense_layer_walker
- flash_count: 80
- ncu rows: 3382
- labeled: 3381 {'elementwise': 1769, 'gemm': 562, 'cast': 326, 'copy': 322, 'reduce': 162, 'splitk_reduce': 160, 'flash_attn': 80}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 562, 'labeled_by_shape': 0, 'dropped': 0}

### A100 / Llama-3.1-8B-Instruct  (held_out=False)
- path: dense_layer_walker
- flash_count: 32
- ncu rows: 1462
- labeled: 1461 {'elementwise': 713, 'gemm': 226, 'splitk_reduce': 160, 'cast': 134, 'copy': 130, 'reduce': 66, 'flash_attn': 32}
- gemm stats: {'total_gemm': 226, 'labeled_by_position': 226, 'labeled_by_shape': 0, 'dropped': 0}

### A100 / Llama-3.3-70B-Instruct  (held_out=True)
- path: dense_layer_walker
- flash_count: 80
- ncu rows: 3382
- labeled: 3381 {'elementwise': 1769, 'gemm': 562, 'cast': 326, 'copy': 322, 'reduce': 162, 'splitk_reduce': 160, 'flash_attn': 80}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 562, 'labeled_by_shape': 0, 'dropped': 0}

### A100 / Mixtral-8x7B-Instruct  (held_out=False)
- path: moe_layer_walker
- flash_count: 32
- ncu rows: 5894
- labeled: 4665 {'elementwise': 1562, 'copy': 1070, 'gemm': 915, 'splitk_reduce': 475, 'cast': 449, 'reduce': 162, 'flash_attn': 32}
- gemm stats: {'total_gemm': 915, 'labeled_by_position': 132, 'labeled_by_shape': 783, 'dropped': 0}

### A100 / Qwen2.5-72B-Instruct  (held_out=True)
- path: dense_layer_walker
- flash_count: 80
- ncu rows: 3382
- labeled: 3381 {'elementwise': 1769, 'gemm': 562, 'cast': 326, 'copy': 322, 'reduce': 162, 'splitk_reduce': 160, 'flash_attn': 80}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 562, 'labeled_by_shape': 0, 'dropped': 0}

### A100 / Qwen3.5-27B  (held_out=False)
- path: dense_layer_walker
- flash_count: 32
- ncu rows: 26206
- labeled: 25917 {'cast': 10698, 'elementwise': 10508, 'reduce': 3331, 'gemm': 1122, 'splitk_reduce': 128, 'copy': 98, 'flash_attn': 32}
- gemm stats: {'total_gemm': 1122, 'labeled_by_position': 113, 'labeled_by_shape': 1009, 'dropped': 0}

### A100 / Qwen3.5-9B  (held_out=False)
- path: dense_layer_walker
- flash_count: 16
- ncu rows: 13206
- labeled: 13061 {'cast': 5354, 'elementwise': 5260, 'reduce': 1667, 'gemm': 562, 'splitk_reduce': 152, 'copy': 50, 'flash_attn': 16}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 57, 'labeled_by_shape': 505, 'dropped': 0}

### A100 / gpt-oss-20b  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 1778
- labeled: 1723 {'elementwise': 964, 'gemm': 218, 'cast': 201, 'reduce': 145, 'copy': 123, 'splitk_reduce': 72}
- gemm stats: {'total_gemm': 218, 'labeled_by_position': 1, 'labeled_by_shape': 217, 'dropped': 0}

### RTX3090 / Llama-3.1-70B-Instruct  (held_out=True)
- path: dense_layer_walker
- flash_count: 80
- ncu rows: 3382
- labeled: 3381 {'elementwise': 1769, 'gemm': 562, 'cast': 326, 'copy': 322, 'reduce': 162, 'splitk_reduce': 160, 'flash_attn': 80}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 562, 'labeled_by_shape': 0, 'dropped': 0}

### RTX3090 / Llama-3.1-8B-Instruct  (held_out=False)
- path: dense_layer_walker
- flash_count: 32
- ncu rows: 1366
- labeled: 1365 {'elementwise': 713, 'gemm': 226, 'cast': 134, 'copy': 130, 'reduce': 66, 'splitk_reduce': 64, 'flash_attn': 32}
- gemm stats: {'total_gemm': 226, 'labeled_by_position': 226, 'labeled_by_shape': 0, 'dropped': 0}

### RTX3090 / Mixtral-8x7B-Instruct  (held_out=False)
- path: moe_layer_walker
- flash_count: 32
- ncu rows: 5575
- labeled: 4376 {'elementwise': 1541, 'copy': 1044, 'gemm': 894, 'cast': 442, 'splitk_reduce': 261, 'reduce': 162, 'flash_attn': 32}
- gemm stats: {'total_gemm': 894, 'labeled_by_position': 132, 'labeled_by_shape': 762, 'dropped': 0}

### RTX3090 / Qwen2.5-72B-Instruct  (held_out=True)
- path: dense_layer_walker
- flash_count: 80
- ncu rows: 3222
- labeled: 3221 {'elementwise': 1769, 'gemm': 562, 'cast': 326, 'copy': 322, 'reduce': 162, 'flash_attn': 80}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 562, 'labeled_by_shape': 0, 'dropped': 0}

### RTX3090 / Qwen3.5-27B  (held_out=False)
- path: dense_layer_walker
- flash_count: 32
- ncu rows: 26206
- labeled: 25917 {'cast': 10698, 'elementwise': 10508, 'reduce': 3331, 'gemm': 1122, 'splitk_reduce': 128, 'copy': 98, 'flash_attn': 32}
- gemm stats: {'total_gemm': 1122, 'labeled_by_position': 114, 'labeled_by_shape': 1008, 'dropped': 0}

### RTX3090 / Qwen3.5-9B  (held_out=False)
- path: dense_layer_walker
- flash_count: 16
- ncu rows: 13118
- labeled: 12973 {'cast': 5354, 'elementwise': 5260, 'reduce': 1667, 'gemm': 562, 'splitk_reduce': 64, 'copy': 50, 'flash_attn': 16}
- gemm stats: {'total_gemm': 562, 'labeled_by_position': 58, 'labeled_by_shape': 504, 'dropped': 0}

### RTX3090 / gpt-oss-20b  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 1778
- labeled: 1723 {'elementwise': 964, 'gemm': 218, 'cast': 201, 'reduce': 145, 'copy': 123, 'splitk_reduce': 72}
- gemm stats: {'total_gemm': 218, 'labeled_by_position': 1, 'labeled_by_shape': 217, 'dropped': 0}

### RTX2080Ti / Llama-3.1-8B-Instruct  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 2070
- labeled: 2037 {'elementwise': 1033, 'cast': 390, 'gemm': 290, 'copy': 130, 'reduce': 130, 'splitk_reduce': 64}
- gemm stats: {'total_gemm': 290, 'labeled_by_position': 1, 'labeled_by_shape': 289, 'dropped': 0}

### RTX2080Ti / Qwen3.5-9B  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 13214
- labeled: 13061 {'cast': 5410, 'elementwise': 5340, 'reduce': 1683, 'gemm': 578, 'copy': 50}
- gemm stats: {'total_gemm': 578, 'labeled_by_position': 1, 'labeled_by_shape': 577, 'dropped': 0}
