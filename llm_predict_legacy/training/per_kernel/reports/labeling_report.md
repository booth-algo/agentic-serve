# Phase A1 Shape Labeling Report

- Output: `/tmp/h100_labeled_v2.csv`
- Total labeled rows: 10327
- GPUs: H100
- Training rows (held_out=False): 10327
- Held-out rows (held_out=True): 0

## Counts by GPU
{'H100': 10327}

## Counts by kernel_family
{'cast': 1483, 'copy': 893, 'elementwise': 5530, 'flash_attn': 234, 'gemm': 1262, 'reduce': 571, 'splitk_reduce': 354}

## Counts by GPU × family
kernel_family  cast  copy  elementwise  flash_attn  gemm  reduce  splitk_reduce
gpu                                                                            
H100           1483   893         5530         234  1262     571            354

## Per-CSV details
### H100 / Llama-3.1-8B-Instruct  (held_out=False)
- path: dense_layer_walker
- flash_count: 4
- ncu rows: 215
- labeled: 212 {'elementwise': np.int64(108), 'gemm': np.int64(32), 'cast': np.int64(26), 'copy': np.int64(20), 'splitk_reduce': np.int64(12), 'reduce': np.int64(10), 'flash_attn': np.int64(4)}
- gemm stats: {'total_gemm': 32, 'labeled_by_position': 29, 'labeled_by_shape': 3, 'dropped': 0}

### H100 / Mixtral-8x7B-Instruct  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 10
- labeled: 10 {'copy': np.int64(10)}
- gemm stats: {'total_gemm': 0, 'labeled_by_position': 0, 'labeled_by_shape': 0, 'dropped': 0}

### H100 / gemma-2-9b-it  (held_out=False)
- path: dense_layer_walker
- flash_count: 152
- ncu rows: 5000
- labeled: 4999 {'elementwise': np.int64(2726), 'cast': np.int64(910), 'gemm': np.int64(528), 'copy': np.int64(305), 'reduce': np.int64(303), 'flash_attn': np.int64(152), 'splitk_reduce': np.int64(75)}
- gemm stats: {'total_gemm': 528, 'labeled_by_position': 526, 'labeled_by_shape': 2, 'dropped': 0}

### H100 / gpt-oss-20b  (held_out=False)
- path: noflash_fallback
- flash_count: 0
- ncu rows: 1776
- labeled: 1560 {'elementwise': np.int64(804), 'copy': np.int64(244), 'cast': np.int64(228), 'gemm': np.int64(152), 'reduce': np.int64(100), 'splitk_reduce': np.int64(32)}
- gemm stats: {'total_gemm': 152, 'labeled_by_position': 0, 'labeled_by_shape': 152, 'dropped': 0}

### H100 / granite-3.0-8b-instruct  (held_out=False)
- path: dense_layer_walker
- flash_count: 78
- ncu rows: 3547
- labeled: 3546 {'elementwise': np.int64(1892), 'gemm': np.int64(550), 'cast': np.int64(319), 'copy': np.int64(314), 'splitk_reduce': np.int64(235), 'reduce': np.int64(158), 'flash_attn': np.int64(78)}
- gemm stats: {'total_gemm': 550, 'labeled_by_position': 548, 'labeled_by_shape': 2, 'dropped': 0}
