# RTX2080Ti — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX2080Ti perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 113 | 158 | 1 | 279.36 | 69.11 | 304.2% | 63.87 | 32.72 | 95.2% | 10370.23 | 5266.87 | 96.9% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 304.24% | 1 |
| TPOT | 95.17% | 1 |
| E2EL | 96.90% | 1 |
