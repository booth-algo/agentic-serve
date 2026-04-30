# RTX2080Ti — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX2080Ti perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 113 | 158 | 1 | 348.16 | 69.11 | 403.8% | 62.51 | 32.72 | 91.0% | 10224.30 | 5266.87 | 94.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 403.80% | 1 |
| TPOT | 91.02% | 1 |
| E2EL | 94.12% | 1 |
