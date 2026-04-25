# RTX2080Ti — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX2080Ti perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 120 | 254 | 1 | 348.16 | 60.43 | 476.2% | 62.39 | 32.89 | 89.7% | 16194.86 | 7974.15 | 103.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 476.18% | 1 |
| TPOT | 89.67% | 1 |
| E2EL | 103.09% | 1 |
