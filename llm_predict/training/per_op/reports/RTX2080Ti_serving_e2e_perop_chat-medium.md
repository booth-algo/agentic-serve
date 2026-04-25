# RTX2080Ti — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX2080Ti perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 120 | 254 | 1 | 279.36 | 60.43 | 362.3% | 63.79 | 32.89 | 93.9% | 16482.71 | 7974.15 | 106.7% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 362.31% | 1 |
| TPOT | 93.94% | 1 |
| E2EL | 106.70% | 1 |
