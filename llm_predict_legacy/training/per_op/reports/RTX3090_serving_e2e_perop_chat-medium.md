# RTX3090 — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX3090 perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 120 | 238 | 1 | 81.44 | 35.48 | 129.5% | 70.37 | 19.86 | 254.3% | 16828.94 | 4596.41 | 266.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 129.51% | 1 |
| TPOT | 254.35% | 1 |
| E2EL | 266.13% | 1 |
