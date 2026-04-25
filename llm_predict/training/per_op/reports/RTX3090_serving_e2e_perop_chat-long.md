# RTX3090 — per-op serving_e2e Validation: chat-long

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX3090 perop_v5_shape.pkl
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 218 | 288 | 1 | 101.61 | 41.74 | 143.4% | 70.52 | 19.91 | 254.3% | 20412.79 | 5582.54 | 265.7% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 143.42% | 1 |
| TPOT | 254.26% | 1 |
| E2EL | 265.65% | 1 |
