# RTX3090 — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: RTX3090 perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 127 | 150 | 1 | 81.65 | 37.93 | 115.3% | 70.24 | 19.75 | 255.6% | 10617.66 | 3041.37 | 249.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 115.29% | 1 |
| TPOT | 255.57% | 1 |
| E2EL | 249.11% | 1 |
