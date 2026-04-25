# RTX3090 — serving_e2e Validation: chat-medium

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: RTX3090 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 120 | 238 | 1 | 53.67 | 35.48 | 51.3% | 32.80 | 19.86 | 65.2% | 7859.67 | 4596.41 | 71.0% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 51.26% | 1 |
| TPOT | 65.16% | 1 |
| E2EL | 71.00% | 1 |
