# RTX3090 — serving_e2e Validation: chat-long

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: RTX3090 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 218 | 288 | 1 | 61.83 | 41.74 | 48.1% | 32.86 | 19.91 | 65.1% | 9525.35 | 5582.54 | 70.6% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 48.13% | 1 |
| TPOT | 65.06% | 1 |
| E2EL | 70.63% | 1 |
