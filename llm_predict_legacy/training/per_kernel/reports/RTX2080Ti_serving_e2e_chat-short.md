# RTX2080Ti — serving_e2e Validation: chat-short

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: RTX2080Ti pkls (['decode_attn', 'elementwise', 'gemm', 'misc'])
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 113 | 158 | 1 | 40.63 | 69.11 | 41.2% | 32.88 | 32.72 | 0.5% | 5236.02 | 5266.87 | 0.6% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 41.21% | 1 |
| TPOT | 0.48% | 1 |
| E2EL | 0.59% | 1 |
