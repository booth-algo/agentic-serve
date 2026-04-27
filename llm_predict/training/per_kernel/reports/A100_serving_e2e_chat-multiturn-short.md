# A100 — serving_e2e Validation: chat-multiturn-short

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-multiturn-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | vllm | 673 | 297 | 1 | 54.89 | 52.97 | 3.6% | 12.82 | 13.04 | 1.7% | 3863.65 | 3815.76 | 1.3% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 3.62% | 1 |
| TPOT | 1.68% | 1 |
| E2EL | 1.26% | 1 |
