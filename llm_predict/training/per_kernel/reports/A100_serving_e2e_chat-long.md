# A100 — serving_e2e Validation: chat-long

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 258 | 299 | 1 | 13.20 | 33.67 | 60.8% | 3.17 | 4.94 | 35.9% | 960.89 | 1399.07 | 31.3% |
| Llama-8B | supported | vllm | 218 | 285 | 1 | 33.89 | 34.62 | 2.1% | 8.51 | 12.95 | 34.3% | 2458.03 | 3633.09 | 32.3% |
| Qwen3.5-9B | hybrid_attn | vllm | 206 | 300 | 1 | 19.19 | 87.58 | 78.1% | 14.97 | 13.78 | 8.6% | 4511.41 | 3920.65 | 15.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 2.09% | 1 |
| TPOT | 34.31% | 1 |
| E2EL | 32.34% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
