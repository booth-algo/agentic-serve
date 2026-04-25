# A100 — serving_e2e Validation: chat-short

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 150 | 171 | 1 | 11.93 | 32.99 | 63.8% | 4.72 | 4.93 | 4.2% | 819.75 | 979.27 | 16.3% |
| Llama-8B | supported | vllm | 127 | 149 | 1 | 35.25 | 35.18 | 0.2% | 12.66 | 12.89 | 1.8% | 1921.82 | 1993.33 | 3.6% |
| Qwen3.5-9B | hybrid_attn | vllm | 95 | 172 | 1 | 29.66 | 89.43 | 66.8% | 22.33 | 13.76 | 62.2% | 3869.84 | 2752.05 | 40.6% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 0.20% | 1 |
| TPOT | 1.77% | 1 |
| E2EL | 3.59% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
