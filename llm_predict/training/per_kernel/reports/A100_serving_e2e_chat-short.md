# A100 — serving_e2e Validation: chat-short

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 150 | 171 | 1 | 11.93 | 32.99 | 63.8% | 3.16 | 4.93 | 36.0% | 551.56 | 979.27 | 43.7% |
| Llama-8B | supported | vllm | 127 | 149 | 1 | 35.25 | 35.18 | 0.2% | 8.46 | 12.89 | 34.4% | 1295.48 | 1993.33 | 35.0% |
| Qwen3.5-9B | hybrid_attn | vllm | 95 | 172 | 1 | 29.66 | 89.43 | 66.8% | 14.91 | 13.76 | 8.4% | 2594.91 | 2752.05 | 5.7% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 0.20% | 1 |
| TPOT | 34.38% | 1 |
| E2EL | 35.01% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
