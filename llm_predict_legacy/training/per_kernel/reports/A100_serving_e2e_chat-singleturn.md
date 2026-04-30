# A100 — serving_e2e Validation: chat-singleturn

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-singleturn` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 258 | 299 | 1 | 13.20 | 33.67 | 60.8% | 4.74 | 4.94 | 4.0% | 1431.89 | 1399.07 | 2.3% |
| Llama-8B | supported | vllm | 218 | 285 | 1 | 33.89 | 34.62 | 2.1% | 12.73 | 12.95 | 1.7% | 3662.83 | 3633.09 | 0.8% |
| Qwen3.5-9B | hybrid_attn | vllm | 206 | 300 | 1 | 19.19 | 87.58 | 78.1% | 22.42 | 13.78 | 62.6% | 6744.04 | 3920.65 | 72.0% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 2.09% | 1 |
| TPOT | 1.66% | 1 |
| E2EL | 0.82% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
