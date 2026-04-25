# A100 — serving_e2e Validation: chat-medium

- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)
- Predictor: A100 pkls (['elementwise', 'flash_attn', 'gemm', 'misc'])
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 159 | 270 | 1 | 12.02 | 33.49 | 64.1% | 3.16 | 4.94 | 36.0% | 866.44 | 1336.96 | 35.2% |
| Llama-8B | supported | vllm | 120 | 257 | 1 | 34.76 | 34.81 | 0.1% | 8.49 | 12.95 | 34.4% | 2216.18 | 3203.68 | 30.8% |
| Qwen3.5-9B | hybrid_attn | vllm | 103 | 271 | 1 | 30.52 | 89.55 | 65.9% | 14.95 | 13.78 | 8.5% | 4082.93 | 3594.96 | 13.6% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 0.15% | 1 |
| TPOT | 34.44% | 1 |
| E2EL | 30.82% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
