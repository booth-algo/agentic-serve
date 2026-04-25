# A100 — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 150 | 171 | 1 | 73.31 | 32.99 | 122.3% | 41.99 | 4.93 | 751.3% | 7253.20 | 979.27 | 640.7% |
| Llama-8B | supported | vllm | 127 | 149 | 1 | 32.29 | 35.18 | 8.2% | 29.25 | 12.89 | 127.0% | 4391.15 | 1993.33 | 120.3% |
| Qwen3.5-9B | hybrid_attn | vllm | 95 | 172 | 1 | 31.61 | 89.43 | 64.7% | 27.88 | 13.76 | 102.6% | 4826.15 | 2752.05 | 75.4% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 8.21% | 1 |
| TPOT | 126.95% | 1 |
| E2EL | 120.29% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
