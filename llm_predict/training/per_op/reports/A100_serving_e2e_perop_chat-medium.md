# A100 — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 159 | 270 | 1 | 73.08 | 33.49 | 118.2% | 42.19 | 4.94 | 754.0% | 11465.11 | 1336.96 | 757.5% |
| Llama-8B | supported | vllm | 120 | 257 | 1 | 32.17 | 34.81 | 7.6% | 29.49 | 12.95 | 127.8% | 7612.05 | 3203.68 | 137.6% |
| Qwen3.5-9B | hybrid_attn | vllm | 103 | 271 | 1 | 31.64 | 89.55 | 64.7% | 27.90 | 13.78 | 102.5% | 7593.86 | 3594.96 | 111.2% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 7.58% | 1 |
| TPOT | 127.80% | 1 |
| E2EL | 137.60% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
