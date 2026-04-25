# A100 — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 159 | 270 | 1 | 74.94 | 33.49 | 123.8% | 42.85 | 4.94 | 767.3% | 11645.19 | 1336.96 | 771.0% |
| Llama-8B | supported | vllm | 120 | 257 | 1 | 39.15 | 34.81 | 12.5% | 32.10 | 12.95 | 148.0% | 8289.41 | 3203.68 | 158.7% |
| Qwen3.5-9B | hybrid_attn | vllm | 103 | 271 | 1 | 38.71 | 89.55 | 56.8% | 29.37 | 13.78 | 113.1% | 7997.48 | 3594.96 | 122.5% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 12.47% | 1 |
| TPOT | 147.95% | 1 |
| E2EL | 158.75% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
