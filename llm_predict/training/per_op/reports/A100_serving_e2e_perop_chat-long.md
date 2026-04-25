# A100 — per-op serving_e2e Validation: chat-long

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 258 | 299 | 1 | 160.63 | 33.67 | 377.1% | 42.63 | 4.94 | 762.7% | 12906.11 | 1399.07 | 822.5% |
| Llama-8B | supported | vllm | 218 | 285 | 1 | 43.44 | 34.62 | 25.5% | 30.93 | 12.95 | 138.9% | 8859.76 | 3633.09 | 143.9% |
| Qwen3.5-9B | hybrid_attn | vllm | 206 | 300 | 1 | 47.18 | 87.58 | 46.1% | 28.91 | 13.78 | 109.7% | 8719.69 | 3920.65 | 122.4% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 25.49% | 1 |
| TPOT | 138.91% | 1 |
| E2EL | 143.86% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
