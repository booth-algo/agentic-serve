# H100 — per-op serving_e2e Validation: chat-medium

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: H100 perop_v5_shape.pkl
- Profile: `chat-medium` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 175 | 312 | 1 | 42.03 | 18.84 | 123.2% | 29.92 | 3.85 | 676.4% | 9377.91 | 1194.35 | 685.2% |
| Llama-8B | supported | sglang | 136 | 289 | 1 | 20.89 | 15.33 | 36.3% | 19.34 | 6.28 | 208.0% | 5611.40 | 1913.92 | 193.2% |
| Llama-8B | supported | vllm | 136 | 291 | 1 | 20.89 | 28.17 | 25.8% | 19.34 | 6.26 | 209.1% | 5650.22 | 1916.57 | 194.8% |
| Qwen3.5-9B | hybrid_attn | sglang | 122 | 324 | 1 | 22.87 | 46.46 | 50.8% | 20.55 | 7.22 | 184.5% | 6681.41 | 2410.39 | 177.2% |
| Qwen3.5-9B | hybrid_attn | vllm | 122 | 324 | 1 | 22.87 | 61.57 | 62.9% | 20.55 | 6.71 | 206.5% | 6681.41 | 2250.96 | 196.8% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 31.05% | 2 |
| TPOT | 208.55% | 2 |
| E2EL | 194.00% | 2 |

_Out-of-scope: 1 moe, 2 hybrid_attn — excluded from headline._
