# A100 — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 150 | 171 | 1 | 75.55 | 32.99 | 129.0% | 42.75 | 4.93 | 766.8% | 7386.14 | 979.27 | 654.2% |
| Llama-8B | supported | vllm | 127 | 149 | 1 | 41.44 | 35.18 | 17.8% | 32.51 | 12.89 | 152.2% | 4884.74 | 1993.33 | 145.1% |
| Qwen3.5-9B | hybrid_attn | vllm | 95 | 172 | 1 | 39.85 | 89.43 | 55.4% | 30.26 | 13.76 | 119.9% | 5243.74 | 2752.05 | 90.5% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 17.80% | 1 |
| TPOT | 152.18% | 1 |
| E2EL | 145.05% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
