# H100 — per-op serving_e2e Validation: chat-short

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: H100 perop_v5_shape.pkl
- Profile: `chat-short` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 166 | 167 | 1 | 42.06 | 18.73 | 124.6% | 29.87 | 3.84 | 678.9% | 5030.81 | 679.58 | 640.3% |
| Llama-8B | supported | sglang | 127 | 157 | 1 | 19.94 | 15.53 | 28.4% | 19.26 | 6.27 | 207.3% | 3043.86 | 1035.54 | 193.9% |
| Llama-8B | supported | vllm | 127 | 151 | 1 | 19.94 | 27.50 | 27.5% | 19.25 | 6.24 | 208.3% | 2926.18 | 972.92 | 200.8% |
| Qwen3.5-9B | hybrid_attn | sglang | 114 | 168 | 1 | 22.69 | 47.74 | 52.5% | 20.46 | 7.22 | 183.3% | 3459.89 | 1383.52 | 150.1% |
| Qwen3.5-9B | hybrid_attn | vllm | 114 | 167 | 1 | 22.69 | 61.18 | 62.9% | 20.46 | 6.89 | 197.0% | 3439.31 | 1327.94 | 159.0% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 27.94% | 2 |
| TPOT | 207.80% | 2 |
| E2EL | 197.35% | 2 |

_Out-of-scope: 1 moe, 2 hybrid_attn — excluded from headline._
