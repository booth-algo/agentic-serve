# A100 — per-op serving_e2e Validation: chat-long

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: A100 perop_v5_shape.pkl
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 258 | 299 | 1 | 74.75 | 33.67 | 122.0% | 42.26 | 4.94 | 755.2% | 12710.44 | 1399.07 | 808.5% |
| Llama-8B | supported | vllm | 218 | 285 | 1 | 38.24 | 34.62 | 10.5% | 29.64 | 12.95 | 128.9% | 8484.36 | 3633.09 | 133.5% |
| Qwen3.5-9B | hybrid_attn | vllm | 206 | 300 | 1 | 32.30 | 87.58 | 63.1% | 27.96 | 13.78 | 102.8% | 8419.65 | 3920.65 | 114.8% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 10.47% | 1 |
| TPOT | 128.88% | 1 |
| E2EL | 133.53% | 1 |

_Out-of-scope: 1 moe, 1 hybrid_attn — excluded from headline._
