# H100 — per-op serving_e2e Validation: chat-long

- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)
- Predictor: H100 perop_v5_shape.pkl
- Profile: `chat-long` (concurrency=1)
- Ground truth: `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}`
- Headline MAPE = supported architectures only.

## Per-row

| Model | arch | backend | ISL | OSL | bs | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | moe | vllm | 227 | 299 | 1 | 42.51 | 18.93 | 124.5% | 29.90 | 3.85 | 677.0% | 8981.35 | 1066.27 | 742.3% |
| Llama-8B | supported | sglang | 187 | 278 | 1 | 22.54 | 16.17 | 39.4% | 19.35 | 6.28 | 208.1% | 5402.06 | 1673.73 | 222.8% |
| Llama-8B | supported | vllm | 187 | 284 | 1 | 22.54 | 27.79 | 18.9% | 19.35 | 6.26 | 209.3% | 5518.58 | 1716.99 | 221.4% |
| Qwen3.5-9B | hybrid_attn | sglang | 174 | 302 | 1 | 24.72 | 45.56 | 45.7% | 20.55 | 7.23 | 184.1% | 6230.49 | 2052.93 | 203.5% |
| Qwen3.5-9B | hybrid_attn | vllm | 174 | 296 | 1 | 24.72 | 61.61 | 59.9% | 20.55 | 6.70 | 206.5% | 6106.79 | 1907.84 | 220.1% |

## Summary (supported architectures only)

| Metric | MAPE | n rows |
|---|---:|---:|
| TTFT | 29.13% | 2 |
| TPOT | 208.67% | 2 |
| E2EL | 222.08% | 2 |

_Out-of-scope: 1 moe, 2 hybrid_attn — excluded from headline._
