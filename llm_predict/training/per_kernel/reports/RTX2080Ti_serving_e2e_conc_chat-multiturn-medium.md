# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.8 | 38.4% | 56.3% | 55.0% | 1 |
| 20 | 19.7 | 44.6% | 25.5% | 16.5% | 1 |
| 40 | 39.6 | 59.9% | 21.4% | 2.3% | 1 |
| 80 | 79.2 | 73.7% | 45.6% | 13.7% | 1 |

Overall supported MAPE: TPOT 37.2%, E2EL 21.9%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 833 | 246 | 9.8 | 248.5 | 403.2 | 38.4% | 69.5 | 44.5 | 56.3% | 17345.4 | 11187.2 | 55.0% |
| Llama-8B | supported | 20 | 833 | 245 | 19.7 | 299.2 | 540.4 | 44.6% | 73.2 | 58.3 | 25.5% | 18245.4 | 15658.1 | 16.5% |
| Llama-8B | supported | 40 | 833 | 246 | 39.6 | 1721.2 | 4294.6 | 59.9% | 82.6 | 68.1 | 21.4% | 22052.8 | 22576.7 | 2.3% |
| Llama-8B | supported | 80 | 833 | 245 | 79.2 | 1845.5 | 7029.9 | 73.7% | 102.7 | 70.5 | 45.6% | 27012.4 | 31285.6 | 13.7% |
