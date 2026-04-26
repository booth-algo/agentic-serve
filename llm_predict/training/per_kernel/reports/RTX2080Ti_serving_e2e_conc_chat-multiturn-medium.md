# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.8 | 38.4% | 30.2% | 29.5% | 1 |
| 20 | 19.7 | 44.6% | 1.1% | 7.8% | 1 |
| 40 | 39.3 | 59.9% | 9.6% | 25.3% | 1 |
| 80 | 78.8 | 73.7% | 2.5% | 37.5% | 1 |

Overall supported MAPE: TPOT 10.8%, E2EL 25.0%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 833 | 246 | 9.8 | 248.5 | 403.2 | 38.4% | 57.9 | 44.5 | 30.2% | 14487.8 | 11187.2 | 29.5% |
| Llama-8B | supported | 20 | 833 | 245 | 19.7 | 299.2 | 540.4 | 44.6% | 57.7 | 58.3 | 1.1% | 14437.3 | 15658.1 | 7.8% |
| Llama-8B | supported | 40 | 833 | 246 | 39.3 | 1721.2 | 4294.6 | 59.9% | 61.5 | 68.1 | 9.6% | 16858.0 | 22576.7 | 25.3% |
| Llama-8B | supported | 80 | 833 | 245 | 78.8 | 1845.5 | 7029.9 | 73.7% | 72.3 | 70.5 | 2.5% | 19551.3 | 31285.6 | 37.5% |
