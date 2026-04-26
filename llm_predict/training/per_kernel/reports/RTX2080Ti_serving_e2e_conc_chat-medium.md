# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 29.4% | 0.2% | 5.5% | 1 |
| 10 | 10.0 | 54.4% | 41.3% | 49.2% | 1 |
| 20 | 19.9 | 41.5% | 66.0% | 57.3% | 1 |
| 40 | 39.9 | 67.3% | 67.7% | 67.2% | 1 |

Overall supported MAPE: TPOT 43.8%, E2EL 44.8%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 120 | 254 | 1.0 | 42.7 | 60.4 | 29.4% | 33.0 | 32.9 | 0.2% | 8414.9 | 7974.2 | 5.5% |
| Llama-8B | supported | 10 | 136 | 289 | 10.0 | 64.1 | 140.6 | 54.4% | 58.4 | 41.3 | 41.3% | 16946.8 | 11358.8 | 49.2% |
| Llama-8B | supported | 20 | 136 | 288 | 19.9 | 77.2 | 132.0 | 41.5% | 69.1 | 41.6 | 66.0% | 19972.5 | 12697.7 | 57.3% |
| Llama-8B | supported | 40 | 136 | 279 | 39.9 | 103.3 | 316.2 | 67.3% | 81.7 | 48.7 | 67.7% | 22891.9 | 13690.5 | 67.2% |
