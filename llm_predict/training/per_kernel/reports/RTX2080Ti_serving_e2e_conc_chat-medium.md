# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-medium

- Predictor: steady-state batch size model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 29.4% | 0.2% | 5.5% | 1 |
| 10 | 9.9 | 54.4% | 10.8% | 5.6% | 1 |
| 20 | 19.9 | 41.5% | 2.0% | 3.2% | 1 |
| 40 | 39.8 | 67.3% | 6.6% | 6.6% | 1 |

Overall supported MAPE: TPOT 4.9%, E2EL 5.2%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 120 | 254 | 1.0 | 42.7 | 60.4 | 29.4% | 33.0 | 32.9 | 0.2% | 8414.9 | 7974.2 | 5.5% |
| Llama-8B | supported | 10 | 136 | 289 | 9.9 | 64.1 | 140.6 | 54.4% | 36.9 | 41.3 | 10.8% | 10726.4 | 11358.8 | 5.6% |
| Llama-8B | supported | 20 | 136 | 288 | 19.9 | 77.2 | 132.0 | 41.5% | 42.4 | 41.6 | 2.0% | 12296.1 | 12697.7 | 3.2% |
| Llama-8B | supported | 40 | 136 | 279 | 39.8 | 103.3 | 316.2 | 67.3% | 51.9 | 48.7 | 6.6% | 14591.9 | 13690.5 | 6.6% |
