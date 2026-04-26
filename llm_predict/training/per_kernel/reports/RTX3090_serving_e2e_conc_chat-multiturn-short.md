# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.6 | 28.3% | 39.7% | 36.8% | 1 |
| 20 | 19.2 | 553.5% | 50.6% | 34.8% | 1 |
| 40 | 38.6 | 7.6% | 59.5% | 53.3% | 1 |
| 80 | 77.5 | 75.4% | 53.6% | 65.9% | 1 |
| 120 | 116.7 | 86.7% | 46.7% | 67.3% | 1 |

Overall supported MAPE: TPOT 50.0%, E2EL 51.6%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 673 | 297 | 9.6 | 257.4 | 200.6 | 28.3% | 16.6 | 27.4 | 39.7% | 5173.8 | 8186.5 | 36.8% |
| Llama-8B | supported | 20 | 673 | 298 | 19.2 | 1724.7 | 263.9 | 553.5% | 17.0 | 34.4 | 50.6% | 6784.8 | 10399.5 | 34.8% |
| Llama-8B | supported | 40 | 673 | 296 | 38.6 | 1849.8 | 1719.0 | 7.6% | 19.1 | 47.2 | 59.5% | 7501.4 | 16053.5 | 53.3% |
| Llama-8B | supported | 80 | 673 | 297 | 77.5 | 1981.9 | 8047.6 | 75.4% | 23.9 | 51.5 | 53.6% | 9075.5 | 26646.6 | 65.9% |
| Llama-8B | supported | 120 | 673 | 296 | 116.7 | 2064.6 | 15561.6 | 86.7% | 28.8 | 54.0 | 46.7% | 10589.6 | 32360.4 | 67.3% |
