# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-long

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.8 | 67.3% | 37.1% | 37.0% | 1 |
| 20 | 19.7 | 81.2% | 5.7% | 5.1% | 1 |
| 40 | 39.4 | 68.0% | 8.0% | 20.3% | 1 |
| 80 | 78.9 | 66.5% | 24.1% | 15.4% | 1 |

Overall supported MAPE: TPOT 18.7%, E2EL 19.5%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 937 | 149 | 9.8 | 137.1 | 419.3 | 67.3% | 57.8 | 42.1 | 37.1% | 8744.2 | 6384.8 | 37.0% |
| Llama-8B | supported | 20 | 937 | 148 | 19.7 | 165.1 | 878.0 | 81.2% | 57.6 | 54.5 | 5.7% | 8688.5 | 9159.6 | 5.1% |
| Llama-8B | supported | 40 | 937 | 148 | 39.4 | 941.8 | 2941.9 | 68.0% | 61.4 | 56.9 | 8.0% | 10030.6 | 12593.1 | 20.3% |
| Llama-8B | supported | 80 | 937 | 148 | 78.9 | 1009.4 | 3015.9 | 66.5% | 72.1 | 58.1 | 24.1% | 11685.2 | 13814.4 | 15.4% |
