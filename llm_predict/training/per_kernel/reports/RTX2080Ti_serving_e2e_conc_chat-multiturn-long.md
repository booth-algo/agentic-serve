# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-long

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.9 | 67.3% | 64.6% | 64.0% | 1 |
| 20 | 19.7 | 81.2% | 34.2% | 19.9% | 1 |
| 40 | 39.5 | 68.0% | 44.9% | 4.4% | 1 |
| 80 | 79.3 | 66.5% | 76.4% | 17.1% | 1 |

Overall supported MAPE: TPOT 55.0%, E2EL 26.4%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 937 | 149 | 9.9 | 137.1 | 419.3 | 67.3% | 69.4 | 42.1 | 64.6% | 10472.6 | 6384.8 | 64.0% |
| Llama-8B | supported | 20 | 937 | 148 | 19.7 | 165.1 | 878.0 | 81.2% | 73.1 | 54.5 | 34.2% | 10985.3 | 9159.6 | 19.9% |
| Llama-8B | supported | 40 | 937 | 148 | 39.5 | 941.8 | 2941.9 | 68.0% | 82.4 | 56.9 | 44.9% | 13141.8 | 12593.1 | 4.4% |
| Llama-8B | supported | 80 | 937 | 148 | 79.3 | 1009.4 | 3015.9 | 66.5% | 102.5 | 58.1 | 76.4% | 16182.5 | 13814.4 | 17.1% |
