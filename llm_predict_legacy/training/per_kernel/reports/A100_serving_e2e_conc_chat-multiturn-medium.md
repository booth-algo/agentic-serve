# A100 -- serving_e2e Concurrency Sweep: chat-multiturn-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.8 | 10.1% | 17.1% | 14.2% | 1 |
| 20 | 19.6 | 11.5% | 18.7% | 17.4% | 1 |
| 40 | 39.3 | 233.0% | 8.1% | 1.3% | 1 |
| 80 | 78.9 | 4.3% | 2.5% | 4.2% | 1 |
| 120 | 118.6 | 23.7% | 30.9% | 23.5% | 1 |
| 160 | 158.3 | 17.0% | 57.2% | 49.2% | 1 |

Overall supported MAPE: TPOT 22.4%, E2EL 18.3%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 833 | 245 | 9.8 | 81.2 | 73.7 | 10.1% | 12.3 | 14.8 | 17.1% | 3091.9 | 3602.6 | 14.2% |
| Llama-8B | supported | 20 | 833 | 246 | 19.6 | 97.7 | 87.7 | 11.5% | 14.0 | 17.2 | 18.7% | 3542.8 | 4290.4 | 17.4% |
| Llama-8B | supported | 40 | 833 | 246 | 39.3 | 552.5 | 165.9 | 233.0% | 17.8 | 19.4 | 8.1% | 4927.5 | 4994.5 | 1.3% |
| Llama-8B | supported | 80 | 833 | 245 | 78.9 | 592.4 | 568.0 | 4.3% | 25.4 | 24.8 | 2.5% | 6815.2 | 6543.5 | 4.2% |
| Llama-8B | supported | 120 | 833 | 244 | 118.6 | 617.1 | 809.2 | 23.7% | 32.9 | 25.1 | 30.9% | 8649.7 | 7005.4 | 23.5% |
| Llama-8B | supported | 160 | 832 | 245 | 158.3 | 634.9 | 764.9 | 17.0% | 39.7 | 25.3 | 57.2% | 10368.3 | 6948.5 | 49.2% |
