# RTX3090 -- serving_e2e Concurrency Sweep: chat-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 43.1% | 1.6% | 2.4% | 1 |
| 10 | 9.8 | 13.1% | 13.3% | 14.9% | 1 |
| 20 | 19.7 | 9.6% | 15.3% | 17.9% | 1 |
| 40 | 39.4 | 46.7% | 14.6% | 20.0% | 1 |
| 80 | 79.0 | 20.6% | 16.3% | 14.0% | 1 |

Overall supported MAPE: TPOT 12.2%, E2EL 13.8%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 127 | 150 | 1.0 | 54.3 | 37.9 | 43.1% | 19.4 | 19.8 | 1.6% | 2969.8 | 3041.4 | 2.4% |
| Llama-8B | supported | 10 | 127 | 154 | 9.8 | 66.5 | 76.6 | 13.1% | 20.1 | 23.2 | 13.3% | 3167.2 | 3719.7 | 14.9% |
| Llama-8B | supported | 20 | 115 | 165 | 19.7 | 78.6 | 87.0 | 9.6% | 21.5 | 25.4 | 15.3% | 3631.7 | 4426.1 | 17.9% |
| Llama-8B | supported | 40 | 127 | 155 | 39.4 | 445.7 | 835.9 | 46.7% | 24.6 | 28.8 | 14.6% | 4253.5 | 5316.6 | 20.0% |
| Llama-8B | supported | 80 | 115 | 167 | 79.0 | 464.6 | 385.3 | 20.6% | 30.7 | 36.7 | 16.3% | 5594.9 | 6509.0 | 14.0% |
