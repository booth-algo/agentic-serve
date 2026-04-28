# RTX3090 -- serving_e2e Concurrency Sweep: chat-long

- Predictor: steady-state batch size model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 48.1% | 1.9% | 1.9% | 1 |
| 10 | 9.9 | 3.5% | 12.9% | 4.1% | 1 |
| 20 | 19.8 | 8.6% | 21.8% | 17.0% | 1 |
| 40 | 39.6 | 34.0% | 27.0% | 22.3% | 1 |
| 80 | 79.4 | 21.2% | 25.1% | 19.7% | 1 |

Overall supported MAPE: TPOT 17.7%, E2EL 13.0%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 218 | 288 | 1.0 | 61.8 | 41.7 | 48.1% | 19.5 | 19.9 | 1.9% | 5685.9 | 5582.5 | 1.9% |
| Llama-8B | supported | 10 | 184 | 286 | 9.9 | 73.3 | 76.0 | 3.5% | 20.4 | 23.4 | 12.9% | 5911.0 | 6161.5 | 4.1% |
| Llama-8B | supported | 20 | 184 | 286 | 19.8 | 88.2 | 96.5 | 8.6% | 22.0 | 28.2 | 21.8% | 6391.0 | 7701.7 | 17.0% |
| Llama-8B | supported | 40 | 184 | 288 | 39.6 | 118.2 | 179.1 | 34.0% | 25.5 | 35.0 | 27.0% | 7469.0 | 9615.4 | 22.3% |
| Llama-8B | supported | 80 | 184 | 286 | 79.4 | 499.8 | 412.3 | 21.2% | 32.5 | 43.4 | 25.1% | 9795.2 | 12198.4 | 19.7% |
