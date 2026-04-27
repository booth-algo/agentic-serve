# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.5 | 21.6% | 25.2% | 23.8% | 1 |
| 20 | 19.1 | 569.9% | 34.2% | 12.9% | 1 |
| 40 | 38.6 | 13.9% | 35.6% | 30.8% | 1 |
| 80 | 77.8 | 41.7% | 24.4% | 36.2% | 1 |
| 120 | 117.2 | 47.2% | 5.8% | 27.1% | 1 |

Overall supported MAPE: TPOT 25.1%, E2EL 26.2%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 833 | 245 | 9.5 | 317.1 | 260.8 | 21.6% | 21.4 | 28.7 | 25.2% | 5569.1 | 7304.2 | 23.8% |
| Llama-8B | supported | 20 | 833 | 245 | 19.1 | 2212.3 | 330.2 | 569.9% | 23.9 | 36.3 | 34.2% | 8068.2 | 9268.3 | 12.9% |
| Llama-8B | supported | 40 | 833 | 246 | 38.6 | 2370.1 | 2080.5 | 13.9% | 29.4 | 45.7 | 35.6% | 9599.7 | 13879.6 | 30.8% |
| Llama-8B | supported | 80 | 833 | 245 | 77.8 | 2541.2 | 4360.8 | 41.7% | 40.3 | 53.3 | 24.4% | 12403.1 | 19448.5 | 36.2% |
| Llama-8B | supported | 120 | 833 | 244 | 117.2 | 2647.5 | 5009.5 | 47.2% | 51.2 | 54.3 | 5.8% | 15136.9 | 20758.5 | 27.1% |
