# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-medium

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.4 | 21.6% | 42.0% | 39.9% | 1 |
| 20 | 18.9 | 569.9% | 52.5% | 30.5% | 1 |
| 40 | 38.0 | 13.9% | 57.2% | 48.3% | 1 |
| 80 | 76.5 | 41.7% | 53.6% | 55.8% | 1 |
| 120 | 115.7 | 47.2% | 44.5% | 51.8% | 1 |

Overall supported MAPE: TPOT 50.0%, E2EL 45.3%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 833 | 245 | 9.4 | 317.1 | 260.8 | 21.6% | 16.6 | 28.7 | 42.0% | 4387.7 | 7304.2 | 39.9% |
| Llama-8B | supported | 20 | 833 | 245 | 18.9 | 2212.3 | 330.2 | 569.9% | 17.3 | 36.3 | 52.5% | 6441.9 | 9268.3 | 30.5% |
| Llama-8B | supported | 40 | 833 | 246 | 38.0 | 2370.1 | 2080.5 | 13.9% | 19.5 | 45.7 | 57.2% | 7178.7 | 13879.6 | 48.3% |
| Llama-8B | supported | 80 | 833 | 245 | 76.5 | 2541.2 | 4360.8 | 41.7% | 24.7 | 53.3 | 53.6% | 8590.8 | 19448.5 | 55.8% |
| Llama-8B | supported | 120 | 833 | 244 | 115.7 | 2647.5 | 5009.5 | 47.2% | 30.2 | 54.3 | 44.5% | 10004.9 | 20758.5 | 51.8% |
