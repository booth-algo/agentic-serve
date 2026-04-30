# RTX3090 -- serving_e2e Concurrency Sweep: chat-medium

- Predictor: steady-state batch size model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 51.3% | 1.8% | 2.1% | 1 |
| 10 | 9.9 | 6.4% | 11.6% | 8.3% | 1 |
| 20 | 19.8 | 0.4% | 14.4% | 11.2% | 1 |
| 40 | 39.7 | 7.4% | 16.3% | 18.4% | 1 |
| 80 | 79.4 | 24.8% | 18.3% | 14.9% | 1 |

Overall supported MAPE: TPOT 12.5%, E2EL 11.0%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 120 | 238 | 1.0 | 53.7 | 35.5 | 51.3% | 19.5 | 19.9 | 1.8% | 4692.7 | 4596.4 | 2.1% |
| Llama-8B | supported | 10 | 138 | 289 | 9.9 | 69.4 | 74.2 | 6.4% | 20.3 | 23.0 | 11.6% | 5946.8 | 6485.1 | 8.3% |
| Llama-8B | supported | 20 | 138 | 286 | 19.8 | 83.6 | 83.2 | 0.4% | 21.9 | 25.6 | 14.4% | 6342.4 | 7145.3 | 11.2% |
| Llama-8B | supported | 40 | 138 | 292 | 39.7 | 111.9 | 104.2 | 7.4% | 25.2 | 30.1 | 16.3% | 7479.8 | 9166.2 | 18.4% |
| Llama-8B | supported | 80 | 138 | 287 | 79.4 | 470.7 | 377.3 | 24.8% | 31.9 | 39.1 | 18.3% | 9630.8 | 11312.1 | 14.9% |
