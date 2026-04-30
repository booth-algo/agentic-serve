# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-short

- Predictor: steady-state batch size model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 41.2% | 0.5% | 0.6% | 1 |
| 10 | 9.9 | 70.9% | 18.2% | 37.8% | 1 |
| 20 | 19.6 | 18.2% | 4.8% | 21.8% | 1 |
| 40 | 39.3 | 314.9% | 7.6% | 17.4% | 1 |

Overall supported MAPE: TPOT 7.8%, E2EL 19.4%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 113 | 158 | 1.0 | 40.6 | 69.1 | 41.2% | 32.9 | 32.7 | 0.5% | 5236.0 | 5266.9 | 0.6% |
| Llama-8B | supported | 10 | 113 | 156 | 9.9 | 49.8 | 170.7 | 70.9% | 35.3 | 43.2 | 18.2% | 5560.8 | 8942.1 | 37.8% |
| Llama-8B | supported | 20 | 127 | 152 | 19.6 | 184.9 | 226.0 | 18.2% | 40.7 | 38.9 | 4.8% | 6377.6 | 8150.7 | 21.8% |
| Llama-8B | supported | 40 | 127 | 152 | 39.3 | 1064.0 | 256.4 | 314.9% | 51.2 | 47.6 | 7.6% | 8846.3 | 7532.1 | 17.4% |
