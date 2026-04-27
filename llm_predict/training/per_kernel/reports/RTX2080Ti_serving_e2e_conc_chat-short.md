# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 41.2% | 0.5% | 0.6% | 1 |
| 10 | 9.9 | 70.9% | 29.4% | 1.9% | 1 |
| 20 | 19.7 | 18.2% | 71.0% | 26.2% | 1 |
| 40 | 39.5 | 314.9% | 69.6% | 77.0% | 1 |

Overall supported MAPE: TPOT 42.6%, E2EL 26.4%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 113 | 158 | 1.0 | 40.6 | 69.1 | 41.2% | 32.9 | 32.7 | 0.5% | 5236.0 | 5266.9 | 0.6% |
| Llama-8B | supported | 10 | 113 | 156 | 9.9 | 49.8 | 170.7 | 70.9% | 55.9 | 43.2 | 29.4% | 8772.0 | 8942.1 | 1.9% |
| Llama-8B | supported | 20 | 127 | 152 | 19.7 | 184.9 | 226.0 | 18.2% | 66.4 | 38.9 | 71.0% | 10282.2 | 8150.7 | 26.2% |
| Llama-8B | supported | 40 | 127 | 152 | 39.5 | 1064.0 | 256.4 | 314.9% | 80.7 | 47.6 | 69.6% | 13331.6 | 7532.1 | 77.0% |
