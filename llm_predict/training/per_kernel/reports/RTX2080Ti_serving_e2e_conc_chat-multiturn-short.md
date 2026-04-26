# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 10.0 | 72.8% | 57.7% | 60.4% | 1 |
| 20 | 19.9 | 72.7% | 32.6% | 26.5% | 1 |
| 40 | 39.8 | 95.5% | 22.4% | 8.6% | 1 |
| 80 | 79.7 | 97.3% | 42.3% | 25.1% | 1 |

Overall supported MAPE: TPOT 38.7%, E2EL 30.1%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 673 | 298 | 10.0 | 87.9 | 322.7 | 72.8% | 68.4 | 43.4 | 57.7% | 20479.9 | 12766.8 | 60.4% |
| Llama-8B | supported | 20 | 673 | 299 | 19.9 | 105.8 | 387.0 | 72.7% | 73.3 | 55.3 | 32.6% | 22030.8 | 17419.1 | 26.5% |
| Llama-8B | supported | 40 | 673 | 296 | 39.8 | 141.6 | 3117.2 | 95.5% | 82.8 | 67.6 | 22.4% | 24643.4 | 26954.1 | 8.6% |
| Llama-8B | supported | 80 | 673 | 297 | 79.7 | 576.8 | 21032.6 | 97.3% | 103.0 | 72.3 | 42.3% | 31156.9 | 41573.0 | 25.1% |
