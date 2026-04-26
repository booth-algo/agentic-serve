# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.9 | 72.8% | 31.2% | 33.6% | 1 |
| 20 | 19.9 | 72.7% | 4.4% | 0.3% | 1 |
| 40 | 39.8 | 95.5% | 8.9% | 31.8% | 1 |
| 80 | 79.6 | 97.3% | 0.2% | 46.8% | 1 |

Overall supported MAPE: TPOT 11.2%, E2EL 28.1%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 673 | 298 | 9.9 | 87.9 | 322.7 | 72.8% | 56.9 | 43.4 | 31.2% | 17055.8 | 12766.8 | 33.6% |
| Llama-8B | supported | 20 | 673 | 299 | 19.9 | 105.8 | 387.0 | 72.7% | 57.7 | 55.3 | 4.4% | 17364.3 | 17419.1 | 0.3% |
| Llama-8B | supported | 40 | 673 | 296 | 39.8 | 141.6 | 3117.2 | 95.5% | 61.6 | 67.6 | 8.9% | 18386.2 | 26954.1 | 31.8% |
| Llama-8B | supported | 80 | 673 | 297 | 79.6 | 576.8 | 21032.6 | 97.3% | 72.5 | 72.3 | 0.2% | 22116.0 | 41573.0 | 46.8% |
