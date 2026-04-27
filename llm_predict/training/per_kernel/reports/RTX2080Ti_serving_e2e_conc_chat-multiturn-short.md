# RTX2080Ti -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.9 | 72.8% | 0.4% | 1.6% | 1 |
| 20 | 19.9 | 72.7% | 18.6% | 22.1% | 1 |
| 40 | 39.8 | 95.5% | 22.2% | 41.7% | 1 |
| 80 | 79.6 | 97.3% | 6.3% | 50.2% | 1 |

Overall supported MAPE: TPOT 11.9%, E2EL 28.9%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 673 | 298 | 9.9 | 87.9 | 322.7 | 72.8% | 43.2 | 43.4 | 0.4% | 12969.2 | 12766.8 | 1.6% |
| Llama-8B | supported | 20 | 673 | 299 | 19.9 | 105.8 | 387.0 | 72.7% | 45.0 | 55.3 | 18.6% | 13570.1 | 17419.1 | 22.1% |
| Llama-8B | supported | 40 | 673 | 296 | 39.8 | 141.6 | 3117.2 | 95.5% | 52.6 | 67.6 | 22.2% | 15717.7 | 26954.1 | 41.7% |
| Llama-8B | supported | 80 | 673 | 297 | 79.6 | 576.8 | 21032.6 | 97.3% | 67.8 | 72.3 | 6.3% | 20702.0 | 41573.0 | 50.2% |
