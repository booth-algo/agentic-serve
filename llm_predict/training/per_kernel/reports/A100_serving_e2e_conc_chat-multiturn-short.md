# A100 -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 3.6% | 1.7% | 1.3% | 1 |
| 10 | 9.9 | 17.6% | 19.8% | 17.9% | 1 |
| 20 | 19.7 | 21.4% | 23.5% | 23.9% | 1 |
| 40 | 39.5 | 233.7% | 16.6% | 13.8% | 1 |
| 80 | 79.2 | 47.1% | 19.4% | 18.6% | 1 |
| 120 | 119.0 | 29.5% | 11.3% | 14.3% | 1 |
| 160 | 158.8 | 70.9% | 3.1% | 14.2% | 1 |

Overall supported MAPE: TPOT 13.6%, E2EL 14.9%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 1 | 673 | 297 | 1.0 | 54.9 | 53.0 | 3.6% | 12.8 | 13.0 | 1.7% | 3863.6 | 3815.8 | 1.3% |
| Llama-8B | supported | 10 | 673 | 296 | 9.9 | 67.2 | 81.6 | 17.6% | 12.1 | 15.1 | 19.8% | 3650.1 | 4446.2 | 17.9% |
| Llama-8B | supported | 20 | 673 | 296 | 19.7 | 81.0 | 102.9 | 21.4% | 13.7 | 17.9 | 23.5% | 4125.8 | 5420.9 | 23.9% |
| Llama-8B | supported | 40 | 673 | 297 | 39.5 | 440.7 | 132.1 | 233.7% | 17.3 | 20.7 | 16.6% | 5572.7 | 6467.5 | 13.8% |
| Llama-8B | supported | 80 | 673 | 296 | 79.2 | 472.5 | 321.3 | 47.1% | 24.3 | 30.1 | 19.4% | 7659.2 | 9413.8 | 18.6% |
| Llama-8B | supported | 120 | 673 | 297 | 119.0 | 491.9 | 697.8 | 29.5% | 31.3 | 35.3 | 11.3% | 9796.4 | 11427.9 | 14.3% |
| Llama-8B | supported | 160 | 673 | 296 | 158.8 | 506.4 | 1738.1 | 70.9% | 37.7 | 38.9 | 3.1% | 11668.5 | 13607.3 | 14.2% |
