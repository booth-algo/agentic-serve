# A100 -- serving_e2e Concurrency Sweep: chat-multiturn-long

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.6 | 31.4% | 13.9% | 6.9% | 1 |
| 20 | 19.4 | 585.6% | 13.4% | 9.7% | 1 |
| 40 | 38.9 | 130.9% | 4.6% | 16.1% | 1 |
| 80 | 78.5 | 71.5% | 36.9% | 42.1% | 1 |
| 120 | 118.1 | 129.2% | 77.5% | 81.2% | 1 |
| 160 | 157.9 | 93.1% | 115.5% | 113.6% | 1 |

Overall supported MAPE: TPOT 43.6%, E2EL 44.9%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 937 | 148 | 9.6 | 82.8 | 63.1 | 31.4% | 12.3 | 14.3 | 13.9% | 1908.6 | 2050.4 | 6.9% |
| Llama-8B | supported | 20 | 937 | 149 | 19.4 | 554.3 | 80.8 | 585.6% | 14.0 | 16.1 | 13.4% | 2635.1 | 2402.4 | 9.7% |
| Llama-8B | supported | 40 | 937 | 149 | 38.9 | 594.1 | 257.3 | 130.9% | 18.0 | 17.2 | 4.6% | 3270.3 | 2816.6 | 16.1% |
| Llama-8B | supported | 80 | 937 | 149 | 78.5 | 636.7 | 371.2 | 71.5% | 25.7 | 18.8 | 36.9% | 4461.7 | 3140.3 | 42.1% |
| Llama-8B | supported | 120 | 937 | 148 | 118.1 | 663.5 | 289.5 | 129.2% | 33.4 | 18.8 | 77.5% | 5605.3 | 3093.8 | 81.2% |
| Llama-8B | supported | 160 | 937 | 149 | 157.9 | 682.4 | 353.3 | 93.1% | 40.5 | 18.8 | 115.5% | 6722.0 | 3147.0 | 113.6% |
