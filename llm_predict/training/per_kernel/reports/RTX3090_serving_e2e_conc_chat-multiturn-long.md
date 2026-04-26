# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-long

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.0 | 654.0% | 40.1% | 6.0% | 1 |
| 20 | 18.1 | 484.1% | 48.3% | 13.0% | 1 |
| 40 | 36.5 | 98.2% | 52.2% | 33.2% | 1 |
| 80 | 74.2 | 71.4% | 41.3% | 26.0% | 1 |
| 120 | 112.6 | 77.6% | 28.0% | 15.8% | 1 |

Overall supported MAPE: TPOT 42.0%, E2EL 18.8%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 937 | 149 | 9.0 | 2366.9 | 313.9 | 654.0% | 16.7 | 27.9 | 40.1% | 4856.8 | 4582.2 | 6.0% |
| Llama-8B | supported | 20 | 937 | 148 | 18.1 | 2538.5 | 434.6 | 484.1% | 17.2 | 33.4 | 48.3% | 5090.5 | 5852.0 | 13.0% |
| Llama-8B | supported | 40 | 937 | 148 | 36.5 | 2720.7 | 1372.7 | 98.2% | 19.5 | 40.9 | 52.2% | 5609.7 | 8392.2 | 33.2% |
| Llama-8B | supported | 80 | 937 | 148 | 74.2 | 2915.9 | 1701.1 | 71.4% | 24.8 | 42.2 | 41.3% | 6581.3 | 8888.3 | 26.0% |
| Llama-8B | supported | 120 | 937 | 149 | 112.6 | 3034.5 | 1708.2 | 77.6% | 30.3 | 42.1 | 28.0% | 7546.9 | 8963.2 | 15.8% |
