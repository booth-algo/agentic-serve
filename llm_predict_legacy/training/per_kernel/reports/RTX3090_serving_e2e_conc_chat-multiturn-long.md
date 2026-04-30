# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-long

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.2 | 654.0% | 23.8% | 20.8% | 1 |
| 20 | 18.5 | 484.1% | 28.8% | 3.5% | 1 |
| 40 | 37.5 | 98.2% | 28.5% | 16.1% | 1 |
| 80 | 76.1 | 71.4% | 4.4% | 0.0% | 1 |
| 120 | 115.6 | 77.6% | 23.1% | 19.9% | 1 |

Overall supported MAPE: TPOT 21.7%, E2EL 12.1%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 937 | 149 | 9.2 | 2366.9 | 313.9 | 654.0% | 21.3 | 27.9 | 23.8% | 5536.5 | 4582.2 | 20.8% |
| Llama-8B | supported | 20 | 937 | 148 | 18.5 | 2538.5 | 434.6 | 484.1% | 23.8 | 33.4 | 28.8% | 6054.1 | 5852.0 | 3.5% |
| Llama-8B | supported | 40 | 937 | 148 | 37.5 | 2720.7 | 1372.7 | 98.2% | 29.2 | 40.9 | 28.5% | 7042.9 | 8392.2 | 16.1% |
| Llama-8B | supported | 80 | 937 | 148 | 76.1 | 2915.9 | 1701.1 | 71.4% | 40.3 | 42.2 | 4.4% | 8884.2 | 8888.3 | 0.0% |
| Llama-8B | supported | 120 | 937 | 149 | 115.6 | 3034.5 | 1708.2 | 77.6% | 51.8 | 42.1 | 23.1% | 10747.6 | 8963.2 | 19.9% |
