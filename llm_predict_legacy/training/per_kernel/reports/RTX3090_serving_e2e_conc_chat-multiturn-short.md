# RTX3090 -- serving_e2e Concurrency Sweep: chat-multiturn-short

- Predictor: Little's Law concurrency model + per-kernel XGBoost
- Concurrencies: [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]

## Per-concurrency MAPE (supported architectures)

| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.7 | 28.3% | 22.6% | 19.8% | 1 |
| 20 | 19.4 | 553.5% | 31.5% | 16.0% | 1 |
| 40 | 38.9 | 7.6% | 39.4% | 35.8% | 1 |
| 80 | 78.4 | 75.4% | 24.9% | 49.5% | 1 |
| 120 | 118.1 | 86.7% | 9.5% | 48.9% | 1 |

Overall supported MAPE: TPOT 25.6%, E2EL 34.0%

## Per-row detail

| Model | arch | Conc | ISL | OSL | bs_eff | pred TTFT | meas TTFT | TTFT err | pred TPOT | meas TPOT | TPOT err | pred E2EL | meas E2EL | E2EL err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | supported | 10 | 673 | 297 | 9.7 | 257.4 | 200.6 | 28.3% | 21.2 | 27.4 | 22.6% | 6564.9 | 8186.5 | 19.8% |
| Llama-8B | supported | 20 | 673 | 298 | 19.4 | 1724.7 | 263.9 | 553.5% | 23.5 | 34.4 | 31.5% | 8734.5 | 10399.5 | 16.0% |
| Llama-8B | supported | 40 | 673 | 296 | 38.9 | 1849.8 | 1719.0 | 7.6% | 28.6 | 47.2 | 39.4% | 10307.4 | 16053.5 | 35.8% |
| Llama-8B | supported | 80 | 673 | 297 | 78.4 | 1981.9 | 8047.6 | 75.4% | 38.7 | 51.5 | 24.9% | 13461.4 | 26646.6 | 49.5% |
| Llama-8B | supported | 120 | 673 | 296 | 118.1 | 2064.6 | 15561.6 | 86.7% | 48.9 | 54.0 | 9.5% | 16542.4 | 32360.4 | 48.9% |
