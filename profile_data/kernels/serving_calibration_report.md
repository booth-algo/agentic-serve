# Serving Calibration Report

Calibration excludes legacy `chat-short`, `chat-medium`, and `chat-long`.
The active scope is canonical single-turn/stress, high concurrency, and multi-turn cache analysis.

## Calibration Coverage

| GPU | Backend | Version | Model | Status | C=1 rows | Profiles | Long rows | Raw TTFT MAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| A100 | sglang | 0.5.9 | Llama-3.1-8B | high_confidence | 3 | 1 | 0 | 10.7% |
| A100 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 2 | 1 | 0 | 35.48% |
| A100 | vllm | 0.19.0 | Qwen3.5-9B | poor_kernel_fit | 2 | 1 | 0 | 75.72% |
| A100 | vllm | 0.19.0 | gpt-oss-20b | low_confidence | 2 | 1 | 0 | 47.85% |
| H100 | sglang | 0.5.9 | Llama-3.1-70B | poor_kernel_fit | 1 | 1 | 0 | 62.63% |
| H100 | sglang | 0.5.9 | Llama-3.1-8B | low_confidence | 4 | 4 | 2 | 36.53% |
| H100 | sglang | 0.5.9 | Llama-3.3-70B | poor_kernel_fit | 1 | 1 | 0 | 62.2% |
| H100 | sglang | 0.5.9 | Qwen2.5-72B | poor_kernel_fit | 1 | 1 | 0 | 57.01% |
| H100 | sglang | 0.5.9 | Qwen3.5-9B | poor_kernel_fit | 4 | 4 | 2 | 65.22% |
| H100 | vllm | 0.19.0 | Llama-3.1-8B | poor_kernel_fit | 4 | 4 | 2 | 51.9% |
| H100 | vllm | 0.19.0 | Qwen3.5-9B | poor_kernel_fit | 4 | 4 | 2 | 64.95% |
| H100 | vllm | 0.19.0 | gpt-oss-20b | poor_kernel_fit | 4 | 4 | 2 | 80.88% |
| H100 | vllm | 0.19.1 | Llama-3.1-70B | poor_kernel_fit | 1 | 1 | 0 | 206.31% |
| H100 | vllm | 0.19.1 | Llama-3.1-8B | poor_kernel_fit | 1 | 1 | 0 | 75.42% |
| H100 | vllm | 0.19.1 | Llama-3.3-70B | poor_kernel_fit | 1 | 1 | 0 | 196.92% |
| H100 | vllm | 0.19.1 | Qwen2.5-72B | poor_kernel_fit | 1 | 1 | 0 | 183.55% |
| H100 | vllm | 0.19.1 | gpt-oss-20b | poor_kernel_fit | 1 | 1 | 0 | 72.35% |
| RTX3090 | sglang | 0.5.9 | Llama-3.1-8B | medium_confidence | 2 | 1 | 0 | 17.3% |
| RTX3090 | sglang | 0.5.9 | gpt-oss-20b | poor_kernel_fit | 2 | 1 | 0 | 75.49% |
| RTX3090 | vllm | 0.19.0 | Llama-3.1-8B | poor_kernel_fit | 1 | 1 | 0 | 64.91% |
| RTX3090 | vllm | 0.19.0 | Qwen3.5-9B | low_confidence | 3 | 3 | 1 | 47.32% |
| RTX3090 | vllm | 0.19.0 | gpt-oss-20b | medium_confidence | 4 | 3 | 1 | 26.62% |

## Prefix Cache Multi-turn Summary

| GPU | Backend | Model | Profile | Rows | Median ctx | Median new | Cache hit | Full E2EL | Cache-aware E2EL | Cache-aware TTFT | Cache-aware TPOT |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 | sglang | Llama-3.1-8B | chat-multiturn-synth | 10 | 1712.0 | 232.5 | 0.862 | 24.1% | 14.6% | 51.2% | 39.3% |
| A100 | sglang | Llama-3.1-8B | osworld-multiturn-synth | 2 | 9760.0 | 98.5 | 0.99 | 51.2% | 43.4% | 96.9% | 120.1% |
| A100 | sglang | Llama-3.1-8B | swebench-multiturn-synth | 2 | 7179.0 | 133.5 | 0.981 | 88.6% | 91.6% | 98.6% | 33.2% |
| A100 | sglang | Llama-3.1-8B | terminalbench-multiturn-synth | 2 | 3491.0 | 126.5 | 0.964 | 83.8% | 90.8% | 97.6% | 51.4% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn | 7 | 8732.0 | 1241.0 | 0.848 | 48.8% | 43.0% | 98.7% | 109.1% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-long | 10 | 920.0 | 168.0 | 0.817 | 93.6% | 96.8% | 80.5% | 95.6% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 10 | 969.0 | 277.0 | 0.714 | 44.2% | 48.2% | 21.7% | 54.7% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-short | 11 | 860.0 | 289.0 | 0.664 | 14.7% | 19.5% | 70.9% | 13.5% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-synth | 11 | 1712.0 | 236.0 | 0.862 | 18.2% | 19.7% | 47.5% | 28.2% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn | 5 | 2675.0 | 27.0 | 0.99 | 43.7% | 25.6% | 80.6% | 86.7% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-long | 10 | 3421.0 | 30.0 | 0.991 | 48.4% | 59.1% | 50.9% | 109.5% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 10 | 3246.0 | 65.0 | 0.98 | 43.2% | 49.2% | 63.8% | 58.1% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-short | 10 | 3180.0 | 146.0 | 0.954 | 42.6% | 49.5% | 79.5% | 45.8% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-synth | 11 | 9760.0 | 98.0 | 0.99 | 52.0% | 46.6% | 92.6% | 37.5% |
| A100 | vllm | Llama-3.1-8B | swebench-multiturn | 5 | 13049.0 | 114.0 | 0.991 | 118.8% | 67.1% | 61.0% | 142.0% |
| A100 | vllm | Llama-3.1-8B | swebench-multiturn-short | 10 | 4253.0 | 186.0 | 0.956 | 58.3% | 61.9% | 82.4% | 52.2% |
| A100 | vllm | Llama-3.1-8B | swebench-multiturn-synth | 11 | 7124.0 | 133.0 | 0.981 | 88.6% | 91.1% | 96.4% | 35.5% |
| A100 | vllm | Llama-3.1-8B | terminalbench-multiturn | 5 | 13084.0 | 134.0 | 0.99 | 234.2% | 95.0% | 111.6% | 142.4% |
| A100 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 10 | 3506.0 | 247.0 | 0.93 | 50.5% | 56.5% | 78.2% | 52.9% |
| A100 | vllm | Llama-3.1-8B | terminalbench-multiturn-synth | 11 | 3491.0 | 128.0 | 0.964 | 82.6% | 87.3% | 90.8% | 30.1% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn | 5 | 516.0 | 227.0 | 0.56 | 103.8% | 116.4% | 322.8% | 75.7% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-long | 9 | 922.0 | 172.0 | 0.813 | 31.6% | 29.9% | 23.5% | 47.6% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-medium | 9 | 980.0 | 287.0 | 0.707 | 21.3% | 12.2% | 62.9% | 9.8% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-short | 9 | 876.0 | 297.0 | 0.661 | 14.4% | 11.5% | 81.6% | 9.5% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-synth | 11 | 1692.0 | 236.0 | 0.861 | 39.7% | 31.2% | 76.2% | 18.4% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn | 5 | 2795.0 | 28.0 | 0.99 | 50.6% | 29.3% | 76.1% | 102.7% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-long | 9 | 3195.0 | 32.0 | 0.99 | 32.5% | 47.3% | 52.6% | 64.3% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-medium | 8 | 3087.0 | 52.0 | 0.983 | 38.9% | 48.5% | 68.8% | 75.4% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-short | 6 | 2933.0 | 53.0 | 0.982 | 34.6% | 35.7% | 75.2% | 69.6% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-synth | 11 | 2100.0 | 98.0 | 0.953 | 56.5% | 56.6% | 82.7% | 25.5% |
| A100 | vllm | Qwen3.5-9B | swebench-multiturn-synth | 11 | 2578.0 | 130.0 | 0.948 | 82.4% | 74.6% | 85.3% | 40.5% |
| A100 | vllm | Qwen3.5-9B | terminalbench-multiturn-short | 5 | 2250.0 | 152.0 | 0.932 | 37.0% | 38.3% | 70.2% | 29.2% |
| A100 | vllm | Qwen3.5-9B | terminalbench-multiturn-synth | 11 | 2053.0 | 128.0 | 0.937 | 77.6% | 62.1% | 75.6% | 40.4% |
| A100 | vllm | gpt-oss-20b | chat-multiturn | 5 | 558.0 | 220.0 | 0.606 | 227.1% | 269.3% | 28.2% | 91.8% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-long | 10 | 947.0 | 166.0 | 0.825 | 132.6% | 129.3% | 63.1% | 130.8% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-medium | 10 | 1001.0 | 277.0 | 0.723 | 113.0% | 115.6% | 70.7% | 102.2% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-short | 10 | 893.0 | 290.0 | 0.675 | 60.7% | 63.1% | 61.5% | 39.8% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-synth | 11 | 1748.0 | 236.0 | 0.865 | 66.2% | 58.2% | 78.1% | 99.5% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn | 5 | 2686.0 | 26.0 | 0.99 | 154.5% | 156.2% | 71.0% | 206.8% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-long | 10 | 3453.0 | 30.0 | 0.991 | 51.7% | 54.8% | 90.2% | 124.3% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-medium | 10 | 3279.0 | 64.0 | 0.98 | 49.9% | 52.1% | 91.3% | 71.0% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-short | 10 | 3213.0 | 145.0 | 0.955 | 43.0% | 45.1% | 95.2% | 49.7% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-synth | 11 | 9796.0 | 98.0 | 0.99 | 46.8% | 55.1% | 87.8% | 52.8% |
| A100 | vllm | gpt-oss-20b | swebench-multiturn | 4 | 4263.0 | 114.0 | 0.973 | 146.2% | 103.8% | 52.1% | 284.1% |
| A100 | vllm | gpt-oss-20b | swebench-multiturn-short | 8 | 4453.0 | 171.0 | 0.962 | 27.2% | 30.9% | 92.4% | 61.1% |
| A100 | vllm | gpt-oss-20b | swebench-multiturn-synth | 11 | 7270.0 | 133.0 | 0.981 | 91.2% | 77.3% | 96.1% | 44.8% |
| A100 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 10 | 3572.5 | 250.0 | 0.93 | 24.1% | 22.7% | 90.7% | 35.3% |
| A100 | vllm | gpt-oss-20b | terminalbench-multiturn-synth | 11 | 3527.0 | 128.0 | 0.964 | 69.3% | 77.1% | 92.8% | 43.6% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 76.9% | 77.2% | 26.6% | 72.2% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-medium | 7 | 969.0 | 277.0 | 0.714 | 76.5% | 80.9% | 34.4% | 70.5% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-short | 7 | 860.0 | 289.0 | 0.664 | 69.3% | 75.0% | 154.2% | 65.0% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-synth | 11 | 1712.0 | 236.0 | 0.862 | 111.3% | 122.5% | 52.8% | 136.8% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 29.6% | 34.6% | 97.3% | 24.5% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-short | 5 | 4262.0 | 74.0 | 0.983 | 45.6% | 53.6% | 91.4% | 40.4% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-synth | 11 | 9760.0 | 98.0 | 0.99 | 37.4% | 30.2% | 80.2% | 84.6% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14317.0 | 375.0 | 0.974 | 82.1% | 81.6% | 94.6% | 73.1% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-short | 7 | 8015.0 | 451.0 | 0.944 | 77.9% | 77.1% | 78.4% | 71.9% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-synth | 11 | 7234.0 | 133.0 | 0.981 | 83.9% | 86.9% | 90.0% | 51.6% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 382.0 | 0.965 | 72.6% | 73.9% | 92.1% | 71.2% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-short | 7 | 4976.0 | 461.0 | 0.907 | 43.2% | 45.5% | 47.9% | 40.6% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-synth | 11 | 3491.0 | 128.0 | 0.964 | 57.0% | 56.6% | 87.5% | 24.9% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 9.6% | 2.3% | 55.6% | 14.4% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-medium | 7 | 980.0 | 287.0 | 0.707 | 12.2% | 9.2% | 70.9% | 6.7% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-short | 7 | 876.0 | 297.0 | 0.661 | 14.6% | 12.8% | 72.6% | 5.8% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-synth | 11 | 1692.0 | 236.0 | 0.861 | 3.5% | 11.7% | 87.7% | 46.5% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4774.0 | 100.0 | 0.979 | 39.1% | 47.9% | 89.8% | 27.6% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-short | 5 | 4541.0 | 74.0 | 0.984 | 56.9% | 59.3% | 92.4% | 44.1% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-synth | 11 | 9740.0 | 98.0 | 0.99 | 8.2% | 16.1% | 90.4% | 62.7% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-medium | 7 | 15060.0 | 358.0 | 0.976 | 81.9% | 82.7% | 89.5% | 76.8% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-short | 7 | 8894.0 | 506.0 | 0.943 | 81.5% | 81.8% | 77.7% | 76.3% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-synth | 11 | 7214.0 | 133.0 | 0.981 | 85.3% | 88.3% | 96.0% | 80.8% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11992.0 | 391.0 | 0.967 | 64.0% | 64.5% | 84.8% | 56.8% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-short | 7 | 5462.0 | 502.0 | 0.908 | 58.0% | 58.7% | 58.2% | 54.6% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-synth | 11 | 3471.0 | 128.0 | 0.963 | 77.6% | 85.6% | 95.1% | 69.9% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-long | 3 | 920.0 | 168.0 | 0.817 | 1015.0% | 1033.0% | 519.8% | 1018.2% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-medium | 3 | 969.0 | 277.0 | 0.714 | 879.6% | 914.2% | 376.9% | 940.9% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-short | 3 | 860.0 | 289.0 | 0.664 | 659.3% | 700.3% | 228.4% | 748.7% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn | 7 | 9980.0 | 1305.0 | 0.872 | 23.7% | 20.9% | 65.7% | 68.7% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 77.4% | 75.4% | 30.2% | 81.4% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 8 | 969.0 | 277.0 | 0.714 | 65.0% | 67.2% | 57.5% | 62.8% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-short | 8 | 860.0 | 289.0 | 0.664 | 61.3% | 67.2% | 81.7% | 60.6% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-synth | 11 | 1712.0 | 236.0 | 0.862 | 106.2% | 104.3% | 68.7% | 141.3% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn | 5 | 14571.0 | 521.0 | 0.964 | 219.1% | 186.3% | 32.5% | 329.3% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 32.5% | 40.1% | 98.0% | 19.7% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-short | 5 | 4478.0 | 74.0 | 0.983 | 47.4% | 48.2% | 89.1% | 38.3% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-synth | 11 | 9760.0 | 98.0 | 0.99 | 31.8% | 24.2% | 85.0% | 72.8% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14132.0 | 427.0 | 0.97 | 80.5% | 80.8% | 95.9% | 69.5% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-short | 8 | 8015.0 | 451.0 | 0.944 | 76.4% | 76.1% | 90.3% | 67.0% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-synth | 11 | 7234.0 | 133.0 | 0.981 | 81.2% | 85.4% | 93.0% | 42.6% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 390.0 | 0.965 | 71.0% | 72.6% | 93.7% | 71.9% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 8 | 4976.0 | 461.0 | 0.907 | 29.7% | 24.7% | 71.8% | 19.0% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-synth | 11 | 3491.0 | 129.0 | 0.963 | 39.9% | 56.5% | 90.8% | 37.9% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-long | 3 | 920.0 | 168.0 | 0.817 | 1052.7% | 1024.0% | 565.1% | 1014.7% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-medium | 3 | 969.0 | 277.0 | 0.714 | 886.0% | 920.9% | 392.3% | 942.8% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-short | 3 | 860.0 | 289.0 | 0.664 | 671.0% | 715.4% | 224.1% | 747.8% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-long | 3 | 905.0 | 169.0 | 0.813 | 975.1% | 971.7% | 511.7% | 964.9% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-medium | 3 | 955.0 | 280.0 | 0.707 | 849.1% | 882.7% | 379.8% | 877.9% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-short | 3 | 848.0 | 291.0 | 0.657 | 637.5% | 680.0% | 238.4% | 694.1% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 28.1% | 24.7% | 61.2% | 46.5% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-medium | 8 | 980.0 | 287.0 | 0.707 | 26.5% | 26.3% | 58.0% | 27.3% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-short | 8 | 876.0 | 297.0 | 0.661 | 25.4% | 25.2% | 62.0% | 22.7% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-synth | 11 | 1692.0 | 236.0 | 0.861 | 47.6% | 52.1% | 78.5% | 117.3% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4778.0 | 100.0 | 0.979 | 43.2% | 52.8% | 92.0% | 28.0% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-short | 5 | 4833.0 | 74.0 | 0.984 | 59.6% | 61.8% | 94.0% | 45.7% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-synth | 11 | 9740.0 | 98.0 | 0.99 | 40.8% | 46.1% | 83.7% | 130.3% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-medium | 7 | 14502.0 | 424.0 | 0.971 | 72.7% | 72.0% | 91.9% | 60.1% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-short | 8 | 8894.0 | 506.0 | 0.943 | 52.5% | 49.9% | 84.7% | 42.9% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-synth | 11 | 7214.0 | 133.0 | 0.981 | 76.2% | 62.6% | 91.8% | 46.7% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11673.0 | 456.0 | 0.962 | 19.0% | 15.2% | 82.3% | 18.1% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-short | 8 | 5462.0 | 502.0 | 0.908 | 31.1% | 22.9% | 71.0% | 27.3% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-synth | 11 | 3471.0 | 128.0 | 0.963 | 51.5% | 66.2% | 91.6% | 41.2% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-long | 5 | 947.0 | 166.0 | 0.825 | 44.1% | 38.3% | 55.4% | 35.2% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-medium | 5 | 1001.0 | 277.0 | 0.723 | 39.7% | 39.0% | 59.8% | 32.8% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-short | 5 | 893.0 | 290.0 | 0.675 | 39.2% | 37.9% | 97.9% | 31.6% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-synth | 2 | 1752.5 | 237.0 | 0.865 | 161.5% | 203.8% | 90.4% | 272.6% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-medium | 5 | 4542.0 | 5.0 | 0.999 | 30.3% | 32.6% | 96.1% | 16.1% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-short | 5 | 4494.0 | 75.0 | 0.983 | 44.3% | 45.7% | 94.3% | 30.8% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-synth | 2 | 2263.0 | 99.5 | 0.956 | 63.1% | 43.6% | 91.6% | 203.3% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-medium | 5 | 14017.0 | 433.0 | 0.969 | 34.1% | 24.4% | 66.1% | 17.5% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-short | 5 | 8079.0 | 454.0 | 0.944 | 51.9% | 53.0% | 90.4% | 31.7% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-medium | 5 | 11213.0 | 409.0 | 0.963 | 20.7% | 16.8% | 61.9% | 21.5% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 5 | 5037.0 | 465.0 | 0.907 | 78.5% | 57.0% | 52.7% | 28.2% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-synth | 2 | 3431.0 | 62.0 | 0.982 | 7.9% | 36.4% | 93.8% | 99.8% |
| RTX3090 | sglang | Llama-3.1-8B | chat-multiturn | 7 | 516.0 | 213.0 | 0.587 | 89.8% | 218.2% | 809.4% | 177.8% |
| RTX3090 | sglang | Llama-3.1-8B | chat-multiturn-synth | 2 | 1716.5 | 237.0 | 0.862 | 32.7% | 30.4% | 94.6% | 174.9% |
| RTX3090 | sglang | Llama-3.1-8B | osworld-multiturn | 7 | 10239.0 | 521.0 | 0.949 | 41.6% | 67.6% | 62.7% | 193.8% |
| RTX3090 | sglang | Llama-3.1-8B | osworld-multiturn-synth | 2 | 9835.0 | 99.0 | 0.99 | 64.1% | 57.2% | 97.1% | 444.5% |
| RTX3090 | sglang | Llama-3.1-8B | swebench-multiturn | 5 | 11616.0 | 111.0 | 0.99 | 50.7% | 51.1% | 74.3% | 173.6% |
| RTX3090 | sglang | Llama-3.1-8B | terminalbench-multiturn | 5 | 11348.0 | 146.0 | 0.987 | 20.5% | 49.6% | 53.3% | 184.1% |
| RTX3090 | sglang | Llama-3.1-8B | terminalbench-multiturn-synth | 2 | 3102.5 | 129.0 | 0.958 | 88.8% | 93.7% | 97.5% | 21.3% |
| RTX3090 | sglang | gpt-oss-20b | chat-multiturn | 7 | 2285.0 | 1665.0 | 0.587 | 41.9% | 280.4% | 97.7% | 74.9% |
| RTX3090 | sglang | gpt-oss-20b | chat-multiturn-synth | 11 | 1752.0 | 236.0 | 0.865 | 74.0% | 71.9% | 80.9% | 69.8% |
| RTX3090 | sglang | gpt-oss-20b | osworld-multiturn | 4 | 4488.5 | 297.5 | 0.953 | 49.5% | 62.9% | 95.5% | 85.5% |
| RTX3090 | sglang | gpt-oss-20b | osworld-multiturn-synth | 11 | 2250.0 | 99.0 | 0.956 | 80.8% | 81.5% | 88.6% | 77.6% |
| RTX3090 | sglang | gpt-oss-20b | swebench-multiturn | 5 | 2078.0 | 118.0 | 0.943 | 159.6% | 103.5% | 84.2% | 36.0% |
| RTX3090 | sglang | gpt-oss-20b | swebench-multiturn-synth | 11 | 4554.0 | 131.0 | 0.971 | 96.9% | 95.4% | 90.9% | 94.2% |
| RTX3090 | sglang | gpt-oss-20b | terminalbench-multiturn | 5 | 4137.0 | 110.0 | 0.973 | 275.7% | 169.0% | 86.4% | 39.4% |
| RTX3090 | sglang | gpt-oss-20b | terminalbench-multiturn-synth | 11 | 3316.0 | 125.0 | 0.959 | 94.2% | 95.1% | 88.4% | 94.8% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-long | 30 | 920.0 | 168.0 | 0.817 | 31.6% | 31.2% | 20.0% | 55.1% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-medium | 30 | 969.0 | 277.0 | 0.714 | 17.7% | 16.2% | 56.3% | 43.8% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-short | 30 | 860.0 | 289.0 | 0.664 | 17.5% | 14.1% | 89.4% | 44.0% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-synth | 11 | 1712.0 | 236.0 | 0.862 | 35.4% | 22.4% | 85.5% | 21.6% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-long | 30 | 3421.0 | 30.0 | 0.991 | 44.5% | 54.1% | 64.4% | 68.6% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 30 | 3246.0 | 65.0 | 0.98 | 40.6% | 51.5% | 74.8% | 57.5% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-short | 30 | 3180.0 | 146.0 | 0.954 | 39.1% | 52.4% | 83.8% | 60.3% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-synth | 11 | 9760.0 | 98.0 | 0.99 | 65.4% | 58.3% | 91.8% | 102.8% |
| RTX3090 | vllm | Llama-3.1-8B | swebench-multiturn-short | 30 | 4242.5 | 193.0 | 0.954 | 75.2% | 76.4% | 88.3% | 60.2% |
| RTX3090 | vllm | Llama-3.1-8B | swebench-multiturn-synth | 8 | 6519.5 | 134.0 | 0.98 | 88.9% | 89.8% | 93.0% | 39.7% |
| RTX3090 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 30 | 3530.0 | 247.0 | 0.931 | 71.9% | 75.0% | 87.9% | 56.4% |
| RTX3090 | vllm | Llama-3.1-8B | terminalbench-multiturn-synth | 11 | 3458.0 | 128.0 | 0.963 | 86.7% | 88.3% | 91.7% | 28.4% |
| RTX3090 | vllm | Qwen3.5-9B | chat-multiturn-synth | 6 | 742.0 | 313.0 | 0.582 | 76.8% | 77.1% | 93.3% | 61.2% |
| RTX3090 | vllm | Qwen3.5-9B | osworld-multiturn-synth | 10 | 1341.0 | 300.0 | 0.822 | 84.6% | 86.3% | 94.2% | 48.9% |
| RTX3090 | vllm | Qwen3.5-9B | swebench-multiturn-synth | 11 | 1541.0 | 134.0 | 0.914 | 86.4% | 86.2% | 89.8% | 48.8% |
| RTX3090 | vllm | Qwen3.5-9B | terminalbench-multiturn-synth | 11 | 1466.0 | 124.0 | 0.916 | 89.2% | 90.1% | 92.6% | 47.6% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn | 5 | 558.0 | 220.0 | 0.606 | 265.3% | 288.3% | 1179.5% | 94.2% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-long | 10 | 947.0 | 166.0 | 0.825 | 71.3% | 51.9% | 4.3% | 42.1% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-medium | 10 | 1001.0 | 277.0 | 0.723 | 37.3% | 28.8% | 42.4% | 24.0% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-short | 10 | 893.0 | 290.0 | 0.675 | 19.2% | 13.5% | 80.1% | 24.1% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-synth | 11 | 1748.0 | 236.0 | 0.865 | 53.3% | 46.8% | 66.1% | 28.8% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn | 5 | 2686.0 | 26.0 | 0.99 | 85.1% | 60.6% | 81.3% | 152.1% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-long | 10 | 3453.0 | 30.0 | 0.991 | 38.5% | 53.1% | 55.9% | 48.1% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-medium | 10 | 3279.0 | 64.0 | 0.98 | 36.7% | 42.6% | 66.9% | 44.9% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-short | 10 | 3213.0 | 145.0 | 0.955 | 48.2% | 53.1% | 82.2% | 49.8% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-synth | 11 | 2246.0 | 99.0 | 0.956 | 56.7% | 59.6% | 73.7% | 39.9% |
| RTX3090 | vllm | gpt-oss-20b | swebench-multiturn | 4 | 4263.0 | 114.0 | 0.973 | 228.3% | 164.9% | 141.7% | 226.2% |
| RTX3090 | vllm | gpt-oss-20b | swebench-multiturn-short | 9 | 4453.0 | 171.0 | 0.962 | 79.2% | 78.4% | 87.7% | 79.0% |
| RTX3090 | vllm | gpt-oss-20b | swebench-multiturn-synth | 11 | 4550.0 | 132.0 | 0.971 | 90.1% | 86.3% | 85.8% | 82.2% |
| RTX3090 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 10 | 3560.5 | 250.0 | 0.93 | 68.3% | 67.8% | 84.2% | 74.2% |
| RTX3090 | vllm | gpt-oss-20b | terminalbench-multiturn-synth | 11 | 3312.0 | 125.0 | 0.962 | 89.0% | 90.0% | 82.3% | 84.4% |

## Notes

- Calibration artifacts are diagnostic only; serving predictions do not consume empirical multipliers.
- Multi-turn TTFT should be evaluated against cache-aware TTFT, not cumulative full-prefill TTFT.
- Prefix-cache rows without `perTurn` remain unsupported rather than using inferred cache state.
- MoE decode gaps remain visible as raw analytical error until a kernel-level MoE model is added.
