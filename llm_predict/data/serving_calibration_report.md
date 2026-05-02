# Serving Calibration Report

Calibration excludes legacy `chat-short`, `chat-medium`, and `chat-long`.
The active scope is canonical single-turn/stress, high concurrency, and multi-turn cache analysis.

## Calibration Coverage

| GPU | Backend | Version | Model | Status | C=1 rows | Profiles | Long rows | Raw TTFT MAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| A100 | sglang | 0.5.9 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 10.7% |
| A100 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 19.1% |
| A100 | vllm | 0.19.0 | Qwen3.5-9B | low_confidence | 1 | 1 | 0 | 72.23% |
| A100 | vllm | 0.19.0 | gpt-oss-20b | low_confidence | 1 | 1 | 0 | 239.84% |
| H100 | sglang | 0.5.9 | Llama-3.1-70B | low_confidence | 1 | 1 | 0 | 62.63% |
| H100 | sglang | 0.5.9 | Llama-3.1-8B | low_confidence | 4 | 4 | 2 | 36.53% |
| H100 | sglang | 0.5.9 | Llama-3.3-70B | low_confidence | 1 | 1 | 0 | 62.2% |
| H100 | sglang | 0.5.9 | Qwen2.5-72B | low_confidence | 1 | 1 | 0 | 57.01% |
| H100 | sglang | 0.5.9 | Qwen3.5-9B | low_confidence | 4 | 4 | 2 | 65.22% |
| H100 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 4 | 4 | 2 | 51.9% |
| H100 | vllm | 0.19.0 | Qwen3.5-9B | low_confidence | 4 | 4 | 2 | 64.95% |
| H100 | vllm | 0.19.0 | gpt-oss-20b | low_confidence | 4 | 4 | 2 | 31.01% |
| H100 | vllm | 0.19.1 | Llama-3.1-70B | low_confidence | 1 | 1 | 0 | 206.31% |
| H100 | vllm | 0.19.1 | Llama-3.3-70B | low_confidence | 1 | 1 | 0 | 196.92% |
| H100 | vllm | 0.19.1 | Qwen2.5-72B | low_confidence | 1 | 1 | 0 | 183.55% |
| RTX3090 | sglang | 0.5.9 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 0.8% |
| RTX3090 | sglang | 0.5.9 | gpt-oss-20b | low_confidence | 1 | 1 | 0 | 172.0% |
| RTX3090 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 64.91% |
| RTX3090 | vllm | 0.19.0 | Qwen3.5-9B | low_confidence | 3 | 3 | 1 | 47.32% |
| RTX3090 | vllm | 0.19.0 | gpt-oss-20b | low_confidence | 3 | 3 | 1 | 825.31% |

## Prefix Cache Multi-turn Summary

| GPU | Backend | Model | Profile | Rows | Median ctx | Median new | Cache hit | Full E2EL | Cache-aware E2EL | Cache-aware TTFT | Cache-aware TPOT |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 | vllm | Llama-3.1-8B | chat-multiturn | 5 | 516.0 | 213.0 | 0.587 | 41.4% | 56.6% | 65.9% | 23.9% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-long | 10 | 920.0 | 168.0 | 0.817 | 16.3% | 17.6% | 93.0% | 28.5% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 10 | 969.0 | 277.0 | 0.714 | 15.2% | 14.0% | 94.7% | 10.8% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-short | 11 | 860.0 | 289.0 | 0.664 | 19.4% | 15.3% | 88.9% | 13.6% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn | 5 | 2675.0 | 27.0 | 0.99 | 16.3% | 30.0% | 97.3% | 49.2% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-long | 10 | 3421.0 | 30.0 | 0.991 | 46.0% | 51.8% | 98.9% | 61.9% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 10 | 3246.0 | 65.0 | 0.98 | 42.1% | 47.6% | 99.2% | 43.2% |
| A100 | vllm | Llama-3.1-8B | osworld-multiturn-short | 10 | 3180.0 | 146.0 | 0.954 | 41.1% | 46.8% | 99.4% | 48.8% |
| A100 | vllm | Llama-3.1-8B | swebench-multiturn | 5 | 13049.0 | 114.0 | 0.991 | 102.0% | 53.2% | 97.2% | 187.4% |
| A100 | vllm | Llama-3.1-8B | swebench-multiturn-short | 10 | 4253.0 | 186.0 | 0.956 | 70.2% | 72.3% | 99.4% | 51.5% |
| A100 | vllm | Llama-3.1-8B | terminalbench-multiturn | 5 | 13084.0 | 134.0 | 0.99 | 214.5% | 50.6% | 95.4% | 214.2% |
| A100 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 10 | 3506.0 | 247.0 | 0.93 | 65.3% | 67.8% | 98.9% | 55.3% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn | 5 | 516.0 | 227.0 | 0.56 | 45.8% | 52.1% | 85.6% | 35.1% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-long | 9 | 922.0 | 172.0 | 0.813 | 20.2% | 18.3% | 97.6% | 12.7% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-medium | 9 | 980.0 | 287.0 | 0.707 | 22.1% | 19.3% | 96.5% | 15.3% |
| A100 | vllm | Qwen3.5-9B | chat-multiturn-short | 9 | 876.0 | 297.0 | 0.661 | 24.4% | 23.9% | 89.6% | 22.8% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn | 5 | 2795.0 | 28.0 | 0.99 | 11.9% | 21.0% | 96.7% | 62.3% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-long | 9 | 3195.0 | 32.0 | 0.99 | 41.9% | 47.1% | 99.0% | 21.6% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-medium | 8 | 3087.0 | 52.0 | 0.983 | 33.9% | 43.2% | 99.1% | 44.0% |
| A100 | vllm | Qwen3.5-9B | osworld-multiturn-short | 6 | 2933.0 | 53.0 | 0.982 | 44.8% | 45.9% | 99.4% | 34.8% |
| A100 | vllm | Qwen3.5-9B | terminalbench-multiturn-short | 5 | 2250.0 | 152.0 | 0.932 | 63.0% | 65.6% | 99.1% | 47.7% |
| A100 | vllm | gpt-oss-20b | chat-multiturn | 5 | 558.0 | 220.0 | 0.606 | 958.6% | 1050.9% | 17.6% | 540.4% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-long | 10 | 947.0 | 166.0 | 0.825 | 495.4% | 478.8% | 72.9% | 460.8% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-medium | 10 | 1001.0 | 277.0 | 0.723 | 448.2% | 453.7% | 72.5% | 413.8% |
| A100 | vllm | gpt-oss-20b | chat-multiturn-short | 10 | 893.0 | 290.0 | 0.675 | 307.8% | 309.3% | 89.3% | 276.4% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn | 5 | 2686.0 | 26.0 | 0.99 | 863.0% | 735.5% | 50.2% | 909.3% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-long | 10 | 3453.0 | 30.0 | 0.991 | 299.0% | 253.7% | 95.2% | 436.8% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-medium | 10 | 3279.0 | 64.0 | 0.98 | 199.1% | 188.5% | 96.1% | 327.3% |
| A100 | vllm | gpt-oss-20b | osworld-multiturn-short | 10 | 3213.0 | 145.0 | 0.955 | 111.2% | 93.4% | 97.1% | 219.1% |
| A100 | vllm | gpt-oss-20b | swebench-multiturn | 4 | 4263.0 | 114.0 | 0.973 | 816.9% | 455.2% | 70.8% | 1071.9% |
| A100 | vllm | gpt-oss-20b | swebench-multiturn-short | 8 | 4453.0 | 171.0 | 0.962 | 102.2% | 86.4% | 95.0% | 319.0% |
| A100 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 10 | 3572.5 | 250.0 | 0.93 | 222.5% | 206.5% | 91.3% | 302.9% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 6.4% | 7.3% | 92.9% | 4.1% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-medium | 7 | 969.0 | 277.0 | 0.714 | 2.4% | 3.6% | 73.5% | 2.6% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-short | 7 | 860.0 | 289.0 | 0.664 | 2.9% | 2.7% | 65.5% | 3.0% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 41.9% | 48.6% | 98.9% | 32.9% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-short | 5 | 4262.0 | 74.0 | 0.983 | 59.3% | 64.0% | 99.1% | 49.7% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14317.0 | 375.0 | 0.974 | 85.2% | 84.6% | 99.7% | 75.2% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-short | 7 | 8015.0 | 451.0 | 0.944 | 83.6% | 83.4% | 98.4% | 75.1% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 382.0 | 0.965 | 78.4% | 79.5% | 99.5% | 74.9% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-short | 7 | 4976.0 | 461.0 | 0.907 | 55.5% | 64.6% | 96.1% | 35.9% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 26.0% | 27.7% | 97.7% | 14.7% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-medium | 7 | 980.0 | 287.0 | 0.707 | 27.2% | 24.3% | 90.3% | 16.8% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-short | 7 | 876.0 | 297.0 | 0.661 | 35.0% | 34.1% | 80.1% | 35.7% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4774.0 | 100.0 | 0.979 | 49.9% | 56.9% | 99.1% | 44.5% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-short | 5 | 4541.0 | 74.0 | 0.984 | 70.2% | 75.6% | 99.3% | 53.2% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-medium | 7 | 15060.0 | 358.0 | 0.976 | 86.6% | 87.5% | 99.5% | 77.5% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-short | 7 | 8894.0 | 506.0 | 0.943 | 85.4% | 88.0% | 98.6% | 80.8% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11992.0 | 391.0 | 0.967 | 72.1% | 74.3% | 99.2% | 60.6% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-short | 7 | 5462.0 | 502.0 | 0.908 | 70.6% | 74.1% | 97.3% | 69.5% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-long | 3 | 920.0 | 168.0 | 0.817 | 703.9% | 715.6% | 78.0% | 769.6% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-medium | 3 | 969.0 | 277.0 | 0.714 | 607.8% | 638.6% | 77.8% | 692.6% |
| H100 | vllm | Llama-3.1-70B | chat-multiturn-short | 3 | 860.0 | 289.0 | 0.664 | 441.7% | 479.7% | 84.3% | 538.1% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 7.2% | 5.3% | 95.9% | 9.7% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 8 | 969.0 | 277.0 | 0.714 | 5.7% | 8.2% | 81.2% | 8.6% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-short | 8 | 860.0 | 289.0 | 0.664 | 1.8% | 5.7% | 77.4% | 6.2% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 44.2% | 51.0% | 99.1% | 31.8% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-short | 5 | 4478.0 | 74.0 | 0.983 | 61.4% | 65.8% | 99.1% | 45.7% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14132.0 | 427.0 | 0.97 | 83.9% | 84.6% | 99.7% | 72.0% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-short | 8 | 8015.0 | 451.0 | 0.944 | 81.0% | 82.5% | 98.5% | 70.9% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 390.0 | 0.965 | 77.1% | 78.0% | 99.6% | 73.8% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 8 | 4976.0 | 461.0 | 0.907 | 44.3% | 48.4% | 96.4% | 28.9% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-long | 3 | 920.0 | 168.0 | 0.817 | 730.2% | 708.2% | 76.4% | 766.9% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-medium | 3 | 969.0 | 277.0 | 0.714 | 612.6% | 643.6% | 77.1% | 694.1% |
| H100 | vllm | Llama-3.3-70B | chat-multiturn-short | 3 | 860.0 | 289.0 | 0.664 | 451.8% | 490.9% | 84.5% | 537.4% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-long | 3 | 905.0 | 169.0 | 0.813 | 674.2% | 669.4% | 78.3% | 726.8% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-medium | 3 | 955.0 | 280.0 | 0.707 | 585.1% | 615.1% | 77.7% | 644.2% |
| H100 | vllm | Qwen2.5-72B | chat-multiturn-short | 3 | 848.0 | 291.0 | 0.657 | 427.3% | 465.0% | 83.8% | 496.8% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 7.0% | 9.7% | 98.0% | 12.1% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-medium | 8 | 980.0 | 287.0 | 0.707 | 16.3% | 15.7% | 91.6% | 12.4% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-short | 8 | 876.0 | 297.0 | 0.661 | 21.2% | 20.1% | 89.7% | 20.9% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4778.0 | 100.0 | 0.979 | 53.2% | 61.0% | 99.4% | 38.8% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-short | 5 | 4833.0 | 74.0 | 0.984 | 71.2% | 75.4% | 99.5% | 50.8% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-medium | 7 | 14502.0 | 424.0 | 0.971 | 79.8% | 80.2% | 99.5% | 64.7% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-short | 8 | 8894.0 | 506.0 | 0.943 | 58.4% | 67.4% | 97.9% | 51.7% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11673.0 | 456.0 | 0.962 | 42.2% | 35.7% | 99.0% | 19.7% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-short | 8 | 5462.0 | 502.0 | 0.908 | 43.6% | 38.9% | 95.9% | 24.7% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-long | 5 | 947.0 | 166.0 | 0.825 | 269.1% | 254.5% | 40.3% | 276.0% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-medium | 5 | 1001.0 | 277.0 | 0.723 | 256.3% | 256.8% | 3.1% | 255.8% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-short | 5 | 893.0 | 290.0 | 0.675 | 255.4% | 254.6% | 4.3% | 249.0% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-medium | 5 | 4542.0 | 5.0 | 0.999 | 128.4% | 100.3% | 95.6% | 166.5% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-short | 5 | 4494.0 | 75.0 | 0.983 | 44.4% | 21.1% | 96.8% | 119.9% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-medium | 5 | 14017.0 | 433.0 | 0.969 | 222.8% | 109.2% | 82.7% | 313.3% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-short | 5 | 8079.0 | 454.0 | 0.944 | 64.3% | 36.6% | 90.2% | 190.3% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-medium | 5 | 11213.0 | 409.0 | 0.963 | 309.6% | 220.3% | 74.4% | 260.0% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 5 | 5037.0 | 465.0 | 0.907 | 455.2% | 350.9% | 22.7% | 304.3% |
| RTX3090 | sglang | Llama-3.1-8B | chat-multiturn | 5 | 516.0 | 213.0 | 0.587 | 101.1% | 128.5% | 45.0% | 62.2% |
| RTX3090 | sglang | Llama-3.1-8B | osworld-multiturn | 5 | 10239.0 | 521.0 | 0.949 | 35.1% | 56.5% | 97.2% | 286.8% |
| RTX3090 | sglang | Llama-3.1-8B | swebench-multiturn | 5 | 11616.0 | 111.0 | 0.99 | 52.3% | 76.5% | 99.6% | 331.7% |
| RTX3090 | sglang | Llama-3.1-8B | terminalbench-multiturn | 5 | 11348.0 | 146.0 | 0.987 | 18.1% | 66.0% | 98.9% | 346.9% |
| RTX3090 | sglang | gpt-oss-20b | chat-multiturn | 5 | 561.0 | 220.0 | 0.608 | 847.2% | 872.4% | 10.5% | 654.9% |
| RTX3090 | sglang | gpt-oss-20b | osworld-multiturn | 5 | 2687.0 | 26.0 | 0.99 | 269.9% | 210.2% | 94.0% | 583.7% |
| RTX3090 | sglang | gpt-oss-20b | swebench-multiturn | 5 | 2078.0 | 118.0 | 0.943 | 737.6% | 358.4% | 67.9% | 769.9% |
| RTX3090 | sglang | gpt-oss-20b | terminalbench-multiturn | 5 | 4137.0 | 110.0 | 0.973 | 1068.4% | 727.2% | 66.6% | 801.3% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-long | 10 | 920.0 | 168.0 | 0.817 | 30.9% | 32.0% | 96.7% | 41.4% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-medium | 10 | 969.0 | 277.0 | 0.714 | 27.0% | 25.8% | 98.2% | 32.5% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-short | 10 | 860.0 | 289.0 | 0.664 | 24.5% | 25.7% | 99.4% | 27.4% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-long | 10 | 3421.0 | 30.0 | 0.991 | 45.2% | 54.2% | 99.6% | 63.5% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 10 | 3246.0 | 65.0 | 0.98 | 44.6% | 49.9% | 99.6% | 58.7% |
| RTX3090 | vllm | Llama-3.1-8B | osworld-multiturn-short | 10 | 3180.0 | 146.0 | 0.954 | 51.0% | 59.1% | 99.6% | 60.9% |
| RTX3090 | vllm | Llama-3.1-8B | swebench-multiturn-short | 10 | 4242.5 | 193.0 | 0.954 | 79.2% | 82.5% | 99.6% | 61.1% |
| RTX3090 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 10 | 3530.0 | 247.0 | 0.931 | 78.4% | 81.1% | 99.6% | 55.6% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn | 5 | 558.0 | 220.0 | 0.606 | 1597.1% | 1679.0% | 273.8% | 902.8% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-long | 10 | 947.0 | 166.0 | 0.825 | 716.6% | 672.4% | 77.1% | 694.0% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-medium | 10 | 1001.0 | 277.0 | 0.723 | 513.0% | 514.1% | 79.0% | 479.3% |
| RTX3090 | vllm | gpt-oss-20b | chat-multiturn-short | 10 | 893.0 | 290.0 | 0.675 | 382.1% | 379.0% | 92.3% | 299.5% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn | 5 | 2686.0 | 26.0 | 0.99 | 777.2% | 642.8% | 81.1% | 1129.4% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-long | 10 | 3453.0 | 30.0 | 0.991 | 391.9% | 306.5% | 95.1% | 584.1% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-medium | 10 | 3279.0 | 64.0 | 0.98 | 241.9% | 214.4% | 96.1% | 409.9% |
| RTX3090 | vllm | gpt-oss-20b | osworld-multiturn-short | 10 | 3213.0 | 145.0 | 0.955 | 130.4% | 102.7% | 96.5% | 260.5% |
| RTX3090 | vllm | gpt-oss-20b | swebench-multiturn | 4 | 4263.0 | 114.0 | 0.973 | 1407.7% | 697.5% | 52.3% | 1939.1% |
| RTX3090 | vllm | gpt-oss-20b | swebench-multiturn-short | 9 | 4453.0 | 171.0 | 0.962 | 35.7% | 30.8% | 97.2% | 38.5% |
| RTX3090 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 10 | 3560.5 | 250.0 | 0.93 | 39.0% | 34.7% | 95.1% | 47.1% |

## Notes

- Calibration artifacts are diagnostic only; serving predictions do not consume empirical multipliers.
- Multi-turn TTFT should be evaluated against cache-aware TTFT, not cumulative full-prefill TTFT.
- Prefix-cache rows without `perTurn` remain unsupported rather than using inferred cache state.
- MoE decode gaps remain visible as raw analytical error until a kernel-level MoE model is added.
