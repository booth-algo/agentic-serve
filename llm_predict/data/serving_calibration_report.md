# Serving Calibration Report

Calibration excludes legacy `chat-short`, `chat-medium`, and `chat-long`.
The active scope is canonical single-turn/stress, high concurrency, and multi-turn cache analysis.

## Calibration Coverage

| GPU | Backend | Version | Model | Status | C=1 rows | Profiles | Long rows | TTFT fit MAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| A100 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 0.0% |
| A100 | vllm | 0.19.0 | Qwen3.5-9B | low_confidence | 1 | 1 | 0 | 0.0% |
| A100 | vllm | 0.19.0 | gpt-oss-20b | low_confidence | 1 | 1 | 0 | 0.0% |
| H100 | sglang | 0.5.9 | Llama-3.1-8B | high_confidence | 4 | 4 | 2 | 7.05% |
| H100 | sglang | 0.5.9 | Qwen3.5-9B | medium_confidence | 4 | 4 | 2 | 19.47% |
| H100 | vllm | 0.19.0 | Llama-3.1-8B | high_confidence | 4 | 4 | 2 | 1.28% |
| H100 | vllm | 0.19.0 | Qwen3.5-9B | high_confidence | 4 | 4 | 2 | 7.85% |
| H100 | vllm | 0.19.0 | gpt-oss-20b | medium_confidence | 4 | 4 | 2 | 24.12% |
| RTX3090 | vllm | 0.19.0 | Llama-3.1-8B | low_confidence | 1 | 1 | 0 | 0.0% |

## Prefix Cache Multi-turn Summary

| GPU | Backend | Model | Profile | Rows | Median ctx | Median new | Cache hit | Full E2EL | Cache raw E2EL | Cache+contention E2EL | TTFT raw→cal | TPOT raw→cal |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 | vllm | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 12.6% | 12.0% | 12.0% | 86.5%→86.5% | 12.5%→12.5% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 7 | 969.0 | 277.0 | 0.714 | 25.4% | 24.2% | 24.2% | 79.1%→79.1% | 18.9%→18.9% |
| A100 | vllm | Llama-3.1-8B | chat-multiturn-short | 8 | 860.0 | 289.0 | 0.664 | 27.5% | 26.0% | 26.0% | 70.1%→70.1% | 24.9%→24.9% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 11.9% | 8.2% | 5.2% | 40.4%→0.0% | 9.1%→0.0% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-medium | 7 | 969.0 | 277.0 | 0.714 | 7.7% | 7.7% | 3.2% | 21.7%→0.0% | 5.4%→0.0% |
| H100 | sglang | Llama-3.1-8B | chat-multiturn-short | 7 | 860.0 | 289.0 | 0.664 | 3.4% | 4.4% | 4.0% | 12.2%→0.0% | 2.7%→0.0% |
| H100 | sglang | Llama-3.1-8B | coding-agent | 10 | 6061.0 | 1469.0 | 0.758 | 10.4% | 22.5% | 2.4% | 35.8%→0.0% | 30.7%→0.0% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 53.5% | 62.4% | 4.4% | 96.5%→0.0% | 48.3%→0.0% |
| H100 | sglang | Llama-3.1-8B | osworld-multiturn-short | 5 | 4262.0 | 74.0 | 0.983 | 67.7% | 74.9% | 6.1% | 97.5%→0.0% | 68.7%→0.0% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14317.0 | 375.0 | 0.974 | 69.3% | 94.2% | 4.6% | 96.7%→0.0% | 93.3%→0.0% |
| H100 | sglang | Llama-3.1-8B | swebench-multiturn-short | 7 | 8015.0 | 451.0 | 0.944 | 69.5% | 91.8% | 12.0% | 92.7%→0.0% | 90.7%→0.0% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 382.0 | 0.965 | 63.2% | 91.3% | 1.1% | 94.7%→0.0% | 91.0%→0.0% |
| H100 | sglang | Llama-3.1-8B | terminalbench-multiturn-short | 7 | 4976.0 | 461.0 | 0.907 | 24.8% | 71.1% | 11.2% | 73.3%→0.0% | 67.1%→0.0% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 3.9% | 8.7% | 4.2% | 32.8%→0.0% | 15.6%→0.0% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-medium | 7 | 980.0 | 287.0 | 0.707 | 3.9% | 4.3% | 1.9% | 36.6%→0.0% | 10.2%→0.0% |
| H100 | sglang | Qwen3.5-9B | chat-multiturn-short | 7 | 876.0 | 297.0 | 0.661 | 3.3% | 4.9% | 2.6% | 71.8%→0.0% | 7.2%→0.0% |
| H100 | sglang | Qwen3.5-9B | coding-agent | 10 | 6316.0 | 5076.0 | 0.196 | 12.0% | 12.6% | 3.8% | 250.1%→0.0% | 21.2%→0.0% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4774.0 | 100.0 | 0.979 | 34.8% | 49.6% | 3.4% | 75.7%→0.0% | 40.3%→0.0% |
| H100 | sglang | Qwen3.5-9B | osworld-multiturn-short | 5 | 4541.0 | 74.0 | 0.984 | 50.2% | 69.1% | 9.6% | 82.9%→0.0% | 54.2%→0.0% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-medium | 7 | 15060.0 | 358.0 | 0.976 | 60.1% | 85.9% | 2.4% | 89.2%→0.0% | 88.7%→0.0% |
| H100 | sglang | Qwen3.5-9B | swebench-multiturn-short | 7 | 8894.0 | 506.0 | 0.943 | 48.1% | 78.1% | 6.2% | 81.2%→0.0% | 87.9%→0.0% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11992.0 | 391.0 | 0.967 | 48.6% | 71.5% | 17.5% | 83.8%→0.0% | 77.7%→0.0% |
| H100 | sglang | Qwen3.5-9B | terminalbench-multiturn-short | 7 | 5462.0 | 502.0 | 0.908 | 33.4% | 50.3% | 30.5% | 54.6%→0.0% | 73.9%→0.0% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-long | 7 | 920.0 | 168.0 | 0.817 | 15.3% | 11.9% | 5.9% | 25.6%→0.0% | 12.5%→0.0% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-medium | 8 | 969.0 | 277.0 | 0.714 | 8.8% | 8.7% | 2.5% | 12.6%→0.0% | 8.3%→0.0% |
| H100 | vllm | Llama-3.1-8B | chat-multiturn-short | 8 | 860.0 | 289.0 | 0.664 | 6.1% | 7.6% | 2.9% | 12.7%→0.0% | 6.2%→0.0% |
| H100 | vllm | Llama-3.1-8B | coding-agent | 10 | 6061.0 | 1494.0 | 0.754 | 14.8% | 29.3% | 3.6% | 15.3%→0.0% | 32.9%→0.0% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-medium | 5 | 4519.0 | 8.0 | 0.998 | 57.4% | 65.7% | 3.5% | 95.4%→0.0% | 48.8%→0.0% |
| H100 | vllm | Llama-3.1-8B | osworld-multiturn-short | 5 | 4478.0 | 74.0 | 0.983 | 68.8% | 75.4% | 10.8% | 96.8%→0.0% | 66.8%→0.0% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-medium | 7 | 14132.0 | 427.0 | 0.97 | 73.8% | 94.5% | 3.4% | 97.1%→0.0% | 92.7%→0.0% |
| H100 | vllm | Llama-3.1-8B | swebench-multiturn-short | 8 | 8015.0 | 451.0 | 0.944 | 71.6% | 90.0% | 3.7% | 93.4%→0.0% | 89.4%→0.0% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-medium | 7 | 11050.0 | 390.0 | 0.965 | 70.8% | 91.9% | 12.3% | 95.0%→0.0% | 91.5%→0.0% |
| H100 | vllm | Llama-3.1-8B | terminalbench-multiturn-short | 8 | 4976.0 | 461.0 | 0.907 | 32.2% | 62.8% | 10.1% | 75.9%→0.0% | 60.2%→0.0% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-long | 7 | 922.0 | 172.0 | 0.813 | 6.5% | 4.4% | 2.6% | 36.7%→0.0% | 9.7%→0.0% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-medium | 8 | 980.0 | 287.0 | 0.707 | 1.8% | 3.1% | 1.4% | 28.4%→0.0% | 8.0%→0.0% |
| H100 | vllm | Qwen3.5-9B | chat-multiturn-short | 8 | 876.0 | 297.0 | 0.661 | 3.2% | 3.8% | 1.9% | 30.6%→0.0% | 8.6%→0.0% |
| H100 | vllm | Qwen3.5-9B | coding-agent | 10 | 6316.0 | 1788.0 | 0.717 | 16.2% | 6.8% | 6.9% | 83.8%→0.0% | 20.5%→0.0% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-medium | 5 | 4778.0 | 100.0 | 0.979 | 42.7% | 57.2% | 5.0% | 81.2%→0.0% | 41.6%→0.0% |
| H100 | vllm | Qwen3.5-9B | osworld-multiturn-short | 5 | 4833.0 | 74.0 | 0.984 | 51.0% | 70.7% | 7.2% | 84.7%→0.0% | 60.6%→0.0% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-medium | 7 | 14502.0 | 424.0 | 0.971 | 26.6% | 82.7% | 2.6% | 86.3%→0.0% | 81.2%→0.0% |
| H100 | vllm | Qwen3.5-9B | swebench-multiturn-short | 8 | 8894.0 | 506.0 | 0.943 | 28.6% | 56.7% | 7.8% | 59.9%→0.0% | 63.1%→0.0% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-medium | 7 | 11673.0 | 456.0 | 0.962 | 50.7% | 38.9% | 26.5% | 62.0%→0.0% | 52.2%→0.0% |
| H100 | vllm | Qwen3.5-9B | terminalbench-multiturn-short | 8 | 5462.0 | 502.0 | 0.908 | 59.9% | 28.0% | 38.0% | 35.1%→0.0% | 40.9%→0.0% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-long | 5 | 947.0 | 166.0 | 0.825 | 3.8% | 3.4% | 5.9% | 48.5%→0.0% | 7.3%→0.0% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-medium | 5 | 1001.0 | 277.0 | 0.723 | 1.4% | 2.1% | 5.3% | 36.6%→0.0% | 4.0%→0.0% |
| H100 | vllm | gpt-oss-20b | chat-multiturn-short | 5 | 893.0 | 290.0 | 0.675 | 2.9% | 3.0% | 3.3% | 27.1%→0.0% | 3.3%→0.0% |
| H100 | vllm | gpt-oss-20b | coding-agent | 10 | 6100.0 | 1804.0 | 0.704 | 23.4% | 27.8% | 26.2% | 67.8%→0.0% | 10.4%→0.0% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-medium | 5 | 4542.0 | 5.0 | 0.999 | 50.2% | 57.6% | 2.6% | 95.0%→0.0% | 32.6%→0.0% |
| H100 | vllm | gpt-oss-20b | osworld-multiturn-short | 5 | 4494.0 | 75.0 | 0.983 | 62.8% | 68.8% | 8.2% | 96.7%→0.0% | 57.2%→0.0% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-medium | 5 | 14017.0 | 433.0 | 0.969 | 45.6% | 54.3% | 2.6% | 90.6%→0.0% | 31.5%→0.0% |
| H100 | vllm | gpt-oss-20b | swebench-multiturn-short | 5 | 8079.0 | 452.0 | 0.944 | 38.9% | 51.3% | 3.1% | 79.8%→0.0% | 39.6%→0.0% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-medium | 5 | 11213.0 | 409.0 | 0.963 | 53.2% | 28.6% | 29.2% | 85.4%→0.0% | 32.5%→0.0% |
| H100 | vllm | gpt-oss-20b | terminalbench-multiturn-short | 5 | 5037.0 | 465.0 | 0.907 | 47.6% | 23.3% | 34.0% | 51.2%→0.0% | 8.3%→0.0% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-long | 6 | 920.0 | 168.0 | 0.817 | 30.3% | 30.6% | 30.6% | 93.7%→93.7% | 14.4%→14.4% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-medium | 6 | 969.0 | 277.0 | 0.714 | 42.4% | 41.3% | 41.3% | 92.7%→92.7% | 23.3%→23.3% |
| RTX3090 | vllm | Llama-3.1-8B | chat-multiturn-short | 6 | 860.0 | 289.0 | 0.664 | 39.7% | 38.5% | 38.5% | 90.9%→90.9% | 20.4%→20.4% |

## Notes

- `low_confidence` calibrations are recorded for coverage visibility but are not applied by default.
- Multi-turn TTFT should be evaluated against cache-aware TTFT, not cumulative full-prefill TTFT.
- Prefix-cache rows without `perTurn` use a C=1 TTFT-inverted cache prior before contention factors are fitted.
- Prefix-cache contention factors are fitted at GPU/backend/model/profile/concurrency granularity; this v1 is a fitted artifact, not a holdout cache-residency model.
- MoE decode factors are model-specific because fused expert kernels do not follow dense GEMM timing.
