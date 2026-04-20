# inference-benchmark

Realistic LLM inference benchmarking tool.

Fixes the core problems with InferenceX:
- Real text workloads instead of random tokens
- Proper SSE-parsed TTFT (not HTTP chunk approximation)
- Explicit failed request tracking
- Separate input/output token throughput
- Multi-turn conversation simulation (unique — nobody else benchmarks this)

## Quick start

```bash
# Start vLLM server
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key test

# Run benchmark
python -m src.benchmark.runner --config configs/baseline_vllm_a6000.yaml
```

## Structure

- `src/benchmark/` — async client, metrics, runner
- `src/workloads/` — profiles, dataset, arrival patterns, multi-turn
- `configs/` — YAML benchmark configs
- `results/` — JSON benchmark outputs
