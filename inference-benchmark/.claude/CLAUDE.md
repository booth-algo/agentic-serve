# inference-benchmark — Sequrity.ai LLM Inference Benchmarking Tool

## Project Goal

Build a realistic LLM inference benchmarking tool that tests production-representative workloads
across vLLM, SGLang, and TensorRT-LLM. Key differentiator vs InferenceX: real text prompts,
prefix caching ON for single-turn/multi-turn, and proper multi-turn KV cache reuse measurement.

## Repo Layout

```
inference-benchmark/
├── benchmark.sh              # Canonical benchmark runs (edit CONFIG section per server)
├── scripts/
│   ├── bench.sh              # Single profile runner with CLI flags
│   └── launch_server.sh      # Mode-aware vLLM launcher (stress-test vs single-turn)
├── src/
│   ├── benchmark/
│   │   ├── runner.py         # Async runner, semaphore concurrency, JSON output
│   │   ├── metrics.py        # p50/p90/p99 TTFT/TPOT/E2EL, input/output tok/s
│   │   └── client.py         # aiohttp SSE streaming, true TTFT measurement
│   ├── engines/
│   │   ├── openai_chat.py    # vLLM/SGLang OpenAI-compatible endpoint
│   │   └── trtllm.py         # TRT-LLM /generate_stream endpoint
│   ├── modes/
│   │   ├── stress_test.py    # random tokens, --ignore-eos required, prefix cache OFF
│   │   ├── single_turn.py    # ShareGPT real prompts, prefix cache ON
│   │   └── multi_turn.py     # growing conversation history, prefix cache reuse
│   └── workloads/
│       ├── profiles.py       # WorkloadProfile dataclasses, PROFILES dict
│       ├── dataset.py        # ShareGPT + FileDataset + random token generation
│       └── arrival.py        # steady, Poisson, ramp arrival patterns
├── .claude/docs/
│   ├── results.md            # All benchmark results with flag annotations
│   └── notes.md              # Architecture notes and lit review
└── results/                  # JSON output from benchmark runs
```

## Three Benchmark Modes

See `src/modes/` for documentation. Always launch server with `scripts/launch_server.sh`:

| Mode | Server flags | Client flags | Profiles |
|------|-------------|--------------|----------|
| stress-test | No prefix cache | `--ignore-eos` (auto) | random-inferencex |
| single-turn | `--enable-prefix-caching` | none | chatbot-*, rag-*, coding-*, etc. |
| multi-turn | `--enable-prefix-caching` | none | multi-turn-short, multi-turn-long |

## Server

- Local: Llama 3.1 8B, GPU 2, localhost:8000, api-key=test
- RunPod: Llama 3.1 70B FP8, 2x H100, URL in benchmark.sh CONFIG section
- Launch: `CUDA_VISIBLE_DEVICES=2 ./scripts/launch_server.sh [stress-test|single-turn]`
- **Use `--gpu-memory-utilization 0.75`** (not 0.90) — run_aime_test.py uses ~6GB on GPU 2

## Key Flags

- `--ignore-eos`: required for FP8 models with random tokens (restores 100% OSL hit rate)
- `--enable-prefix-caching`: vLLM flag for prefix/APC caching (single-turn mode)
- `--enable-chunked-prefill`: improves throughput under load (always recommended)
- `--request-rate inf` (InferenceX): fires all requests at t=0 → TTFT includes queue wait
- Our tool uses steady semaphore concurrency → TTFT reflects actual scheduling latency

## Methods

When investigating problems (performance regressions, unexpected behavior, comparing approaches),
follow the experimental method in `.claude/methods/experiment.md`:
Observe → Hypothesize → Design experiment → Execute → Conclude.

## Cross-Validation with InferenceX

Use `./benchmark.sh cross_validate` and compare **TPOT only** (not TTFT):
- InferenceX uses `--request-rate inf` → TTFT not comparable
- Both must use `--ignore-eos` for FP8 models
- Both must have prefix caching OFF
- InferenceX officially uses SGLang + `--disable-radix-cache`; we test vLLM endpoints
