# Running Benchmarks

## Quick Reference

| Goal | Command |
|------|---------|
| Single profile, single concurrency | `./scripts/bench.sh --profile chatbot-short --concurrency 20 --mode single-turn` |
| Chat sweep | `./benchmark.sh chat` |
| All production profiles | `./benchmark.sh production` |
| Cross-validate with InferenceX | `./benchmark.sh cross_validate` |
| Full matrix (all models × engines) | `bash run_all_benchmarks.sh all` |
| SGLang only | `bash run_all_benchmarks.sh sglang` |
| vLLM only | `bash run_all_benchmarks.sh vllm` |

## Three Entry Points

### 1. `scripts/bench.sh` — Single Run
Runs one profile at one concurrency level.
```bash
./scripts/bench.sh --profile chatbot-short --concurrency 20 --mode single-turn
```

### 2. `benchmark.sh` — Canonical Sweeps
Runs a named target across multiple concurrency levels. Edit the CONFIG section for your server.
```bash
BENCH_URL=http://localhost:8000/v1/chat/completions ./benchmark.sh chatbot
```
- **Targets:** `chat`, `production`, `stress`, `cross_validate`
- **Config env vars:** `BENCH_URL`, `BENCH_MODEL`, `PYTHON`
- **Default concurrency:** `1 10 20 40 80 120`

### 3. `run_all_benchmarks.sh` — Full Matrix Orchestrator
Runs all models across SGLang and vLLM. Manages server lifecycle (start, wait, benchmark, kill).
```bash
bash run_all_benchmarks.sh all > /tmp/master_bench.log 2>&1 &
```
- **Resume support:** Skips existing result files
- **Concurrency sweep:** `1 10 20 40 80 120 160 200 256 320`
- **Profiles:** See `src/workloads/profiles.py` for current profile names (Tier 1-4)
- **Model registry** defined in `.claude/tools/model-registry.md`

## Server Launch

Always start the server before benchmarking:
```bash
# vLLM — single-turn (prefix caching ON)
CUDA_VISIBLE_DEVICES=0 ./scripts/launch_server.sh single-turn --model /workspace/models/Llama-3.1-8B-Instruct

# vLLM — stress-test (prefix caching OFF)
CUDA_VISIBLE_DEVICES=0 ./scripts/launch_server.sh stress-test --model /workspace/models/Llama-3.1-8B-Instruct

# SGLang (run_all_benchmarks.sh handles this automatically)
python -m sglang.launch_server --model <MODEL> --port 8000 --tp <N>
```
Wait for health: `curl http://localhost:8000/health`

## Benchmark Modes

| Mode | Prefix Cache | `--ignore-eos` | Use Case |
|------|:-----------:|:--------------:|----------|
| stress-test | OFF | YES (auto) | Cross-validation with InferenceX |
| single-turn | ON | NO | Realistic workload measurement |
| multi-turn | ON | NO | KV cache reuse (not yet implemented) |

## Common Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--num-requests` | 200 | Requests per run (50-100 at conc=1) |
| `--warmup` | 5 | Warmup requests (excluded from timing) |
| `--timeout` | 300 | Per-request timeout in seconds |
| `--ignore-eos` | off | Required for FP8 + random tokens |

## Results

- **JSON output:** `results/<model>_tp<N>_<engine>/<descriptive-name>.json`
- **Human summary:** `markdown/results.md`
- **Metrics:** TTFT p50/p90/p99, TPOT p50/p90/p99, E2EL, output tok/s, req/s

## Monitoring a Long Run

```bash
tail -f /tmp/master_bench.log
grep -E "DONE:|FAIL:|══.*\|" /tmp/master_bench.log
```
