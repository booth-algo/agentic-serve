# run-benchmark

Run a benchmark in the correct mode with proper server configuration.

## Steps

1. **Identify the mode**: stress-test, single-turn, or multi-turn (not yet implemented)

2. **Start the server** with the correct flags:
   ```bash
   # Stress-test (prefix cache OFF)
   CUDA_VISIBLE_DEVICES=2 ./scripts/launch_server.sh stress-test --model <MODEL>

   # Single-turn (prefix cache ON)
   CUDA_VISIBLE_DEVICES=2 ./scripts/launch_server.sh single-turn --model <MODEL>
   ```
   Wait for health: `curl http://localhost:8000/health`

3. **Run the benchmark**:
   ```bash
   # Using top-level benchmark.sh (recommended)
   ./benchmark.sh cross_validate      # stress-test mode
   ./benchmark.sh chatbot             # single-turn mode
   ./benchmark.sh output_short_long   # single-turn mode

   # Or use scripts/bench.sh for a single run
   ./scripts/bench.sh --profile chatbot-short --concurrency 20 --mode single-turn
   ```

4. **Check results**: JSON saved to `results/`, summary in `markdown/results.md`

## Common flags

| Flag | When to use |
|------|------------|
| `--ignore-eos` | FP8 models + random tokens (auto-set by --mode stress-test) |
| `--num-requests 200` | Standard; use 50-100 at concurrency=1 |
| `--warmup 5` | Always warm up (populates prefix cache for single-turn) |
| `--timeout 300` | Increase for large models or high concurrency |
