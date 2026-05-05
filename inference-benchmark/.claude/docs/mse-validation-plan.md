# MSE Validation Plan — 2026-05-05

## Goal

Prove that bucketed distributional trace replay (MSE) produces per-turn latency matching
real archived trace_replay runs on the same model/hardware. Validate head-to-head at the
per-turn level: ISL, OSL, TPOT, TTFT, and success-rate survival curve.

## What was done (Phase 0 — prep)

- **h100-2** set up with miniconda, vllm 0.20.1 env, sglang 0.5.10.post1 env, Llama-3.1-8B-Instruct model
- **4 bucketed distributional JSONs** built using same `word_count * 1.35` estimator as `scripts/build_trace_distributions.py`, ISL cap ≤32K:

| File | Sessions | Turn range | Median turns | ISL p50 | ISL p90 |
|------|----------|-----------|-------------|---------|---------|
| `swebench_multiturn_short_tracereplay_filtered-mse.json` | 5 | 13-29 | 22 | 2,445 | 9,791 |
| `swebench_multiturn_medium_tracereplay_filtered-mse.json` | 96 | 50-124 | 78 | 7,862 | 17,896 |
| `terminalbench_multiturn_short_tracereplay_filtered-mse.json` | 67 | 2-30 | 18 | 2,125 | 5,167 |
| `terminalbench_multiturn_medium_tracereplay_filtered-mse.json` | 118 | 50-124 | 76 | 6,343 | 17,833 |

- **4 new MSE profiles** added to `src/workloads/profiles.py`:
  - `swebench-multiturn-mse-short` / `swebench-multiturn-mse-medium`
  - `terminalbench-multiturn-mse-short` / `terminalbench-multiturn-mse-medium`

- **`--min-success-rate 0.75` gate** in `src/benchmark/runner.py` (added earlier)

## Remaining ISL gap

Bucketed distributions still show ISL 1.5-1.8x below REAL benchmark at early turns (0-10)
where success rate is high. Likely causes:
1. `word_count * 1.35` underestimates vs Llama tokenizer on code-heavy agent traces
2. REAL profile samples 100 specific sessions which may differ from the full bucketed set

This gap needs actual benchmark runs to determine if it translates to TPOT differences.

## Phase 1: Run REAL trace_replay profiles for per_turn ground truth

Goal: get fresh per_turn data at high success rate. These runs must succeed ≥85%.

### On h100 (gpu-13, 4x H100, Llama-3.1-8B-Instruct, tp=1)

Server launch:
```bash
# h100, in vllm env
vllm serve /data/models/Llama-3.1-8B-Instruct \
  --enable-prefix-caching --gpu-memory-utilization 0.75 \
  --max-model-len 32768 --port 8089
```

Run (from inference-benchmark dir, with `DASHBOARD_SCOPE=mse_validation`):
```bash
# swebench-multiturn-short (13-30 turns, real trajectories)
for conc in 5 40 160; do
  num_req=$((conc * 2 > 100 ? conc * 2 : 100))
  python -m src.benchmark.runner \
    --profile swebench-multiturn-short --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_real_swebench-short_conc${conc}.json
done

# swebench-multiturn-medium (50-125 turns, real trajectories)
for conc in 5 40 160; do
  num_req=$((conc * 2 > 200 ? conc * 2 : 200))
  python -m src.benchmark.runner \
    --profile swebench-multiturn-medium --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_real_swebench-medium_conc${conc}.json
done

# terminalbench-multiturn-short
for conc in 5 40 160; do
  num_req=$((conc * 2 > 100 ? conc * 2 : 100))
  python -m src.benchmark.runner \
    --profile terminalbench-multiturn-short --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_real_terminalbench-short_conc${conc}.json
done

# terminalbench-multiturn-medium
for conc in 5 40 160; do
  num_req=$((conc * 2 > 200 ? conc * 2 : 200))
  python -m src.benchmark.runner \
    --profile terminalbench-multiturn-medium --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_real_terminalbench-medium_conc${conc}.json
done
```

### On gpu-4 (a100, 4x A100-40GB, Llama-3.1-8B-Instruct, tp=2)

Same commands, but:
- Python: `/data/kevinlau/miniconda3/bin/python`
- Model: `/data/models/Llama-3.1-8B-Instruct`
- Output prefix: `results/mse_validation/a100_real_*`
- GPU: `CUDA_VISIBLE_DEVICES=0,1` with tp=2 (add `--tensor-parallel-size 2` to vllm serve)

## Phase 2: Run MSE distributional profiles

Goal: run the new bucketed MSE profiles for head-to-head comparison.

### On h100 (same config)

```bash
# swebench-multiturn-mse-short (5 session pool, 13-30 turns)
for conc in 5 40 160; do
  num_req=$((conc * 2 > 100 ? conc * 2 : 100))
  python -m src.benchmark.runner \
    --profile swebench-multiturn-mse-short --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_mse_swebench-short_conc${conc}.json
done

# swebench-multiturn-mse-medium (96 session pool, 50-125 turns)
for conc in 5 40 160; do
  num_req=$((conc * 2 > 200 ? conc * 2 : 200))
  python -m src.benchmark.runner \
    --profile swebench-multiturn-mse-medium --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_mse_swebench-medium_conc${conc}.json
done

# terminalbench-multiturn-mse-short
for conc in 5 40 160; do
  num_req=$((conc * 2 > 100 ? conc * 2 : 100))
  python -m src.benchmark.runner \
    --profile terminalbench-multiturn-mse-short --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_mse_terminalbench-short_conc${conc}.json
done

# terminalbench-multiturn-mse-medium
for conc in 5 40 160; do
  num_req=$((conc * 2 > 200 ? conc * 2 : 200))
  python -m src.benchmark.runner \
    --profile terminalbench-multiturn-mse-medium --concurrency $conc \
    --num-requests $num_req --min-success-rate 0.75 \
    --url http://localhost:8089/v1/chat/completions \
    --model /data/models/Llama-3.1-8B-Instruct \
    --output results/mse_validation/h100_mse_terminalbench-medium_conc${conc}.json
done
```

### On gpu-4 (a100, tp=2, same as Phase 1 config)

Same commands with a100 paths and `results/mse_validation/a100_mse_*` output prefix.

## Phase 3: Comparison and analysis

For each (profile, conc, hardware) pair where both REAL and MSE runs completed:

1. **Per-turn ISL**: MSE median_isl_tokens vs REAL avg_input_tokens, turns 0-30
2. **Per-turn OSL**: MSE median_osl_tokens vs REAL avg_output_tokens, turns 0-30
3. **Per-turn TPOT**: MSE median_tpot_ms vs REAL median_tpot_ms
4. **Per-turn survival**: MSE successful/num_requests vs REAL, by turn index
5. **Aggregate**: TPOT p50/p90, TTFT p50, output tok/s

**Success criterion**: MSE per-turn TPOT within ±20% of REAL at turns where REAL success rate ≥80%.

## Run summary

| Phase | Hardware | # Profiles | # Concs | # Runs |
|-------|----------|-----------|---------|--------|
| 1 (REAL) | h100 | 4 | 3 | 12 |
| 1 (REAL) | gpu-4 (a100) | 4 | 3 | 12 |
| 2 (MSE) | h100 | 4 | 3 | 12 |
| 2 (MSE) | gpu-4 (a100) | 4 | 3 | 12 |
| **Total** | | | | **48** |

## Key files

| What | Path |
|------|------|
| Profiles | `src/workloads/profiles.py` (lines 608-653) |
| Distributions | `data/distributions/*_tracereplay_filtered-mse.json` |
| Runner | `src/benchmark/runner.py` |
| Plan | `.claude/docs/mse-validation-plan.md` |
| Rules | `.claude/rules/benchmark-modes.md`, `.claude/rules/results-tracking.md` |

## Notes

- `num_requests` must be ≥2x concurrency to saturate the semaphore
- The 5-session swebench short distribution is thin — MSE will re-sample from the same pool
- h100-2 (gpu-15) is set up but not used in this plan; available for parallel runs
- All multi-turn runs need prefix caching ON (`--enable-prefix-caching` on vllm serve)
- `--min-success-rate 0.75` will abort runs that fall below 75% success
