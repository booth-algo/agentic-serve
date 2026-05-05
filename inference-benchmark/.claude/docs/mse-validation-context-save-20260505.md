# MSE Validation — Context Save 2026-05-05 ~10:20 UTC

## Current Status

### h100-2 (H100 tp=1, Llama-3.1-8B-Instruct)
- **Phase 1 REAL**: 12/12 COMPLETE
- **Phase 2 MSE**: 5/12 (script auto-advancing via nohup)
  - Done: swebench-short conc 5,40,160 + swebench-medium conc 5,40
  - Running: `swebench-mse-medium conc=160` (PID 428090, started 09:57, ~26 min elapsed)
  - Remaining: terminalbench-mse-short ×3 + terminalbench-mse-medium ×3 (7 runs)
- vLLM on port 8089 with `--enable-prefix-caching --gpu-memory-utilization 0.75 --max-model-len 32768`

### gpu-4 (A100 tp=2, Llama-3.1-8B-Instruct)
- **Phase 1 REAL**: 10/12 (script auto-advancing via nohup)
  - Done: swebench-{short,medium} ×3 concs + terminalbench-short ×3 + terminalbench-medium conc=5
  - Running: `terminalbench-medium conc=40` (PID 2338862, started 09:52, ~31 min elapsed)
  - Remaining: terminalbench-medium conc=160 (1 run)
- **Phase 2 MSE**: NOT STARTED (script not yet written/copied)
- vLLM on port 8089 with `--enable-prefix-caching --gpu-memory-utilization 0.75 --max-model-len 32768`

## All monitors and cron jobs STOPPED
- Monitor bxc17fgk8 stopped
- Cron 65a1ea88 cancelled
- Cron 8dddbd6e cancelled
- Both bash scripts continue running via nohup — they auto-advance

## File inventory

### h100-2 results/mse_validation/
```
h100_mse_swebench-medium_conc5.json
h100_mse_swebench-medium_conc40.json
h100_mse_swebench-short_conc160.json
h100_mse_swebench-short_conc40.json
h100_mse_swebench-short_conc5.json
h100_real_swebench-medium_conc{5,40,160}.json
h100_real_swebench-medium_conc{5,40,160}_per_turn.json
h100_real_swebench-short_conc{5,40,160}.json
h100_real_swebench-short_conc{5,40,160}_per_turn.json
h100_real_terminalbench-medium_conc{5,40,160}.json
h100_real_terminalbench-medium_conc{5,40,160}_per_turn.json
h100_real_terminalbench-short_conc{5,40,160}.json
h100_real_terminalbench-short_conc{5,40,160}_per_turn.json
```

### gpu-4 results/mse_validation/
```
a100_real_swebench-medium_conc{5,40,160}.json
a100_real_swebench-short_conc{5,40,160}.json
a100_real_terminalbench-short_conc{5,40,160}.json
a100_real_terminalbench-medium_conc5.json
```

## Phase 3 Analysis — Key Findings

### REAL vs MSE comparison on h100-2 swebench (conc=5)

**Validation FAILED.** Only 29% of turns within ±20% TPOT (target: >80%).

| Metric | swebench-short | swebench-medium |
|--------|---------------|-----------------|
| ISL ratio (MSE/REAL) | 0.67 | 0.56 |
| OSL ratio (MSE/REAL) | 0.88 | 0.85 |
| TPOT ratio (MSE/REAL) | 0.61 | 0.72 |
| Turns within ±20% TPOT | 11/29 (38%) | 10/43 (23%) |

### Root cause: Context accumulation

The MSE distributional sampler accumulates context differently from real traces:

1. At `distributional.py:142`, `_sample_turn(turn_index)` samples a turn from the distribution's `turns_by_index[turn_index]`
2. At line 144: `new_user_tokens = max(1, sample.new_prefill_tokens - previous_output_tokens)`
3. `sample.new_prefill_tokens` was computed in the SOURCE session as `total_context[N] - total_context[N-1]`, but `previous_output_tokens` is from the SYNTHETIC session's prior turn

This mismatch between source-session prefill deltas and synthetic-session outputs causes the synthetic context to grow slower than real agent traces. Real context grows 1.5-1.8x faster than synthetic at later turns.

**Fix direction:** Instead of sampling `new_prefill_tokens` (a delta from an unrelated session), sample `total_context_tokens` for the target turn index and compute `new_user_tokens = desired_total_context - context_before_user`. This way synthetic context follows the same growth curve as real sessions.

### Other findings
- Cache hit rates are fine (86-98% after turn 0) in both REAL and MSE
- MSE success rate stays at 100% while REAL drops naturally (e.g., 92% by turn 29 for short)
- MSE pool size affects survival: swebench-short has only 5 sessions (13-29 turns), causing sharp drop at turn 25

## Key files

| What | Path |
|------|------|
| Context accumulation bug | `src/workloads/distributional.py:142-146` |
| Distribution build script | `scripts/build_trace_distributions.py` |
| Trace distributions | `src/workloads/trace_distributions.py` |
| Profiles | `src/workloads/profiles.py` (lines 608+) |
| Runner | `src/benchmark/runner.py` |
| MSE validation plan | `.claude/docs/mse-validation-plan.md` |
| Benchmark rules | `.claude/rules/benchmark-modes.md` |
| Results tracking | `.claude/rules/results-tracking.md` |
| This file | `.claude/docs/mse-validation-context-save-20260505.md` |

## To resume
```bash
# h100-2 (script still running via nohup, check progress):
ssh h100-2 'ls /home/kevinlau/inference-benchmark/results/mse_validation/h100_mse_*.json | grep -v per_turn | wc -l'

# gpu-4 (script still running via nohup, check progress):
ssh gpu-4 'ls /home/kevinlau/inference-benchmark/results/mse_validation/a100_real_*.json | grep -v per_turn | wc -l'

# When gpu-4 Phase 1 finishes, launch Phase 2 MSE:
# Copy run_mse_benchmarks.sh from h100-2, edit output prefix to a100_mse_*, launch with nohup
```

## Tasks remaining
- [ ] #14: gpu-4 Phase 1 REAL — 10/12, 2 remaining (conc=40 running, conc=160 pending)
- [ ] #15: h100-2 Phase 2 MSE — 5/12, 7 remaining (conc=160 running)
- [ ] #16: gpu-4 Phase 2 MSE — not started (12 runs)
- [ ] #17: Phase 3 analysis — per-turn REAL vs MSE comparison
  - [ ] Fix context accumulation bug in `distributional.py`
  - [ ] Rebuild distributions with corrected sampling
  - [ ] Re-run MSE profiles and validate
- [ ] #18: h100-2 REAL vs MSE comparison — partial (conc=5 done, waiting for higher concs)
