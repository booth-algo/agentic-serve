# Morphology-Aware Synthetic Filler — SWE-bench MSE Validation

**Date:** 2026-05-05
**Status:** chars/token fixed; turn-10+ E2EL gap partially improved but not closed

## Hypothesis

SWE-bench distributional MSE E2EL gap (turn 10-19: +45.5%) is partly caused by synthetic prompts having ~8.4 chars/token (English filler) vs REAL ~3.8 chars/token (code/tool-output text). Matching the chars/token ratio should reduce request-preprocessing and prefix-handling differences if character density is a causal driver rather than just a correlated symptom.

## Method

Source-locked MSE+REAL pair on same vLLM instance, same 40 SWE sessions. Two conditions:

| Condition | Host | `DISTRIBUTIONAL_SYNTHETIC_STYLE` | `DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN` |
|-----------|------|----------------------------------|----------------------------------------|
| English (baseline) | h100 (gpu-13) | (default) | (default ~8.4) |
| Code (experiment) | h100-2 (gpu-15) | `code` | `3.8` |

Both runs: `SESSIONS=40`, `CONC=5`, `PORT=8091/8092`, `GPU_MEM=0.75`, `MAX_LEN=32768`, `SOURCE_SESSION_IDS_FILE` source-locked to same 40 SWE session IDs extracted from the English REAL run.

Source IDs extracted from:
```
results/mse_validation_source_locked_pair/h100_swebench_c5_s40/Llama-3.1-8B_tp1_vllm/swebench-multiturn-short_conc5.json
```

Command:
```
DISTRIBUTIONAL_SYNTHETIC_STYLE=code \
DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN=3.8 \
SOURCE_SESSION_IDS_FILE=/tmp/swebench_source_locked_ids.txt \
bash scripts/run_mse_validation.sh /data/models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm \
  swebench 5 results/mse_validation_morphology/h100_swebench_c5_s40_codechars \
  /home/kevinlau/miniconda3/envs/vllm/bin/python 0.75 32768
```

## Results

### Criterion 1: chars/token ratio

| Condition | Median chars | Median tokens | chars/token | vs REAL |
|-----------|-------------|--------------|-------------|---------|
| English MSE | 57,004 | 6,798 | 8.39 | 2.2× |
| **Code MSE** | 25,037 | 6,798 | **3.68** | **1.0×** |
| REAL | 24,618 | 6,472 | 3.80 | — |

Per-bin prompt_chars Δ (Code MSE vs REAL): −2% to +6%. **Target hit.**

### Criterion 2: E2EL p50 per turn bin

| Turn bin | English MSE | English REAL | English Δ | Code MSE | Code REAL | Code Δ | Improvement |
|----------|------------|-------------|-----------|----------|----------|--------|-------------|
| 00–04 | 214ms | 207ms | +3.2% | 221ms | 217ms | +1.9% | +1pp |
| 05–09 | 348ms | 340ms | +2.2% | 343ms | 339ms | +1.0% | +1pp |
| **10–19** | 675ms | 464ms | **+45.5%** | 670ms | 511ms | **+31.1%** | **+14pp** |
| 20–29 | 2,267ms | 1,957ms | +15.8% | 2,128ms | 1,893ms | +12.4% | +3pp |

Input token Δ per bin: identical between conditions (+2.3% to +4.0% across bins). Session overlap: 40/40 (Jaccard 1.00).

### Aggregate

| Metric | English Δ | Code Δ |
|--------|-----------|--------|
| TTFT p50 | +46.8% | +23.8% |
| TPOT p50 | +8.9% | +14.7% |
| E2EL p50 | +27.0% | +21.6% |

## Verdict

**Chars/token ratio is a partial cause, not the whole cause.** Matching the morphology from 2.2× to 1.0× closed ~14pp of the 45% turn-10–19 E2EL gap, but the remaining +31% gap is outside the observed noise floor. The residual likely stems from content structure differences beyond simple character density: real SWE traces contain recurrent boilerplate (traceback frames, pytest output, git diffs), while code-like filler generates independent random fragments. The experiment does not establish a different KV-cache block layout; same token count should imply similar sequence length and KV block count.

### Confounding factor

The English and Code runs used different GPU hosts:
- English: h100 / gpu-13
- Code: h100-2 / gpu-15

REAL baselines differ (E2EL 515ms vs 551ms), so ~5–10pp of the residual gap may be cross-host noise. A same-host A/B test would be needed to isolate this.

## Result Files

### Morphology (code-like filler) run
```
results/mse_validation_morphology/h100_swebench_c5_s40_codechars/Llama-3.1-8B_tp1_vllm/
  swebench-multiturn-mse-short_conc5.json           ← MSE (code filler)
  swebench-multiturn-mse-short_conc5_per_turn.json
  swebench-multiturn-short_conc5.json               ← REAL
  swebench-multiturn-short_conc5_per_turn.json
```

### English filler baseline (same-instance source-locked pair)
```
results/mse_validation_source_locked_pair/h100_swebench_c5_s40/Llama-3.1-8B_tp1_vllm/
  swebench-multiturn-mse-short_conc5.json           ← MSE (English filler)
  swebench-multiturn-short_conc5.json               ← REAL
```

### Source session IDs
```
/tmp/swebench_source_locked_ids.txt   (40 IDs, on h100-2)
```

## Next Probe

Recurrent content filler. Instead of independent code fragments per request, reuse actual prefix text chunks from SWE real traces. If the gap closes, recurrent boilerplate/content structure is the root cause; if not, the remaining error is somewhere else in the serving input path or runtime variance rather than simple chars/token morphology.

## Prefix-Aware Synthetic Follow-Up

vLLM automatic prefix caching reuses KV blocks when later requests share an exact token prefix with previous requests. The distributional generator already builds a growing transcript within each synthetic session, so each turn is prefix-eligible relative to the previous turn in the same session. What it did not model was the shared cross-session harness prefix that real SWE/TerminalBench traces usually have.

Local change added on 2026-05-05:
- `DISTRIBUTIONAL_PREFIX_AWARE=1` enables a fixed shared system-prefix for distributional synthetic sessions.
- `DISTRIBUTIONAL_SHARED_PREFIX_TOKENS=1024` sets the content-token target for that shared prefix.
- `DISTRIBUTIONAL_PREFIX_BLOCK_SIZE=16` aligns the shared prefix target to vLLM-style cache blocks.
- `PREFIX_AWARE_SYNTHETIC=on` in `scripts/run_mse_validation.sh` enables this only for the MSE/distributional side of a paired run.

This preserves sampled total-context token targets by subtracting the shared prefix from the first-turn synthetic user payload. The result is an APC-aware ablation: same token budget, same morphology controls, but with cross-session shared prefix structure.

Suggested next run:
```
PREFIX_AWARE_SYNTHETIC=on \
DISTRIBUTIONAL_SYNTHETIC_STYLE=code \
DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN=3.8 \
SHARED_PREFIX_TOKENS=1024 \
SOURCE_SESSION_IDS_FILE=/tmp/swebench_source_locked_ids.txt \
bash scripts/run_mse_validation.sh /data/models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm \
  swebench 5 results/mse_validation_prefix_aware/h100_swebench_c5_s40_codechars_sharedprefix \
  /home/kevinlau/miniconda3/envs/vllm/bin/python 0.75 32768
```

## Code

- `inference-benchmark/src/workloads/distributional.py:136` — `DISTRIBUTIONAL_SYNTHETIC_STYLE` / `DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN`
- `inference-benchmark/src/workloads/distributional.py:147` — `DISTRIBUTIONAL_PREFIX_AWARE` / shared-prefix controls
- `inference-benchmark/src/workloads/distributional.py:381` — `_synthetic_text()` style dispatch
- `inference-benchmark/src/workloads/distributional.py:483` — `_calibrated_morphology_text()` with `target_chars_per_token`
