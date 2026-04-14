# Plan: Extended Multi-Turn Concurrency + TP=4 Sweep

**Created:** 2026-04-08
**Status:** NOT STARTED

## Goal

Extend multi-turn concurrency beyond 40 for profiles that haven't saturated, and add TP=4 multi-turn benchmarks (never done before).

## Evidence: Saturation Analysis (2026-04-07)

Chat-multiturn profiles still GROWING 10-55% at max tested concurrency:
- Llama-8B chat-multiturn-short: +54.8% at conc=40
- Qwen3.5-9B chat-multiturn-medium: +37.7% at conc=40
- Qwen3.5-27B chat-multiturn-long: +26.3% at conc=40
- Llama-70B chat-multiturn-short: +8.3% at conc=20

Already saturated (SKIP):
- swebench-multiturn-short/medium: saturated at conc=20-40
- terminalbench-multiturn-medium: saturated at conc=20-40

## Profiles to Run

- chat-multiturn-short
- chat-multiturn-medium
- chat-multiturn-long
- terminalbench-multiturn-short

## Models & TP Configs

### TP=1 (extend existing, conc 80/120/160)
- Llama-3.1-8B TP=1: conc 80, 120, 160
- Qwen3.5-9B TP=1: conc 80, 120, 160

### TP=2 (extend existing)
- Qwen3.5-27B TP=2: conc 80, 120, 160
- Llama-3.1-70B TP=2: conc 40, 80 (tight VRAM, low-conc)

### TP=4 (NEW — full sweep from scratch)
- Llama-3.1-8B TP=4: conc 5, 10, 20, 40, 80, 120, 160
- Qwen3.5-9B TP=4: conc 5, 10, 20, 40, 80, 120, 160
- Qwen3.5-27B TP=4: conc 5, 10, 20, 40, 80, 120, 160
- Qwen2.5-72B TP=4: conc 5, 10, 20, 40, 80
- Llama-3.1-70B TP=4: conc 5, 10, 20, 40, 80
- Llama-3.3-70B TP=4: conc 5, 10, 20, 40, 80

## Execution Order (GPU Parallelism)

1. **TP=1 batch** (2 parallel slots on GPU 0+1):
   - Slot 0: Llama-3.1-8B, 4 profiles × 3 conc = 12 runs
   - Slot 1: Qwen3.5-9B, 4 profiles × 3 conc = 12 runs

2. **TP=2 batch** (2 parallel slots on GPUs 0,1 + 2,3):
   - Slot 0: Qwen3.5-27B, 4 profiles × 3 conc = 12 runs
   - Slot 1: Llama-3.1-70B, 4 profiles × 2 conc = 8 runs

3. **TP=4 batch** (sequential, all 4 GPUs):
   - Llama-3.1-8B TP=4: 4 profiles × 7 conc = 28 runs
   - Qwen3.5-9B TP=4: 4 profiles × 7 conc = 28 runs
   - Qwen3.5-27B TP=4: 4 profiles × 7 conc = 28 runs
   - Qwen2.5-72B TP=4: 4 profiles × 5 conc = 20 runs
   - Llama-3.1-70B TP=4: 4 profiles × 5 conc = 20 runs
   - Llama-3.3-70B TP=4: 4 profiles × 5 conc = 20 runs

## Total Runs

- TP=1: 24 runs
- TP=2: 20 runs
- TP=4: 144 runs
- **Total: 188 runs** (× 2 files each = 376 files)

## Implementation

Update `run_multiturn_benchmarks.sh`:
- Add TP=4 configs to MODELS array
- Update CONC_SWEEP to "5 10 20 40 80 120 160"
- Add selective profile list (skip saturated swebench/terminalbench-medium)
- Use GPU parallelism (scripts/gpu_scheduler.sh)

## Estimated Time

- TP=1 batch: ~2 hours (chat profiles are faster)
- TP=2 batch: ~3 hours
- TP=4 batch: ~5-7 hours (6 models sequential)
- **Total: ~10-12 hours**

## Prerequisites

- torch 2.10.0 installed (vLLM)
- All GPUs clean (0 MiB)
- Disk space: ~28GB free (need ~5GB for results)
