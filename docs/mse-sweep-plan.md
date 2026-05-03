# MSE Validation Sweep Plan

Date: 2026-05-03

## Goal

Prove synthetic distributional traces reproduce legacy real-trace aggregate
metrics within bounded MAPE. Both profiles use the same ISL=32768 filter,
same 100-session count, same server instance — the only difference is
synthetic vs real trace content.

## Hardware

| Host | GPU | TP | Model |
|------|-----|----|-------|
| h100 | H100 | 1 | Llama-3.1-8B |
| gpu-4 | A100-40GB | 1 | Llama-3.1-8B |
| 3090 | RTX 3090 | 1 | Llama-3.1-8B |

## Matrix

3 datasets × 3 hosts × 2 concurrencies = 18 comparisons:

| Dataset | C | Distributional | Legacy |
|---------|---|---------------|--------|
| swebench | 40 | swebench-multiturn-mse | swebench-multiturn-short |
| swebench | 80 | swebench-multiturn-mse | swebench-multiturn-short |
| terminalbench | 40 | terminalbench-multiturn-mse | terminalbench-multiturn-short |
| terminalbench | 80 | terminalbench-multiturn-mse | terminalbench-multiturn-short |
| osworld | 40 | osworld-multiturn-mse | osworld-multiturn-short |
| osworld | 80 | osworld-multiturn-mse | osworld-multiturn-short |

## Run Script

`scripts/run_mse_sweep.sh` — runs all 3 datasets × 2 concurrencies on a single GPU host, using the same vLLM instance for all comparisons.

```bash
# On each GPU host:
bash scripts/run_mse_sweep.sh \
  /data/models/Llama-3.1-8B-Instruct 1 H100 \
  /data/kevinlau/miniconda3/bin/python \
  /tmp/results/mse_sweep
```

## Output

Per-run JSON files under `{OUT_DIR}/`:
```
{H100,A100,RTX3090}_{swebench,terminalbench,osworld}_{mse,legacy}_conc{40,80}.json
```

## Comparison

After collection: compute MAPE between mse and legacy pairs on same hardware. Expected to produce the "characteristic traces reproduce real traces within X% MAPE" claim for the paper.
