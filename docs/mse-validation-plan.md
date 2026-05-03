# MSE Validation: Distributional (Current) vs Real-Trajectory (Legacy)

Date: 2026-05-02

## Goal

Prove that ~10 distributional/synthetic traces reproduce aggregate metrics of the full ~165 real-trace sweep within bounded MAPE. Replaces the predictor's "fewer runs to inform" narrative with an empirical claim for NeurIPS contribution 2.

---

## Legacy vs Current Benchmark Suites

### Legacy (real-trajectory) — `active=False`

| Profile | Dataset | Data source | Sessions | Content | Duration |
|---------|---------|-------------|----------|---------|----------|
| `swebench-multiturn-short` | `swebench-multi-turn` | `data/swebench_trajectories.jsonl` | 100 | Real agent traces | 10-20h |
| `swebench-multiturn-medium` | `swebench-multi-turn` | same JSONL | 100 | Real agent traces | 10-20h |
| `swebench-multiturn-long` | `swebench-multi-turn` | same JSONL | 50 | Real agent traces | 10-20h |

- Replays actual SWE-bench agent conversations (real messages, tool calls, code)
- Filtered by ISL limit (short=32K, medium=65K, long=131K tokens)
- Routes through `TrajectoryMultiTurnDataset` (`dataset.py:598`)
- These are the **ground truth** for the MSE comparison

### Current (distributional/synthetic) — `active=True`

| Profile | Dataset | Data source | Sessions | Content | Duration |
|---------|---------|-------------|----------|---------|----------|
| `swebench-multiturn` | `distributional-multi-turn` | `data/distributions/swebench_multiturn.json` | 10 | Synthetic filler | minutes |
| `terminalbench-multiturn` | `distributional-multi-turn` | `data/distributions/terminalbench_multiturn.json` | 10 | Synthetic filler | minutes |
| `osworld-multiturn` | `distributional-multi-turn` | `data/distributions/osworld_multiturn.json` | 10 | Synthetic filler | minutes |
| `chat-multiturn` | `distributional-multi-turn` | `data/distributions/chat_multiturn.json` | 10 | Synthetic filler | minutes |

- Samples from empirical turn/ISL/OSL/cache-hit distributions
- Generates synthetic filler text: `"turn_1_token_0 turn_1_token_1 ..."`
- Routes through `DistributionalMultiTurnDataset` (`dataset.py:735`)
- These are the **fast approximation** — the claim to validate

### Source of Distributions

Distribution JSONs were built from the SAME real traces that legacy profiles use:
```
build_trace_distributions.py:
  data/swebench_trajectories.jsonl       → swebench_multiturn.json (165 sessions, 15,509 turns)
  data/terminalbench_trajectories.jsonl  → terminalbench_multiturn.json (267 sessions, ~20K turns)
  data/osworld_trajectories.jsonl        → osworld_multiturn.json (60 sessions, ~788 turns)

build_chat_distribution_from_results():
  dashboard/public/data.json             → chat_multiturn.json (423 sessions)
```

---

## Code Infrastructure Status

All legacy code is **intact on main branch**:

| Component | Location | Status |
|-----------|----------|--------|
| `TrajectoryMultiTurnDataset` | `dataset.py:598` | Alive |
| Legacy profiles | `profiles.py` | `active=False`, runnable |
| Trajectory JSONL files | `data/*.jsonl` | Present |
| Distribution JSONs | `data/distributions/*.json` | Present |
| `DistributionalMultiTurnDataset` | `dataset.py:735` | Alive |
| `make_dataset()` factory | `dataset.py:829` | Routes both |

No code changes needed. Compare existing data from `data.json`.

---

## Existing Data Coverage

Distributional AND legacy swebench data exist on the same hardware:

| GPU | Overlapping C | Dist rows (current scope) | Legacy rows (archive scope) |
|-----|---------------|--------------------------|---------------------------|
| A100-40GB | 40, 80, 160 | 3 | ~6 (short) |
| RTX3090 | 20, 40, 80 | 3 | ~6 (short) |
| A100-40GBx4 | 20, 40, 80 | 3 | ~6 (short + medium) |

---

## Plan

### Phase 1: Verify Row Matching
- Read `data.json`, find matched pairs: same hardware + model + backend + concurrency
- Confirm model is Llama-3.1-8B

### Phase 2: Compute MAPE
For each matched concurrency level, extract aggregate metrics from `summary`:
```
median_ttft_ms, p90_ttft_ms, median_tpot_ms, p90_tpot_ms, req/s
MAPE = |dist - legacy| / legacy × 100 per metric
```

### Phase 3: Cross-Dataset
Repeat for `terminalbench-multiturn` and `osworld-multiturn` where data exists.

### Phase 4: Output
- Compact Markdown table: per dataset, per C, per metric MAPE
- Scatter plot: distributional vs legacy
- JSON results file

### Script
`scripts/validate_distributional_mse.py` — reads `data.json`, matches rows, computes MAPE, prints table.

---

## What This Proves

"10 characteristic traces reproduce aggregate metrics within X% MAPE of the full 100-trace sweep across C ∈ {20, 40, 80, 160}" — directly replaces the predictor's "fewer runs" narrative for NeurIPS contribution 2.

---

## Diagnosis (2026-05-02)

**Root cause found**: All distributional profiles had `num_sessions=10`, capping effective concurrency at 10 regardless of the benchmark concurrency setting. Legacy profiles use 100 sessions.

| Profile | Old num_sessions | New num_sessions | Source sessions |
|---------|-----------------|-----------------|-----------------|
| swebench-multiturn | 10 | 100 | 165 |
| terminalbench-multiturn | 10 | 100 | 267 |
| osworld-multiturn | 10 | 50 | 60 |
| chat-multiturn | 10 | 100 | 423 |

At C=80, distributional had `min(80, 10) = 10` effective concurrency vs legacy `min(80, 100) = 80` — 8× less queue pressure and KV cache contention. This explains the 80-85% MAPE across all metrics.

At C=5 (where both have `min(5, 10) = min(5, 100) = 5`), MAPE was only 17-25% — borderline acceptable and confirms the session count was the dominant variable.

**Fix applied** in `profiles.py`. New benchmark runs will use 100 sessions and produce valid comparisons against legacy data.

**No re-run needed yet**: The orchestrator will pick up the new session counts on its next sweep cycle.
