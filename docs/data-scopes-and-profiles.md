# Data Scopes & Profiles — Agentic-Serve Benchmark Dataset

Date: 2026-05-03 | For: HuggingFace dataset setup

2026-05-11 rename: the dashboard/storage boundary now uses
`trace_replay` for the old `archive` real-trace subset,
`synthetic_distributional` for the synthetic Hugging Face subset, and
`archived` for retired canonical, fixed-grid, and MSE result surfaces. Older
sections below use the original names for historical context.

2026-05-11 synthetic grid update: newly compiled `synthetic_distributional`
jobs derive from fixed launch cells but use the mode-specific trace-replay
concurrency grid. Single-turn synthetic uses
{1,10,20,40,80,120,160,200,256,320,500}; multi-turn synthetic uses
{1,5,10,20,40,80,120,160,200,256,320}.

This document explains the four data scopes (`archive`, `current`, `fixed`, and MSE validation) — what profiles they contain, what's been run, what hasn't, how they're filtered, and how they produce the `data.json` consumed by the dashboard.

---

## Quick Reference

| Scope | Status | Profiles | Concurrency grid | Data in `data.json`? |
|-------|--------|----------|-----------------|----------------------|
| **archive** | Historical, done | Legacy real-trace (short/medium/long) | 1-500 | Yes (~4K rows) |
| **current** | Active but partially bugged | Distributional synthetic | 1-320 | Yes (~530 rows, C>10 bugged) |
| **fixed** | Ready, not yet run | Distributional synthetic (post-fix) | 40,80,200,320 | No (empty) |
| **MSE validation** | Ready, not yet run | Filtered distributional + legacy pairs | 40,80 | No (empty) |

---

## How Data Gets Into `data.json`

```
GPU Host                          Hetzner                              Dashboard
────────                          ───────                              ─────────
benchmark.runner                  bench_orchestrator.sh                build-data.ts
    │                                  │                                  │
    │  writes results/*.json           │  rsyncs from GPU hosts            │  reads all result files
    │  config.dashboard_scope:         │  uploads to R2                    │  detects scope from
    │    "current" (active profiles)   │  s3://agent-bench/results/       │    config.dashboard_scope
    │    "archive" (inactive)          │                                  │    or result path
    │    "fixed" (--scope flag)        │                                  │
    │                                  │                                  │  produces:
    │                                  │                                  │    dashboard/public/data.json
    │                                  │                                  │
    ▼                                  ▼                                  ▼
  JSON per run                       R2 bucket                          { config: {...},
                                                                            summary: {...},
                                                                            perTurn: [...],
                                                                            dataScope: "current",
                                                                            hardware: "A100-40GB" }
```

Each result JSON becomes one row in `data.json`. The `dataScope` field is set at collection time by the benchmark runner (`--scope` flag, defaulting to `fixed` for active profiles as of commit `6b590c2`). The dashboard reads `data.json` and filters by `dataScope` for the scope selector (Current / Archive / Fixed tabs).

---

## Profile Families

### Legacy Real-Trajectory Profiles (`active=False`)

| Profile | Dataset type | Data source | Sessions | ISL filter | Turn count | Duration |
|---------|-------------|-------------|----------|------------|------------|----------|
| `swebench-multiturn-short` | real-trajectory | `data/swebench_trajectories.jsonl` | 100 | ≤32K | ~30 | 10-20h |
| `swebench-multiturn-medium` | real-trajectory | same | 100 | ≤65K | ~30 | 10-20h |
| `swebench-multiturn-long` | real-trajectory | same | 50 | ≤131K | ~30 | 10-20h |
| `terminalbench-multiturn-short` | real-trajectory | `data/terminalbench_trajectories.jsonl` | 100 | ≤32K | ~30 | hours |
| `terminalbench-multiturn-medium` | real-trajectory | same | 100 | ≤65K | ~30 | hours |
| (etc.) | | | | | | |

**What they do**: Replay the actual SWE-bench/TerminalBench/OSWorld agent conversations — real messages, real tool calls, real code. Filter sessions where max turn ISL exceeds the limit. These are the **ground truth** for MSE validation.

**Scope**: All legacy data lands in `archive`.

### Distributional Synthetic Profiles (`active=True`)

| Profile | Dataset type | Data source | Sessions | Turn count | Duration |
|---------|-------------|-------------|----------|------------|----------|
| `swebench-multiturn` | distributional | `data/distributions/swebench_multiturn.json` | 1 (runtime→C) | up to 320 | minutes |
| `terminalbench-multiturn` | distributional | `data/distributions/terminalbench_multiturn.json` | 1 (runtime→C) | up to 876 | minutes |
| `osworld-multiturn` | distributional | `data/distributions/osworld_multiturn.json` | 1 (runtime→C) | up to 30 | minutes |
| `chat-multiturn` | distributional | `data/distributions/chat_multiturn.json` | 1 (runtime→C) | up to 20 | minutes |

**What they do**: Sample from empirical turn/ISL/OSL/cache-hit distributions extracted from real traces. Generate synthetic filler text (`"s0_t0_user_0 s0_t0_user_1..."`). 10× faster than legacy. `num_sessions=1` is a floor — the runner's `resolve_multi_turn_num_sessions()` applies `max(1, concurrency)` at runtime, so C=80 effectively creates 80 synthetic sessions.

**Scope**: Pre-fix data lands in `current`. Post-fix data (with Codex's `max(num_sessions, concurrency)` runtime fix) lands in `fixed`.

### MSE Validation Profiles (`active=False`)

| Profile | Data source | Sessions | ISL filter | Turn cap | Purpose |
|---------|-------------|----------|------------|----------|---------|
| `swebench-multiturn-mse` | `data/distributions/swebench_multiturn_filtered.json` | 100 | ≤32K | 30 | Match legacy `-short` exactly |
| `terminalbench-multiturn-mse` | `data/distributions/terminalbench_multiturn_filtered.json` | 100 | ≤32K | 30 | Match legacy `-short` exactly |
| `osworld-multiturn-mse` | `data/distributions/osworld_multiturn_filtered.json` | 50 | ≤32K | 30 | Match legacy `-short` exactly |

**What they do**: Same as distributional profiles, but use ISL-filtered distributions (sessions with any turn >32K ISL are excluded — same population as legacy `-short` profiles). 100 sessions (matching legacy). Used for head-to-head MSE validation: synthetic vs real on same hardware with same session population.

**Scope**: Not yet run. Will produce `fixed` scope data. Intended for paper contribution 2 validation.

**Filtered distribution stats** (from `scripts/filter_distribution.py`):
- swebench: 138/165 sessions kept (27 excluded for ISL > 32K)
- terminalbench: 242/267 sessions kept (25 excluded)
- osworld: 60/60 sessions kept (none exceed 32K)

---

## What's Been Run vs What Hasn't

### archive — DONE

Legacy real-trajectory profiles on all 4 GPU hosts (H100, A100, 3090, 2080Ti) across concurrencies C ∈ {1,5,10,20,40,80,120,160,200,256,320,500}. ~4,854 rows in `data.json`.

**Known bug**: `num_sessions=100` means rows at C > 100 have effective concurrency capped at 100 (labeled as C=160 but actually running C=100). 928 rows affected. This is visible in the dashboard at high C.

### current — PARTIALLY DONE, PARTIALLY BUGGED

Distributional profiles on all 4 GPU hosts at C ∈ {5,20,40,80,160}. ~530 rows in `data.json`.

**Known bug**: `num_sessions=10` (original value) means rows at C > 10 have effective concurrency capped at 10. 432 rows affected. C=5 rows are valid. Fixed now in code (`num_sessions=1` + runtime `max(1, C)`), but existing data in `data.json` is from pre-fix runs.

### fixed — NOT YET RUN

Same canonical distributional profiles as `current`, but with the concurrency bug fixed. Reduced concurrency grid: C ∈ {40,80,200,320} for single-turn, C ∈ {40,80,200,320} for multi-turn. All new runs default to `--scope fixed` as of commit `235d934`. Sweep cells exist in `sweep.yaml`. No data yet — first orchestrator tick will populate.

### MSE Validation — NOT YET RUN

Filtered distributional profiles (`*-multiturn-mse`) matched head-to-head against legacy `-short` profiles on same GPU. C ∈ {40,80} per dataset, on H100, A100, and 3090. Run via `scripts/run_mse_sweep.sh`. No data yet.

---

## How the Dashboard Shows Different Scopes

The dashboard has a scope selector (Current / Archive / Fixed tabs) in the top nav.

**CoveragePage**: Each scope has its own expected grid:

| Scope | Single-turn profiles | Multi-turn profiles | Single C | Multi C |
|-------|---------------------|---------------------|----------|---------|
| current | chat-singleturn, coding-singleturn | chat-mt, swebench-mt, terminalbench-mt, osworld-mt | 1,10,20,40,80,160,256,320 | 5,20,40,80,160 |
| archive | (all historical, including stress) | (all historical) | 1,10,20,40,80,120,160,200,256,320,500 | 5,10,20,40,80,120,160,200,256,320 |
| fixed | chat-singleturn, coding-singleturn | chat-mt, swebench-mt, terminalbench-mt, osworld-mt | 40,80,200,320 | 40,80,200,320 |

**Row filtering**: `dataScope` field in `data.json` determines which scope a row belongs to. The dashboard's `useData.ts` hook groups rows by `row.dataScope` (defaulting to `archive` if missing). Only rows whose `dataScope` matches the selected scope and whose profile is in that scope's set are displayed.

**ServingPredictionsPage**: Only shows for `current` and `fixed` scopes. Uses canonical profiles. `archive` scope hides the serving predictions tab.

---

## For HuggingFace Dataset

The dataset should include all three completed scopes (`archive`, `current`) as separate splits, plus note that `fixed` and MSE validation are pending:

```python
# Proposed HuggingFace dataset structure
{
  "archive": [...],   # ~4,854 rows — legacy real-trace data
  "current": [...],   # ~530 rows — distributional data (C>10 bugged, documented)
  # "fixed": pending   # will be added when sweeps complete
  # "mse": pending     # will be added when MSE validation completes
}
```

Each row includes `dataScope`, `config` (profile, model, backend, concurrency, hardware), `summary` (aggregate metrics), and optional `perTurn` (per-turn breakdown for multi-turn rows).

The `archive` scope is the largest and most diverse — it's the ground truth benchmark dataset. The `current` scope is the canonical paper-facing dataset (distributional profiles). Both should be clearly labeled with their known limitations in the dataset card.
