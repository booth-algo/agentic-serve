# GCS Results Structure

## Bucket

`gs://sequrity-experiments/inference-benchmark/results/`

## Current State (as of 2026-04-03)

**5,506 files, heavily duplicated across 3 organizational schemes:**

1. **Model-first** (03-24 to 03-27): `{model}_tp{N}_{engine}/{date}/` — original uploads, sometimes partial
2. **Date-first 03-28**: `2026-03-28/{model}_tp{N}_{engine}/` — first full consolidation
3. **Date-first 03-30**: `2026-03-30/{model}_tp{N}_{engine}/` — second full consolidation, ~identical to 03-28

Verified: 03-28 and 03-30 are 99-100% identical by md5. Model-first dirs are strict subsets of 03-28/03-30.

### Canonical data: `2026-03-30/` (most recent complete sweep)

Models benchmarked (all on 2x H100 SXM5):

| Model | TP | SGLang | vLLM | Profiles |
|-------|-----|--------|------|----------|
| Llama-3.1-8B | 1 | 104 | 74 | old names (chatbot-short, agentic-tool-use, etc.) |
| Llama-3.1-8B | 2 | 104 | 74 | old names |
| Llama-3.1-70B | 2 | 104 | 74 | old names |
| Qwen3.5-9B | 1 | 104 | 54 | old names (vLLM partial) |
| Qwen3.5-9B | 2 | 104 | 50 | old names (vLLM partial) |
| Qwen3.5-27B | 2 | 104 | 50 | old names (vLLM partial) |
| Qwen3-32B | 2 | 104 | 74 | old names |
| Qwen2.5-72B | 2 | 104 | 74 | old names |
| gpt-oss-20b | 1 | 100 | 70 | old names |
| gpt-oss-120b | 1 | 100 | 70 | old names |

Concurrency levels: 1, 5, 10, 20, 40, 80, 120, 160

Also present:
- `roofline/` — 46 parsed + 92 raw kernel profiles (Llama-8B/70B, Qwen-32B/72B, gpt-oss-20b)
- `archive/` — early A6000 and crossval runs

---

## Proposed Clean Structure (for new round, April 2026)

### Principles
1. **One canonical path per result** — no duplication
2. **Model-first** hierarchy — easier to find all results for a model
3. **Hardware tag** — distinguish 2xH100 (old) from 4xH100 (new) runs
4. **Date in filename** — not in directory path (avoids duplication temptation)

### Path format

```
results/
  v1/                              # old 2xH100 runs (moved from 2026-03-30/)
    {Model}_tp{N}_{engine}/
      {Model}_tp{N}_{engine}_{profile}_conc{C}.json
  v2/                              # new 4xH100 runs (April 2026+)
    {Model}_tp{N}_{engine}/
      {Model}_tp{N}_{engine}_{profile}_conc{C}.json
  roofline/
    parsed/
    raw/
  archive/                         # everything else (old model-first, 03-28 dups)
```

### Naming conventions

- **Model**: Short name matching download dir, e.g. `Llama-3.1-70B`, `MiniMax-M2.5`, `GLM-4.6-FP8`
- **Engine**: `vllm` or `sglang`
- **TP**: Tensor parallelism degree
- **Profile**: New canonical profile name from `profiles.py` (e.g. `chat-short`, `coding-agent`, `swebench-multiturn-short`)
- **Concurrency**: `conc{N}`

### Example paths

```
v2/Llama-3.1-70B_tp4_vllm/Llama-3.1-70B_tp4_vllm_chat-short_conc20.json
v2/MiniMax-M2.5_tp4_sglang/MiniMax-M2.5_tp4_sglang_coding-agent_conc10.json
v2/GLM-4.6-FP8_tp4_vllm/GLM-4.6-FP8_tp4_vllm_swebench-multiturn-short_conc5.json
```

### Migration plan

1. Keep `2026-03-30/` as `v1/` (rename, don't copy)
2. Move all model-first and `2026-03-28/` dirs into `archive/`
3. New benchmarks go into `v2/`
4. Update `src/reporting/loader.py` to scan `v1/` + `v2/` for dashboard
5. Update `gcs-storage.md` upload scripts to use new paths

### What v2 adds over v1

- New models: Llama-3.3-70B, MiniMax-M2.5, GLM-4.6-FP8
- New TP configs: TP=4 for all 70B+ models
- New profiles: Tier 1 real agent data (coding-agent, swebench-multiturn-*, terminalbench-multiturn-*)
- New profiles: Tier 2 renamed chat profiles (chat-short/medium/long)
- Hardware: 4x H100 SXM5 80GB
