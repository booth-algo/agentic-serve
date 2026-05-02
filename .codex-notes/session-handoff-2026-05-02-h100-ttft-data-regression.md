# Session Handoff: H100 TTFT Data Regression

Date: 2026-05-02

## Saved Markdown Notes

This session saved these repo-local notes:

- `.codex-notes/ttft-prefix-cache-miscalculation.md`
  - Problem record for TTFT prediction errors.
  - Captures evidence that catastrophic TTFT error is mostly prefix-cache
    affected rows falling back to full-prefill behavior, especially
    `coding-singleturn`.

- `.codex-notes/prefix-cache-ttft-fix-plan.md`
  - Implementation plan for handling prefix caching first.
  - Explicitly leaves MoE and GPU-specific kernel differences out of scope.

- `.codex-notes/session-handoff-2026-05-02-h100-ttft-data-regression.md`
  - This handoff note.

There is also a short breadcrumb in `.omx/notepad.md` pointing to the TTFT
prefix-cache diagnosis.

## User Context

The user wants to hand this work to a new Codex session without OMX hooks.

Relevant local config from earlier in this session:

- `~/.codex/config.toml` was changed to set `[features].codex_hooks = false`.
- The current Codex session may still have hook-injected context, so use a new
  Codex session to get the hook-free behavior.
- The user wants to keep the HUD if possible, but avoid Codex hooks.

## Current Question Being Investigated

The user remembers previously seeing a lot of H100x1 data with multi-turn
caching metadata and hit rate. Current dashboard/predictor behavior appears to
show little or none.

They suspect three separate issues:

1. Prefix caching is not modeled properly.
2. MoE is not supported properly.
3. GPUs have different kernel characteristics, though that should mostly be
   handled by per-GPU kernel training.

The current investigation is about the first-order data disappearance issue:
why H100 data with multi-turn cache metadata is no longer visible/used.

Another Codex session is reportedly working on the separate issue that H100x2
and H100x4 were being included as H100.

## Read-Only Findings So Far

### 1. Current `data.json` has no exact `H100|current` rows

Local artifact inspected:

- `inference-benchmark/dashboard/public/data.json`

Structured count from the local generated file:

```text
total rows: 5358
archive rows: 4845
current rows: 513

H100|archive: 755
H100x2|archive: 1449
H100x4|archive: 201
H100x4|current: 114
H100|current: 0
```

Implication:

- If UI or predictor code asks for exact `H100` plus `current`, it will see
  nothing.
- Current H100 data is actually under `H100x4`, not `H100`.

### 2. Current H100x4 has cache-rich per-turn metadata

Current H100-ish rows in local `data.json`:

```text
H100x4|current|per=false|cache=false: 45
H100x4|current|per=true|cache=true: 69
```

Representative current H100x4 `perTurn` rows include:

- `avg_new_prefill_tokens`
- `median_new_prefill_tokens`
- `avg_cached_context_tokens`
- `median_cached_context_tokens`
- `avg_cache_hit_rate`
- `median_cache_hit_rate`
- block-aligned cache fields
- uncached prefix tail fields

So cache metadata exists for current H100x4 rows.

### 3. Archive exact H100 per-turn rows exist, but they do not have cache-hit metadata

Archive exact H100 rows in local `data.json`:

```text
H100|archive|per=false|cache=false: 439
H100|archive|per=true|cache=false: 316
```

Representative archive exact H100 `perTurn` rows include only older fields like:

- `turn_index`
- `num_requests`
- `successful`
- TTFT / TPOT / E2EL stats
- input/output token summaries

They do not include cache-hit or new-prefill/cached-context fields.

Historical committed `data.json` checks showed the same pattern:

```text
HEAD:
  H100|archive|per=true|cache=false: 316
  H100x4|current|per=true|cache=true: 69

c6449d6:
  H100|archive|per=true|cache=false: 316
  H100x4|current|per=true|cache=true: 60

d63ed36:
  H100|archive|per=true|cache=false: 316
  H100x4|current|per=true|cache=true: 69

584a9fb / c587a5e / 546ed86:
  H100|per=true|cache=false: 289
  H100x2|per=true|cache=false: 501
```

Inference:

- The committed dashboard data does not show a past state where exact H100x1 had
  many cache-hit-rate fields.
- The remembered "lots of H100 cache metadata" may have been:
  - H100x4 current rows being displayed under a collapsed `H100` label,
  - derived serving prediction/calibration rows,
  - or a local/uncommitted UI state that was lost after a `git pull`.

### 4. Local `serving-predictions.json` is stale relative to current `HEAD`

Local checked/generated artifact:

- `inference-benchmark/dashboard/public/serving-predictions.json`

It currently has keys:

```text
H100
A100
RTX3090
RTX2080Ti
```

It does not have `H100x2` or `H100x4` buckets.

Counts from local checked/generated file:

```text
H100 total: 595
H100 current: 0

A100 current: 44
RTX3090 current: 49
RTX2080Ti current: 14
```

Fresh export to `/tmp` from current `HEAD` produced:

```text
H100: total 595, current 0, current_cacheaware 0
H100x2: total 1140, current 0, current_cacheaware 0
H100x4: total 225, current 114, current_cacheaware 69
A100: total 291, current 44, current_cacheaware 44
RTX3090: total 245, current 49, current_cacheaware 34
RTX2080Ti: total 74, current 14, current_cacheaware 14
```

Inference:

- The checked local serving prediction artifact predates the latest H100 tensor
  parallel split.
- Rebuilding predictions from current code would expose `H100x4` current rows,
  but exact `H100` current rows still remain zero.

### 5. Current code has strict current/archive profile filtering

Relevant code paths:

- `inference-benchmark/dashboard/src/profileMeta.ts`
  - `CURRENT_PROFILES` only includes:
    - `chat-singleturn`
    - `coding-singleturn`
    - `chat-multiturn`
    - `swebench-multiturn`
    - `terminalbench-multiturn`
    - `osworld-multiturn`
  - short/medium/long profiles live under `ARCHIVE_PROFILES`.

- `inference-benchmark/dashboard/src/components/GemmPage.tsx`
  - `scopedServingRows(...)` filters rows by both data scope and profile scope.
  - A row must satisfy:
    - `row.data_scope === dataScope`
    - `isProfileInScope(row.profile, dataScope)`

Inference:

- Switching the dashboard/predictor view from broad archive data to
  `dashboard_scope=current` can make lots of older H100 rows disappear from the
  current view even though they remain in archive data.

### 6. Build-data underloaded filter exists, but is not the main explanation for multi-turn cache metadata disappearing

Relevant code:

- `inference-benchmark/dashboard/scripts/build-data.ts`

Behavior:

- Single-turn rows are skipped when `loadReqs < concurrency`.
- Multi-turn rows are exempt because `num_requests=num_sessions` can be
  intentionally lower than concurrency.

Inference:

- The underloaded filter may remove some high-concurrency single-turn rows.
- It does not explain the absence of exact `H100|current` multi-turn cache
  metadata. That absence is better explained by current rows being `H100x4`,
  while exact H100 rows are archive-only and older/no-cache.

## Best Current Explanation

Ranked diagnosis:

1. **Exact H100 current rows are absent because current H100 data is labeled
   `H100x4`, and the latest split correctly separates H100x1/H100x2/H100x4.**
   Confidence: high.

2. **The current view hides archive exact-H100 rows because dashboard serving
   rows are filtered by both data scope and profile scope.**
   Confidence: high.

3. **Older exact-H100 archive per-turn data predates cache-hit telemetry, so
   even reading archive H100 does not recover the cache-hit fields the user
   wants.**
   Confidence: high for committed local artifacts.

4. **The checked/generated `serving-predictions.json` is stale and should be
   regenerated after the H100 split.**
   Confidence: high.

5. **The user's remembered H100x1 cache-rich view may have been a collapsed UI
   label, calibration/prediction artifact, or uncommitted local UI/data state
   overwritten by `git pull`.**
   Confidence: medium, because local git history does not prove the old UI state.

## Commands Already Run

Important read-only checks:

```bash
node - <<'NODE'
# summarized inference-benchmark/dashboard/public/data.json
NODE
```

```bash
node - <<'NODE'
# summarized inference-benchmark/dashboard/public/serving-predictions.json
NODE
```

```bash
python3 -m llm_predict.export_serving_predictions \
  --output /tmp/agentic-serving-predictions-check.json
```

```bash
node - <<'NODE'
# compared committed data.json at HEAD, c6449d6, d63ed36, 584a9fb, c587a5e, 546ed86
NODE
```

## Interrupted Step

The user interrupted a read-only R2 check. The command was intended to download:

- `s3://agent-bench/json/current/serving-predictions.json`
- `s3://agent-bench/json/current/data.json`

to `/tmp`, then summarize H100/H100x2/H100x4 current/archive/cache counts.

Because it was interrupted, do not assume R2 was verified in this investigation.
If the next session needs live evidence, rerun a safe read-only R2 check.

## Recommended Next Read-Only Probes

1. Verify live R2 current artifacts:

```bash
aws --profile r2 s3 cp s3://agent-bench/json/current/serving-predictions.json /tmp/r2-serving-predictions.json --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com
aws --profile r2 s3 cp s3://agent-bench/json/current/data.json /tmp/r2-data.json --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com
```

Then compare keys/counts to the local `/tmp` export.

2. Inspect whether the dashboard currently loaded by the browser reads R2 or
   local public files, depending on dev/prod mode:

- `inference-benchmark/dashboard/src/dataUrls.ts`
- browser network tab or Playwright response URLs

3. If the other Codex session finishes the H100 split fix, regenerate serving
   predictions and verify:

```text
serving-predictions.json has H100x4 current rows
H100 exact current remains zero unless new H100x1 runs exist
H100x4 current cache-aware rows are visible in current scope
```

4. If the user specifically needs H100x1 cache-rich data, check raw R2 result
   files for exact H100 current runs with `_per_turn.json` sidecars containing
   cache fields. Current committed `data.json` does not show them.

## Do Not Conflate

Keep these issues separate:

- H100x1/H100x2/H100x4 labeling and scope visibility.
- Prefix-cached TTFT modeling.
- MoE modeling.
- Per-GPU kernel predictor fidelity.
- Stale generated dashboard artifacts vs source code behavior.

The immediate data disappearance question is mostly labeling/scope/stale
artifact behavior, not MoE or kernel training.
