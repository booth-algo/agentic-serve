# RFC: Coverage Outcome Classification Overhaul

- **Status:** Draft (for review — no code yet)
- **Date:** 2026-06-02
- **Scope:** `synthetic_distributional` coverage; generalizes to all scopes
- **Affected:** `inference-benchmark/scripts/bench_orchestrator.sh`, `sweep_all_profiles*.sh`,
  `reconcile_sweep_coverage.py`, `inference-benchmark/dashboard/src/components/CoveragePage.tsx`,
  `types-coverage-blockers.ts`

## 1. Problem

Coverage cells are labelled (TODO / N/A / failed) by **reverse-engineering a free-text
`reason` string** that the launcher writes lossily. The launcher knows exactly what happened,
collapses it into one sentence, and a regex downstream tries to recover the cause. Information
is destroyed at the source and guessed back.

### Evidence (2026-06-02)

Six `skipped` H100x2 (tp2) jobs — **four genuinely different failures, one identical reason
string**:

| Job | attempt | Actual cause (found via SSH log-diving) | `reason` on disk |
|---|---|---|---|
| gpt-oss-120b tp2 | 0 | model not staged on host / vllm KV-cache OOM | `zero results … no retryable OOM; oom_log=` |
| gpt-oss-20b tp2 | 0 | gpt-oss MXFP4 startup | `zero results … no retryable OOM; oom_log=` |
| Qwen2.5-72B tp2 | 1 | ran once, failed — capacity at tp2 | `zero results … no retryable OOM; oom_log=` |
| Qwen3.5-27B tp2 **sglang** | 0 | sglang-side crash (vllm variant is `done`) | `zero results … no retryable OOM; oom_log=` |
| Llama-3.1-70B tp2 multi | 3 | repeated failure | `zero results … no retryable OOM; oom_log=` |
| Llama-3.3-70B tp2 multi sglang | 0 | zero expected outputs | `zero results … no retryable OOM; oom_log=` |

The same reason was produced for a **missing model download**, a **KV-cache OOM**, an
**sglang engine crash**, and a **capacity limit**. The classifier cannot tell them apart, so
whatever boundary it draws (N/A vs TODO) is wrong for some subset.

### Concrete bugs this caused (all this session)

1. **3090 gpt-oss-120b shown N/A.** Root cause was a *missing model download*; the launcher
   wrote `zero results … no retryable OOM`, the classifier bucketed it `zero_results → N/A`,
   hiding a fixable infra gap as a capacity limit. (Diagnosis required SSH-ing the host and
   discovering `/home/kevinlau/models/gpt-oss-120b` did not exist.)
2. **vllm gpt-oss-120b "zero results"** was actually `ValueError: No available memory for the
   cache blocks` (KV cache = −0.11 GiB at `gpu_mem=0.85`). The launcher saw the OOM and still
   wrote `no retryable OOM; oom_log=` (empty). Fixable with `gpu_mem: 0.95`, but invisible to
   the classifier.
3. **Stale published artifact.** The dashboard read a 4-day-old `coverage-blockers.json` from
   R2 and silently rendered cells as TODO that were already terminal — no provenance guard.
4. **Dual implementation drift.** `N_A_FAILURE_CATEGORIES` exists in both Python and TSX;
   every policy change touches both, rebuilds, redeploys.
5. **Per-job smear.** One disposition is applied to all of a job's missing cells, even when
   low concurrencies succeed and high ones OOM.

## 2. Root-cause diagnosis

| Smell | Consequence |
|---|---|
| Classification derived from free-text `reason` via regex | Lossy; same string for unrelated failures |
| Launcher collapses all "no output" causes into one sentence | Root cause destroyed at the source |
| `N/A` assigned on **absence** of evidence ("no captured OOM") | Missing model / crash hidden as capacity N/A |
| Two classifiers (Python + TSX) | Drift; double-edit; rebuild/redeploy per tweak |
| Disposition is per-job | Cannot express partial (per-concurrency) outcomes |
| Conflates "what happened" with "what the UI shows" | Policy and observation tangled together |
| Separately-uploaded artifact, no freshness guard | Stale blob renders silently |

## 3. Goals / non-goals

**Goals**

- A cell is never labelled `N/A` without **positive evidence** of an irreducible limit.
- Root cause captured **once, at the source**, as structured data — never re-parsed from prose.
- **One** place defines the observation→label policy; the dashboard only renders it.
- Per-cell (`profile × concurrency`) granularity.
- Policy changes are a one-line table edit plus a test — no regex hunt, no dashboard rebuild.

**Non-goals**

- Changing what workloads are swept (`sweep.yaml` grid is unchanged).
- Replacing the orchestrator/dispatch model. This is about *outcome capture + labelling*.
- Re-running historical jobs (covered by migration, §7).

## 4. Design

### 4.1 Structured outcome record (written by the runner/launcher)

Replace the free-text `reason` with a machine-readable record per attempt. The launcher already
computes every field below; it just stops throwing them away.

```jsonc
// <job_id>.<conc?>.outcome.json   (see §4.5 for granularity)
{
  "schema": 1,
  "phase_reached": "model_load",        // preflight|model_load|engine_init|warmup|serving|complete
  "failure_class": "model_missing",     // see enum below; "none" on success
  "evidence": {
    "kv_cache_gib": null,               // engine-reported KV cache headroom (e.g. -0.11)
    "oom": false,                       // true iff a real OOM/cache-block failure was observed
    "oom_log_excerpt": null,            // captured text when oom=true
    "success_rate": null,               // 0.0–1.0 when the benchmark actually ran
    "http_ok": false,                   // server /health passed
    "outputs_present": 0,
    "outputs_expected": 11,
    "gpu_mem_util": 0.85
  },
  "attempts": 0,
  "max_attempts": 3,
  "reason_human": "model dir /home/.../gpt-oss-120b not found",  // tooltip ONLY, never parsed
  "remote_log": "/tmp/vllm_8090.log",
  "updated_at": "2026-06-02T00:00:00Z"
}
```

`failure_class` enum (closed set):

| class | meaning | typical source signal |
|---|---|---|
| `none` | success | outputs_present == expected |
| `model_missing` | weights/config absent or unloadable | `OSError: Can't load configuration` / dir absent |
| `hw_infeasible` | static: won't fit / unsupported arch | preflight vram/sm check |
| `oom_kv_cache` | engine init or runtime OOM / no cache blocks | `No available memory for the cache blocks`, kv_cache_gib<0 |
| `engine_crash` | server/engine died on startup | EngineCore stack trace, non-zero exit |
| `requests_aborted` | server up, 100% requests failed | `ABORT: N/N failed` with http_ok |
| `low_success_rate` | ran, success rate < threshold | success_rate < min |
| `timeout` | warmup/serving exceeded budget | watchdog |
| `incomplete_partial` | some cells done, some missing | outputs 0<present<expected |
| `not_attempted` | never dispatched | no run record |

### 4.2 Preflight gate (before dispatch)

A cheap check on the target host, recorded as an outcome without burning a benchmark attempt:

- model directory present + has `config.json` + index shards? → else `model_missing`.
- vram/sm feasibility for `(model, tp, dtype)`? → else `hw_infeasible`.

This alone would have turned the entire 3090 + h100 gpt-oss saga into an explicit
`model_missing` signal instead of a regex guess.

### 4.3 Single disposition function (Python, one place)

```python
def disposition(o: Outcome) -> Disposition:
    if o.failure_class == "none":            return DONE
    if o.failure_class == "hw_infeasible":   return NA_INFEASIBLE
    if o.failure_class == "oom_kv_cache" and o.evidence.gpu_mem_util >= MAX_UTIL:
        return NA_OOM                        # captured OOM at max util = irreducible
    if o.failure_class == "low_success_rate":return NA_QUALITY
    if o.failure_class in ("model_missing", "engine_crash", "requests_aborted", "timeout"):
        return FAILED                        # actionable: we have a cause + a log
    # zero/partial output, no positive evidence of a limit  ->  fillable work
    return TODO
```

The artifact stores `disposition`, a short `label`, and the `evidence` that justifies it. The
**dashboard renders `cell.disposition` directly** — all TSX regex / `N_A_FAILURE_CATEGORIES`
is deleted.

### 4.4 The invariant

> **A cell is `NA_*` only with positive evidence of an irreducible limit** — a captured OOM at
> max `gpu_mem`, a measured `success_rate` below threshold, or static infeasibility. Anything
> that produced no output for a reason we did not positively classify is `TODO` (fixable) or
> `FAILED` (has a cause to inspect) — **never N/A.**

Every bug in §1 violates this one rule.

### 4.5 Per-cell granularity

Outcomes are recorded per `(profile, concurrency)` where the runner produces them. A server
that serves but OOMs at `conc ≥ 200` marks 1–160 `DONE` and 200+ `NA_OOM`. Startup failures
(model_missing, engine_crash) apply to the whole job and fan out to all its cells.

### 4.6 Disposition taxonomy (fixed, evidence-backed)

`DONE · TODO · FAILED · NA_INFEASIBLE · NA_OOM · NA_QUALITY`

- `FAILED` (we have a stack trace → fix it) is intentionally distinct from `TODO` (never ran).
- Coverage denominator = grid − `NA_*`. `TODO` and `FAILED` stay in the fillable denominator.

### 4.7 Provenance / freshness

The artifact carries `generated_at` + `source_commit`. The dashboard surfaces it and shows a
staleness banner if older than N minutes, instead of silently rendering an old blob.

## 5. Mapping table (single source of truth)

| failure_class | evidence gate | disposition | label |
|---|---|---|---|
| none | — | DONE | — |
| hw_infeasible | static check | NA_INFEASIBLE | "N/A — won't fit" |
| oom_kv_cache | gpu_mem_util ≥ MAX_UTIL | NA_OOM | "N/A — OOM at max util" |
| oom_kv_cache | gpu_mem_util < MAX_UTIL | TODO | "retry: raise gpu_mem" |
| low_success_rate | success_rate < min | NA_QUALITY | "N/A — low success rate" |
| model_missing | — | FAILED | "model not staged" |
| engine_crash | — | FAILED | "engine crash — inspect" |
| requests_aborted | — | FAILED | "server up, requests aborted" |
| timeout | — | FAILED | "timeout" |
| incomplete_partial | — | TODO (missing cells) | "partial — re-queue" |
| not_attempted | — | TODO | "not attempted yet" |

Note: the `oom_kv_cache @ util<MAX` → TODO row is exactly the vllm gpt-oss-120b case — it is
**not** infeasible, it just needs more `gpu_mem`.

## 6. Worked examples (this session, under the new model)

| Real failure | Old label | New `failure_class` → disposition |
|---|---|---|
| 3090 gpt-oss-120b, dir absent | N/A | `model_missing` → **FAILED** (now actionable) |
| vllm gpt-oss-120b, KV −0.11 @ 0.85 | N/A / TODO | `oom_kv_cache` @ util<max → **TODO (raise gpu_mem)** |
| Qwen2.5-72B tp2, ran once failed | N/A | `oom_kv_cache`/`engine_crash` → **NA_OOM or FAILED** (evidence decides) |
| Qwen3.5-27B tp2 sglang crash | N/A | `engine_crash` → **FAILED** |
| success rate 38% | N/A | `low_success_rate` → **NA_QUALITY** |

## 7. Migration plan (incremental, backwards-compatible)

1. **Add the outcome writer** in the launcher alongside the existing `reason`/`failure.json`
   (no removals yet). Populate `failure_class` + `evidence` from signals already in the logs
   (config OSError, KV-cache line, ABORT count, success rate, outputs present/expected).
2. **Add the preflight gate** (model-present + vram/sm) → emits `model_missing`/`hw_infeasible`.
3. **Add the Python `disposition()` mapping** + emit `disposition`/`label`/`evidence` into the
   coverage artifact, *in addition to* the legacy fields. Dashboard keeps working unchanged.
4. **Switch the dashboard** to read `cell.disposition` directly; delete TSX classification.
   Behind a feature flag / artifact-schema-version check for one release.
5. **Backfill**: re-run reconcile to re-derive `failure_class` for existing terminal jobs from
   their captured logs where possible; otherwise mark `not_attempted`/`FAILED` (never silently
   N/A). Then remove the legacy `reason`-regex path.

Each step ships independently and is reversible.

## 8. Testing

- **Golden test**: a fixture set of real outcome records (the six H100x2 + 3090 cases above)
  asserts each maps to the intended disposition. Policy changes = edit the table + update the
  golden expectations; nothing else.
- **Property**: no input yields `NA_*` unless its evidence gate is satisfied (encodes §4.4).
- **Parity test**: dashboard renders exactly the artifact's `disposition` (no independent logic).

## 9. Open questions

- `MAX_UTIL` per hardware? (24 GB cards may cap lower than 80 GB before runtime OOM risk.)
- Should `requests_aborted` with a known engine signature auto-downgrade to a more specific
  class, or always stay `FAILED` until inspected?
- Where does per-cell granularity actually come from today — does the runner emit per-conc
  failure, or only per-job? (Determines §4.5 effort.)
- Auto-action hooks: should `model_missing` enqueue a (sequential) download, and
  `oom_kv_cache @ util<max` auto-bump `gpu_mem` and re-queue?

## 10. Summary

The launcher records *what it saw* as structured data; one tested function decides the label;
the dashboard paints it; nothing is N/A without proof. The whack-a-mole stops because the
information is no longer thrown away at the source and re-guessed downstream.
