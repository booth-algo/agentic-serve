# Session Summary: Serving TP Split + H100 Multi-Turn Provenance

Date: 2026-05-02

## User Preference / Operating Context

- User wants important dashboard/predictor changes committed and pushed quickly to `main`; local-only UI/predictor work has been lost after pulls.
- Source of truth for dashboard data should be GitHub Actions rebuilding from R2 and uploading refreshed JSON back to R2.
- Avoid heavy local predictor rebuilds when GitHub Actions can regenerate R2 artifacts.

## Pushed Commits This Session

- `5dae2c6` — `Split serving H100 tensor-parallel buckets`
  - First split `H100`, `H100x2`, `H100x4` serving buckets while using base `H100` predictor internals.
  - GitHub Actions rebuild `25241434451` succeeded.
  - R2 output then had `H100: 595`, `H100x2: 1140`, `H100x4: 227`, plus non-H100 base buckets.

- `772c60c` — `Generalize serving TP hardware buckets`
  - Generalized serving display/export buckets for all TP hardware labels:
    - `H100`, `H100x2`, `H100x4`
    - `A100`, `A100x2`, `A100x4`, `A100x8`
    - `RTX3090`, `RTX3090x2`, `RTX3090x4`, `RTX3090x8`
    - `RTX2080Ti`, `RTX2080Tix2`, `RTX2080Tix4`
  - Internally `_predictor_gpu_key()` strips `xN` and maps to base predictor GPU.
  - Dashboard serving tabs now load available GPU buckets from `serving-predictions.json`, not from `gemm-eval.json`.
  - GitHub Actions rebuild `25241642700` succeeded and deployed.
  - R2 serving output:
    - `H100: 595`
    - `H100x2: 1140`
    - `H100x4: 236`
    - `A100: 291`
    - `A100x2: 110`
    - `A100x4: 393`
    - `A100x8: 372`
    - `RTX3090: 265`
    - `RTX3090x2: 256`
    - `RTX3090x4: 174`
    - `RTX3090x8: 75`
    - `RTX2080Ti: 74`
    - `RTX2080Tix2: 200`
    - `RTX2080Tix4: 167`
  - R2 verification printed:
    - `current serving prediction rows: 544`
    - `current H100x4 serving prediction rows: 125`

## Validation Run Locally

- `python3 -m pytest llm_predict/tests/test_export_serving_predictions.py`
- `npm run lint` in `inference-benchmark/dashboard`
- `npm run build` in `inference-benchmark/dashboard`
- Temp smoke export:
  - `python3 -m llm_predict.export_serving_predictions --output /tmp/serving-predictions-all-tp-split.json --concurrency 80 --concurrency 120 --concurrency 160 --concurrency 200 --concurrency 256 --concurrency 320`
  - Confirmed buckets such as `RTX3090x4`, `A100x4`, and `RTX2080Tix4` exist locally before Actions rebuild.

## H100x1 Multi-Turn / Cache Provenance

Question investigated: user remembered current/canonical `H100` multi-turn cache rows, but after TP split only current `H100x4` appeared.

Findings:

- Plain `H100` multi-turn rows do exist in local `data.json`, but they are `archive`, not `current`.
- The archive plain `H100` multi-turn rows came from result directories like:
  - `inference-benchmark/results/h100_Llama-3.1-70B_tp4_vllm`
  - `inference-benchmark/results/h100_Llama-3.3-70B_tp4_vllm`
  - `inference-benchmark/results/h100_Qwen2.5-72B_tp4_vllm`
  - Also some `tp8` H100 archive dirs.
- Synced file mtimes for those archive H100 per-turn rows were around 2026-04-25 UTC.
- They are classified as archive by `inference-benchmark/dashboard/scripts/build-data.ts` because `detectDataScope()` only returns current when:
  - `raw.config.dashboard_scope === "current"`, or
  - the path is under `results/current/`.
- Those archive H100 sidecars have `_per_turn.json`, but the checked sample did not have explicit cache telemetry fields like:
  - `median_cache_hit_rate`
  - `median_new_prefill_tokens`
  - `median_cached_context_tokens`
- Current canonical rows under `results/current/...` do have `dashboard_scope: current`, `prefix_caching_state: on`, and explicit cache/new/cached token fields.

Important interpretation:

- The “archive cache rows” are better described as archive per-turn rows with derived cache features.
- In `llm_predict/cache_aware.py`, `derive_turn_cache_features()` falls back to deriving cache when measured cache fields are absent:
  - `new_prefill_tokens = total_context - previous_context`
  - `cached_context_tokens = total_context - new_prefill_tokens`
  - `cache_hit_rate = cached_context_tokens / total_context`
- Before the TP split, current `H100x4` serving rows were collapsed into the `H100` serving tab, likely making it look like current H100x1 canonical data existed.

## Current Dirty Worktree Warning

As of this note, `git status --short` showed dirty/untracked files. Do not revert them casually; some may be user/other-agent work.

Tracked dirty files observed:

- `inference-benchmark/dashboard/public/sweep-state.json`
- `inference-benchmark/dashboard/src/components/GemmPage.tsx`
- `inference-benchmark/scripts/bench_jobs.txt`
- `inference-benchmark/scripts/sweep.yaml`
- `llm_predict/cache_aware.py`
- `llm_predict/export_serving_predictions.py`
- `llm_predict/serving.py`
- `llm_predict/tests/test_cache_aware.py`
- `llm_predict/validate.py`

Untracked files observed:

- `.claude/docs/session-summary-2026-04-30-distributional-benchmarks.md`
- `.claude/docs/sweep-delegation-2026-04-30-current-canonical.md`
- `.codex-notes/prefix-cache-ttft-fix-plan.md`
- `.codex-notes/session-handoff-2026-05-02-h100-ttft-data-regression.md`
- `.codex-notes/ttft-prefix-cache-miscalculation.md`
- `README.md`
- `inference-benchmark/tests/test_real_trace_workloads.py`
- `llm_predict/data/flash_attn/`
- `llm_predict/tests/test_attention_prediction.py`

## Recommended Next Resume Point

- If continuing predictor debugging, first inspect the current dirty diffs before editing.
- The high-value next question is likely TTFT error source:
  - current canonical rows have explicit cache telemetry;
  - archive rows often only have per-turn context growth and use derived cache features;
  - mixing these in aggregate MAPE can mislead.
- Be very explicit about data scope (`current` vs `archive`) and hardware bucket (`H100x4` vs `H100`) when interpreting serving predictor errors.

## Local Dashboard Iteration Server

Added 2026-05-02 in the follow-up session.

Pulled current R2 dashboard JSON artifacts into
`inference-benchmark/dashboard/public/`:

- `data.json`
- `serving-predictions.json`
- `sweep-state.json`
- `gemm-eval.json`
- `predictor-coverage.json`
- `profiling-state.json`
- `roofline-quadrant.json`
- `gemm-extrapolation.json`

R2 object mtimes at pull time showed the main current data artifacts rebuilt at:

- `data.json`: 2026-05-02 02:50:42 UTC
- `serving-predictions.json`: 2026-05-02 02:50:47 UTC
- `sweep-state.json`: 2026-05-02 02:49:43 UTC

The dashboard dev server was launched host-bound on port `4173` with local
public JSON URLs:

```bash
cd /root/agentic-serve/inference-benchmark/dashboard
VITE_DATA_JSON_URL=/agentic-serve/data.json \
VITE_SWEEP_STATE_URL=/agentic-serve/sweep-state.json \
VITE_GEMM_EVAL_JSON_URL=/agentic-serve/gemm-eval.json \
VITE_SERVING_PREDICTIONS_JSON_URL=/agentic-serve/serving-predictions.json \
VITE_PROFILING_STATE_JSON_URL=/agentic-serve/profiling-state.json \
VITE_PREDICTOR_COVERAGE_JSON_URL=/agentic-serve/predictor-coverage.json \
npm run dev -- --host 0.0.0.0 --port 4173
```

Verified with `curl -I`:

- `http://127.0.0.1:4173/agentic-serve/` -> 200
- `http://127.0.0.1:4173/agentic-serve/data.json` -> 200
- `http://127.0.0.1:4173/agentic-serve/serving-predictions.json` -> 200

Browser URLs shown by Vite:

- Local: `http://localhost:4173/agentic-serve/`
- Network: `http://49.13.232.148:4173/agentic-serve/`
- Network: `http://10.250.201.35:4173/agentic-serve/`
- Network: `http://172.18.0.1:4173/agentic-serve/`

Important: the dashboard source was also changed so profiling and predictor
coverage hooks use configurable URLs from `src/dataUrls.ts`, matching the rest
of the JSON fetches. Before that change, those two hooks were hardcoded to R2
and would not use locally pulled JSON files.

## Recommended Predictor Iteration Loop

Use R2-pulled JSONs as the measured-data baseline. The running local dashboard
currently serves the pulled R2 artifacts from `dashboard/public`, which is good
for comparing against deployed/current data.

For predictor work, keep `data.json` from R2 unless benchmark ingestion or
measured result data changes. `data.json` is the measured truth source; most
predictor iterations should only regenerate `serving-predictions.json`.

When changing predictor code, regenerate only the serving prediction artifact:

```bash
cd /root/agentic-serve
python3 -m llm_predict.export_serving_predictions \
  --output inference-benchmark/dashboard/public/serving-predictions.json
```

The Vite dashboard should serve the updated JSON without a restart.

To compare deployed/R2 predictor output against a local predictor export:

```bash
cd /root/agentic-serve
cp inference-benchmark/dashboard/public/serving-predictions.json \
  /tmp/serving-predictions-r2-baseline.json

python3 -m llm_predict.export_serving_predictions \
  --output inference-benchmark/dashboard/public/serving-predictions.json
```

Then inspect the local dashboard. Restore the R2 baseline if needed:

```bash
cp /tmp/serving-predictions-r2-baseline.json \
  inference-benchmark/dashboard/public/serving-predictions.json
```

For TTFT/cache predictor iteration, focus the dashboard on:

- data scope: `current`
- hardware bucket: `H100x4`
- cache-affected profiles, especially `coding-singleturn` and multi-turn
  profiles

Before trusting charts, run a quick JSON summary to verify cache/unsupported
counts. Example:

```bash
node -e 'const fs=require("fs"); const data=JSON.parse(fs.readFileSync("inference-benchmark/dashboard/public/serving-predictions.json","utf8")); const h=data.H100x4||[]; const cur=h.filter(r=>r.data_scope==="current"); console.log({current:cur.length, cacheaware:cur.filter(r=>r.cache_aware_applied).length, unsupported:cur.filter(r=>r.cache_prediction_regime==="unknown_prefix_cache").length});'
```

Focused validation after each predictor change:

```bash
python3 -m pytest llm_predict/tests
npm run lint --prefix inference-benchmark/dashboard
```

Remember the distinction:

- R2-pulled `serving-predictions.json` shows the deployed/GitHub Actions
  predictor output.
- Locally regenerated `serving-predictions.json` shows the current dirty
  worktree predictor output.

## 2026-05-02 A100x1 No-Affine TTFT Predictor Iteration

Removed the affine TTFT correction path entirely. The predictor no longer
applies or exports `ttft_correction_alpha`, `ttft_correction_beta_ms`,
`ttft_correction_note`, `ttft_correction_applied`, or calibration
`ttft_correction` blocks. TTFT now remains raw-kernel plus explicit queue/cache
factors; decode/MoE factors remain separate.

Regenerated artifacts:

- `llm_predict/data/serving_calibration.json`
- `llm_predict/data/serving_calibration_report.md`
- `inference-benchmark/dashboard/public/serving-predictions.json`

Validation:

- `python3 -m pytest llm_predict/tests` -> 24 passed.
- JSON string checks on `serving-predictions.json` and
  `serving_calibration.json` found no alpha/beta TTFT correction fields.
- A100 current rows versus the pre-change local baseline:
  - E2EL mean error: 302.2% -> 112.5%
  - TPOT mean error: 408.4% -> 153.3%
  - TTFT mean error: 72.3% -> 177.3%

Interpretation: removing the hidden additive TTFT floor improves A100x1 E2EL
and TPOT by letting low-confidence MoE decode/queue information apply, but TTFT
gets worse. That is expected: cached-prefill TTFT still needs an explicit
cache/queue model instead of a beta floor.

Local dashboard status:

- Existing host listener still on `http://49.13.232.148:4173/agentic-serve/`.
- New Vite session also started on `http://49.13.232.148:4174/agentic-serve/`
  because 4173 was already occupied.
- Host curl verification on 4174 confirmed 291 A100 rows and no alpha/beta
  TTFT correction fields in the served JSON.

## 2026-05-02 A100x1 Low-Confidence TTFT Queue Gate

Controlled A100-only what-ifs after removing alpha/beta showed two separate
issues:

- Low-confidence TTFT queue tables caused severe short-context overprediction
  on A100 current rows, especially `gpt-oss-20b` chat/osworld at C80/C160.
- Adding a first decode token to TTFT helped long-context Llama rows a bit, but
  worsened those already-overpredicted queue rows.

Applied the narrow fix first: `ttft_queue_factor()` now uses the normal
high/medium confidence gate for generated calibration artifacts. Low-confidence
TTFT queue factors are recorded for visibility but not applied. The separate
experimental gate remains available for MoE decode factors.

Regenerated:

- `inference-benchmark/dashboard/public/serving-predictions.json`

Validation:

- `python3 -m pytest llm_predict/tests/test_cache_aware.py llm_predict/tests/test_attention_prediction.py`
  -> 21 passed.
- Host curl on `http://127.0.0.1:4174/agentic-serve/` -> HTTP 200.
- Served/regenerated JSON has 291 A100 rows and no alpha/beta TTFT fields.

A100 current metrics after this gate:

- TTFT mean error: 177.3% -> 72.3%
- E2EL mean error: 112.5% -> 95.6%
- TPOT mean error: 153.3% -> 164.9%
- A100 current rows with applied TTFT queue factors: 0

Remaining gap: long-context cached-prefill A100 Llama rows are now the worst
TTFT misses. Example: `swebench-multiturn` C20 predicts 25.44 ms vs measured
727.51 ms. Flash attention already reads full-context KV for cached prefill, so
the next candidate should be explicit long-context/first-token serving physics
rather than restoring empirical alpha/beta floors.

## 2026-05-02 A100x1 Prefix-Cache Contention Overlay

Follow-up on the remaining A100x1 long-context gap: per-turn inspection showed
the miss is concentrated in prefix-cache affected multi-turn rows. Individual
turns report hundreds of milliseconds of TTFT while only ~100-200 new tokens are
prefilled against a long cached prefix. The raw kernel model treats those turns
as nearly free once the prefix is cached.

Controlled what-if:

- Base after low-confidence TTFT queue gate:
  - A100 current TTFT mean error: 72.3%
  - A100 current TPOT mean error: 164.9%
  - A100 current E2EL mean error: 95.6%
- Applying low-confidence prefix-cache contention factors for A100 only:
  - A100 current TTFT mean error: 17.7%
  - A100 current TPOT mean error: 20.9%
  - A100 current E2EL mean error: 17.5%
  - H100x4 current metrics unchanged by scope.

Applied the A100-scoped experimental prefix-cache path in
`prefix_cache_contention_factors()`: low-confidence prefix-cache contention
factors may apply only for A100. TTFT queue factors remain high/medium gated.
Prefix-cache priors remain high/medium gated.

Validation:

- `python3 -m pytest llm_predict/tests` -> 25 passed.
- Regenerated `inference-benchmark/dashboard/public/serving-predictions.json`.
- Local dashboard JSON verified reachable:
  `http://127.0.0.1:4174/agentic-serve/serving-predictions.json` -> HTTP 200.
- A100 current prefix-cache contention applied rows: 44/44.
- No alpha/beta TTFT correction fields in A100 JSON rows.

Important local-dashboard correction:

- The earlier `4174` dev server was started without `VITE_*_JSON_URL`
  overrides, so the React app still defaulted to R2 JSON even though
  `/agentic-serve/serving-predictions.json` existed locally.
- A new host-accessible dev server is running on
  `http://49.13.232.148:4175/agentic-serve/` with local JSON overrides:
  `VITE_DATA_JSON_URL`, `VITE_SWEEP_STATE_URL`,
  `VITE_SERVING_PREDICTIONS_JSON_URL`, `VITE_PROFILING_STATE_JSON_URL`, and
  `VITE_PREDICTOR_COVERAGE_JSON_URL` all point at `/agentic-serve/*.json`.
- Verified transformed `src/dataUrls.ts` on port `4175` contains the local
  `/agentic-serve/serving-predictions.json` URL. Use `4175` for local predictor
  iteration; `4174` may show deployed/R2 metrics.

Remaining concern: the prefix-cache factors are still low-confidence and
profile/concurrency-specific. This is a useful local A100x1 predictor iteration,
not final physics. The next cleanup should replace the large multiplicative
factors with an explicit serving model for prefix-cache lookup/residency and
first-token scheduling overhead.

## 2026-05-02 Analytical-Only Serving Predictor Reset

User clarified that low-confidence factors should not be used at all and that
the serving predictor should be analytical-only. The A100-only prefix-cache
factor overlay above is now treated as a diagnostic what-if, not an active
predictor path.

Removed empirical factor application and factor-field export from the serving
prediction path:

- `predict_serving()` no longer applies TTFT queue factors, dense decode
  correction factors, MoE decode factors, or prefix-cache contention factors.
- Multi-turn aggregation no longer carries factor/applied fields.
- Dashboard serving prediction rows no longer export `ttft_queue_factor`,
  `decode_correction_factor`, `prefix_cache_ttft_factor`,
  `prefix_cache_decode_factor`, or related applied flags.
- Prefix-cache affected rows without `perTurn` cache features are marked
  unsupported instead of using calibration-derived prefix-cache priors.

Regenerated:

- `inference-benchmark/dashboard/public/serving-predictions.json`

Validation:

- `python3 -m pytest llm_predict/tests` -> 22 passed.
- Local dashboard JSON verified reachable on the local-data server:
  `http://127.0.0.1:4175/agentic-serve/serving-predictions.json` -> HTTP 200.
- JSON string check found no exported empirical factor/correction fields.

A100 current analytical-only metrics after reset:

- TTFT mean/median error: 72.3% / 80.6%
- TPOT mean/median error: 408.4% / 162.2%
- E2EL mean/median error: 302.2% / 84.3%

Interpretation: the regression is expected and useful. It exposes the true
analytical gaps that empirical factors were hiding, especially MoE decode and
prefix-cache/first-token serving overhead. Next work should add explicit
analytical terms rather than restoring multiplicative factors.

## 2026-05-02 Empirical Factor Surface Purge

User clarified: remove usage of factors; serving prediction should remain
analytical rather than fitted by empirical multipliers.

Additional cleanup completed after the analytical-only predictor reset:

- `framework_corrections.py` is now metadata-only: calibration lookup status
  plus TTFT validation scope. Static fallback tables and multiplier APIs were
  removed.
- `training/calibrate_serving.py` no longer fits or writes TTFT queue, decode,
  MoE decode, prefix-cache contention, or prefix-cache prior artifacts.
- `serving_calibration.json` and `serving_calibration_report.md` were
  regenerated as diagnostic coverage/raw-error artifacts only.
- Dashboard TypeScript no longer defines or displays prefix-cache contention
  multiplier fields.
- Regenerated `inference-benchmark/dashboard/public/serving-predictions.json`.

Validation:

- `python3 -m compileall -q llm_predict` -> passed.
- `python3 -m pytest llm_predict/tests` -> 22 passed.
- String search across `llm_predict`, dashboard source, generated
  `serving_calibration.json`, generated `serving_calibration_report.md`, and
  generated `serving-predictions.json` found no remaining factor/contention
  multiplier symbols.
- Port `4175` is the only active local dashboard port among `4173-4176` and
  serves the refreshed JSON:
  `http://127.0.0.1:4175/agentic-serve/serving-predictions.json` -> HTTP 200.

A100 current analytical-only metrics after the stricter purge:

- TTFT mean/median/best/worst error: 72.3% / 80.7% / 17.5% / 96.5%
- TPOT mean/median/best/worst error: 408.4% / 201.0% / 2.1% / 2017.0%
- E2EL mean/median/best/worst error: 302.2% / 102.4% / 0.2% / 1506.6%
- Unsupported current A100 rows: 0

Current interpretation: the predictor is now honestly analytical and exposes
large unresolved physics/modeling gaps. Do not reintroduce empirical
multipliers. Next useful iteration is an explicit analytical model for MoE
decode cost and prefix-cache/first-token serving overhead.

## 2026-05-02 E2EL Off-by-One Fix

Ground-truth benchmark semantics: TTFT is time to first output token, and TPOT
is mean inter-token latency after the first output token. Therefore the
request-level identity is:

`E2EL = TTFT + TPOT * max(output_tokens - 1, 0)`

Implemented this in the analytical predictor:

- Added `decode_interval_count(output_tokens)` in `llm_predict/serving.py`.
- `predict_serving()` now integrates decode over `OSL - 1` intervals and
  computes `TPOT = decode_total / max(OSL - 1, 1)`.
- Multi-turn aggregate TPOT is weighted by successful post-TTFT decode
  intervals, not raw output-token count.
- Calibration diagnostics now use `decode_total_raw_ms` from the predictor
  instead of recomposing with `tpot * OSL`.
- Export skips TPOT error when predicted TPOT is zero for one-token outputs.

Evidence from `data.json` means:

- `mean_ttft + mean_tpot * (avg_osl - 1)` median error vs mean E2EL: 1.247%.
- `mean_ttft + mean_tpot * avg_osl` median error vs mean E2EL: 1.523%.

Validation:

- `python3 -m compileall -q llm_predict` -> passed.
- `python3 -m pytest llm_predict/tests` -> 25 passed.
- Regenerated `llm_predict/data/serving_calibration.json`.
- Regenerated `inference-benchmark/dashboard/public/serving-predictions.json`.
- Local dashboard JSON on port `4175` -> HTTP 200, last modified
  2026-05-02 12:56:57 UTC.

A100 current after this change:

- TTFT mean/median/best/worst error: 837.3% / 419.3% / 17.5% / 2759.3%
- TPOT mean/median/best/worst error: 185.3% / 13.5% / 0.1% / 657.9%
- E2EL mean/median/best/worst error: 201.7% / 98.8% / 0.7% / 740.5%

Interpretation: the off-by-one fix improves TPOT/E2EL accounting but does not
solve prefix-cache TTFT physics. Current multi-turn TTFT can under-predict
badly because the analytical cache-aware path models only new prefill tokens
and not serving scheduler/cache lookup/chunking overhead.

## 2026-05-02 Sweep Profile Infeasibility

Built the profile-level OOM/context-infeasible mechanism for current sweep
coverage:

- Added `profile_infeasible` to `inference-benchmark/scripts/sweep.yaml`.
- Current rule marks `swebench-multiturn` and `terminalbench-multiturn` as
  N/A when `mode=multi` and resolved `max_len < 32768`.
- `compile_sweep.py` now removes only those blocked profiles from emitted
  jobs instead of skipping the whole host/model/tp/backend cell.
- `publish_sweep_state.py` now publishes concrete
  `profile_infeasible[]` records with `hw_label`, model, backend, profile,
  max_len, and reason.
- `CoveragePage.tsx` consumes those records, excludes N/A profile rows from
  expected coverage counts, and shows the context-length reason in tooltips.
- `bench_orchestrator.sh` now writes a job signature and reopens legacy
  terminal `skipped` states when the generated job shape changed, so old
  full-profile skips do not block reduced `chat-multiturn/osworld-multiturn`
  reruns.

Validation:

- `python3 -m py_compile inference-benchmark/scripts/compile_sweep.py
  inference-benchmark/scripts/publish_sweep_state.py` -> passed.
- `bash -n inference-benchmark/scripts/bench_orchestrator.sh` -> passed.
- `python3 inference-benchmark/scripts/compile_sweep.py` -> 92 emitted rows,
  27 profile-infeasible reductions, 2 VRAM infeasible skips, 1 known OOM.
- `python3 inference-benchmark/scripts/publish_sweep_state.py --no-upload`
  -> 97 cells and 56 concrete `profile_infeasible` records.
- `npm run build` and `npm run lint` in
  `inference-benchmark/dashboard` -> passed.
- Port `4175` serves the regenerated sweep state:
  `http://127.0.0.1:4175/agentic-serve/sweep-state.json` -> HTTP 200.
