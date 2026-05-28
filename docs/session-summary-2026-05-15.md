# Session Summary - 2026-05-15

Generated: 2026-05-15 12:20:38 UTC

Branch at summary time: `mse-prefix-aware-replay`

Current HEAD at summary time: `294ba8d Keep GPU state available during dashboard refresh`

Live dashboard check at summary time:

```text
https://agenticserve.tail2bcc6a.ts.net/agentic-serve/gpu-state.json
HTTP/2 200
content-type: application/json; charset=utf-8
content-length: 84750
```

## Purpose

This document is a high-detail handoff for the current agentic-serve session.
It is meant to survive context compaction and let a future agent quickly recover:

- the dashboard hosting and Tailscale setup context,
- the synthetic distributional benchmark data flow,
- the GPU scheduler, dashboard, drain/block, and reclaim work,
- the R2/result-layout cleanup decisions,
- the serving prediction, simulator, and simulator v2 dashboard changes,
- the multi-turn benchmark discussion around vLLM's upstream client,
- the latest `gpu-state.json` 404 diagnosis,
- the important dirty-worktree warning.

The repository has a large dirty worktree. Do not revert files unless the user
explicitly asks. Many changes predate this summary and may belong to the user or
another agent.

## User-Level Goal

The user has been trying to turn `agentic-serve` into the source of truth for:

1. benchmark result storage and dashboard publication,
2. synthetic distributional coverage execution across multiple GPU hosts,
3. live GPU scheduling state and control through the private dashboard,
4. serving prediction and simulator views focused on synthetic runs,
5. a stable Tailscale-hosted dashboard without needing a public domain.

The repeated product requirement is that the dashboard should be useful during
active experiments, not just after offline data rebuilds. That means local
state on `/mnt/100g` should be fresh, GPU state should be live, and R2 should
be a mirror/public artifact store rather than the only source of truth.

## Workspace / Repo State

Repository root:

```text
/root/agentic-serve
```

Important branch state:

```text
branch: mse-prefix-aware-replay
HEAD:   294ba8d Keep GPU state available during dashboard refresh
```

Important current directories:

| Path | Role |
|---|---|
| `docs/` | Root-level plans, handoffs, session summaries |
| `inference-benchmark/` | Benchmark runner, dashboard, scripts, sweep/orchestrator logic |
| `inference-benchmark/dashboard/` | React/Vite private dashboard |
| `inference-benchmark/scripts/` | Orchestration, dashboard rebuild, GPU state refresh, R2 sync |
| `llm_predict/` | Serving predictor and kernel/model prediction code |
| `simulator/` | Standalone simulator v1 package |
| `simulator_v2/` | New simulator v2 package/page data source |
| `deploy/systemd/` | Systemd units for dashboard, orchestrator, GPU refresh, cleanup |

Important dirty-worktree note:

- There are many modified files and many deleted `llm_predict_legacy/*` paths.
- `llm_predict_legacy` deletion was requested earlier by the user.
- Do not assume every dirty file was touched in the latest turn.
- Avoid broad cleanup unless explicitly scoped.

## Dashboard Hosting / Tailscale

The dashboard is intended to be reachable privately through Tailscale:

```text
https://agenticserve.tail2bcc6a.ts.net/agentic-serve/
```

The setup direction established in the session:

- Use Tailscale rather than a public domain.
- Teammates install Tailscale, join the tailnet, and get access through the
  Tailscale admin console / device ACLs.
- The dashboard is private because GPU state includes host names, users, PIDs,
  ports, process commands, and scheduling control surfaces.

Important files:

| File | Role |
|---|---|
| `deploy/systemd/agentic-serve-dashboard.service` | Dashboard service unit |
| `deploy/systemd/agentic-serve-gpu-state-refresh.service` | GPU-state refresh service |
| `deploy/systemd/agentic-serve-gpu-state-refresh.timer` | Periodic GPU-state refresh |
| `inference-benchmark/dashboard/scripts/serve-control.mjs` | Static dashboard server plus drain/block APIs |
| `inference-benchmark/scripts/rebuild-local-dashboard.sh` | Safe local bundle rebuild path |
| `inference-benchmark/scripts/refresh-gpu-state.sh` | GPU-state-only refresh path |

The dashboard service serves `dashboard/dist` under `/agentic-serve`. The
control server handles:

- static dashboard assets,
- `GET/POST /api/host-drain`,
- `GET/POST /api/gpu-block`.

## Dashboard Data Flow

The main design established in this session:

```text
benchmark hosts
  -> local durable result/state store under /mnt/100g/agent-bench
  -> dashboard/public generated JSON artifacts
  -> dashboard/dist private bundle for Tailscale
  -> optional R2 mirror under json/current/
```

The key architecture decision was to stop treating R2 as the only ground truth.
R2 remains useful for public/static artifacts and backup, but active dashboard
freshness should come from the local server store.

Important local roots:

| Root | Purpose |
|---|---|
| `/mnt/100g/agent-bench/results` | Local durable benchmark result ground truth |
| `/mnt/100g/agent-bench/state` | Local orchestrator/sweep/control state |
| `/mnt/100g/agent-bench/state/control/drained-hosts.txt` | Hosts drained from new dispatch |
| `/mnt/100g/agent-bench/state/control/blocked-gpus.txt` | Host/GPU pairs blocked from new dispatch |

Important generated dashboard JSONs:

| Artifact | Purpose |
|---|---|
| `data.json` | Aggregated dashboard rows |
| `data.trace_replay.json` | Trace replay scoped rows |
| `data.synthetic_distributional.json` | Synthetic distributional scoped rows |
| `data.archived.json` | Archived scoped rows |
| `sweep-state.json` | Coverage and job state |
| `gpu-state.json` | Private live GPU/orchestrator/process state |
| `serving-predictions.json` | Serving prediction rows |
| `gemm-eval.json` | GEMM prediction/evaluation data |
| `simulator-predictions.json` | Simulator v1 rows |
| `simulator-v2-predictions.json` | Simulator v2 rows |

R2 public base used by the dashboard in public mode:

```text
https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current
```

Private Tailscale dashboard uses:

```text
VITE_R2_JSON_BASE=/agentic-serve
```

so private JSON URLs resolve to local static files under the dashboard service.

## R2 Layout Decisions

The user identified R2 as messy, with root JSONs and multiple historical
directories. The direction established:

- Keep generated dashboard JSON under `json/current/`.
- Treat bucket-root `*.json` as legacy.
- Archive old non-current prefixes such as `bench_mse/`, `perkernel/`,
  `predictor/`, and `profiling-data/`.
- Normalize active results under:

```text
results/<scope>/<run-dir>/
```

The naming decision:

| Old / Ambiguous | New / Intended |
|---|---|
| `archive` | `trace_replay` |
| `synthetic` | `synthetic_distributional` |
| `mse`, `fixed-grid`, `canonical` | `archived` |

Important docs already present:

- `docs/data-scopes-and-profiles.md`
- `docs/synthetic-scope-sweep-handoff-2026-05-05.md`
- `inference-benchmark/docs/r2-data-layout.md`

## Synthetic Distributional Coverage

The main active benchmark scope is now:

```text
synthetic_distributional
```

The dashboard runtime tabs are intended to be synthetic-only:

- GPUs
- GEMM
- Predictions
- Simulator
- Simulator V2

The user asked why coverage showed a large number such as `1881` but the UI
showed a smaller "runs loaded" count such as `223`. The explanation established:

- The larger number is expanded expected coverage cells or profile/concurrency
  coverage.
- The smaller number was a confusing raw loaded-run/UI count.
- The UI number was removed/simplified so users do not compare unrelated
  denominators.

The synthetic grid was expanded to use trace-replay-like concurrency levels.
The hardcoded concurrency behavior was identified as a problem and made more
configurable through sweep/job config paths.

Important files involved:

| File | Role |
|---|---|
| `inference-benchmark/scripts/sweep.yaml` | Canonical sweep/job source |
| `inference-benchmark/scripts/bench_jobs.txt` | Generated/flat dispatch manifest |
| `inference-benchmark/scripts/compile_sweep.py` | Sweep config compilation |
| `inference-benchmark/scripts/reconcile_sweep_coverage.py` | Coverage reconciliation |
| `inference-benchmark/scripts/bench_orchestrator.sh` | Dispatch/orchestration |
| `inference-benchmark/dashboard/src/components/CoveragePage.tsx` | Coverage UI |

The user explicitly questioned why hardcodes existed. The plan was to move
toward YAML/JSON-backed config and generated manifests rather than hidden
hardcoded grids.

## Bench Orchestrator / GPU Scheduler

The orchestrator is responsible for launching benchmark jobs from the synthetic
coverage queue onto available GPU hosts.

Important behavior established:

- It should run continuously or through systemd/timer.
- It should not dispatch to drained hosts.
- It should respect blocked host/GPU pairs.
- It should detect stale servers/listeners that block ports/GPU memory.
- It should use local state under `/mnt/100g/agent-bench/state`.
- It should update R2 only as a downstream mirror, not as the live state source.

Important service/script paths:

| File | Role |
|---|---|
| `deploy/systemd/agentic-serve-bench-orchestrator.service` | Orchestrator systemd unit |
| `inference-benchmark/scripts/run-bench-orchestrator-service.sh` | Service entrypoint |
| `inference-benchmark/scripts/bench_orchestrator.sh` | Main orchestrator |
| `inference-benchmark/scripts/sweep_progress_report.py` | GPU/process/sweep reporter |
| `inference-benchmark/scripts/clean_orphan_gpus.py` | Reclaimer/cleanup script |
| `inference-benchmark/scripts/gpu_cleanup.json` | Reclaim policy config |

During the session, the user noticed many GPUs classified as local
non-sweeps/same-user-nonsweep. The diagnosis:

- Some processes were stale SGLang/vLLM servers from completed or abandoned
  jobs.
- Some were still considered "sweep" because their process tree contained
  sweep-shaped commands even though no live assignment mapped to them.
- Some listener processes held scheduler ports even after GPU processes were
  gone.
- The cleaner initially audited but did not execute in the installed systemd
  unit.

## GPU Reclaim Policy

The agreed reclaim design:

A same-user-nonsweep or stale sweep-shaped process is reclaimable only if all
of these are true:

- user matches the benchmark SSH user,
- process is vLLM/SGLang server-shaped,
- listener port is in scheduler range `8089-8096`,
- no live sweep state maps to that port/GPU,
- process age exceeds the threshold, usually at least one hour,
- observed in at least two consecutive cleanup scans,
- it is not other-user,
- it is not unknown-busy,
- it is not protected by config.

The user also wanted a future-proof run identity / lease concept so that the
scheduler, dispatcher, and reclaimer can coordinate:

- every launched run should have a run/lease id,
- the scheduler should know which GPU/port/job owns a process,
- the reclaimer should distinguish live assignment from stale leftover,
- future instant reclaim or run reordering should use the same identity.

Related plan document:

```text
docs/run-lease-reclaim-plan-2026-05-12.md
```

Important cleanup outcome from prior session summary:

- stale sweep GPU processes were reclaimed,
- stale listener ports were reclaimed,
- cleaner service was changed from audit-only to execute,
- SIGKILL fallback was enabled after TERM for eligible candidates,
- idle memory threshold was raised to reduce false busy classification.

## GPU Dashboard Controls

The dashboard GPU page was expanded during the broader session:

- shows GPU states,
- shows error states,
- shows which GPUs are used by sweep jobs,
- shows other-user occupancy,
- shows local same-user non-sweep occupancy,
- shows scheduler/orchestrator status,
- supports host drain controls,
- supports per-GPU block controls,
- uses compact/shrinkable GPU host panels/cards so users do not have to scroll
  to the bottom to find a host such as `h100`.

Important user requests captured:

- "keep GPU 6 of h100 free"
- "add some button which i can use to block it from being used for sweep"
- "how do i stop the 3090 runs after the current ones finish?"
- "is there a button on the dashboard for that?"

The resulting design:

- drain a host to let current jobs finish but prevent new dispatch,
- block a specific GPU to keep it free,
- expose control through the dashboard API,
- reflect the control state in `gpu-state.json`.

Important files:

| File | Role |
|---|---|
| `inference-benchmark/dashboard/src/components/GpuStatePage.tsx` | GPU UI/control page |
| `inference-benchmark/dashboard/src/hooks/useGpuState.ts` | GPU-state polling |
| `inference-benchmark/dashboard/src/types-gpu-state.ts` | GPU-state types |
| `inference-benchmark/dashboard/scripts/serve-control.mjs` | API for drain/block control |
| `inference-benchmark/scripts/sweep_progress_report.py` | Generates private GPU JSON |

## `gpu-state.json` 404 Diagnosis

The user reported:

```text
Failed to load gpu-state.json: HTTP 404
```

The analysis:

- The dashboard loads `gpu-state.json` from the configured JSON base.
- For private local/Tailscale builds that base is `/agentic-serve`.
- `useGpuState.ts` fetches it with a timestamp and `cache: no-store`, so browser
  caching is unlikely to be the root cause.
- `serve-control.mjs` returns 404 for missing static files with an extension.
- The older `build:local` path rebuilt directly into live `dashboard/dist` and
  only refreshed `gpu-state.json` after the Vite build finished.
- That created a race where the dashboard bundle was live but `gpu-state.json`
  was not present yet.

Important evidence paths:

| File | Evidence |
|---|---|
| `inference-benchmark/dashboard/src/dataUrls.ts` | `gpuStateJsonUrl` uses `jsonBase/gpu-state.json` |
| `inference-benchmark/dashboard/src/hooks/useGpuState.ts` | Fetch uses no-store plus timestamp |
| `inference-benchmark/dashboard/scripts/serve-control.mjs` | Static missing `*.json` returns 404 |
| `inference-benchmark/dashboard/package.json` | `build:local` validates, builds, then refreshes GPU state |
| `inference-benchmark/scripts/rebuild-local-dashboard.sh` | Safe path builds `dist.next`, writes GPU state, then promotes |

Current HEAD at summary time is:

```text
294ba8d Keep GPU state available during dashboard refresh
```

That commit touches:

```text
inference-benchmark/scripts/rebuild-local-dashboard.sh
```

and is intended to keep GPU state available during dashboard refresh.

Current live check at summary time:

```text
HTTP/2 200
content-length: 84750
```

## Dashboard Navigation / UI Direction

The user wanted top-level tabs, not hidden/back buttons:

- Home
- Coverage
- Runtime group:
  - GPUs
  - GEMM
  - Predictions
  - Simulator
  - Simulator V2

Specific UI changes made in the recent session:

- The old "Serving" tab was renamed to "Predictions".
- GPU/GEMM/Predictions/Simulator tabs are synthetic-only runtime tabs.
- The "back to home" concern was resolved by keeping Home in the top nav.
- Sim2 was made visually closer to the Simulator page:
  - nav label changed from `Sim2` to `Simulator V2`,
  - simulator-style icon reused,
  - top status reads `Simulator V2 target loaded`,
  - scope bar reads `H100 / Llama-3.1-8B`,
  - App focus title is `Simulator V2 Target`.

Important files:

| File | Role |
|---|---|
| `inference-benchmark/dashboard/src/App.tsx` | Page routing and simulator/sim2 focus props |
| `inference-benchmark/dashboard/src/components/Layout.tsx` | Top nav, runtime group, status text |
| `inference-benchmark/dashboard/src/dataUrls.ts` | URLs for simulator and simulator v2 JSON |

Recent build verification for the sim2 UI work:

```text
npm run build:local
validate:data passed
Vite build passed
refresh-gpu-state wrote /tmp/agentic-serve-gpu-state-latest.md
Tailscale dashboard route returned HTTP 200
```

## Predictions / Serving Page

The serving page was changed conceptually to "Predictions".

User requirements:

- only synthetic rows for GPU/GEMM/Serving/Predictions pages,
- show serving prediction data based on synthetic runs only,
- add TTFT and TPOT MAPE to GPU config buttons,
- keep MAPE visible on the right side of the table without awkward horizontal
  scrolling,
- make the UI cleaner than a jarring split scroll/pinned-column treatment.

Implemented/covered during the session:

- Prediction page filtered to synthetic runtime surface.
- GPU config buttons were adjusted to include relevant error summaries.
- Table MAPE columns were made visible at the right side.
- A cleaner MAPE rail design was explored and applied.
- The confusing "runs loaded" UI count was removed.

Important files:

| File | Role |
|---|---|
| `inference-benchmark/dashboard/src/components/ServingPredictionsPage.tsx` | Predictions table, filters, MAPE columns/rail, simulator focus |
| `inference-benchmark/dashboard/src/index.css` | Sticky/pinned rail styling |
| `inference-benchmark/dashboard/src/dataUrls.ts` | `serving-predictions.json`, simulator JSON URLs |
| `llm_predict/export_serving_predictions.py` | Serving predictions export |
| `llm_predict/serving.py` | Serving prediction model |
| `llm_predict/serving_sim.py` | Simulator-related serving prediction code |

The user asked why one card showed a different MAPE than the table. The answer
was that different aggregations were being used. The requested direction was to
use the same metric consistently across UI surfaces.

## Simulator / Simulator V2

Simulator v1 existed from the previous session:

- standalone `simulator/` package,
- H100 + Llama-3.1-8B focus,
- kernel-composed prefill/decode prediction,
- dashboard Simulator page,
- exported `simulator-predictions.json`.

Simulator v2 was added by the user and then aligned visually with the simulator
page in this session.

Important simulator v1 doc:

```text
docs/session-summary-simulator-2026-05-14.md
docs/simulator-backend-emulation-plan-2026-05-14.md
```

Important simulator v2 dashboard artifacts:

| File | Role |
|---|---|
| `simulator_v2/` | New simulator v2 package/source |
| `inference-benchmark/dashboard/public/simulator-v2-predictions.json` | Dashboard payload |
| `inference-benchmark/dashboard/src/App.tsx` | `sim2` route/focus |
| `inference-benchmark/dashboard/src/components/Layout.tsx` | `Simulator V2` nav/status |
| `inference-benchmark/dashboard/src/dataUrls.ts` | `simulatorV2PredictionsJsonUrl` |

The simulator pages are intentionally locked to:

```text
GPU:   H100
Model: Llama-3.1-8B
```

until the modeling approach is accurate and explainable.

## Multi-Turn Benchmark Discussion

The user asked how our multi-turn benchmark differs from vLLM's upstream
`benchmarks/multi_turn` client:

```text
https://github.com/vllm-project/vllm/tree/releases/v0.19.0/benchmarks/multi_turn
```

Conclusion:

- vLLM's benchmark client is good upstream infrastructure for generic
  multi-turn serving and KV-cache/offloading experiments.
- Our runner is a controlled benchmark suite plus dashboard/prediction pipeline.
- The two are related but not interchangeable.

Key differences:

| Topic | vLLM multi-turn client | agentic-serve runner |
|---|---|---|
| Main purpose | Generic multi-turn serving/KV-cache benchmark | Paper/dashboard workload suite and synthetic coverage pipeline |
| Inputs | Generated synthetic conversations or converted ShareGPT/OpenAI-style JSON | Profile registry: ShareGPT, SWE-bench, TerminalBench, OSWorld, synthetic distributional |
| Turn history | Writes model output back into future context | Prebuilds deterministic growing-history requests for stable cross-backend comparison |
| Scheduling | Active conversations per client, round-robin/random | Global round-robin barrier such as `A1, B1, C1, A2, B2, C2` |
| Backend scope | vLLM benchmark client | vLLM/SGLang/OpenAI-like backend abstraction in our runner |
| Output schema | vLLM's benchmark metrics | Dashboard/predictor schema with per-turn/cache/scope metadata |

Important local files for our path:

| File | Role |
|---|---|
| `inference-benchmark/src/benchmark/runner.py` | Multi-turn benchmark runner |
| `inference-benchmark/src/workloads/dataset.py` | Multi-turn datasets and deterministic request construction |
| `inference-benchmark/src/workloads/distributional.py` | Synthetic distributional multi-turn sampler |
| `inference-benchmark/src/benchmark/metrics.py` | Per-turn/cache metric annotation |
| `inference-benchmark/scripts/sweep_multiturn_profiles.sh` | vLLM multi-turn sweep wrapper |
| `inference-benchmark/scripts/sweep_multiturn_profiles_sglang.sh` | SGLang multi-turn sweep wrapper |

The answer to "can vLLM run with other workload profiles?" was:

- The vLLM server can run any workload our runner sends to `/v1/chat/completions`.
- vLLM's own benchmark client cannot directly understand our profile names.
- It can run other workloads only if we export/convert them into its expected
  conversation JSON format.

## GPU Hosts / Operational Notes

Hosts mentioned throughout the session:

| Host | Notes |
|---|---|
| `a100` | A100 host, used for synthetic sweep work |
| `3090` | User wanted drain behavior; several 3090 jobs were running/draining |
| `2080ti` | Had stale/malformed archived JSON warnings earlier and sweep/orphan cleanup concerns |
| `h100` | User specifically wanted GPU 6 kept free |
| `h100-2` | Earlier summaries noted SSH timeout/unreachable state |

The user observed specific GPU state oddities:

- A100 GPU had two sweep assignments at once.
- 2080ti was still orphaned after expected cleanup.
- 3090 GPUs 1, 6, 7 were still being swept despite drain expectations.
- SGLang appeared more implicated than vLLM for some stuck/drained host cases.

Likely classes of bugs discussed:

- assignment mapping duplicated for multi-GPU/tensor-parallel jobs,
- stale sweep state not matching live process tree,
- listener-only process still blocking scheduler port,
- completed/done jobs not being reclaimed automatically,
- same-user-nonsweep not being audited/executed by reclaimer,
- block/drain controls needed to be visible and easy in dashboard.

## Important User Preferences Captured

These are standing product/UX preferences from the session:

- Dashboard should be an actual operational tool, not just a static results
  viewer.
- GPU/GEMM/Predictions/Simulator pages should be synthetic-only for now.
- Top nav should contain all major pages; no hidden "back to home" behavior.
- The GPU page should be compact with expandable host/GPU cards.
- H100 GPU 6 should be keep-free/blockable.
- 3090 should be drainable so current jobs finish but no new jobs launch.
- Confusing UI counts should be removed rather than explained in-place.
- MAPE/error metrics should use the same aggregation everywhere.
- Reclaim should be automatic, not manual, but gated by safety policy.
- Hardcoded grids should move into config/manifests.
- R2 should be cleaned/normalized, with active generated JSON under
  `json/current/`.

## Validation Evidence Collected During Latest Work

Recent sim2/dashboard validation:

```text
npm run build:local
validate:data passed
Vite build passed
GPU state refresh completed
Tailscale dashboard route returned HTTP 200
```

Recent live GPU state validation:

```text
curl -I --max-time 8 https://agenticserve.tail2bcc6a.ts.net/agentic-serve/gpu-state.json
HTTP/2 200
content-length: 84750
```

Recent generated GPU state note:

```text
/tmp/agentic-serve-gpu-state-latest.md
```

Earlier GPU reclaim validation from prior summary:

```bash
python3 -m unittest \
  inference-benchmark/tests/test_sweep_progress_report.py \
  inference-benchmark/tests/test_orphan_gpu_cleaner.py

bash -n inference-benchmark/scripts/bench_orchestrator.sh

python3 -m py_compile \
  inference-benchmark/scripts/sweep_progress_report.py \
  inference-benchmark/scripts/clean_orphan_gpus.py
```

Prior result:

```text
20 unit tests passed
Bash syntax check passed
Python compile check passed
```

## Open Risks / Things To Check Next

1. `gpu-state.json` availability
   - HEAD now indicates a fix exists in `rebuild-local-dashboard.sh`.
   - Confirm whether all deploy paths use `rebuild-local-dashboard.sh` rather
     than raw `npm run build:local`.
   - If any manual flow still uses `build:local`, it may still expose a short
     missing-file window.

2. Dashboard generated artifacts
   - `dashboard/public/*.json` and `dashboard/dist/*` are generated artifacts.
   - Avoid committing huge generated JSON unless the user explicitly wants it.

3. Dirty worktree
   - Many files are modified/deleted/untracked.
   - Carefully separate intentional session changes from pre-existing state.

4. Systemd install state
   - Repo service files may differ from installed units.
   - For any runtime claim, verify with `systemctl status` or the service logs.

5. Orchestrator state
   - If coverage stalls, inspect:
     - drained hosts,
     - blocked GPUs,
     - live assignments,
     - same-user-nonsweep processes,
     - listener ports `8089-8096`,
     - job status in `/mnt/100g/agent-bench/state`.

6. vLLM benchmark client integration
   - If desired later, add an export path from our workload profiles to vLLM's
     conversation JSON format.
   - Do not replace our runner unless preserving deterministic prompt replay,
     synthetic distributional profiles, and dashboard schema.

7. Simulator v2
   - UI now matches simulator more closely.
   - Need verify the simulator v2 modeling/data path separately if accuracy or
     artifact generation becomes the next task.

## Practical Recovery Commands

Check branch and dirty state:

```bash
cd /root/agentic-serve
git branch --show-current
git log -1 --oneline --decorate
git status --short
```

Rebuild private local dashboard safely:

```bash
cd /root/agentic-serve/inference-benchmark
bash scripts/rebuild-local-dashboard.sh
```

Refresh only GPU state:

```bash
cd /root/agentic-serve/inference-benchmark
bash scripts/refresh-gpu-state.sh
```

Check live dashboard:

```bash
curl -I --max-time 8 https://agenticserve.tail2bcc6a.ts.net/agentic-serve/
curl -I --max-time 8 https://agenticserve.tail2bcc6a.ts.net/agentic-serve/gpu-state.json
```

Inspect current GPU state report:

```bash
sed -n '1,220p' /tmp/agentic-serve-gpu-state-latest.md
```

Audit GPU cleanup without executing:

```bash
cd /root/agentic-serve
python3 inference-benchmark/scripts/clean_orphan_gpus.py \
  --config inference-benchmark/scripts/gpu_cleanup.json \
  --jobs-config inference-benchmark/scripts/sweep.yaml \
  --scope synthetic_distributional \
  --state-dir /mnt/100g/agent-bench/state \
  --dry-run
```

Run targeted cleaner/reporter tests:

```bash
cd /root/agentic-serve
python3 -m unittest \
  inference-benchmark/tests/test_sweep_progress_report.py \
  inference-benchmark/tests/test_orphan_gpu_cleaner.py
```

Build dashboard:

```bash
cd /root/agentic-serve/inference-benchmark/dashboard
npm run build
```

Build private local dashboard route artifacts:

```bash
cd /root/agentic-serve/inference-benchmark/dashboard
VITE_R2_JSON_BASE=/agentic-serve npm run build
```

## Files Most Likely Relevant For Next Turns

Dashboard:

- `inference-benchmark/dashboard/src/App.tsx`
- `inference-benchmark/dashboard/src/components/Layout.tsx`
- `inference-benchmark/dashboard/src/components/GpuStatePage.tsx`
- `inference-benchmark/dashboard/src/components/ServingPredictionsPage.tsx`
- `inference-benchmark/dashboard/src/dataUrls.ts`
- `inference-benchmark/dashboard/src/hooks/useGpuState.ts`
- `inference-benchmark/dashboard/scripts/serve-control.mjs`
- `inference-benchmark/dashboard/package.json`

Orchestration and GPU state:

- `inference-benchmark/scripts/rebuild-local-dashboard.sh`
- `inference-benchmark/scripts/refresh-gpu-state.sh`
- `inference-benchmark/scripts/bench_orchestrator.sh`
- `inference-benchmark/scripts/run-bench-orchestrator-service.sh`
- `inference-benchmark/scripts/sweep_progress_report.py`
- `inference-benchmark/scripts/clean_orphan_gpus.py`
- `inference-benchmark/scripts/gpu_cleanup.json`
- `inference-benchmark/scripts/sweep.yaml`
- `inference-benchmark/scripts/bench_jobs.txt`

Benchmark runner and workloads:

- `inference-benchmark/src/benchmark/runner.py`
- `inference-benchmark/src/benchmark/metrics.py`
- `inference-benchmark/src/workloads/dataset.py`
- `inference-benchmark/src/workloads/distributional.py`
- `inference-benchmark/src/workloads/profiles.py`

Prediction/simulator:

- `llm_predict/export_serving_predictions.py`
- `llm_predict/serving.py`
- `llm_predict/serving_sim.py`
- `simulator/`
- `simulator_v2/`

Docs:

- `docs/session-summary-2026-05-13.md`
- `docs/session-summary-2026-05-14.md`
- `docs/session-summary-simulator-2026-05-14.md`
- `docs/run-lease-reclaim-plan-2026-05-12.md`
- `docs/gpu-orphan-cleanup.md`
- `docs/data-scopes-and-profiles.md`
- `inference-benchmark/docs/r2-data-layout.md`
- `inference-benchmark/docs/sweep-dispatch.md`

## Final State At Time Of This Summary

- Root summary file created as `docs/session-summary-2026-05-15.md`.
- Branch is `mse-prefix-aware-replay`.
- HEAD is `294ba8d Keep GPU state available during dashboard refresh`.
- Live private dashboard `gpu-state.json` returned HTTP 200 during summary.
- Worktree remains dirty and should be handled carefully.
- The user's latest explicit request was documentation only; no existing dirty
  code was reverted.
