# L6 — roofline utils (audit-v2 G4 + G7)

Lane: `a100-roofline-utils` / `/root/agentic-serve-a100`. Scope: own the roofline-utils
RECIPE (deterministic builder over serving-wall traces); re-derive H100 `util_bw = 0.93`
with the pinned recipe; A100 measurement (G7) is **DEFERRED** — 7/8 GPUs on the A100 host
are running someone else's latency-sensitive benchmark campaign (decided at lane launch);
this entry ships the recipe, the H100 re-derivation, and a ready-to-run A100 runbook +
preflight instead.

Baseline: replay-ON `gate_scoped_rows` captured in this worktree BEFORE any edit
(`/tmp/l6_base.json`, `/tmp/l6_base.metrics.json`).

## Input data (pinned)

The four H100 serving-wall step traces documented by audit-v2's refuted-KV-pool item
(`fitted_constants_audit_v2.md`, `available_kv_blocks` row — 4/4 traces parse):

| trace | steps |
|---|---|
| `vllm_engine_step_trace_swe_c40_t12_benchmark_serving_wall.jsonl` | 1993 |
| `vllm_engine_step_trace_swe_c80_t12_benchmark_serving_wall.jsonl` | 1822 |
| `vllm_engine_step_trace_swe_c320_t2_benchmark_serving_wall.jsonl` | 454 |
| `vllm_engine_step_trace_terminal_c80_t16_benchmark_serving_wall.jsonl` | 3587 |

Provenance correction (recorded here, fixes the audit's wording): these traces are NOT
git-committed — `profile_data/_archive/` is gitignored and the files exist only in the
main checkout's working tree (`git log --all` has no trace of them). They are the pinned
working-set artifacts the audit verified. This lane worktree symlinks
`profile_data/_archive -> /root/agentic-serve/profile_data/_archive` (untracked).

## PRE-REGISTERED RECIPE (written before any util number was computed)

Registered 2026-06-10, before running the builder. The builder
(`profiling/process/build_roofline_utils.py`) implements exactly this; any deviation
found later must be recorded below as a deviation, not silently edited here.

Constants are read from `profile_data/kernels/roofline_params_H100_llama31_8b.json`:
`n_params = 8.03e9`, `bytes_per_param = 2.0`, `kv_bytes_per_token = 131072`,
`peak_flops_per_s = 989e12`, `peak_bw_bytes_per_s = 3.35e12`. `SCHED_PRIOR = 5.7 ms`
(the current pinned value) is used ONLY as an eligibility threshold, never in the
output arithmetic.

Step classification (per JSONL record, file order):
- **decode-only**: `decode_batch > 0`, `prefill_tokens == 0`, `model_executed == "true"`.
- **pure-prefill**: `prefill_tokens > 0`, `decode_batch == 0`, `model_executed == "true"`.
- mixed steps are excluded from all three constants.

Per-request context reconstruction (deterministic, for decode KV-read bytes):
- `prompt_tokens[request_id]` from each step's `engine_cache_truth.requests[*].prompt_tokens`
  (first sighting wins; chunked prefill repeats the same value).
- `generated[request_id]` = count of PRIOR steps (any kind) in which the id appears in
  `decode_request_ids`. Context of request r at step s = `prompt_tokens[r] + generated[r]`
  (prefix-cached tokens are part of the prompt and ARE read at decode — no shared-prefix
  dedup, matching the formula `closed_form_tpot._decode_step_ms` prices in its default path
  and the G4 anchor arithmetic).
- A decode step containing any request with unknown `prompt_tokens` is EXCLUDED (count
  reported in the artifact).

Roofline times per step (the sim's own pricing conventions, `closed_form_tpot.py:16-30`):
- decode: `bw_roofline_ms = (bytes_per_param·n_params + Σ_r ctx_r·kv_bpt + decode_batch·kv_bpt) / peak_bw · 1e3`
- prefill: `compute_roofline_ms = 2·n_params·prefill_tokens / peak_flops · 1e3`
  (`prefill_tokens` = scheduled NEW tokens; cached tokens compute nothing).

### util_bw (headline)

Eligible steps: decode-only, AND
1. full batch: `decode_batch >= 0.9 × per-trace max decode_batch` (steady state; also
   excludes ramp/drain),
2. warmup: `step_id > 5`,
3. bandwidth-dominated vs host floor: `bw_roofline_ms >= 2 × SCHED_PRIOR` (= 11.4 ms) —
   the ratio must measure bandwidth, not the additive host overhead; this mirrors the
   original anchor's ~16 ms byte scale without hand-picking steps,
4. outlier rule (catches the audit's 290 ms step): drop steps whose `engine_step_wall_ms`
   lies outside `[0.5×, 2×]` the per-trace median wall of steps passing 1–3.

`util_bw = median over eligible steps (all four traces pooled) of
bw_roofline_ms / engine_step_wall_ms`.

Wall convention: `engine_step_wall_ms`, exactly what the pinned anchor's arithmetic used
(`roofline_params_H100_llama31_8b.json` `_notes.util_bw_anchor`: 16.06/17) and what
`closed_form_tpot` predicts (it adds NO separate scheduler term to tpot — util_bw absorbs
host overhead at the anchor byte scale). Diagnostic (reported, not headline):
the same median against `engine_step_wall_ms − sched` (the `ttft_queue_sim:762`
convention, which DOES add the scheduler term), and per-trace + per-byte-quartile medians
so the additive-host-overhead scale dependence is visible instead of hidden.

### util_flops

Eligible steps: pure-prefill, `prefill_tokens >= 1024` (compute roofline ≥ 3× bandwidth
roofline there, so the step is compute-bound), `step_id > 5` dropped — DEVIATION RISK
NOTED UP FRONT: big prefills cluster at trace start (turn-0 cohort), so the warmup rule
for prefill is instead the per-token outlier trim: drop steps whose per-token
`model_submit_wall_ms/prefill_tokens` lies outside `[0.5×, 2×]` the per-trace median.

`util_flops = median over eligible steps (pooled) of
compute_roofline_ms / model_submit_wall_ms`.

Wall convention: `model_submit_wall_ms`, following the pinned anchor's own documented
arithmetic (`_notes.util_flops_anchor`: 116.6/178). Diagnostic: same vs
`engine_step_wall_ms`. Honesty note registered up front: pure-prefill steps ≥1024 tokens
are RARE in these decode-heavy serving traces (schema scan: ~7 steps ≥2048 across all
four), so this re-derivation is thin by construction; the measured per-step util curve
(`prefill_gemm_util_H100.json`, R1 artifact, plateau 0.754) remains the better prefill
source. We report n per trace and refuse a headline if n_total < 5.

### scheduler_overhead_ms_per_step

Eligible: decode-only steps with `decode_batch == 1` (minimal work; the pinned comment's
"lowest-work decode steps" population, `closed_form_tpot.py:74`).
`sched = median over eligible steps (pooled) of engine_step_wall_ms − model_submit_wall_ms`.
Median, not mean (robust; no trim needed). Per-trace medians + counts reported.

### Decision rule (registered before the numbers)

- `matches_093` iff `|util_bw − 0.93| <= 0.01`.
- If it does NOT match: keep `0.93` in `configs/gpus/*.json` / `closed_form_tpot.py`
  defaults for now (honest stop-point: document the measured disagreement; rewiring is a
  prediction-moving change gated H100 ±0.3 and is deferred to integration), but fix the
  PROVENANCE comments to point at the builder artifact instead of the irreproducible
  "16.06/17 = 0.945 ≈ 0.93" anchor.
- Artifact: `profile_data/kernels/roofline_utils_H100.json` (medians, counts, per-trace
  and per-quartile breakdowns, exclusion counts, recipe echo).

## Results (builder run 2026-06-10, after the pre-registration commit)

Artifact: `profile_data/kernels/roofline_utils_H100.json` (deterministic — two
consecutive runs byte-identical, sha256 `a8e644b6…`). No recipe deviations were needed;
the builder implements the registered text as written.

| constant | pinned | re-derived (pre-registered recipe) | n steps |
|---|---|---|---|
| `util_bw` | 0.93 | **0.8111** | 62 (26/21/2/13 per trace after trim) |
| `util_flops` | 0.65 | 0.5886 (THIN — see below) | 7 |
| `scheduler_overhead_ms_per_step` | 5.7 | 4.5574 | 2666 |

**G4 verdict: 0.93 does NOT fall out** (|0.8111 − 0.93| = 0.12 ≫ 0.01). Neither does
the audit's "documented recipe" value 0.945 nor its "real step walls" range 0.90-0.92 —
those used the anchor's hand-written byte count (T≈7200/req ⇒ 53.8 GB); the
trace-reconstructed contexts of the actual steady-state steps top out at a 21.3 ms
roofline and give stable medians per byte-quartile (0.804/0.830/0.805/0.818 — flat, so
the 0.81 is not an artifact of byte scale within the eligible range). Per-trace medians
0.828 (swe_c40) / 0.810 (swe_c80) / 0.724 (swe_c320, n=2) / 0.748 (terminal_c80).
The 290 ms outlier class the audit flagged is handled by the registered trim rule
(11 steps trimmed across traces).

Diagnostics worth recording:
- `util_bw_net_of_sched_prior` = **1.2536 (> 1, unphysical)**: you cannot subtract the
  full pinned 5.7 ms scheduler term from these walls and still attribute the remainder
  to bandwidth — i.e. the `ttft_queue_sim:762` convention (decode_ms + sched) and
  util_bw=0.93 cannot BOTH be honest descriptions of these steps. The closed-form
  convention (util absorbs host overhead; no separate sched term) is the one the data
  supports, at util ≈ 0.81.
- `util_flops` is thin by construction (7 pure-prefill steps ≥1024 tokens in 7856
  steps) AND the pinned `model_submit_wall` convention is unreliable under
  AsyncScheduler: swe_c80's two steps give util 2.69 (> 1, impossible — submit returns
  before the GPU finishes). Against `engine_step_wall` the median is 0.5441. The R1
  measured curve (`prefill_gemm_util_H100.json`, plateau 0.754) remains the
  authoritative prefill-util source; this re-derivation only shows 0.65 is not
  reproducible from the serving-wall traces either.
- `sched` = 4.56 ms (median over 2666 batch-1 decode steps) vs pinned 5.7 (mean over 99
  steps of one trace). Per-trace medians 4.62/4.58/5.61/4.53 — the pinned value sits at
  the top of the range (swe_c320's quiet steps), not the center.

### Action taken (per the registered decision rule)

- `configs/gpus/*.json`, `closed_form_tpot.py` defaults, `roofline_params_*` values:
  **unchanged** (0.93/0.65/5.7 stay). Rewiring moves predictions and is deferred to
  integration where the H100 ±0.3 gate is re-checked; this lane's deliverable is the
  pinned recipe + the measured disagreement (honest stop-point, knee/util-cap
  precedent).
- Provenance comments/_notes updated to point at the builder artifact and state the
  re-derived values, replacing the irreproducible "16.06/17 = 0.945 ≈ 0.93" story.
- Byte-identity of predictions after the comment edits is established by construction
  + equivalence proof rather than a second 25-min gate run: the baseline
  (`/tmp/l6_base.*`, replay-ON, zero warnings) completed at 16:46 BEFORE any
  simulator/config edit; the edits touch only Python comments and free-text `_notes`
  (verified consumers: `RooflineParams.from_json` filters to dataclass fields;
  `configs/loader.py` / `generate_deployments.py` read named fields only); the
  functional-field equality pre/post edit is asserted programmatically (see commit) and
  the full pinned test suites pass. Integration re-gates anyway (merge protocol).

### A100 (G7) — deferred, ready to run

- Runbook: `profiling/docs/a100_roofline_utils_runbook.md`.
- Preflight: `profiling/process/preflight_a100_roofline_utils.sh` — executed read-only
  2026-06-10 against the live host: reachability PASS (alias `a100` = hostname
  `gpu-4`), **gpu_quiet FAIL (7 compute apps — the deferral condition, by design)**,
  vllm 0.19.0 PASS, `/data/models/Llama-3.1-8B-Instruct` (30G) PASS, 1879G free PASS.
- Known gap recorded in the runbook: the serving-wall capture harness was never
  committed; the runbook pins the exact JSONL field contract the builder consumes.
