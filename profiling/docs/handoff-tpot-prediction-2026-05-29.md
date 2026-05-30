# TPOT prediction stack — session handoff 2026-05-29

Canonical record of the per-turn **TPOT (inter-token latency / ITL)** prediction
work for vLLM on H100 / Llama-3.1-8B. Supersedes and folds in the four
2026-05-28 TPOT docs (see [Superseded documents](#superseded-documents)).

> **Update 2026-05-30.** Two things changed since this was written:
> 1. **TTFT + E2EL prediction shipped** (the "next phase" below). Forward queue
>    model `simulator/ttft_predict.py` (`TTFT = baseline_prefill +
>    0.5·min(1,pressure)·sched·decode_step + max(0,pressure−1)·output·tpot`;
>    strictly physical) → **TTFT 61.95% / E2EL 30.0% MAPE**, wired via
>    `augment_simulator_predictions_with_ttft.py` + `validate_ttft.py`
>    (additive, TPOT headline untouched). E2EL is decode-dominated so it tracks
>    far better than TTFT. TTFT is a wall-clock queue quantity — static models
>    cap ~60%; the faithful path is a wall-clock queue sim. See the
>    `ttft-e2el-scoping` memory.
> 2. **Dead/superseded infra retired.** The `llm-d`, `two-roofline`, and
>    `three-regime` comparison chains (predictor modules + augmenters +
>    validators + tests + dashboard lines) and five diagnostic/one-off scripts
>    were deleted (19 files), and the stale JSON keys stripped. Mentions of those
>    below are historical — the **live TPOT lines are now roofline / kernel /
>    kernel+hint / fwd-ramp**.

---

## 1. Status / headline

Four TPOT predictors (roofline / kernel / kernel+hint / fwd-ramp) live
side-by-side on the dashboard per-turn chart (plus TTFT/E2EL on the metric
toggle). The **production headline is `tpot_pred_kernel`** (overall MAPE
**16.48%**); a
classifier soft-hint and a forward ramp predictor sit beside it as comparison
lines. The matrix MAPE / KPI badge report the kernel headline (byte-identical
repoint). Whole stack since commit `e2b19ed` is **uncommitted**.

Measured (forward, all 1043 turns), MAPE = mean |pred−meas|/meas:

| metric | roofline | **kernel** | kernel+hint | fwd-ramp |
|---|--:|--:|--:|--:|
| overall | 35.7 | **16.5** | 16.2 | 19.1 |
| chat | 13.8 | **6.1** | 6.1 | 6.1 |
| osworld | 32.1 | **18.7** | 17.9 | 22.8 |
| swebench | 44.4 | **15.7** | 15.7 | 17.6 |
| terminalbench | 46.2 | **23.4** | 23.2 | 23.6 |
| **plateau** (meas>100) overall | 77.0 | 14.1 | **13.0** | 13.3 |
| plateau osworld | 72.4 | 19.5 | **16.2** | 20.2 |
| plateau swebench | 75.3 | 9.2 | 8.9 | **8.7** |
| plateau terminalbench | 84.7 | 18.3 | 17.7 | **15.3** |

(The `llm-d` / `two-roofline` / `three-regime` comparison columns were retired
2026-05-30.) **TTFT/E2EL** (forward, all 1043 turns): TTFT **61.95%**, E2EL
**30.0%** MAPE — see `validate_ttft.py` and the 2026-05-30 update banner.

129 tests + 12 subtests pass (`simulator/tests/` + `profiling/tests/`); dashboard builds clean.

---

## 2. The predictor stack (lineage)

```
RooflineParams (H100 spec + 3 single-anchor constants)
  ├─ closed-form roofline      → tpot_pred              max(compute, bw) decode floor
  ├─ kernel (HEADLINE)         → tpot_pred_kernel        measured decode grid × pressure amplifier
  ├─ kernel + classifier hint  → tpot_pred_kernel_hint   one-sided pull to classifier ramp window
  └─ forward ramp              → tpot_pred_ramp          eviction-deficit ramp, forecast cohort + recovery cap
```

**What each newer one adds (this session's work):**

- **kernel** (`simulator/kernel_tpot.py`) — `ITL = kernel_step + smoothstep(pressure;
  P_LO,p_hi)·(T_upper − kernel_step)`. `kernel_step = decode_step_ms(b_eff, ctx)`
  (measured decode kernel grid, 7.4% MAPE), output-gated upper knee, output-sustain
  gate, de-swung output-keyed ceiling `saturated_ceiling_ms(out)=min(260,118.7+3263/out)`.
  89%→16.48% overall over the session.
- **kernel+hint** (`simulator/kernel_tpot_hint.py` + `simulator/session_regime_classifier.py`)
  — the classifier (standalone, pure-workload) emits a stepping ramp window
  `{jump_start, jump_end, confidence}` via the **KV-eviction watermark** crossing
  (`pcross(0.85)`, pressure-slope width to `pcross(2.0)`); the kernel consumes it as a
  **one-sided pull** `pred = pressure_path + conf·max(0, ramp_target − pressure_path)`
  capped at `max(pressure_path[t:])` (forward-max recovery cap). FLAT/chat/coding-PERTURB
  get confidence 0 → byte-identical. Fixes the ramp phase-lag; overall 16.48→16.18.
- **fwd-ramp** (`simulator/ramp_tpot.py`) — **fully forward** 3D-roofline eviction-deficit
  ramp: forecasts the cohort `sched_hat[t]=round(C·survival(t))` from the profile
  session-length histogram (no measured `scheduled_requests`), then
  `ITL = T_bw + smoothstep(defcap; -0.12, 0.22)·sustain·(T_ceiling − T_bw)`,
  `defcap=pressure-1`. A fit-free **forward watermark-recovery cap** lets a draining
  cohort recover: `ITL=min(s_hat, max(W[t:]))`, `W=kstep+rel·(t_ceil−kstep)`,
  `rel=load/max(load[0..t])`, `load=sched_hat·blk`. Targeted ramp-tracking win on
  swe/terminal; the watermark/ceiling story is the deepest mechanistic model we have.

---

## 3. Physical model + shared constants (reuse for prefill/E2EL)

The saturated ITL plateau is an **eviction/recompute/queueing** ceiling, NOT the
dense compute roofline. Decode is KV-read-bandwidth-bound (the measured kernel
grid floor); above the KV-pool eviction **watermark (~88–92% of the 27250-block
pool, pressure ≈ 0.88–1.22)** vLLM v1 preempts+**recomputes** sequences, and
chunked-prefill injects that recompute into decode steps, lifting ITL toward an
output-amortized ceiling. Short-output workloads saturate high (~237ms), long-output
recover (osworld ~120 then drains). Both a two-roofline-amortized ceiling and a
KV-read ceiling were measured to **collapse to the floor** — the plateau is not an
amortizable roofline term — so the fitted output-keyed ceiling magnitude is kept.

`RooflineParams` (`simulator/closed_form_tpot.py`, `profiling/data/roofline_params_H100_llama31_8b.json`):

| constant | value | basis |
|---|--|--|
| `available_kv_blocks` | 27250 | measured H100 free KV blocks (max across 4 vLLM traces) |
| `cache_block_size` | 16 | vLLM v1 PagedAttention block size |
| `peak_flops_per_s` | 989e12 | H100 bf16 |
| `util_flops` | 0.65 | single-anchor (swe_c40 prefill) |
| `peak_bw_bytes_per_s` | 3.35e12 | H100 HBM |
| `util_bw` | 0.93 | single-anchor (swe_c40 decode) |
| `scheduler_overhead_ms_per_step` | 5.7 | single-anchor (99 minimal-work decode steps) |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | confirmed across 4 traces; `T_upper(compute)≈204.7ms` |
| pressure | `sched·ceil((cached+new_prefill+0.5·output)/16)/27250` | KV-pool commit fraction |
| eviction watermark `DEF_LO/DEF_HI` | −0.12 / 0.22 (defcap) | measured jump-pressure cluster (not a MAPE fit) |
| `saturated_ceiling_ms` | `min(260,118.7+3263/out)` | **FITTED** 1/output (R²=0.64) — the one fitted block |

Measured kernel grids (reusable for prefill): `simulator/kernel_step_cost.py`
(decode B×T grid) and `simulator/cached_prefill_lookup.py` (`cached_prefill_step_ms(U,P)`,
the prefill step with U pending tokens over P cached — **directly relevant to TTFT**).

---

## 4. Key files

```
simulator/
├── closed_form_tpot.py        RooflineParams + decode/prefill compute_ms|bandwidth_ms|wave-factor
├── kernel_step_cost.py        decode_step_ms(B,T) measured grid (bw floor)
├── cached_prefill_lookup.py   cached_prefill_step_ms(U,P) measured prefill grid
├── kernel_tpot.py             HEADLINE predictor (+ _kernel_step_ms, saturated_ceiling_ms, _smoothstep)
├── session_regime_classifier.py  standalone pure-workload class + ramp-window hint
├── kernel_tpot_hint.py        kernel + soft hint (forward-max cap)
├── ramp_tpot.py               forward eviction-deficit ramp + forward recovery cap
└── ttft_predict.py            forward TTFT queue model (baseline + queue amplifier); E2EL composes
profiling/process/
├── emitters/augment_simulator_predictions_with_kernel.py   writes tpot_pred_kernel(+_hint,_ramp); repoints tpot_err to kernel only
├── emitters/augment_simulator_predictions_with_ttft.py     writes ttft/e2el meas + pred (additive, no repoint)
├── emitters/update_tpot_fit_with_kernel.py                 dashboard KPI fit → kernel
├── predictors/validate_kernel_tpot.py                      per-profile + plateau MAPE gate (the measurement gate)
└── predictors/validate_ttft.py                             TTFT/E2EL MAPE gate (overall + high-TTFT slice)
inference-benchmark/dashboard/src/components/ServingPredictionsPage.tsx
    per-turn chart: actual + TPOT lines (kernel yellow, kernel+hint rose, fwd-ramp teal #2dd4bf);
    metric toggle TPOT/TTFT/E2EL (TTFT/E2EL show measured + "queue v1")
profiling/docs/
├── handoff-tpot-prediction-2026-05-29.md   ← THIS FILE
├── table_of_ramp_bad_tracking.md           ramp lag analysis (memory-linked)
├── h100_setup.md, useful_links.md          environment/reference (retained)
└── predictor.md                            user scratch/prompt notes (do not edit)
```

Memory: `/root/.claude/projects/-root-agentic-serve/memory/tpot-amplifier-pressure-law.md`
holds the full derivation chain (amplifier law → eviction-gate → pcross jump → forward ramp → recovery cap).

---

## 5. Reproduce / measure

```bash
python3 -m pytest simulator/tests/ profiling/tests/ -q          # 184 pass
python3 -m profiling.process.emitters.augment_simulator_predictions_with_kernel  # regen JSON (1043 turns)
python3 -m profiling.process.predictors.validate_kernel_tpot    # per-profile + plateau MAPE; GATE: kernel 16.48 / chat 6.1
cd inference-benchmark/dashboard && npm run build               # bundle picks up dist/
```

Dashboard: served at `http://127.0.0.1:4180/agentic-serve/?scope=synthetic_distributional#simulator`
(serve-control.mjs serves `dist/`; Tailscale `agenticserve.tail2bcc6a.ts.net`). To screenshot
headlessly: drive cached chromium at `/root/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`
via `playwright-core` from the dashboard `node_modules` (the MCP playwright server lacks Chrome).

**Reporting convention (be consistent):** MAPE = mean APE = what the dashboard shows;
median APE = tail-robust. "plateau" = turns with `tpot_meas > 100ms` (where the jump lives).
`engine_*` fields in the JSON are **SIMULATED, not ground truth** — only `tpot_meas` is measured.

---

## 6. Open issues / known limitations

- **fwd-ramp osworld residual** (plateau 20.2 vs kernel+hint 16.2): the oracle
  (measured-cohort) run scores osworld 18.9, so the gap is the forward survival
  **over-counting osworld's steep, concurrency-sensitive drain** — a cohort-forecast
  problem (a per-concurrency survival would help), not ceiling/cap.
- **Turn-0 MAPE outliers**: bench excludes initial prefill from TPOT (it's in TTFT) but
  the models amortize it in → turn-0 over-prediction (capped ~240% APE). Becomes a
  TTFT concern next phase.
- ~~**Stale JSON key**: `tpot_pred_kernel_jump`~~ — RESOLVED 2026-05-30: stripped from
  `simulator-predictions.json` (along with the retired `tpot_pred_llm_d/_two_roofline/_three_regime` keys).
- **saturated_ceiling_ms** (118.7 / 3263, R²=0.64) is the one fitted block; externally
  only the qualitative TPOT∝KV-tokens-read / output-length law is corroborated.

---

## 7. NEXT PHASE — prefill (TTFT) + E2EL prediction

**Data gap (scope this first):** `ttft_meas`, `ttft_pred`, `e2el_meas`, `e2el_pred`
are present in the dashboard type schema and the TTFT/E2EL metric toggle exists, but
**all are 0/1043 in `simulator-predictions.json`** — measured TTFT/E2EL must be sourced
(from the benchmark per-request results / extractors) and a prediction column added,
mirroring the TPOT augmenter → validate → dashboard pipeline.

**Physics to reuse:**
- TTFT = queue wait + prefill compute of the new chunk. For multiturn-synth, prefix
  caching is ON, so per-turn TTFT is dominated by the **fresh** prefill (`new_prefill_tokens`
  / `fresh_prefill_tokens`) over cached prefix `cached_context_tokens` → use
  `cached_prefill_step_ms(U=new_prefill, P=cached)` (already measured) + scheduler/queue term.
  `closed_form_tpot` has the prefill `compute_ms`/`bandwidth_ms`; `two_roofline` T_upper is
  the prefill-compute roofline. Simulated prefill fields exist (`prefill_total_ms`,
  `engine_prefill_attention_ms`, `prefill_chunks`, `prefill_intrusion_ms`) — useful as
  cross-checks, NOT ground truth.
- E2EL = TTFT + Σ(output decode). Natural composition: **E2EL_pred ≈ TTFT_pred +
  output_tokens · TPOT_pred** per turn — so a good E2EL predictor mostly falls out of
  composing the new TTFT predictor with the existing kernel TPOT predictor; validate the
  composition against `e2el_meas` once sourced.
- Benchmark caveat (`inference-benchmark/.claude/rules`): TTFT validity depends on
  arrival pattern + prefix-cache state — annotate ⚠️ for prefix-cache-hit (file-based) vs
  ✓ for ShareGPT; never compare TTFT to InferenceX (`--request-rate inf` differs).

**Suggested first steps:** (1) locate/extract measured TTFT & E2EL per turn into the JSON;
(2) build a cached-prefill-grounded TTFT predictor (reuse `cached_prefill_lookup`);
(3) compose E2EL = TTFT + output·TPOT; (4) wire as columns + validate + dashboard lines,
same measure-gated pattern as TPOT.

---

## Superseded documents

The following 2026-05-28 docs are **deleted** (this handoff supersedes them); their
load-bearing facts are preserved above. Reconstruct full detail from git history if needed.

- `handoff-2026-05-28.md` — prior session handoff; three-regime cut overall MAPE 89%→22.77%;
  predictor stack diagram; constants (`util_flops=0.65`, `util_bw=0.93`,
  `scheduler_overhead=5.7`, `available_kv_blocks=27250`, `max_num_batched_tokens=8192`);
  open paths (turn-0, workload-only coverage). All carried forward into §2–§7.
- `three-regime-tpot-2026-05-28.md` — three-regime validation verdict: per-profile/tier
  MAPE+median+p90+max; regime confusion matrix 34/44=77%; constants `K_SUSTAIN=2`,
  `BURST_COMPLETION_FRACTION=0.30`, `SATURATION_FLOOR=1.2`; SATURATING = 3-turn linear
  T_min→T_max ramp; documented osworld c≥160 misclassification. (three_regime_tpot.py retired 2026-05-30 — recover from git history if needed.)
- `two-roofline-tpot-2026-05-28.md` — two-roofline verdict: `T_lower→T_upper` KV-pressure
  interp `w=clamp((pressure−1)/2,0,1)`; turn-history `effective_c` rules
  (`K_SUSTAIN=2`, `BURST_COMPLETION_FRACTION=0.15`); median APE 18.7%. (two_roofline_tpot.py retired 2026-05-30 — recover from git history if needed.)
- `kernel-composition-tpot-2026-05-28.md` — cached-prefill lookup (`cached_prefill_step_ms(U,P)`,
  8 tests) nails osworld c80 t5 (29.8 vs 29.9); scheduler-overhead anchor 5.7ms; the
  osworld c≥160 SATURATING misclassification analysis. The cached-prefill lookup is the
  asset most relevant to the next-phase TTFT work (§7).
