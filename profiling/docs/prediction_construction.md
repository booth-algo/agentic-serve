# How TPOT / TTFT / E2EL predictions are constructed (from scratch)

_Reference for the simulator's per-turn prediction pipeline, the constants it rests on, and their
provenance. Grounded in `simulator/{kernel_tpot,kernel_step_cost,closed_form_tpot,ttft_queue_sim}.py`.
Last verified 2026-06-05._

**Provenance legend:** 🟢 physics-derived · 🔵 trace-measured (cited artifact) · ⚙️ vLLM/engine default · 🔴 fitted-unexplained

---

## Shared inputs — `RooflineParams` (per config; `closed_form_tpot.py`)

| Symbol | Value (H100 tp1 / H100x2 tp2) | Prov. |
|---|---|---|
| `n_params` | 8.03e9 | 🟢 |
| `peak_flops_per_s` | 989 TFLOP/s (per GPU; compute term divides `n_params` by `tp`) | 🟢 |
| `peak_bw_bytes_per_s` | 3.35 TB/s (per GPU — stays per-GPU at tp>1; each rank streams its own HBM) | 🟢 |
| `util_flops` / `util_bw` | 0.65 / 0.93 | 🔵 |
| `kv_heads` | 8 (GQA; caps KV shard at `min(tp, kv_heads)`) | 🟢 |
| `cache_block_size` | 16 | ⚙️ |
| `available_kv_blocks` | 27250 / **62416** | ⚙️ |
| measured **decode grid** (`kernel_step_cost`) | per-config CSV (`decode_profile_*` / default H100 wide sweep) | 🔵 |
| measured **saturation ceiling** (`kernel_tpot`) | per-config JSON (`saturated_ceiling_*`) | 🔵 |
| measured **prefill floor** (`ttft_queue_sim`) | per-config JSON (`prefill_floor_llama31_8b.json`); fallback 26.0 | 🔵 |

---

## TPOT — `kernel_tpot.predict_turn_tpot(cached, new, output, scheduled)`

A **measured unsaturated decode-kernel step, ramped toward a measured saturation ceiling by KV pressure**
(gated by output length). Per turn:

| Step | Formula | Prov. |
|---|---|---|
| 1 | `ctx = cached + new + 0.5·output` (resident KV at decode midpoint) | 🟢 |
| 2 | `per_session_blocks = ceil(ctx / 16)` | 🟢 |
| 3 | `capacity_batch = available_kv_blocks / per_session_blocks` | 🟢 |
| 4 | `b_eff = min(scheduled, capacity_batch)` — KV-throttled running batch | 🟢 |
| 5 | `pressure = scheduled · per_session_blocks / available_kv_blocks` | 🟢 |
| 6 | `kernel_step = decode_step_ms(b_eff, ctx)` — measured grid (bilinear) where covered 🔵; beyond the grid edge / OOM corners the analytic roofline `max(fixed_floor + b·ctx·kv_bytes/(kv_shards·bw·util_bw), 2·(n_params/tp)·b/(flops·util_flops))` | 🔵 grid; 🟢 roofline; 🔵 `fixed_floor`; 🟢 launch-floor `max(0.0,·)` guard (de-fit 2026-06-05; was `0.3`) |
| 7 | `t_upper = max(kernel_step, saturated_ceiling_ms(median_output))` — measured plateau anchors, interpolated | 🔵 |
| 8 | `p_hi = P_HI_SHORT + smoothstep(output, OUT_KNEE_LO, OUT_KNEE_HI)·(P_HI_LONG − P_HI_SHORT)` | 🔴 `P_HI_SHORT=1.22`, `P_HI_LONG=2.0` (honest tuned-knobs since 2026-06-09; the measured band disagrees & is gate-rejected — see De-fit log); 🔵 knees `28/86` |
| 9 | `sustain = smoothstep(output, SAT_SUSTAIN_LO=9, SAT_SUSTAIN_HI=24)` | 🔵 |
| 10 | `weight = smoothstep(pressure, P_LO, p_hi) · sustain` | 🔴 `P_LO=0.88` (honest tuned-knob since 2026-06-09; measured onset ≈ 0.45, gate-rejected) |
| 11 | **`TPOT = kernel_step + weight·(t_upper − kernel_step)`** | — |

- At **low pressure** `weight≈0` → `TPOT = kernel_step` (the raw measured/analytic decode step).
- At **high pressure** → `TPOT` rises to the measured saturation plateau `t_upper`.
- The fitted constants (🔴) live entirely in the **ramp** (steps 8–10): `P_LO`, `P_HI_SHORT`, `P_HI_LONG` —
  now honestly labelled compensating fits; the reproducible measurement of the real ramp band
  (`build_ramp_knees` → `ramp_knees_*_llama31_8b.json`) disagrees with them and is gate-rejected,
  meaning the smoothstep-in-pressure ramp *shape* is the model error (see De-fit log 2026-06-09).
- `decode_step_ms` is the same composition for all configs; `predict_cell_tpot` calls `predict_turn_tpot`
  per turn using the cell's **median output** as the ceiling output (so the plateau doesn't jitter).

### Why this over-prices tp=2 (context)
tp=2's 2.29× KV pool keeps `pressure` below `P_LO` even at high concurrency, so `weight≈0` and TPOT = the
raw step-6 kernel. At high `b_eff` × high `ctx` that lands beyond the (sparse) tp2 grid edge, where the
analytic roofline's `b·ctx` term grows **linearly** while the real kernel is **sub-linear** → ~1.25–1.30×
over-price. On tp=1 the same regime is hidden because tp1 saturates first and is priced by the (correct)
measured ceiling (step 7). See `.omc/specs/deep-dive-trace-whether-there-are-fitted.md`.

---

## TTFT — `ttft_queue_sim.predict_cell_ttft_qsim(...)` (event-driven closed-loop sim)

Not a closed-form — a simulation. **Cohort** = trajectory replay: real per-(GPU,conc) session
`[cached, new, output]` trajectories 🔵. Sessions arrive in a synchronized **barrier herd**; KV persists
across turns under LRU eviction + RECOMPUTE preemption. The vLLM v1 chunked-prefill budget
(`MAX_NUM_BATCHED_TOKENS=8192` ⚙️, per-req chunk ≤ `LONG_PREFILL_TOKEN_THRESHOLD=1310` ⚙️) governs which
requests advance each step.

Per engine step (`_price_step`) — a mixed prefill+decode step:

| Term | Formula | Prov. |
|---|---|---|
| decode | `decode_step_ms(decode_batch, mean_running_ctx)` | 🔵 |
| prefill NEW | `(gemm_per_tok_loaded(total_chunk) + PREFILL_NEW_DISPATCH_RESIDUAL)·total_chunk` | 🟢 + 🔵 `0.005745` (measured host serving-stack/new tok; de-fit 2026-06-05) |
| └ `gemm_per_tok` | `2·(n_params/tp)/(peak_flops·util-ramp)` + `PREFILL_TP_COMM·(tp>1)` (util ramps `util_flops→PREFILL_GEMM_UTIL_SAT=1.0` over the chunk budget) | 🟢; 🔵 `TP_COMM=0.00585` |
| prefill FA3 | `PREFILL_FA3_MS_PER_TOKEN2 · M·(R + 0.5·M)·frac` (super-linear attn; M=tokens (re)prefilled, R=resident prefix) | 🔵 `8.31e-7` |
| host per-req | `PREFILL_HOST_PERREQ_MS_PER_TOKEN · cached · frac` (re-tokenize cached context, summed over concurrent prefills) | 🔵 `0.0029076` = 0.4764×6.103e-3 (measured split, de-fit 2026-06-10) |
| host shared | `PREFILL_HOST_SHARED_MS_PER_TOKEN · mean_cached` (amortized once/step) | 🔵 `0.0031954` = 0.5236×6.103e-3 (measured split, de-fit 2026-06-10) |
| step cost | `step_ms = max(decode_ms + scheduler_overhead, prefill_ms)` | 🔵 |
| floor (once per req, at first token) | `floor_residual = max(0, prefill_floor − scheduler_overhead)` added to `first_token_epoch` | 🔵 (PR #73: per-config; H100=25.86, H100x2=14.01) |

→ **`TTFT[turn] = first_token_epoch − arrival_epoch`** = queue-wait + prefill service + per-request floor.
A turn no session reaches falls back to the forward static predictor (`_fallback_ttft`).

No 🔴 rows remain in the TTFT step pricing: the **prefill host-split partition** is now the measured
point estimate (shared fraction 0.5236 of the measured sum 6.103e-3 — pooled OLS of the live B-sweep
band [0.40,0.54]; adopted 2026-06-10 on the production replay-ON re-gate after an initial replay-OFF
worktree gate had rejected it — see the De-fit log). (`PREFILL_NEW_DISPATCH_RESIDUAL` was de-fit
2026-06-05 and is 🔵.)

---

## E2EL
`E2EL[turn] = TTFT[turn] + output_tokens · TPOT[turn]` — pure composition. 🟢

---

## Fitted-constant debt (7 found 2026-06-05, workflow `wpa6sviup`; **4 de-fit, 3 resolved-as-tuned, 0 remaining**)

| Constant | Module | Value | Status / issue |
|---|---|---|---|
| ~~`PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.00602 → **0.005745** | ✅ **DE-FIT 2026-06-05** — measured (frontend.new); fit-pin dropped — see log |
| ~~`PREFILL_HOST_SHARED_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.0030515 → **0.0031954** | ✅ **DE-FIT 2026-06-10** — measured partition adopted: shared fraction 0.5236 (pooled-OLS point estimate of the live B-sweep band [0.40,0.54], `build_host_split.py`). An initial replay-OFF worktree gate rejected it (+0.44pt); the production replay-ON re-gate PASSED and improves (H100 TTFT 18.20→18.07) — see log |
| ~~`PREFILL_HOST_PERREQ_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.0030515 → **0.0029076** | ✅ **DE-FIT 2026-06-10** — the complementary 0.4764 share; sum pinned EXACTLY at the measured 6.103e-3 — see log |
| `P_LO` | kernel_tpot | 0.88 | ⚠️ **RESOLVED-AS-TUNED 2026-06-09** — measurement built (measured onset ≈ 0.45); adoption gate-rejected → honest tuned-knob label, false "measured" claim removed — see log |
| `P_HI_SHORT` | kernel_tpot | 1.22 | ⚠️ **RESOLVED-AS-TUNED 2026-06-09** — measured ≈ 1.69; gate-rejected (swe-plateau 8.7→10.4) — see log |
| `P_HI_LONG` | kernel_tpot | 2.0 | ⚠️ **RESOLVED-AS-TUNED 2026-06-09** — measurement data-starved (2 cells, 1 profile; point est. 2.31 consistent) — see log |
| ~~`0.3` launch-floor clamp~~ | kernel_step_cost | 0.3 → **0.0** | ✅ **DE-FIT 2026-06-05** — see log below |

(⚠️ resolved-as-tuned = the spec's sanctioned fallback: the constant remains a fit, but the false
"measured" claim is gone, the real measurement exists as a regenerable artifact documenting the
disagreement, and a test pins both so neither drifts silently.)

`closed_form_tpot.py` and `ramp_tpot.py` are clean. Remediation spec:
`.omc/specs/deep-dive-whether-there-are-fitted.md`.

### De-fit log
- **2026-06-09/10 — `PREFILL_HOST_SHARED`/`PERREQ` 50/50 partition: measurement built → ADOPTED on the production re-gate (de-fit)** (`ttft_queue_sim`; the last 2 debt rows — spec `.omc/specs/deep-dive-whether-there-are-fitted.md` rows 2-3, "use the point estimate of the 40-54% band, not the gate-max").
  The SUM 6.103e-3 ms/cached-token was always measured (the c1 serving-regression cached coefficient;
  live-validated at 5.89 by the 2026-06-03 loopback probe, commit 9dce1dc; corroborated by stage-split
  frontend.cached 5.174 in `serving_stage_split_H100.csv`). Only the shared/per-request PARTITION was a
  fit: 50/50 chosen to "maximize the gate" within a live-measured ~40-54% band (pre-06-03 it was 57/43,
  imported from the offline batch CSV `cached_prefill_batch_ttft_H100.csv` via commit 7748e70). The
  missing measurement now exists, recomputed OFFLINE from the on-disk live-probe CSVs (no GPU, no
  server): `profiling/process/build_host_split.py` → `profile_data/kernels/prefill_host_split_H100.json`
  (deterministic numpy lstsq; band-membership check built in). (1) Denominator: 2-var lstsq
  `ttft = floor + new_rate·new + c1_rate·cached` over the 20 rows of `prefill_live_ttft_H100.csv` →
  c1 cached rate **5.8872e-3 ms/tok** (floor 16.03 ms, new 29.397 ms/1k) — exactly the documented
  "5.89" that reproduces the fitted 6.103. (2) B-sweep endpoint slopes (`prefill_live_split_H100.csv`,
  B∈{1,2,4,8,16} concurrent cache-hits on one primed P-token prefix):
  `shared_frac = 1 − (B-slope/P)/c1_rate` → P=8000: **0.4017**, P=16000: **0.5460** — the in-code
  "~40-54%" band is real and reproduces from raw data on disk; P=2000 excluded exactly as the band
  always did (shared_frac −0.017: a fixed ~12.5 ms/req cost misattributed as per-token at short
  prefixes; recorded in the artifact). (3) Pre-registered point estimate: pooled OLS (per-P intercepts
  + one common B·P slope) over the two band planes → perreq 2.8048e-3 → **shared_frac 0.5236**
  (in-band; sensitivities: endpoint-mean 0.474, sum-denominator 0.540). Candidate constants =
  0.5236/0.4764 × the exact sum: SHARED 0.0031954 / PERREQ 0.0029076. **Gate history — a
  measurement-fidelity lesson:** the first adoption gate (2026-06-09, in the fresh `defit-host-split`
  worktree) REJECTED the split: TTFT-cell H100 33.36→33.79 (+0.44 > 0.3). But that gate ran with
  **trajectory replay OFF** — the per-GPU realized pool files (`*_realized_<slug>.json`) are
  gitignored (~100MB, excluded from PR #72) and absent from a fresh worktree, so the TTFT cohort
  silently fell back to the pooled forward mode (baseline 33.4% instead of the production ~18.2%).
  **Re-gated 2026-06-10 with the pools restored (production replay-ON): PASSED and improves** —
  TTFT-cell H100 **18.20→18.07**, E2EL 11.33→11.20; A100 21.94→22.06 / 15.84→15.93 (within ±0.3);
  H100x2 advisory TTFT 34.71→33.06, E2EL 25.92→24.71; TPOT byte-identical everywhere. **Outcome:
  measured split ADOPTED** (shared 0.0031954 / perreq 0.0029076, sum exactly 6.103e-3);
  `test_ttft_queue_sim` pins the literals to the regenerable artifact AND the artifact's measured
  band (both-ways pin, knee-precedent). Lesson recorded: worktree gates must verify the per-GPU
  realized pools are present, or TTFT/E2EL gates measure a non-production cohort mode.
- **2026-06-09 — ramp-knee corrected-floor follow-up: Phase-0 stop-point, ADOPTED = none** (gated
  re-attempt per `profiling/docs/ramp_knee_adoption_plan.md`; full numbers in its Execution record).
  Hypothesis: the gate-rejection below was floor misattribution — at low pressure the implied weight
  absorbs a pressure-independent host excess `D` over the decode kernel grid, and the tuned late/steep
  band compensates for it. Measured `D = max(0, median(tpot_meas − kernel_step))` over low-pressure
  (pressure < 0.30), sustain-clean turns: **H100 0.0 ms** (raw −0.0382, n=334), **A100 0.1246 ms**
  (n=178), **H100x2 0.0 ms** (raw **−0.7251**, n=486) → the pre-registered stop-point (`D < 0.5 ms`
  everywhere) triggered: the hypothesis is unsupported at the floor level, knees stay tuned, nothing
  wired. The H100x2 raw median is strongly *negative* (grid over-predicts low-batch decode, Spearman
  −0.46 vs scheduled) — pointing at the known tp2 analytic-fill over-pricing, not a host floor term.
  The corrected band (`knees_corrected` in the artifacts) is ~identical to the uncorrected one and
  still fails cross-GPU onset convergence (0.45 / 0.60 / 0.85, spread 0.41 > 0.20), confirming no
  global pressure-onset constant exists; `P_HI_SHORT` convergence passes (1.69 vs 1.41, spread 0.28 ≤
  0.30; moot under the stop, recorded for the audit-item-6 ramp restructure). Production predictions
  verified **byte-identical** to the pre-run baseline; the pin test now also locks per-GPU
  `floor_excess_ms` below the 0.5 ms stop threshold. Tooling kept: `build_ramp_knees.py` v2 (D +
  `knees_corrected` + `--exclude-profile` LOCO) and `profiling/process/gate_scoped_rows.py` (scoped
  rebuild + metrics harness). 124 tests green.
- **2026-06-09 — ramp knees `P_LO`/`P_HI_SHORT`/`P_HI_LONG`: measurement built, adoption gate-rejected → resolved-as-tuned** (`kernel_tpot`).
  The missing measurement now exists: `profiling/process/build_ramp_knees.py` (offline, no GPU; same
  GT data path as `build_saturated_ceiling`) computes per-turn **implied ramp weight**
  `w = clip((tpot_meas − kernel_step)/(t_upper − kernel_step), 0, 1)` vs pressure, detects each cell's
  sustained `w=0.1`/`w=0.9` rolling-median crossings (pre-registered rule: sustain-clean `output ≥ 24`,
  conditioning `t_upper − kernel_step ≥ max(10ms, 0.5·kstep)`, ≥8 turns, both crossings interior),
  inverts the smoothstep through them, and aggregates per-cell band edges by median with a
  leave-one-profile/conc-out jackknife → `profile_data/kernels/ramp_knees_{h100,h100x2,a100}_llama31_8b.json`.
  **Measured H100 band: `P_LO ≈ 0.4456` (n=8, 3 profiles), `P_HI_SHORT ≈ 1.6866` (n=6, 2 profiles —
  every measured short cell reaches w=0.9 only at pressure ≥ 1.35), `P_HI_LONG` data-starved (2 cells,
  1 profile, point est. 2.31).** Both well-supported values **disagree** with production (0.88/1.22) —
  the production ramp turns on later and saturates earlier than measured. **Gate result: adoption
  rejected.** Full adoption: H100 TPOT cell-MAPE 14.5→21.9 (validate overall 15.41→23.30; chat
  5.6→16.7), H100x2 28.7→40.5, A100 15.4→18.3. Per-knee isolation: `P_LO` alone 14.5→28.1 (the
  catastrophic driver); `P_HI_SHORT` alone borderline overall (15.41→15.65) but swe-plateau 8.65→10.38
  and H100x2 +1.3pt → fail. (Curiosity: full adoption *improves* the saturated plateau 11.3→8.6 —
  the two measured knees compensate each other there — while destroying low-pressure cells.)
  **Interpretation:** the implied weight attributes *any* excess over `kernel_step` to KV pressure, so
  at low pressure the measurement absorbs non-ramp excess (host overhead, scheduling jitter) — and the
  tuned late-onset/steep band compensates for exactly that misattribution. The knees are therefore
  **compensating fits for the smoothstep-in-pressure ramp shape itself**; re-anchoring the literals
  cannot fix them. The physical fix is restructuring the ramp (audit item 6: per-cell
  eviction-watermark crossing) — converges with the deferred tp2 sub-linearity spec.
  **Outcome (spec fallback):** production values retained; false "measured eviction-watermark cluster"
  comments replaced with honest TUNED-KNOB labels citing the artifact;
  `test_ramp_knees_tuned_values_and_measured_band_both_pinned` pins both the tuned literals and the
  measured band (drift in either now trips a test). Revert verified byte-identical to baseline;
  124 tests green.
- **2026-06-05 — `0.3` launch-floor clamp → `max(0.0, …)`** (`kernel_step_cost.default_launch_floor_ms`).
  Evidence: the launch floor is a config-independent constant `1.37 ms` (8B/H100: `fixed_floor 6.55 − weight 5.15 − attn 0.02`), so the `0.3` lower bound never binds; and it is read only on the *analytic* decode path (H100/H100x2/A100 tp1 use measured grids and never touch it). Removing the magic literal — replaced by a physical non-negativity guard `max(0.0, ·)` — is a proven **no-op**: TPOT byte-identical (max diff `0.00e+00 ms`) on every analytic Llama config (H100x4, A100x2, …). Tests green (39/39).
- **2026-06-05 — `PREFILL_NEW_DISPATCH_RESIDUAL` 0.00602 → 0.005745, re-attributed** (`ttft_queue_sim`).
  Was a backed-out remainder (fitted `0.0310 − roofline 0.02498`) mis-labelled "GPU framework dispatch", and `test_no_fitted_constants` pinned it via `gemm+residual≈0.0310`. The c1 live-server **stage-split microbench** (built by workflow `wt91dnjkm`, run on h100 GPU 7 → `profile_data/results/serving_stage_split_H100.csv`) measured it directly: it is **HOST serving-stack per new token** (tokenize + chat-template/parse + ZMQ-IPC) = `frontend.new = 5.745 ms/1k`; the GPU forward window is roofline-clean (`prefill_span.new 22.7` ≈ util-ramped GEMM, no above-roofline excess). Value set to the measurement, **fit-pin dropped**. Re-validated: H100 tp1 TTFT 18.21→18.14, A100 21.78→21.86, H100x2 37.87→37.48 (all ±0.4pt, no regression). Serving-level breakdown (additive, c1): wall = frontend(`5.7·new + 5.2·cached`) + prefill(`22.7·new + 0.8·cached`) + queue(~0); the CACHED cost is ~86% host frontend, not GPU. Name retained for diff-minimality (it is a host term, does not shard with tp). Bench/analyzer: `profiling/gpu_profiling/vllm/serving_stage_split.py` + `analyze_serving_stage_split.py`; design `profiling/docs/serving_stage_split_plan.md`.
  - **Lane B device cross-check** (`cuda_events/lane_b_device_split.py` → `lane_b_device_H100.csv`, offline mp=0 tp1, `torch.cuda.Event` around `GPUModelRunner.execute_model`; nsys/`--worker-extension-cls` both unavailable on 0.19.0 so used in-process monkeypatch): `device.new = 23.85 ms/1k` (util ~0.68, roofline-consistent) ≈ Lane A `prefill_span.new 22.7` → the engine prefill window is **~100% GPU kernel, no in-engine dispatch residual**; `device.cached = 0.59` (paged-attn). Confirms the NEW residual is frontend host, not GPU. NOTE: the script's `dispatch_ms`/`forward_wall_ms` columns are INVALID (CUDA-async wall measures launch not execution); only `device_ms` (events) vs `prefill_span` is used.

## Recent change — PR #73 (both fit-free, tp1 byte-identical)
1. **Per-config prefill floor** — replaced the single H100-tp1 `26.0` with measured per-deployment floors
   (`build_prefill_floor.py` → `prefill_floor_llama31_8b.json`); moves the TTFT floor (step "floor" above).
2. **Robust decode `fixed_floor`** — `load_grid` now takes the **min over the B=1 row** instead of the single
   `(B=1, T=512)` warm-up-outlier cell; de-biases the decode analytic-fill (step 6).

Result: H100x2 TTFT 40.3→37.9, TPOT 33.0→29.2, E2EL 30.7→28.0; tp1 unchanged; 123 tests pass.
