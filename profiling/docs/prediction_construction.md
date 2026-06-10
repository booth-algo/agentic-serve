# How TPOT / TTFT / E2EL predictions are constructed (from scratch)

_Reference for the simulator's per-turn prediction pipeline, the constants it rests on, and their
provenance. Grounded in `simulator/{kernel_tpot,kernel_step_cost,closed_form_tpot,ttft_queue_sim,cohort_scale}.py`.
Last verified 2026-06-10._

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
| 6 | `kernel_step = decode_step_ms(b_eff, ctx)` — measured grid (bilinear) where covered 🔵; beyond the grid edge / OOM corners the analytic roofline `max(fixed_floor + b·ctx·kv_bytes/(kv_shards·bw·util_bw), 2·(n_params/tp)·b/(flops·util_flops))` | 🔵 grid (tp2 extended +54 cells 2026-06-10); 🟢 roofline with 🔵 measured-shape SUB-linear beyond-hull fill (L3, de-fit 2026-06-10); 🔵 `fixed_floor`; 🔵 per-class launch floors H100 1.37 / H100x2 1.82 / A100 2.06 ms (G5, de-fit 2026-06-10; was one config-independent 1.37 + `max(0.0,·)` guard) |
| 7 | `t_upper = max(kernel_step, saturated_ceiling_ms(median_output))` — measured plateau anchors, interpolated | 🔵 |
| 8 | `z = pressure · qbar` — distribution-integrated KV demand / pool. `qbar` = trapezoid mean of the MEASURED `context_scale_quantiles` (per-cell, `cohort_scale.cohort_scale_mean`; swe 1.1269, term 1.3463, osworld 0.9834, chat 1.0003 → onsets `1/qbar` = 0.887/0.743/1.017/1.000) | 🔵 quantile artifact; 🟢 trapezoid |
| 9 | `sustain = smoothstep(output, SAT_SUSTAIN_LO=9, SAT_SUSTAIN_HI=24)` | 🔵 |
| 10 | `weight = clamp(n_evicted · chunk_steps / out [· z if chunk_steps ≥ 2], 0, 1) · sustain` — the chunk-quantized eviction-DRAIN fraction of the turn's decode steps (round 2): `n_evicted = (1−1/z)·sched` (LIFO sticky-prefix), `chunk_steps = ceil(ctx·qbar/(M − b_eff))` (per-victim chunked re-prefill steps; `M = max_num_batched_tokens` ⚙️ 8192 H100-class / 2048 A100, the vLLM device-rule resolved default), `out` = THIS turn's own output. Onset gate: `pressure ≥ 1` (pool physically full) AND `z > 1`, with firing-gate HYSTERESIS on the cell path (round 3: armed once a turn fills the pool, held while `z > 1` at effective pressure `max(pressure, 1)`, released at `z ≤ 1` — the backlog persists through block-quantization flicker of the raw pressure). Multi-chunk victims gain ×z (rotation amplification); a cell's first overflow turn is growth-damped (`w ≤ sched·chunk_steps/ctx`, single-chunk only; development + armed state tracked by `predict_cell_tpot`) | 🟢 derived; ⚙️ engine config; 🔵 quantile artifact (**no tuned numeric constant** — 2026-06-10 restructure rounds 2–3) |
| 11 | **`TPOT = kernel_step + weight·(t_upper − kernel_step)`** | — |

- At **pressure < 1 or z ≤ 1** (pool not physically full, or the distribution-summed demand fits)
  `weight = 0` → `TPOT = kernel_step` (the raw measured/analytic decode step).
- In overflow → the non-resident `(1−1/z)` fraction of the cohort (vLLM v1 preemption is LIFO
  `running.pop()` + RECOMPUTE) re-prefills its full context in budget-sized chunks; each chunk occupies
  one engine step's prefill budget, so the drain occupies `n_evicted·chunk_steps` of the turn's `out`
  decode steps. `weight = 1` (every step's budget recompute-filled) is exactly the regime where
  `t_upper` was anchored.
- **The tuned ramp band is GONE** (2026-06-09 honest tuned-knobs `P_LO=0.88` / `P_HI_SHORT=1.22` /
  `P_HI_LONG=2.0` + the `OUT_KNEE` interpolation → ELIMINATED 2026-06-10, audit item 6): onset is the
  physical pool-full + distribution-overflow crossing; the long-output band-widening the output-binned
  knee hand-coded now emerges from the turn's own `out` in the drain-fraction denominator. In z-units
  the measured per-GPU onsets collapse to 0.964/1.188/0.963 (H100/H100x2/A100) vs 0.45/0.85/0.60 in raw
  pressure — see the De-fit log 2026-06-10 (rounds 1–3; round 3 = firing-gate hysteresis, all 9
  pre-registered binding gates PASS).
- `decode_step_ms` is the same composition for all configs; `predict_cell_tpot` calls `predict_turn_tpot`
  per turn using the cell's **median output** as the ceiling output (so the plateau doesn't jitter).

### tp=2 over-pricing — RESOLVED 2026-06-10 (campaign lane L3)
Historically tp=2's 2.29× KV pool kept `z` below the overflow point, so TPOT = the raw step-6 kernel —
which, beyond the sparse tp2 grid edge, used an analytic fill whose `b·ctx` term grew **linearly** while
the real kernel is **sub-linear** (~1.25–1.30× over-price). Fixed by measurement: a 54-cell tp2 grid
extension + a measured-shape sub-linear beyond-hull fill (hold-out worst over-price 1.24×→1.11×) →
**H100x2 TPOT cell-MAPE 28.75 → 21.53 (−7.2pt)**, tp1 within +0.15 binding. See
`defit_log_entries/L3-tp2.md` for the derivation and `.omc/specs/deep-dive-trace-whether-there-are-fitted.md`
for the original trace.

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
| └ `gemm_per_tok` | `2·(n_params/tp)/(peak_flops·util-ramp)` + `PREFILL_TP_COMM·(tp>1)` (util ramps `util_flops→PREFILL_GEMM_UTIL_SAT=1.0` over the chunk budget) | 🟢 GEMM + ⚙️ ramp endpoints; 🔴 `UTIL_SAT=1.0` (compensating fit — the measured curve, plateau 0.754, was gate-rejected 2026-06-10; see De-fit log); 🔵 `TP_COMM=0.0032789` (measured like-for-like tp1/tp2 pair, de-fit 2026-06-10 — was the backed-out 0.00585) |
| prefill FA3 | `PREFILL_FA3_MS_PER_TOKEN2 · M·(R + 0.5·M)·frac` (super-linear attn; M=tokens (re)prefilled, R=resident prefix) | 🔵 `8.31e-7` |
| host per-req | `PREFILL_HOST_PERREQ_MS_PER_TOKEN · cached · frac` (re-tokenize cached context, summed over concurrent prefills) | 🔵 partition 0.4764 × 🔵 sum 5.8872e-3 (both measured; sum de-fit 2026-06-10, R2 closed) |
| host shared | `PREFILL_HOST_SHARED_MS_PER_TOKEN · mean_cached` (amortized once/step) | 🔵 partition 0.5236 × 🔵 sum 5.8872e-3 (R2 closed — see De-fit log) |
| step cost | `step_ms = max(decode_ms + scheduler_overhead, prefill_ms)` | 🔵 |
| floor (once per req, at first token) | `floor_residual = max(0, prefill_floor − scheduler_overhead)` added to `first_token_epoch` | 🔵 (PR #73: per-config; H100=25.86, H100x2=14.01) |

→ **`TTFT[turn] = first_token_epoch − arrival_epoch`** = queue-wait + prefill service + per-request floor.
A turn no session reaches falls back to the forward static predictor (`_fallback_ttft`).

The **host term is now measured both ways**: partition 0.5236 (pooled OLS of the live B-sweep
band, adopted 2026-06-10) × sum 5.8872e-3 (the live c1 lstsq, R2 de-fit 2026-06-10 — replaced the
benchmark-fitted 6.103e-3; see the De-fit log). Remaining 🔴 in this table per audit v2
(`fitted_constants_audit_v2.md`): `PREFILL_GEMM_UTIL_SAT` — now a MEASURED-AND-DOCUMENTED
compensating fit (the real per-step curve, plateau 0.754 in `prefill_gemm_util_H100.json`, was
gate-rejected 2026-06-10; the cap offsets the S7–S10 deep-cohort queue error — De-fit log) — and
`TP_COMM` is now MEASURED (G3 de-fit 2026-06-10: like-for-like pair → 3.279 ms/1k, top of the
NCCL physics band; H100x2 TTFT improved 31.76→29.02). Bonus: the FA3 coefficient gained an
independent cross-check (sweep slopes ≈8.9e-7 vs 8.31e-7, test-pinned).
(`PREFILL_NEW_DISPATCH_RESIDUAL` was de-fit 2026-06-05 and is genuinely 🔵 — regenerates exactly.)

---

## E2EL
`E2EL[turn] = TTFT[turn] + output_tokens · TPOT[turn]` — pure composition. 🟢

---

## Fitted-constant debt — audit v1 (7 found 2026-06-05, workflow `wpa6sviup`; **4 de-fit, 3 retired (note below), 0 remaining**)

**Audit v2 (2026-06-10, post-restructure — `fitted_constants_audit_v2.md`) found NEW debt beyond
these 7:** 2 RED (`PREFILL_GEMM_UTIL_SAT=1.0` — measurement built 2026-06-10, adoption
gate-rejected → resolved-as-compensating-fit, successor = the S7–S10 re-derivation; the host
SUM 6.103e-3 benchmark-fitted — **✅ DE-FIT 2026-06-10**, replaced by the live 5.8872e-3, see the
De-fit log), 9 GRAY
(incl. `util_bw=0.93` matching no documented computation, `TP_COMM` backed-out, `SAT_SUSTAIN 9/24`
unpinned anchors), and **14 STRUCTURAL** — discrete modeling rules adjudicated by the same gate
suite they pass ("the 9-gate suite is the new fitting surface"). The v1 table below remains the
record of the original 7 only.

| Constant | Module | Value | Status / issue |
|---|---|---|---|
| ~~`PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.00602 → **0.005745** | ✅ **DE-FIT 2026-06-05** — measured (frontend.new); fit-pin dropped — see log |
| ~~`PREFILL_HOST_SHARED_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.0030515 → **0.0031954** | ✅ **DE-FIT 2026-06-10** — measured partition adopted: shared fraction 0.5236 (pooled-OLS point estimate of the live B-sweep band [0.40,0.54], `build_host_split.py`). An initial replay-OFF worktree gate rejected it (+0.44pt); the production replay-ON re-gate PASSED and improves (H100 TTFT 18.20→18.07) — see log |
| ~~`PREFILL_HOST_PERREQ_MS_PER_TOKEN`~~ | ttft_queue_sim | 0.0030515 → **0.0029076** | ✅ **DE-FIT 2026-06-10** — the complementary 0.4764 share; sum pinned EXACTLY at the measured 6.103e-3 — see log |
| ~~`P_LO`~~ / ~~`P_HI_SHORT`~~ / ~~`P_HI_LONG`~~ | kernel_tpot | 0.88 / 1.22 / 2.0 → **deleted** | ✅ **RETIRED 2026-06-10** (ramp restructure — note below). History: resolved-as-tuned 2026-06-09 (measurement disagreed + gate-rejected re-anchoring) — see log |
| ~~`0.3` launch-floor clamp~~ | kernel_step_cost | 0.3 → **0.0** | ✅ **DE-FIT 2026-06-05** — see log below |

**Retired — the three `kernel_tpot` ramp knees (2026-06-10 ramp restructure, pre-registered
binding gates 9/9 PASS):** `P_LO = 0.88`, `P_HI_SHORT = 1.22`, `P_HI_LONG = 2.0` (and the
`OUT_KNEE` output-binned interpolation between them) are **deleted from the code**, not
re-anchored. Onset is now the physical pool-full + distribution-overflow gate (`pressure ≥ 1`
AND `z = pressure·qbar > 1`, qbar from the measured `context_scale_quantiles`, with firing-gate
hysteresis on the cell path — the floor is the pool-full boundary 1.0, not a tuned number), and
the transition width is the computed chunk-quantized eviction-drain duty
`w = n_evicted·chunk_steps/out` (full saturation = the drain-fill point
`n_evicted·chunk_steps ≥ out`; the long-output widening emerges from the turn's own `out` in
the denominator) — **zero tuned numeric constants**. The measured-band artifacts
(`ramp_knees_*`) remain the valid measured history, pinned by
`test_ramp_knees_measured_band_remains_pinned_history`. See the De-fit log 2026-06-10
(rounds 1–3) and `ramp_knee_adoption_plan.md` § Restructure outcome.

(In the log below, "resolved-as-tuned" = the spec's sanctioned fallback used as the knees'
interim state on 2026-06-09: value kept, false "measured" claim removed, the real measurement
preserved as a regenerable artifact, both pinned. Superseded for the knees by the 2026-06-10
retirement.)

`closed_form_tpot.py` and `ramp_tpot.py` are clean. Remediation spec:
`.omc/specs/deep-dive-whether-there-are-fitted.md`.

### De-fit log
- **2026-06-10 — PARALLEL DE-FIT CAMPAIGN (6 lanes; see `parallel_defit_campaign.md` Execution record + `defit_log_entries/L*.md` for full per-lane evidence).** Net gate movement (replay-ON, vs the campaign base `5f06393`): **H100x2 TPOT 28.75→21.53 (−7.2), E2EL 21.83→18.55 (−3.3)**; H100 +0.15 TPOT (disclosed L3 fill-consistency trade, binding-clean); all TTFT byte-flat; everything else byte-identical. Lane outcomes:
  - **L3 (tp2 sub-linearity, ADOPTED):** 54-cell tp2 grid + measured-shape sub-linear fill + per-class launch floors (G5) + S12 resolved by re-measurement (drop vindicated, "OOM" docstring fixed).
  - **L4 (eviction cluster, REDERIVED):** S7 whole-session MRU preemption FALSIFIED against engine traces and retired (engine-faithful partial LRU-oldest trims; predictions byte-identical, counterfactual-replay-proven). S8 live hit/miss built and gate-rejected (+3.47 H100 TTFT) → freeze retained as a PINNED compensating rule. S9 kept (trace-validated). Residual localized: the S10 re-prefill-volume→TTFT amplification; successor = re-derive S10, then re-gate S8-unfreeze + the prefill util cap TOGETHER (A100 already improves under both).
  - **L2 (S13 CLOSED):** the 78 per-GPU replay pools are now COMMITTED (minified, 19.9MB) + warn-once/`RAMP_TPOT_REQUIRE_POOLS=1` hard-fail — the silent replay-OFF gate footgun is structurally dead. ramp_tpot's false "measured cluster" provenance (D1–D4) honest-relabeled.
  - **L5 (gray batch CLOSED):** SAT_SUSTAIN anchors regenerable + the population finding (turn-median p5 = 24.0; the 9.0 anchor exists only per-request — band retune deferred honestly); RESERVE_BYTES rule stated + reproduction test; sglang budgets fixed to the real engine tier rule (cited; RTX3090-sglang honestly regresses — the wrong budget had masked a tiny pool); ceiling-cut sensitivity embedded.
  - **L6 (G4 resolved-as-compensating-fit):** pre-registered recipe re-derives H100 `util_bw = 0.8111` ≠ 0.93; candidate gate-rejected (H100 TPOT +0.67) → 0.93 kept, exposed as compensating for a missing decode host-overhead term (net-of-sched util 1.25 > 1, unphysical) — pairs with the S10 successor. A100 measurement deferred (host busy) with a turnkey runbook + preflight.
  - **L7 (dead paths CLOSED):** classifier/hint modules retired to `_legacy` with reachability proofs; D5–D9 + audit-doc S7 misfile fixed; byte-identical.
- **2026-06-10 — `PREFILL_TP_COMM` 5.85 → 3.279 ms/1k (like-for-like measurement): audit-v2 G3 CLOSED** (`ttft_queue_sim`; `ttft_pricing_defit_plan.md` Item 3).
  The retired 5.85 was a backed-out remainder (tp2 ttft.new 18.5 − GEMM/2 12.65) from an
  instrumentation-INCONSISTENT pair (tp2 multiprocess api_server vs tp1 in-process LLM). Measured
  like-for-like: `serving_stage_split.py` gained `--tensor-parallel-size` and the tp2 leg ran on h100
  GPUs 6+7 (same script/stack as the 2026-06-05 tp1 leg) → `prefill_span.new`: tp1 **22.733**, tp2
  **14.645 ms/1k** → `comm = 14.645 − 22.733/2 = 3.279 ms/1k` (`build_tp_comm.py` →
  `prefill_tp_comm_H100.json`, deterministic, pins the literal). Lands at the top of the NCCL
  all-reduce physics band (~1–3 ms/1k) — confirming the audit's hypothesis that the old remainder
  absorbed ~2.5 ms/1k of host IPC under a comm label. **Gate: PASS** — H100/A100 **byte-identical**
  (the term is ×(tp−1)), H100x2 TTFT-cell 31.76→**29.02** (−2.74), E2EL 24.01→**21.83** (−2.17):
  the honest physics value substantially improves the config it prices. 187 tests + 12 subtests green.
- **2026-06-10 — prefill-GEMM util cap: measurement built, adoption gate-rejected → resolved-as-compensating-fit** (`ttft_queue_sim`; audit-v2 R1/S6; `ttft_pricing_defit_plan.md` Item 2).
  Phase A debunked the cap's anchor: the "15.5 ms/1k GT cohort ≈ util-1 roofline" claim traces to ONE
  cell (osworld c5 turn-0) whose token denominator double-counts the 1024-token shared APC prefix
  (implied util 1.05 > 1 — impossible); dedup-corrected GT gives a ~0.62 gross floor, and the offline
  microbench re-derives to 0.754–0.773 in the sim's FLOPs convention (its 0.655–0.672 was the
  executed-GEMM convention). Phase B measured the real per-step curve in-engine
  (`prefill_util_sweep.py`, h100 GPU 6 per `h100_setup.md`; CUDA events on `execute_model` with
  per-step token counts read from SchedulerOutput; budgets 512–8192, 282 full-step samples):
  per-step device time OLS-decomposed against the sim's own FA3 regressor `M·(R+0.5·M)` —
  **zero-prefix GEMM intercepts → util_sim 0.640 (512) → 0.708 (1310) → 0.744 (2048) → 0.752 (4096)
  → 0.754 (8192)**, R² 0.987–0.998 (`build_prefill_gemm_util` →
  `prefill_gemm_util_H100.json`). The curve CONFIRMS `util_flops≈0.65` at small m, saturates at
  **0.754 by m≈2048** (not 1.0 by 8192), and the regression slopes independently re-measure
  **FA3 ≈ 8.9e-7** vs the production 8.31e-7 (~7% — cross-check pinned in tests). Phase C wired the
  measured lookup and **the gate REJECTED it**: H100 TTFT-cell 18.13→21.28 (+3.15), H100x2 advisory
  31.76→34.88 — while **A100 improved** (TTFT −0.38, E2EL −1.79) and TPOT stayed byte-identical.
  Interpretation: the `util→1.0` ramp under-prices saturated H100 steps to compensate a structural
  error in the deep-cohort queue interaction (the audit-v2 S7–S10 cluster); A100 (budget 2048, no
  deep saturated steps) has no such compensation to lose, so the honest curve helps there.
  Per-config adoption would be cherry-picking → **ramp+cap retained with the compensating-fit label**;
  the measurement, builder, sweep tooling, and both-ways test pins stay as the permanent record.
  Revert verified byte-identical to HEAD. Successor filed: re-derive the saturated-step/queue
  interaction (S7–S10), then re-gate the measured curve.
- **2026-06-10 — host cached SUM 6.103e-3 → 5.8872e-3 (LIVE measurement): audit-v2 R2 CLOSED** (`ttft_queue_sim`; `ttft_pricing_defit_plan.md` Item 1).
  The sum both host constants scale was the `760d9bd` c1 **benchmark-regression** coefficient
  (`prefill ≈ 22.5 + 0.0310·new + 0.006103·cached`, lstsq over the benchmark's own c1 cells — i.e.
  fitted to cells inside the scored validation payload; audit-v2 RED). Replaced by the **live
  regenerable measurement**: `build_host_split`'s c1 lstsq over `prefill_live_ttft_H100.csv` →
  **5.8872e-3 ms/tok** (floor 16.03, new 29.397/1k, n=20). Partition unchanged (measured 0.5236) →
  SHARED 3.0824e-3 / PERREQ 2.8048e-3, sum exactly the live value; artifact `constants` block is now
  the adopted source (benchmark sum retired to `benchmark_sum_reference`), pinned by test.
  **Gap decomposition (pre-registered, run before gating):** the benchmark fit exceeds the probe on
  ALL THREE parameters (floor 22.5 vs 16.03 ms, new 31.0 vs 29.4, cached 6.103 vs 5.887 ms/1k) —
  consistent with real chat-templated benchmark prompts exercising heavier tokenize/template host
  paths than the probe's synthetic single-block text; reconciliation with the B-sweep's fixed
  ~12.5 ms/req and the measured 25.86 ms prefill floor is coherent (16.03 + 12.46 ≈ 28.5 band).
  Documented refinement: re-run the live probe replaying ACTUAL benchmark prompts.
  **Gate (replay-ON, vs f165a88 baseline): PASS** — H100 TTFT 18.07→18.13 (+0.06), E2EL
  10.86→10.76 (−0.10); A100 TTFT 22.06→22.22 (+0.16), E2EL 15.77→15.86 (+0.10); TPOT
  **byte-identical** both GPUs (wiring-clean check); H100x2 advisory improves TTFT 33.06→**31.76**,
  E2EL 25.04→**24.01**. 187 tests + 12 subtests green.
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
- **2026-06-10 — ramp restructure ROUND 3 (final): firing-gate HYSTERESIS — all 9 pre-registered
  binding gates PASS; the tuned ramp band is eliminated for good.** The round-2 single gate fail
  (H100 swe-plateau 8.65→9.67) decomposed into three mechanisms (evaluator replication, bit-exact);
  the dominant one (85 % of the overshoot) was **knife-edge gate flicker, not gate refusal**: at
  H100 swe@40 t20–29 the block-quantized pressure (`ceil(ctx/16)` makes it jumpy) oscillates
  0.96–1.05 around the P0b gate while z stays 1.08–1.18 > 1 and the measured ITL develops
  monotonically 28→219 ms — the prediction oscillated 27↔132 ms across adjacent turns. Fix (zero
  new constants — the floor is the pool-full boundary itself): `predict_cell_tpot` tracks an
  **armed** state — arm when a turn physically fills the pool (`pressure ≥ 1` AND `z > 1`), hold
  while the demand overflow persists (`z > 1`), price armed turns at the effective pressure
  `max(pressure, 1.0)`, release at `z ≤ 1`. This is the SAME physical argument the development
  clock already codified ("the backlog persists through pressure-gate flicker"), applied to the
  firing gate. Standalone `predict_turn_tpot` keeps per-turn gating (`armed=False`); cells that
  never reach pressure 1 never arm — the protected twin **A100 term@20 (pressure peak 0.965,
  measured CLEAN) is preserved bit-exactly**. **Gate run (real `gate_scoped_rows` rebuild):
  9/9 PASS** — H100 swe-plateau 9.674→**8.511** (beats baseline 8.652), tpot_cell 14.686→**14.556**,
  e2el 21.387→**21.304**, chat 5.469 (all vs baseline 14.536/21.270/5.559); ttft +0.005; A100 and
  H100x2 predictions **bit-identical to round 2** (no A100/H100x2 cell arms-then-dips). The only
  changed predictions on the whole grid are the four swe@40 dip turns t24/25/26/28 (27 ms floor →
  85–104 ms armed; meas 142–178). Rejected variants (measured, not shipped): latching `developed`
  while z > 1 fails H100 tpot_cell (+0.03 over gate) by amplifying first-fire overshoot;
  t0-developed init fixes A100 swe t0 but un-fixes the A100 term t0 wins (tpot_cell 14.38→15.25).
  **Remaining honest residuals (documented, next-physics):** (a) first-fired-turn overshoot — the
  fresh-crossing damping triggers on "previous turn z ≤ 1" but z crosses 1 one-to-two turns before
  pressure does, so it is dead code at the first ARMED turn (swe@120 t8 pred 243 vs meas 142); its
  magnitude would need recalibration (measured first-fire duty 0.50–0.65 vs the clip's 0.03–0.05) —
  a re-derivation, not a knob, left out of scope; (b) shallow-z duty undershoot when armed
  (w 0.35–0.49 vs implied 0.87–0.89 at swe@40 t27/29) and sub-pool-full saturation the gate cannot
  reach (H100 term@80 t14–17, pressure 0.81–0.88) — both point at runtime effective pool < traced
  `available_kv_blocks` and/or prefix-cache thrash (the documented out-of-core extension);
  (c) H100x2 osworld plateau advisory regression (33.8→49.2, ×z rotation on tp2 marginal
  overflows — known tp2 pool/fill caveat, non-binding). 141 tests green (hysteresis hold/disarm/
  never-arm pinned in `test_kernel_tpot`).
  Round 1's once-per-turn linear recompute-mass duty was falsified by the per-turn implied-duty
  extraction (measured duty 0.6–0.9 across z ∈ [1.1, 2.0] at pressure ≥ 1, 2–3× the linear ramp) and
  its z-only onset over-fired pools that were not physically full. Round-2 law
  (`kernel_tpot._overflow_weight`, all inputs measured/engine/derived, zero tuned numeric constants):
  (a) **onset gate** `pressure ≥ 1 AND z > 1` (P0b: vLLM v1 preempts only on allocation failure; the
  measured spread `qbar` sizes the mass, not the onset); (b) **drain fraction**
  `w = n_evicted·chunk_steps/out` with `n_evicted = (1−1/z)·sched` and
  `chunk_steps = ceil(ctx·qbar/(M − b_eff))` — each victim's chunked re-prefill occupies whole engine
  steps (the quantization IS the measured 2–4× steepening at small ctx/budget, e.g. H100 chat ×3.3);
  (c) **rotation amplification ×z** for multi-chunk victims (standing overflow re-rotates the LIFO
  victim queue; single-chunk victims de-synchronize — measured: A100 chat single-chunk drains sit on
  the once-per-turn drain exactly); (d) **own-output amortization** (P1a: the cell-median de-swing
  prices the ceiling only); (e) **fresh-crossing growth damping** — on a cell's first overflow turn
  (tracked by `predict_cell_tpot`: previous turn overflowed AND its output ≥ the measured sustain
  band's Hermite midpoint, (9+24)/2) single-chunk boundary waves land in the admission burst (TTFT
  side) and only decode-growth evictions are ITL-visible (`w ≤ sched·chunk_steps/ctx`).
  **Gates (vs the same reproduced baseline):** PASS H100 tpot_cell 14.54→14.69, ttft +0.005,
  e2el 21.27→21.39, chat 5.56→**5.47**; A100 tpot_cell 15.37→**14.38**, ttft +0.03,
  e2el 29.08→29.32, chat 19.99→**17.25**. FAIL **H100 swe-plateau 8.65→9.67 (+0.72,
  gate +0.3)** — 100 % the swe@40 cell (plateau turns at raw pressure 0.97–1.0, z ≈ 1.1, measured
  saturated): the P0b gate refuses to fire a pool that is not physically full, and the
  observationally-TWIN cell A100 term@20 (pressure 0.87–0.96, z up to 1.30, measured CLEAN — the
  round-1 mode-2 driver, now fixed: t18/19 preds 95.7/146.7→21.6/22.2 vs meas 28.2/29.9) REQUIRES the
  gate; aggregate inputs cannot split the pair (per-cell realized qbar: swe@40 1.1023 vs term@20
  1.2823 — no separation; sched≈capacity in both). Candidate physical fix recorded: per-(conc,cell)
  realized quantile blocks cannot resolve it; co-residency/arrival data could.
  **Round-1 failure modes closed:** H100 term@320 t0 94.2→22.4 (meas 16.1) and t2 186.6→63.4
  (meas 66.3); A100 term@120 t0 175.4→42.8 (meas 35.2); H100 chat@320 t7-9 32/35/38→41/54/58
  (meas 44/51/61); H100 osworld@200 drain t13 67.4→119.4 (meas 108); A100 chat@200 drain t15-17
  57-61→125.8 (meas 108-114); A100 chat plateau 31.85→12.22. Advisory H100x2: tpot_cell 28.73→28.60,
  swe-plateau 6.73→4.57, term-plateau 29.91→26.02, but osworld-plateau 33.83→49.16 (the ×z rotation
  over-fires tp2's huge-pool marginal overflows — known tp2 pool/fill caveat, non-binding). Honest
  residuals: osworld@200/@256 drains still −20/+15 % band (duty under the measured 0.65 at z≈1.4);
  H100 term mid-conc cells +2–7 vs baseline (development dynamics deeper than the depth-1 state).
  138 tests green; narrow-band w=0.9 crossings now found by bisection on the round-2 law: swe c120
  3.6 %, swe c160 2.4 %, term c200 5.6 % (clamps at the pressure gate vs measured 0.9468).
- **2026-06-10 — ramp RESTRUCTURE: distribution-overflow recompute-duty weight (audit item 6) —
  knees ELIMINATED; pre-registered binding gates 7/10 PASS, 3 FAIL (adoption decision OPEN, round 1/3).**
  The tuned ramp band (`P_LO=0.88`, `P_HI_SHORT=1.22`, `P_HI_LONG=2.0`, `OUT_KNEE` interpolation) was
  replaced by the fully computed eviction-recompute duty cycle (`kernel_tpot._overflow_weight`):
  `z = pressure·qbar` (qbar = trapezoid mean of the measured `context_scale_quantiles`;
  `simulator/cohort_scale.py`; swe 1.1269 / term 1.3463 / osworld 0.9834 / chat 1.0003), onset at the
  pool-overflow crossing `z = 1`, transition `w = clamp((z−1)·pool_tokens/(ceil_out·(M − b_eff)), 0, 1)`
  with `M = max_num_batched_tokens` (vLLM device-rule engine config: 8192 H100-class / 2048 A100, new
  optional deployment-JSON key + `RooflineParams` field). Kept untouched: decode grid, measured ceiling
  anchors, sustain gate 9/24. New inputs: `KernelTurnInput.cohort_scale_mean` (defaulted 1.0 — all
  existing constructors unchanged; `build_simulator_rows` sets it per cell). **Physics validation:** the
  per-GPU measured onset medians collapse in z-units to 0.964/1.188/0.963 (H100/H100x2/A100; spread
  0.225) vs 0.4456/0.8540/0.6048 in raw pressure (spread 0.408) — pinned in
  `test_measured_onsets_collapse_in_z_units`; narrow-band H100x2 w=0.9 crossings predicted within 10%
  with zero constants (pinned, bench-gated). **Gate run (A/B vs pristine-HEAD on identical data —
  the doc's TPOT baselines reproduce exactly; the recorded TTFT baselines do NOT reproduce in this
  environment (env-dependent per-GPU realized-dist files), so the reproduced baseline is authoritative):**
  PASS: H100 ttft +0.005 / e2el +0.17 / swe-plateau 8.65→8.08; A100 tpot_cell 15.37→15.09 /
  ttft ±0.00 / chat 19.99→**14.92** / swe-plateau +0.11. FAIL: **H100 tpot_cell 14.54→15.85 (+1.31)**,
  **H100 chat 5.559→5.861 (+0.3025, over by 0.0025)**, **A100 e2el_cell 29.08→30.33 (+1.25)**.
  Attribution (all three pre-documented in the design's risk list): (a) terminalbench low-conc/early
  turns over-fire — the pooled qbar (1.3463) overstates the low-conc cohort spread (A100 term c20
  +25.5pt; H100 term c320 t0 pred 94 vs meas 16 — a turn-0 cohort has no prior resident KV to
  recompute), needs per-conc `by_concurrency` quantile blocks (resolver already supports them) and/or
  a cached-prior-KV bound on the recompute mass; (b) osworld deep-saturation UNDER-prediction — the
  computed band (full at z≈2.6 for out~86) is wider than measured (p_high ≈1.8–2.0), H100 osworld
  plateau 11.85→28.14 — the requeue-stall escalation the static duty cycle omits; (c) chat c320
  mid-cohort turns under-predict with onset moved 0.88→1.0. Improvements where the physics is right:
  A100 chat −5.06pt (the computed late onset fixes the tuned 0.88 over-fire), H100x2 plateau
  terminalbench 29.91→7.01, H100 plateau terminalbench 15.86→12.17. **Per the adoption-plan protocol a
  binding-gate fail means production keeps tuned values; this branch keeps the restructure for rounds
  2/3 with the failure modes documented above** (candidate fixes are all measured/derived: per-conc
  quantile regeneration, recompute mass bounded by previously-resident cached KV, conc-dependent
  requeue escalation). Artifacts unchanged (`ramp_knees_*` remain the measured history; pins moved to
  `test_ramp_knees_measured_band_remains_pinned_history`). 134 tests green.
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
