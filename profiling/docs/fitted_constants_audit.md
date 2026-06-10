# Fitted-constant audit — tp1 headline TTFT / TPOT / E2EL flow

> **2026-06-10 — SUPERSEDED for open items by `fitted_constants_audit_v2.md`.** The v2 final
> report re-verified this audit's closures against on-disk artifacts (6 of 7 confirmed; the host
> cached-token sum 6.103e-3 reopened as v2 R2) and re-classified the remaining debt (2 RED, 9 GRAY,
> 14 STRUCTURAL, 9 dead-path). Treat anything still listed as open/`de-fit needed` below as
> historical — v2 is the active worklist. Values quoted in the body reflect the 2026-06-02 tree
> and some have since moved (e.g. `SAT_SUSTAIN_LO` 10.0 → 9.0, `OUT_KNEE_HI` 80 → 86 demoted to a
> ceiling-cluster label, the tuned knees `P_LO`/`P_HI_*` deleted in `aea241e`). Also note
> `kernel_tpot_hint.py` and `session_regime_classifier.py` (named out-of-scope below) were retired
> to `simulator/_legacy/` on 2026-06-10 (v2 items D5–D9).

**Date:** 2026-06-02. **Scope:** the constants actually wired into the dashboard headline
(`build_simulator_rows.py` → `predict_cell_tpot` for TPOT, `predict_cell_ttft_qsim` for TTFT,
`e2el = ttft + output·tpot` for E2EL). Comparison-only modules (`kernel_tpot_hint.py`,
`ramp_tpot.py::predict_cell_tpot_ramp`, `session_regime_classifier.py`) and retired code
(`fit_llm_d_config`) are **out of scope** — they don't feed headline numbers.

**Purpose:** a de-fitting playbook. Project rule is *no fitted constants without physical basis*.
This catalogs every constant by provenance and — for the fitted/tuned ones — what it does and how to
replace it with measured/derived physics, one at a time.

---

## Verdict

> **UPDATE 2026-06-02 — the TPOT ceiling (T_upper) is now DE-FITTED.** `SATURATED_BASE_MS` +
> `SATURATED_TURN_OVERHEAD_MS` (the `118.7 + 3263/output` least-squares fit) were removed and replaced
> by **measured plateau anchors** — median benchmark ITL at pressure ≥ 2.5, one per output cluster:
> `(28 tok → 243.1 ms)`, `(86 tok → 134.9 ms)` — linearly interpolated (fit-free, the kernel-grid
> pattern). Artifact `profile_data/kernels/saturated_ceiling_H100_llama31_8b.json`, generator
> `profiling/process/build_saturated_ceiling.py`. Gate held: tp1 TPOT **15.91 → 15.89%**, swe-plateau
> **8.83 → 8.64%**, 148 tests pass — proof the regression added nothing. **The lone remaining regression
> is now the TTFT prefill law** (#2 below).

The stack is **mostly** fit-free, but it is **not** "one fit (T_upper) + everything else physical."
It carries **two independent least-squares regressions** (on two different metrics, against two
different targets) plus **four un-anchored hand-tuned knees**:

1. **TPOT ceiling (T_upper)** — `SATURATED_BASE_MS` + `SATURATED_TURN_OVERHEAD_MS` (the known one).
2. **TTFT prefill law** — `22.5 + 0.0310·new + 0.006103·cached`, a *separate* 3-coefficient fit to the
   c1 benchmark cells (commit `760d9bd`), fanned out into four named constants. **This sits on the TTFT
   headline AND the E2EL composition**, and its cached rate (6.1 ms/1k) is **~4× the measured host
   tokenize rate (~1.5 ms/1k)** — i.e. the fit is absorbing ~4.6 ms/1k of unexplained physics.
3. **Four tuned TPOT knees** — `P_HI_SHORT`, `OUT_KNEE_LO`, `OUT_KNEE_HI`, `SAT_SUSTAIN_LO`
   ("Knees retuned vs the data", their own comment).

E2EL inherits BOTH regressions because `e2el = ttft + output·tpot`.

---

## The two governing formulas (for reference)

**TPOT** (`kernel_tpot.py::predict_turn_tpot`), per turn:
```
ctx         = cached + new + 0.5·output
psb         = ceil(ctx / cache_block_size)             # per-session KV blocks
b_eff       = min(scheduled, available_kv_blocks / psb) # KV-throttled running batch
pressure    = scheduled · psb / available_kv_blocks     # KV oversubscription
kernel_step = decode_step_ms(b_eff, ctx)                # MEASURED decode grid (+ TP-sharded analytic fill)
T_upper     = max(kernel_step, min(T_UPPER_MAX_MS, SATURATED_BASE_MS + SATURATED_TURN_OVERHEAD_MS/output))
p_hi        = P_HI_SHORT + smoothstep(output; OUT_KNEE_LO, OUT_KNEE_HI)·(P_HI_LONG − P_HI_SHORT)
sustain     = smoothstep(output; SAT_SUSTAIN_LO, SAT_SUSTAIN_HI)
weight      = smoothstep(pressure; P_LO, p_hi) · sustain
ITL         = kernel_step + weight·(T_upper − kernel_step)
```

**TTFT prefill law** (`ttft_queue_sim.py::_price_step`, charged per chunked-prefill step inside the
queue sim; TTFT = the accumulated prefill cost until a request's prefill finishes):
```
prefill_step_ms = PREFILL_FLOOR_MS                                        # per-step floor
                + PREFILL_NEW_MS_PER_TOKEN        · (new tokens this step) # GPU prefill GEMM/serving
                + PREFILL_FA3_MS_PER_TOKEN2       · (M·(R+0.5·M))          # super-linear FA3 attention
                + PREFILL_HOST_SHARED_MS_PER_TOKEN · (cached, once/step)   # host re-tokenize (shared)
                + PREFILL_HOST_PERREQ_MS_PER_TOKEN · (cached, per request) # host re-tokenize (per-req)
```

**E2EL** = `ttft_pred + output · tpot_pred` — pure composition, no own constants.

---

## FITTED — least-squares regressions (the bad category)

### `SATURATED_BASE_MS = 118.7` — [kernel_tpot.py:80](../../simulator/kernel_tpot.py#L80) — TPOT, E2EL
- **Does:** the output-*independent* part of the saturated ITL ceiling `T_upper`. It's the per-token
  latency the engine plateaus toward when the KV pool is full and evicting — the height long-output
  workloads (osworld, where `OVERHEAD/output` → small) saturate at.
- **Provenance:** intercept of the least-squares fit of measured ITL at `pressure ≥ 2.5` vs `1/output`
  (120 cells, R²=0.64). The "per-token saturated-decode floor" reading is a post-hoc interpretation of a
  fitted intercept.
- **De-fit:** replace with `decode_step_ms(b_eff_saturated, ctx)` from the measured decode grid at the
  saturated batch/ctx (the roofline bandwidth term `(weights + B·ctx·kv_bpt + B·kv_bpt)/(peak_bw·util_bw)`).
  The grid + `util_bw` are already measured and already in the flow. **Effort: low. No GPU.**

### `SATURATED_TURN_OVERHEAD_MS = 3263.0` — [kernel_tpot.py:81](../../simulator/kernel_tpot.py#L81) — TPOT, E2EL
- **Does:** the output-*amortized* part of the ceiling (`OVERHEAD/output`). It's why short-output
  workloads saturate higher: at out≈28 it adds 3263/28 ≈ 116 ms on top of BASE → ceiling ≈ 235 ms; at
  out≈87 it adds only ≈ 37 ms. Physically the per-turn cohort-prefill + scheduling wall amortized over a
  session's output tokens.
- **Provenance:** slope of the same R²=0.64 regression. The canonical fitted constant.
- **De-fit:** model the per-turn wall it names: `cohort_prefill_ms = _prefill_ms(c·P_fresh) +
  scheduler_overhead·steps`, then `÷ output`. Makes OVERHEAD a *computed* quantity. **Effort: medium
  (must validate it reproduces the 1/output slope). No GPU.** This is the harder half of T_upper.

### `PREFILL_FLOOR_MS = 22.5` — [ttft_queue_sim.py:116](../../simulator/ttft_queue_sim.py#L116) — TTFT, E2EL
- **Does:** the fixed per-(re)prefill overhead floor — the irreducible TTFT any prefill pays regardless
  of token count (launch + scheduler dispatch + first-token emit).
- **Provenance:** intercept of the c1 prefill regression `22.5 + 0.0310·new + 0.006103·cached`
  (commit `760d9bd`, "fit to the c1 cells"). Not an independently measured idle floor.
- **De-fit:** profile one `new=1, cached=0` prefill on H100 → use the measured irreducible floor as a
  MEASURED-ANCHOR. **Effort: low (one tiny microbench). GPU: yes.**

### `PREFILL_NEW_MS_PER_TOKEN = 0.0310` — [ttft_queue_sim.py:117](../../simulator/ttft_queue_sim.py#L117) — TTFT, E2EL
- **Does:** per-newly-prefilled-token GPU rate — the dominant TTFT term for cache-miss / long-new-prompt
  turns. Multiplies the NEW (fresh) tokens prefilled in a step.
- **Provenance:** the fitted slope of the c1 regression. Comment line 102: *"fit to the benchmark c1
  cells."* The kernel-composition route (GEMM ≈ 0.0042 ms/tok) was **tried and rejected** (chat 25→47).
- **De-fit:** a controlled microbench — sweep N concurrent bounded (≤1310-tok) chunked-prefill passes on
  H100 to isolate launch-overhead-per-token, so the rate becomes `FLOOR + measured_launch +
  measured_GEMM`. **Effort: high (needs a profiling run). GPU: yes.** Until then, 0.031 stays a fit.

### `PREFILL_HOST_SHARED_MS_PER_TOKEN = 0.003485` — [ttft_queue_sim.py:119](../../simulator/ttft_queue_sim.py#L119) — TTFT, E2EL
- **Does:** host-side cost of re-processing (re-tokenize + block-hash) the re-sent **cached** context,
  the portion amortized **once per step** across all concurrent prefills (the dominant cost on
  cache-HIT turns). = `0.571 × 6.103e-3`.
- **Provenance:** the `0.571/0.429` shared/per-req split is *measured*; the magnitude `6.103e-3 ms/1k`
  it splits is the **cached-term coefficient of the c1 regression** (fitted).
- **De-fit:** instrument host tokenize + block-hash per cached token directly, then apply the measured
  split. **Effort: medium (host instrumentation). GPU: no (host-side).** ⚠️ See the 4× note below.

### `PREFILL_HOST_PERREQ_MS_PER_TOKEN = 0.002618` — [ttft_queue_sim.py:120](../../simulator/ttft_queue_sim.py#L120) — TTFT, E2EL
- **Does:** the same host re-tokenize cost, the portion charged **per request** (summed over concurrent
  prefills). = `0.429 × 6.103e-3`. Same fitted parent as the shared term.
- **De-fit:** same host instrumentation as above; this is the per-request half of the measured split.

> ⚠️ **The 4× host discrepancy (the most important hidden thing).** The fitted cached rate
> `6.103e-3 ms/1k` is ~4× the measured host-tokenize rate (~1.5 ms/1k). So ~4.6 ms/1k is *unexplained
> physics absorbed by the fit*. De-fitting the HOST terms isn't cosmetic — it will surface whatever
> real cost (block-hash? KV-block allocation? scheduler bookkeeping over re-sent context?) is currently
> hiding inside this coefficient.

---

## TUNED-KNOB — hand-chosen, no measured anchor (TPOT only)

Comment ([kernel_tpot.py:56](../../simulator/kernel_tpot.py#L56)) is explicit: *"Knees retuned vs the data."*

### `P_HI_SHORT = 1.6` — [kernel_tpot.py:59](../../simulator/kernel_tpot.py#L59) — TPOT, E2EL
- **Does:** the upper pressure knee for **short-output** turns — pressure at which the amplifier reaches
  full ceiling (weight=1). Controls how *steep* the saturation step is for swe/terminal-like workloads
  (steep: ceiling by pressure 1.6).
- **De-fit:** anchor to the measured **eviction-watermark crossing** — `ramp_tpot.py` already reads the
  real-jump pressure cluster (`defcap = pressure−1 ∈ [−0.12, +0.22]`, i.e. ~0.88–1.22) off measured
  cells. Replace the swept knee with that per-cell scheduler crossing. **Effort: medium-high
  (restructures the amplifier ramp). No GPU.**

### `OUT_KNEE_LO = 40.0` / `OUT_KNEE_HI = 80.0` — [kernel_tpot.py:61-62](../../simulator/kernel_tpot.py#L61-L62) — TPOT, E2EL
- **Does:** the output-length window over which `p_hi` interpolates `P_HI_SHORT → P_HI_LONG`. Below 40
  tok output a turn is treated as "short" (steep step); above 80 as "long" (gentle ramp). It's the
  *classifier* that picks the ramp steepness by workload output.
- **De-fit:** the whole output-binning exists only because saturation onset is binned by median output
  instead of computed per turn. Replace with the per-cell eviction watermark (above) + the
  output-amortized overhead (already in `saturated_ceiling_ms`) — the [40,80] window then disappears
  entirely. **Effort: medium-high. No GPU.**

### `SAT_SUSTAIN_LO = 10.0` — [kernel_tpot.py:73](../../simulator/kernel_tpot.py#L73) — TPOT, E2EL
- **Does:** lower edge of the output-sustain gate (`weight ×= smoothstep(output; 10, 24)`). A turn
  producing < ~10 output tokens finishes before the eviction queue builds, so it can't reach the
  ceiling regardless of instantaneous pressure (kills spurious high-pressure early-turn saturation).
- **De-fit:** derive the minimum sustained-saturation output from decode physics — tokens for the
  cohort to co-reside until KV-pool exhaustion = (eviction-queue build time) / `kernel_step` at that
  batch/ctx, measured from the H100 trace where tpot first steps up. **Effort: medium. No GPU.** (Its
  partner `SAT_SUSTAIN_HI=24` is already anchored — see below.)

---

## Gray — tuned-but-anchored (defensible; re-ground if convenient)

### `P_LO = 0.8` — [kernel_tpot.py:57](../../simulator/kernel_tpot.py#L57) — TPOT, E2EL
- **Does:** lower pressure knee — below 0.8 the amplifier is off (`ITL = kernel_step`). The saturation
  onset pressure.
- **Anchor:** read off the measured amp-vs-pressure curve (amp ≈ 1 below ~0.7–0.8). Single read-off, not
  regressed. Sharper route: the measured pressure where `engine_max_decode_batch` first drops below
  `scheduled` (~0.85–0.95).

### `P_HI_LONG = 2.5` — [kernel_tpot.py:60](../../simulator/kernel_tpot.py#L60) — TPOT, E2EL
- **Does:** upper pressure knee for **long-output** turns (gentle ramp to ceiling by pressure 2.5).
- **Anchor:** coincides with the measured "saturates by pressure ~2.5" curve read, but as *used* it's one
  end of a tuned interpolation. Re-ground to the eviction watermark like `P_HI_SHORT`.

### `SAT_SUSTAIN_HI = 24.0` — [kernel_tpot.py:74](../../simulator/kernel_tpot.py#L74) — TPOT, E2EL
- **Does:** upper edge of the sustain gate — by 24 output tokens full saturation is reachable.
- **Anchor:** the minimum output observed on any saturated plateau turn (`tpot_meas > 150`) is 22 tok;
  24 = "just above" (the +2 is a structural offset). MEASURED-ANCHOR, fine as-is.

### `T_UPPER_MAX_MS = 260.0` — [kernel_tpot.py:84](../../simulator/kernel_tpot.py#L84) — TPOT, E2EL
- **Does:** hard cap on `T_upper` so tiny-output turns (where `OVERHEAD/output` explodes) don't blow up.
- **Anchor:** ~p90 of measured saturated ITL. A guard/clamp, not a driver. MEASURED-ANCHOR.

---

## Clean — measured / spec / config / structural (no action)

- **MEASURED-ANCHOR:** the decode + FA3 kernel grids and `fixed_floor` (`kernel_step_cost.py`, interp
  only); `PREFILL_FA3_MS_PER_TOKEN2 = 8.31e-7` (FA3(8192)/(8192²/2), kernel-derived); `util_flops=0.65`
  / `util_bw=0.93` (single anchors); `scheduler_overhead_ms_per_step=5.7` (99 quiet steps);
  `available_kv_blocks=27250` (max free across 4 traces); the survival / `context_scale_quantiles` /
  `turn_count` **measured workload distributions** + per-turn medians + `scheduled_requests`.
- **SPEC-DERIVED:** `n_params=8.03e9`, `peak_flops=989e12`, `peak_bw=3.35e12`,
  `kv_bytes_per_token=131072` (= 2·32·8·128·2), `bytes_per_param=2.0`, `kv_heads=8`.
- **VLLM-CONFIG:** `MAX_NUM_SEQS=1024`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `LONG_PREFILL_TOKEN_THRESHOLD=1310` (=32768·0.04), `cache_block_size=16`, `tensor_parallel`.
- **MODEL-CHOICE (relabeled 2026-06-10, audit-v2 S7 — was misfiled under VLLM-CONFIG above):**
  `preempt_policy='tail'`. vLLM's tail/LIFO preemption is a semantics for RUNNING requests; the
  sim applies this policy to IDLE sessions' cached blocks, which the real engine evicts as
  free-queue prefix blocks, LRU-OLDEST block-by-block (`vllm/.../block_pool.py`) — so engine
  semantics do not transfer and 'tail' is a simulator modeling choice, not engine config. The
  engine-faithful combination would be the sim's non-default `'lru'`+partial.
- **STRUCTURAL:** 0.5 decode midpoint, Hermite smoothstep `3u²−2u³`, FA3 triangular-FLOP 0.5,
  `(k+0.5)/C` quantile draw, the e2el composition, `max(1,·)` guards.
- **Verified vestigial / retired (not in headline):** `_prefill_pass_ms`, `_GRID_U_MAX=1024`
  (`_price_step` uses only the per-token law); `fit_llm_d_config` (fits the external llm-d sim *from*
  us — retired).

---

## Suggested de-fit order

1. ~~**`SATURATED_BASE_MS` → roofline saturated-decode floor**~~ — **DONE 2026-06-02, but via MEASURED
   ANCHORS, not roofline.** The roofline route was tested and FAILED (collapses to the kernel step ~27 ms
   vs measured plateau ~259 ms → MAPE 37%): the saturated plateau is queueing/recompute latency, not a
   compute term. Both `SATURATED_BASE_MS` and `SATURATED_TURN_OVERHEAD_MS` were instead replaced by the
   measured output-binned plateau (the `build_saturated_ceiling.py` artifact). A true *from-physics*
   ceiling needs an eviction/queueing model (extend `ttft_queue_sim`), filed as future work.
2. **`PREFILL_FLOOR_MS`** — one tiny `new=1,cached=0` microbench → measured anchor.
3. **`SATURATED_TURN_OVERHEAD_MS` → modeled cohort-prefill wall** — offline, medium; finishes T_upper.
4. **HOST terms** — host instrumentation; resolves the 4× discrepancy (may surface a real model error).
5. **`PREFILL_NEW_MS_PER_TOKEN`** — the chunked-prefill microbench; biggest TTFT fit, needs GPU.
6. ~~**TPOT knees → eviction-watermark crossing**~~ — **RESOLVED 2026-06-10 (distribution-overflow
   eviction-drain ramp; round 3 added firing-gate HYSTERESIS — all 9 pre-registered binding gates
   PASS, H100 swe-plateau 8.51 vs baseline 8.65, see the
   `prediction_construction.md` De-fit log).** The tuned ramp band `P_LO=0.88` / `P_HI_SHORT=1.22` / `P_HI_LONG=2.0` and the
   `OUT_KNEE_LO/HI` upper-knee interpolation were ELIMINATED from `kernel_tpot`: the saturation weight is
   now the computed chunk-quantized eviction-drain fraction
   `w = clamp(n_evicted·chunk_steps/out [·z if multi-chunk], 0, 1)` with `n_evicted = (1−1/z)·sched`,
   `chunk_steps = ceil(ctx·qbar/(M − b_eff))`, `z = pressure·qbar` (qbar = trapezoid mean of the
   measured `context_scale_quantiles`, resolver `simulator/cohort_scale.py`) and
   `M = max_num_batched_tokens` (vLLM device-rule engine config: 8192 H100-class / 2048 A100,
   per-deployment JSON). Onset = the pool being PHYSICALLY full (`pressure ≥ 1` — vLLM v1 preempts
   only on allocation failure) AND distribution overflow (`z > 1`); a cell's first overflow turn is
   growth-damped (development state in `predict_cell_tpot`). The output-binned knee disappears because
   the long-output widening emerges from the turn's own `out` in the drain-fraction denominator
   (round-1's `z=1`-only onset and linear once-per-turn duty were falsified by the implied-duty
   extraction and replaced in round 2). Round 3 closed the last gate fail: the firing gate gains
   HYSTERESIS on the cell path (arm at `pressure ≥ 1` AND `z > 1`, hold at effective pressure
   `max(pressure, 1)` while `z > 1`, release at `z ≤ 1` — the H100 swe@40 plateau was knife-edge
   gate FLICKER of the block-quantized pressure, not sub-pool-full saturation; its observational
   twin A100 term@20, pressure peak 0.965, never arms and stays bit-identical).
   `OUT_KNEE_LO/HI` survive only as the measured ceiling-cluster
   output labels (28/86 = the saturated-ceiling anchor outputs) for `ramp_tpot`/`build_ramp_knees`
   diagnostics. See the 2026-06-10 De-fit log entries (rounds 1–3) in `prediction_construction.md`
   for the gate numbers.

Each change must be measure-gated: tp1 TPOT 15.91% / swe-plateau 8.83% / TTFT 33.0% / E2EL 19.6% must
not regress (and ideally the de-fitted version is within noise of the fitted one — that's the proof the
physics route is right).
