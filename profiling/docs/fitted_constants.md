# Fitted constants — replacement ledger

Pure table of every **fitted** (least-squares regression) or **tuned-knob** (hand-chosen, no
measured/physical anchor) constant in the tp1 headline TTFT/TPOT/E2EL flow. Narrative + provenance in
[fitted_constants_audit.md](fitted_constants_audit.md). Effort: **Low / Med / High**; **(GPU)** = needs a
profiling run, **(host)** = host-side instrumentation, else offline. Status: ⛔ open · ✅ done · ✓ acceptable.

## FITTED — least-squares regressions

| Constant (value) | Location | Metric | Used for | Replace with | Effort | Status |
|---|---|---|---|---|---|---|
| `SATURATED_BASE_MS` (118.7) | kernel_tpot.py (removed) | TPOT, E2EL | output-independent height of the saturation ceiling `T_upper` | measured plateau anchors (median ITL @pressure≥2.5 per output cluster) | — | ✅ done 2026-06-02 |
| `SATURATED_TURN_OVERHEAD_MS` (3263) | kernel_tpot.py (removed) | TPOT, E2EL | output-amortized part of `T_upper` (`/output` term → short outputs saturate higher) | same artifact (`build_saturated_ceiling.py`) | — | ✅ done 2026-06-02 |
| `PREFILL_NEW_MS_PER_TOKEN` (0.0310) | ttft_queue_sim.py:117 | TTFT, E2EL | per **fresh** (cache-miss) prefilled-token rate; dominant TTFT term for long new prompts | **mostly GPU GEMM**: = `util=0.65` roofline **0.02498 ms/tok** (fit-free, EXISTING `roofline_params`) — NEW is **1.24× roofline, NOT 8×**. ~0.006 ms/tok off-GPU tail = framework dispatch (TaxBreak); needs stage-split microbench. Validate at swebench operating point (leverage ±31%). | ◐ Med | ◐ GPU part de-fittable now |
| `PREFILL_FLOOR_MS` (22.5) | ttft_queue_sim.py:116 | TTFT, E2EL | fixed per-prefill cost (schedule + first-token emit + detok + return); genuine intercept ≈ 26 ms min pure-prefill | measured min pure-prefill (turn-0) TTFT, **EXISTING data** (~22.5–26 ms; per-profile spread → terminalbench ~14) | Low (offline) | ◐ de-fittable now |
| `PREFILL_HOST_SHARED_MS_PER_TOKEN` (0.003485) | ttft_queue_sim.py:119 | TTFT, E2EL | host work over re-sent **cached** ctx, amortized once/step (= 0.571 × 6.103e-3) | **split is unidentifiable at c1** → re-fit `per_req(B)=shared/B+perreq` on EXISTING batch B-sweep `cached_prefill_batch_ttft_H100.csv`; subtract EXISTING GPU paged-attn ~1.5 ms/1k (`cached_prefill_v3`) first | Med | ⛔ open |
| `PREFILL_HOST_PERREQ_MS_PER_TOKEN` (0.002618) | ttft_queue_sim.py:120 | TTFT, E2EL | same host work, charged per-request (= 0.429 × 6.103e-3) | same batch-B-sweep joint fit; residual ~2.5 ms/1k host = **SHA-256 prefix-cache block-hash** (lead — vLLM default ≥v0.11; check config) → microbench | Med (host) | ⛔ open |

> The four `PREFILL_*` rows are **one** 3-coefficient c1 regression (`22.5 + 0.0310·new + 0.006103·cached`,
> R²=0.963, verified) fanned into named constants. `new`/`cached` are cleanly separable (VIF=1.00) → the de-fit
> is well-posed. **Audited decomposition** (see `prefill_law_defit_trace.md`): the 6.1 ms/1k cached rate ≈
> **1.5 GPU paged-attn (measured)** + **~1.5 tokenize** + **~2.5 unexplained host**, whose lead is **SHA-256
> prefix-cache block hashing** (block-hash/alloc are negligible only for the *cheap* hash; SHA-256 is vLLM's
> default ≥v0.11). The shared/perreq split is **structurally unidentifiable at c1** (1 req/step → identical
> column) — it must come from the batch B-sweep, not the headline c1 fit.

## TUNED-KNOB — hand-chosen, no measured anchor ("Knees retuned vs the data")

| Constant (value) | Location | Metric | Used for | Replace with | Effort | Status |
|---|---|---|---|---|---|---|
| `P_HI_SHORT` (1.6) | kernel_tpot.py:61 | TPOT, E2EL | upper pressure knee for short-output turns — pressure at which the amplifier reaches full ceiling (step steepness) | anchor to the measured KV-eviction watermark crossing (`ramp_tpot` defcap cluster ~0.88–1.22) | Med | ⛔ open |
| `OUT_KNEE_LO` (40) | kernel_tpot.py:63 | TPOT, E2EL | output below which a turn is "short" (selects `P_HI_SHORT`) | drop entirely — compute saturation onset per-cell from the watermark instead of binning by output | Med | ⛔ open |
| `OUT_KNEE_HI` (80) | kernel_tpot.py:64 | TPOT, E2EL | output above which a turn is "long" (selects `P_HI_LONG`) | drop with `OUT_KNEE_LO` (same per-cell-onset restructure) | Med | ⛔ open |
| `SAT_SUSTAIN_LO` (10) | kernel_tpot.py:75 | TPOT, E2EL | output below which a turn can't sustain saturation (gate → 0) | derive from decode physics: eviction-queue build time ÷ `kernel_step` at that batch/ctx | Med | ⛔ open |

## Anchored knobs — acceptable as-is (optional re-grounding)

| Constant (value) | Location | Metric | Used for | Anchor (why acceptable) | Effort | Status |
|---|---|---|---|---|---|---|
| `P_LO` (0.8) | kernel_tpot.py:59 | TPOT, E2EL | amplifier onset pressure (below it ITL = kernel_step) | read off the measured amp-vs-pressure curve; sharper route: measured `maxdec<sched` crossing | Low | ✓ anchored |
| `P_HI_LONG` (2.5) | kernel_tpot.py:62 | TPOT, E2EL | upper knee for long-output (gentle ramp) | ≈ measured "saturates by pressure ~2.5"; re-ground to watermark | Low | ✓ anchored |
| `SAT_SUSTAIN_HI` (24) | kernel_tpot.py:76 | TPOT, E2EL | output for full sustain weight | = measured 22-tok plateau-min + 2 (structural offset) | — | ✓ measured |
| `PREFILL_FA3_MS_PER_TOKEN2` (8.31e-7) | ttft_queue_sim.py:118 | TTFT, E2EL | super-linear FA3 attention growth term | kernel-derived `FA3(8192)/(8192²/2)` — not a fit | — | ✓ measured |

## Note on the de-fitted ceiling
`SATURATED_BASE_MS` / `SATURATED_TURN_OVERHEAD_MS` were de-fitted to measured anchors (no regression,
gate held 15.91→15.89%). This is **fit-free but still empirical** — it reads the measured plateau, it
does not derive it. A true *from-physics* ceiling failed via roofline (the saturated plateau is ~259 ms
queueing/recompute latency vs a ~27 ms compute roofline) and needs an eviction/queue model
(extend `ttft_queue_sim`) — the only remaining route to a fully-derived `T_upper`.
