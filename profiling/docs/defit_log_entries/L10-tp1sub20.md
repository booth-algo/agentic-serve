# L10-tp1sub20: RTX2080Ti tp1 Llama E2EL < 20 (2026-06-11)

**Lane contract:** branch `tp1-e2el-sub20` (off `defit-campaign-integration` post-L9-merge).
Target: RTX2080Ti tp1 Llama **E2EL cell-MAPE < 20** (baseline 35.47; ttft_cell 68.42,
tpot_cell 26.40) while H100/A100/RTX3090 tp1 stay < 20 and within +0.3 of baselines
(10.78 / 15.87 / 16.00) and the binding trio (H100, A100, H100x2) stays byte-identical.
NO FITTING; no GPU launches (all 8 host GPUs busy with someone else's campaign; ssh
read-only allowed). Gates: replay-ON, `RAMP_TPOT_REQUIRE_POOLS=1`. Baselines reused:
`/tmp/tp1_base.{predictions,metrics}.json`, `/tmp/postmerge_trio.{predictions,metrics}.json`.

## Pre-registered error map (built from `/tmp/tp1_base.predictions.json` BEFORE any edit)

44 cells (4 profiles × 11 concs). Cell err = |pred−meas|/meas on the mean-over-turns;
cell-MAPE = mean |err| over cells (verified: reproduces 35.47/68.42/26.40 exactly).
Turn-level decomposition: `e2el_signed_err ≈ ttft_signed_err + tpot_signed_err·osl`
(verified: e2el ≈ ttft + tpot·output_tokens at turn level).

### Counterfactuals (which lever CAN reach < 20)

| counterfactual (turn-level, re-aggregated per cell) | E2EL cell-MAPE |
|---|---|
| baseline | **35.47** |
| (a) ALL TTFT turn errors zeroed (TPOT-only floor) | **3.77** |
| (b) ALL decode/TPOT turn errors zeroed | **24.09** — still > 20 |
| TTFT zeroed in the 22 OVER-predicted cells only | **12.34** |
| TTFT zeroed in the 22 UNDER-predicted cells only | 17.07 |

**TTFT is the only lever that can reach < 20; TPOT alone cannot (floor 24.09).**
Killing just the OVER-side TTFT error reaches 12.34 — a 7.7 pt margin even if the
under-side never improves.

### Where the 35.47 lives

Contribution to the 35.47 (pts): chat 10.80, terminalbench 9.27, osworld 9.13,
swebench 6.28; by conc band: c1–10 → 7.34, c20–80 → 9.16, c120–320 → 18.98.
Two signed regimes (22 cells each):

1. **OVER-side, 20.28 pts** — chat (all 11 cells, e2el signed +11→+79 % monotone in
   conc; TTFT pred 222.5 s vs meas 119.2 s at c320; 4 198 vs 1 385 ms at c10) +
   osworld (9 cells, +24..+36 % at c≥120) + the four c1 cells. Mechanism: the pool
   lower-bound pin **512 blocks = 8 192 tokens**; chat sessions ≈ 142 blocks, osworld
   ≈ 330 → the queue sim holds only ~1.7–3.6 sessions resident, predicting mass
   eviction/serialized admission, while measured shows mild queueing at c10–20 (real
   pool ≫ 512). Pool consumers: `ramp_tpot.py:339` (pressure), `kernel_tpot.py:221/375`
   (capacity_batch), `ttft_queue_sim.PrefixLRUCache` (admission/eviction).
2. **UNDER-side, 15.19 pts** — swebench c5–320 (10 cells) + terminalbench c5–320 (10)
   + osworld c5/c10 (tiny). e2el signed −16..−51 %. Coupled signature: predicted TPOT is
   clamped at the saturated-ceiling anchors (out 25 → 51.8 ms; preds 46.8–49.5) while
   measured plateaus run 51–73 ms (terminalbench c160: meas 73.1 vs pred 49.0) → the
   queue sim's decode drain is too fast → TTFT under (terminal c256 pred 94.7 s vs meas
   119.6 s). The ceiling artifact was built with pool=512 "pressure ≥ 2.5", a population
   polluted by pseudo-saturated mid-conc turns that bias the anchors LOW; the out-only
   anchor keying also merges swe (~52–54) and terminal (~66–73) plateaus into one 51.8.
3. **c1 floor cells, ~1.8 pts** — TTFT over +40..+66 % at c1 on all profiles (chat c1
   pred 200 vs meas 121 ms): the H100-inherited prefill law on a consumer card (L9
   successor 4 — needs live measurement, OUT of reach this round; documented, not guessed).

**RTX3090 headroom check** (42 cells, e2el 16.00, TRUE pool 2008 from its engine log):
same qualitative shape, ~2.5× smaller (profile means 13.7/16.6/16.7/17.1; c120–320
contributes 7.34 pts). The 3090 is the true-pool reference signature the 2080Ti should
converge toward. No planned lever touches any 3090/H100/A100 input → those slices must
re-build byte-identical.

## Pool truth status (the L9 lead, resolved as far as offline evidence goes)

* **Exact engine log unrecoverable:** the tp1 GT sweep ran 05-11/05-12 on ports
  8089/8090 (`2080ti:/tmp/bench_Llama-3.1-8B_tp1_multi_vllm_p8089/p8090.log`, read-only);
  `/tmp/vllm_8089.log` and `/tmp/vllm_8090.log` were overwritten 05-22 by the Qwen-tp4 /
  Llama-tp4 launches. The two unidentified logs (8091/8092) are Qwen3.5 tp1 starts. GT
  files carry `engine_cache_telemetry: not_available`.
* **Engine-version parity confirmed:** the tp1 GT (`_engine_version.txt`: vllm 0.19.0)
  matches the 3090 Llama tp1 anchor log (`3090:/tmp/vllm_8093.log` 06-02, v0.19.0, same
  model, same launch-flag family, util 0.85, tp1/no-NCCL).
* **Engine-anchored reconstruction** (the exact L9-precedented method used for 2080ti
  Qwen tp2): transfer the 3090-tp1 back-solved non-KV overhead **1.63 GB** →
  pool = (23 622 320 128·0.85 − 16.06e9 − 1.63e9) / 2 097 152 = **1138 blocks**
  (18 208 tokens). Direction of bias: the 2080ti GT ran max_model_len 8192 < 3090's
  16384 → activation-profiling peak ≤ the 3090's → overhead ≤ 1.63 GB → **1138 is
  floor-leaning** within the engine bracket [512, 1916]. Falsification cross-check: the
  tp2-sibling overhead (3.29 GB at max_len 32768, world=2) transfers to 347 < the 512
  startup bound → tp2 transfer provably wrong; tp1 overhead < 2.95 GB.
* `max_num_batched_tokens=2048` in the manifest is VERIFIED engine truth (installed
  vLLM 0.19.0 `arg_utils._set_default_args` read off the host: <70 GiB device → 2048
  OPENAI_API_SERVER default) — no change.

## Pre-registered plan (ranked by expected impact; offline-only; registered BEFORE edits)

* **P1 (dominant, expect −15..−20 pts):** pin `configs/deployments/
  2080ti_Llama-3.1-8B_tp1_vllm.json` `kv_pool` 512 → **1138**, status
  `engine_anchored_reconstruction`, full provenance note (3090 anchor, bracket,
  falsified tp2 transfer, unrecoverable-log evidence). One number; engine-config truth.
* **P2 (coupled, required):** regenerate the RTX2080Ti saturated ceiling
  (`python3 -m profiling.process.build_saturated_ceiling`) — the pressure ≥ 2.5
  population purges the pseudo-saturated turns; pre-registered expectation: anchors RISE
  toward the measured plateaus (out 25: 51.8 → ~55–65; out 95: 42.1 → ~43–46), which
  also pushes the under-side the right way (slower drain → higher swe/terminal TTFT).
  All other ceiling artifacts must regenerate byte-identical.
* **P3 (gate):** rebuild with `RAMP_TPOT_REQUIRE_POOLS=1`; compare vs
  `/tmp/tp1_base.*` (H100/A100/RTX3090 byte-identical expected — no shared inputs
  touched) and `/tmp/postmerge_trio.*` (byte-identical). Pre-registered landing:
  over-side 20.28 → ~5–8 pts, under-side 15.19 → 12–17 pts; central E2EL ≈ 17–20.
* **P4 (decision points if still ≥ 20):** (a) re-map with the same decomposition; if the
  residual is swe/terminal under-cells with TPOT still ceiling-clamped >10 % below the
  measured plateau, the out-only anchor keying is implicated — structural builder choice,
  honest-stop candidate; (b) NO pool scanning against the gate (= fitting); admissible
  alternates are other engine-anchored constructions only; (c) verify remaining
  engine-config-truth scheduler params for the 22 GiB device (`max_num_seqs` resolved
  default, `long_prefill_token_threshold`) from the installed engine source, read-only;
  (d) only after (c): document per the L9 honest-stop protocol.
* **NOT in scope (documented, not guessed):** the consumer prefill law (c1 floor cells,
  ~1.8 pts, needs live measurement — L9 successor 4); per-profile ceiling keying (builder
  structure change).

(Implementation results appended below by the execution phase; the section above is the
pre-registration and was committed before any lever was touched.)

## Round 1 results (2026-06-11, levers P1+P2, gate P3)

**Pool truth superseded the 1138 estimate before any edit:** the read-only harvest
(committed at `3729f53`, `profile_data/engine_logs/2080ti_Llama-3.1-8B_tp1_pool_evidence.txt`)
found a same-host same-model same-util tp1 ENGINE POOL LINE
(`2080ti:/tmp/calib_2080ti_test.log` 04-29, vllm 0.19.0, eager/4096/0.85:
`GPU KV cache size: 23,600 tokens` = **1475 blocks** zero-CG ceiling) plus the tp1
CUDA-graph estimate from the 3090 anchor log (0.575e9 B) → GT-config reconstruction
**1201 blocks** (19,216 tokens), HARD ENGINE BRACKETS **[512, 1475]** (vs the looser
host-arithmetic [512, 1916]). Per the plan's "pool truth first if found", P1 pinned
**1201** (not 1138): `available_kv_blocks` 512 → 1201, `kv_pool.status` kept the repo
convention `derived` with source `engine-log-anchored reconstruction` (the L9 Qwen-tp2
precedent; the pre-registered label `engine_anchored_reconstruction` lives in the source
string), full verbatim citation in the manifest note.

**P2 (ceiling regen):** only `saturated_ceiling_RTX2080Ti_llama31_8b.json` changed —
saturated population 759 → 605 (pseudo-saturated mid-conc turns purged), anchors rose
out25: 51.8 → **54.8** ms, out95: 42.1 → **42.9** ms (pre-registered direction correct;
magnitudes at the low edge of the predicted ~55–65 / ~43–46 bands). G9 sensitivity
TIGHTENED (threshold ±1.4% vs ±3.3%) — the purged population is more homogeneous. All
seven other ceiling artifacts regenerated byte-identical.

**P3 (gates, replay-ON, RAMP_TPOT_REQUIRE_POOLS=1):** pytest 200 passed + 1 skipped +
12 subtests green. H100/A100/RTX3090 prediction slices **byte-identical** to
`/tmp/tp1_base.predictions.json` (10.78 / 15.87 / 16.00 E2EL unchanged); binding trio
**byte-identical** to `/tmp/postmerge_trio.predictions.json` (`cmp` clean, metrics too).

RTX2080Ti, before → after: **e2el_cell 35.47 → 26.96** (−8.5 pts), ttft_cell
68.42 → 42.73, tpot_cell 26.40 → 25.69. Versus the pre-registered landing: the
over-side collapsed 20.28 → **10.86** pts (predicted ~5–8: direction right, residual
bigger), the under-side 15.19 → **16.09** (inside the registered 12–17 band, at the bad
end), central e2el 26.96 vs predicted ~17–20 — **target < 20 NOT reached in round 1**.

Per-cell movement (worst-5 baseline cells): chat c320 76.35→17.33 (TTFT pred
222.5 s→114.8 s vs meas 119.2 s), terminal c10 64.41→37.73, chat c256 63.00→14.50,
chat c200 61.88→10.01, chat c120 58.52→18.60. The chat eviction blow-up is GONE
(all chat c80–320 cells now 10–22%). Honest exposure: 9 cells WORSENED — swebench
c80–320 (23.1→57.8 at c120, 20.8→55.3 at c200; TTFT signed −38..−60%) + terminal
c5/c80/c120/c320 + chat c20 — median signed e2el bias swung −0.83% → −18.97%: the
bigger pool removes predicted re-prefill (fewer evictions → faster drain → lower TTFT),
and the +3.0 ms anchor rise was too small to offset it. This is exactly the P4(a)
signature: swe/terminal TPOT still clamped near the out25 anchor 54.8 while measured
plateaus run 52–73 (out-only anchor keying merges them), and P4(c) (engine-truth
`max_num_seqs` / `long_prefill_token_threshold` for the 22 GiB device) is the remaining
admissible engine-config lever for round 2. No fitting performed; no rollback — every
binding gate is clean and the dominant residual moved from the falsified pool pin to
the named builder-structure successor.

## Round 2 pre-registration (committed BEFORE the lever; 2026-06-11)

**Diagnosis of `/tmp/tp1_r1.predictions.json` (44 cells, mean-over-turns decomposition):**
the residual is now almost pure UNDER-prediction — 37/44 cells signed-negative, median
signed e2el −18.97 %. Two distinct signatures:

1. **TTFT under with TPOT now CORRECT** (the dominant mass): swebench c120 TPOT +1.5 %
   but TTFT −60.7 % (pred 20.5 s vs meas 52.0 s); swe c200 −56.9 %, osworld c160–c256
   −37..−40 %, chat c10/c20 −45/−53 %, terminal c80/c120 −37/−45 %. With pool AND decode
   speed both right, the queue sim still admits/advances prefills too fast.
2. **terminalbench TPOT still ceiling-clamped** (pred ~52 vs meas 63–73 at c40–c320) —
   the P4(a) out-only anchor-keying structural residual, pre-registered NOT in scope.

**Root cause found for (1), engine-config truth (P4(c) executed read-only):** the queue
sim prices admission with MODULE-LEVEL H100 constants — `MAX_NUM_BATCHED_TOKENS = 8192`,
`LONG_PREFILL_TOKEN_THRESHOLD = int(32768·0.04) = 1310`, `MAX_NUM_SEQS = 1024` — for
EVERY GPU, while the 2080Ti GT verifiably ran a 4× smaller scheduler: GT server metadata
(`chat-multiturn-synth_conc10.json` config block): `max_model_len=8192`,
`max_num_batched_tokens=None` (unset), `max_num_seqs=None` (unset); installed engine
(2080ti host, vllm 0.19.0 `g2a69949bd`, read-only) `vllm/engine/arg_utils.py
get_batch_defaults`: `device_memory < 70 GiB → OPENAI_API_SERVER:
max_num_batched_tokens=2048, max_num_seqs=256`. The kernel-TPOT side of the SAME cells
already prices with the per-deployment 2048 (`RooflineParams.max_num_batched_tokens`,
manifest-pinned, L9) — the TTFT queue sim is the one consumer still on the H100
constants. Internal inconsistency + engine truth ⇒ the round-2 lever.

**Threshold caveat (recorded honestly):** in the installed source the
`int(max_model_len·0.04)` rule fires only when `max_num_partial_prefills > 1`
(`vllm/config/scheduler.py:244-246`; default 1 → threshold 0 → a prefill chunk is
bounded by the token budget alone). The sim's adoption of the 0.04 rule is a
PRE-EXISTING gated structural choice (module comment, audit-v2 lineage) — this round
only replaces its `max_model_len` input with the config's own GT-recorded value
(8192 → threshold 327), NOT relitigating the rule. The `_prefill_gemm_per_tok_loaded`
util-ramp endpoints (1310/8192) are NOT touched: that ramp is a documented retained
compensating fit (audit-v2 R1/S6) inside the prefill-law pricing stack — part of the
consumer-prefill-law successor, not scheduler admission arithmetic.

**Lever (offline, no new tuned constants):** thread per-config scheduler truth into the
queue sim — optional `QSimSchedConfig(max_num_batched_tokens, long_prefill_token_threshold,
max_num_seqs)` passed by `build_row` ONLY when the deployment manifest pins
`max_model_len`/`max_num_seqs`; every unpinned config resolves to the existing module
constants (BYTE-IDENTICAL by construction). Pin the 2080Ti tp1 manifest:
`max_model_len: 8192`, `max_num_seqs: 256` (+ provenance note with the verbatim
citations above). A100/RTX3090 GT also ran 2048/256 (same <70 GiB rule) — adopting their
truth is a NAMED SUCCESSOR for their own gated lanes, not silently bundled here (their
slices must stay byte-identical this round; per-lane adoption is the L9 precedent —
pools/floors/ceilings landed one config at a time, each on its own gate).

**Pre-registered expectations:** budget 8192→2048 + per-req chunk 1310→327 slow the
sim's per-step prefill advancement ~4× on the 2080Ti only → TTFT predictions RISE
broadly at c≥5 → the 37-cell under side compresses (swe c120/c200, terminal c80/c120,
chat c10/c20, osworld c160–c256 move toward measured). `max_num_seqs` 1024→256: NO
behavioral change expected (the 1201-block pool binds at ~8–13 resident sessions ≪ 256;
pinned for engine-truth completeness). c1 floor cells may rise slightly (more chunks ×
per-step overhead) — direction WORSE for the +40..+66 % over-side c1 cells, bounded,
documented (prefill-law successor). Terminal TPOT clamp untouched (~3–5 pts residual).
H100/A100/RTX3090 slices and the trio: byte-identical expected. Landing band: e2el
cell-MAPE 15–22 — the target < 20 is INSIDE the band but not guaranteed; if ≥ 20 the
remaining residuals are the two named structural successors (P4(a) anchor keying,
consumer prefill law) → L9 honest-stop protocol.

## Round 2 results (2026-06-11, lever committed e56be4a; gates replay-ON, RAMP_TPOT_REQUIRE_POOLS=1)

pytest 200 passed + 1 skipped + 12 subtests green. H100/A100/RTX3090 prediction slices
**byte-identical** to `/tmp/tp1_base.predictions.json` (10.78 / 15.87 / 16.00 unchanged);
binding trio **byte-identical** to `/tmp/postmerge_trio.{predictions,metrics}.json`
(`cmp` clean on both). RTX2080Ti tpot_cell EXACTLY unchanged (25.6936 → 25.6936) — the
lever is TTFT-only, as constructed.

RTX2080Ti: **e2el_cell 26.96 → 22.21** (−4.75 this round; 35.47 → 22.21 cumulative,
−13.3 pts), ttft_cell 42.73 → 37.56. Landed at the BAD edge of the pre-registered 15–22
band: **target < 20 NOT reached**. Median signed e2el −18.97 % → −8.99 % (under-side
mass 34/44 cells, 18.7 of the 22.2 pts).

Per-cell movement (gate scoring, r1 → r2): the swe/terminal mid-conc under-side the
lever targeted COLLAPSED — swebench c120 57.8→12.8, c256 41.4→10.5, c80 34.7→11.2,
c40 29.2→10.1, c200 55.3→31.7; chat c40 27.7→12.4, c80 21.9→12.1, c120 18.6→10.0;
terminal c200 26.4→14.3, c40 27.9→15.5, c80 34.2→24.4. Honest exposure — 2 cells
WORSENED >5 pts: swebench c160 42.5→53.5 and terminalbench c120 42.1→50.6 (both
TTFT-under, signed −53/−56 %), NON-MONOTONE with their improved neighbors (swe c120 now
+8.3 OVER while c160 is −53 UNDER): the closed-loop sim flips regime cell-to-cell —
this is queue-regime sensitivity around the (pool, budget) operating point, not a
constant-rate misprice; no engine-config knob is implicated.

**Honest stop (L9 protocol).** P4(c) is now EXHAUSTED: every scheduler parameter the sim
consumes (token budget, chunk cap, running-set cap, pool, block size) is per-config
engine truth for this deployment. The remaining 22.2 pts decompose onto the three
pre-registered NON-offline residuals: (1) **consumer prefill law** — the c1 cells
(+24..+25 % over, ttft_err 46–66 at c1–c10 across profiles) and the under-side TTFT
rate at mid/high conc are both priced by H100-measured grids + H100 host constants;
needs the live c1 stage-split microbench on the 2080Ti (L9 successor 4). (2) **P4(a)
out-only ceiling anchor keying** — terminal TPOT still clamped ~52 vs measured 63–73
(turn-level tpot_err 26–41 on terminal, 57–61 on swe high-conc, sign-cancelling in the
cell mean); per-profile anchor keying is a builder-structure change, pre-registered out
of scope. (3) **queue-regime sensitivity** at swe c160/terminal c120 (above). Admissible
next steps, in order: capture the exact tp1 engine pool line at GT flags (one server
start, next host window — removes the [512,1475] residual bracket), run the live
prefill-law microbench, then the P4(a) builder RFC. NO pool scanning, NO per-profile
gate-picking — both are fitting. Successor for the method itself: A100 (mml 32768) and
RTX3090 (mml 16384) GTs also ran the <70 GiB/a100 2048/256 resolved defaults; adopting
their scheduler truth via the same manifest pins belongs to their own gated lanes (this
lane held their slices byte-identical, as required).

## Round 3 (2026-06-11): no admissible offline lever remains — verification + re-gate

**Pre-lever checks (committed BEFORE the gate, per protocol):**

1. **Pool truth re-verified, nothing new.** The orchestrator's independent pool hunt
   returned `2080ti_Llama-3.1-8B_tp1_vllm = 1201`, eager-4096 engine line `1475`,
   hard brackets `[512, 1475]` — exactly the values already pinned at `d6be418`
   (manifest `available_kv_blocks: 1201`, verbatim citation in `data.kv_pool.note`,
   evidence file `profile_data/engine_logs/2080ti_Llama-3.1-8B_tp1_pool_evidence.txt`).
   No new engine-log truth to apply.
2. **GT-artifact builders re-run at HEAD → byte-identical.**
   `python3 -m profiling.process.build_saturated_ceiling` and
   `python3 -m profiling.process.build_prefill_floor` both regenerate every artifact
   byte-identical (`git status` clean; 2080Ti anchors stay out25=54.8 ms n=372 /
   out95=42.9 ms n=233, population 605; 20 config floors unchanged). The builders'
   inputs did not change in round 2 (the ceiling consumes only `available_kv_blocks`
   for pressure — unchanged since R1; the floor is conc=1-only) → no regeneration lever.

**Round-3 diagnosis of `/tmp/tp1_r2.predictions.json` under the GATE's own scoring**
(`gate_scoped_rows._gpu_metrics`: cell = mean over rows of the ROW `e2el_err`, which is
the WITHIN-ROW turn-level MAPE — not the err-of-means used in the round-1 map):

* The same 44 rows score **16.67** on err-of-means vs **22.21** gate-scored: ~5.5 pts of
  the headline is sign-cancelling WITHIN-ROW turn error (e.g. swe c320 ttft signed
  −18.6 % vs abs 48.2 %; terminal c320 signed −4.6 % vs abs 40.3 %) — error in the
  per-turn TTFT *dynamics* (cohort ordering/timing), invisible to any constant-input
  lever by construction.
* **Counterfactuals (gate scoring):** all TTFT turn errors zeroed → **6.56**; all
  priced-TPOT turn errors zeroed → **20.85**. TTFT remains the only path < 20; the
  direct e2el contribution of priced TPOT is ~1.4 pts (its real damage is indirect,
  through the queue-sim drain rate — the P4(a) coupling).
* **Worst cells:** swe c160 53.5, terminal c120 50.6, swe c320 46.5, osworld c256 38.5,
  osworld c160 38.5, terminal c320 36.5, terminal c10 35.7, terminal c5 34.6,
  swe c200 31.7, chat c20 30.7. Three signatures, each mapping onto an already-named
  NON-offline successor:
  1. **TTFT-under at mid/high conc with sign-cancelling per-turn errors** (swe c160/c320,
     terminal c120/c320, osworld c80–c256): the closed-loop sim's per-turn admission
     timing, the round-2 "queue-regime sensitivity" — every scheduler input it consumes
     is already per-config engine truth (P4(c) exhausted); no engine-config knob left.
  2. **Ceiling-clamped TPOT vs wide measured plateaus** (measured per-turn TPOT spans
     28–159 ms on swe/terminal high-conc; predictions clamp at the out-keyed anchors
     34.7–54.8): P4(a) out-only anchor keying — builder-STRUCTURE change, pre-registered
     out of scope; relitigating it now, against a failing gate, is the "per-profile
     gate-picking" this entry already ruled out as fitting.
  3. **c1–c10 over-side** (chat/swe/terminal c1 TTFT +43..+66 %): the consumer prefill
     law (H100-measured grids + H100 host constants) — needs the live c1 stage-split
     microbench (L9 successor 4); guessing a consumer constant offline is fitting.

**Decision (per pre-registered P4(b)/(d) and the round-2 honest stop): NO honest offline
lever remains.** Round 3 is verification + re-gate only; the working tree is unchanged
from `7d92648`, so `/tmp/tp1_r3.*` is expected byte-identical to `/tmp/tp1_r2.*` and the
trio byte-identical to `/tmp/postmerge_trio.*`. Lane outcome: **partial progress,
35.47 → 22.21 (−13.3 pts), target < 20 not reached**; the residual is fully decomposed
onto the three named successors above (exact tp1 engine pool line at GT flags; live
consumer prefill-law microbench; P4(a) per-profile ceiling-keying builder RFC; plus the
A100/RTX3090 scheduler-truth adoptions in their own lanes).

**Round 3 gate results (replay-ON, RAMP_TPOT_REQUIRE_POOLS=1; no lever applied):**
pytest 200 passed + 1 skipped + 12 subtests green. `/tmp/tp1_r3.{predictions,metrics}.json`
**byte-identical** to `/tmp/tp1_r2.*` (`cmp` clean on both — confirms the round was
verification-only); H100/A100/RTX3090 slices byte-identical to `/tmp/tp1_base.predictions.json`
(10.7777 / 15.8653 / 16.0045, all < 20, delta 0); binding trio
`/tmp/tp1_trio_r3.{predictions,metrics}.json` **byte-identical** to
`/tmp/postmerge_trio.*` (`cmp` clean on both). RTX2080Ti final: e2el_cell **22.2064**
(ttft 37.563, tpot 25.6936). **Lane closes at the honest stop: 35.47 → 22.21, < 20 not
reachable with the admissible offline evidence; successors named above.**
