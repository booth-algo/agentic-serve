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

(Implementation results to be appended below by the execution phase; this section is the
pre-registration and was committed before any lever was touched.)
