# L12-a100multi: A100x2 / A100x4 Llama E2EL close-out (2026-06-11)

**Lane contract:** branch `a100-multi-close` (off `defit-campaign-integration`, post-L9/L10
merge — sub-linear fill, QSimSchedConfig scheduler-truth mechanism, engine-log pool patterns
and committed pools all IN). Target: **A100x4 Llama E2EL cell-MAPE < 20** (baseline measured
below) AND **A100x2 stays < 20** (baseline e2el 14.6301; regression > 0.3 forbidden; TPOT
improvement recorded but E2EL is the goal metric) AND **H100/H100x2 byte-identical** to
`/tmp/am_pair_base.predictions.json`; **A100 tp1 is BINDING** — byte-identical until the tp1
scheduler-truth lever (P4 below = the plan's "P3c"), then within +0.3 of e2el 15.8653 /
ttft 22.2222 / tpot 14.3728 (and ideally improving). NO FITTING — engine-config truth,
GT-builder artifacts, cited physics only. NO GPU launches (7/8 a100 GPUs busy with another
campaign; ssh a100 READ-ONLY). Gates: replay-ON, `RAMP_TPOT_REQUIRE_POOLS=1`. Local commits
only; `prediction_construction.md` untouched. Baselines (captured in this worktree at HEAD
`8fd255e` BEFORE any edit; reproduction at HEAD re-verified byte-identical on both
predictions and metrics, replay-ON, 2026-06-11):

| scope | artifact | e2el_cell | ttft_cell | tpot_cell |
|---|---|---|---|---|
| A100x2 | `/tmp/am_base.*` | **14.6301** | 23.8786 | 45.0845 |
| A100x4 | `/tmp/am_base.*` | **17.3047** (already < 20) | 33.1784 | 57.7320 |
| A100 tp1 (binding) | `/tmp/am_a100_base.*` | 15.8653 | 22.2222 | 14.3728 |
| H100 / H100x2 (byte-gate) | `/tmp/am_pair_base.*` | 10.7777 / 18.5506 | 18.1309 / 29.0163 | 14.4697 / 21.5336 |

**Headline finding at baseline:** the stated goal (A100x4 e2el < 20) is ALREADY met at
17.30. This lane's job is therefore (a) put the A100-family inputs on engine truth /
own-GT artifacts (the L10-named successor), (b) record the TPOT improvement the own
artifacts buy (45.08 / 57.73 are the two worst non-consumer TPOT cells in the campaign),
(c) hold every binding gate, (d) name what remains honestly.

## Pre-registered error map (from `/tmp/am_base.predictions.json`, BEFORE any edit)

44 cells each (4 profiles x 11 concs); gate scoring = mean over rows of the within-row
turn-level MAPE; signed = 100·(pred−meas)/meas on the cell means.

**A100x2** (e2el 14.63 / ttft 23.88 / tpot 45.08): e2el median signed **−11.6 %**, 2/42
cells over — broad mild UNDER. ttft median −12.3 % (17 over / 27 under); tpot median
−9.5 % BUT **swebench tpot median +95.9 % OVER** (terminal c120: pred 141.7 vs meas
87.2 ms; the H100-inherited out28 ceiling anchor 243.1 ms never caps an A100 plateau that
measures 87–122 ms). Worst e2el cells: terminal c80 30.8, osworld c160 29.9 (ttft 4754 vs
7607 ms under), terminal c120 24.0, osworld c200 22.6, chat c200 21.8, chat c256 20.3.
Structure: TTFT-under at mid/high conc partially COMPENSATED by ceiling-uncapped TPOT-over
on swe/terminal — removing one side alone can worsen e2el (pre-registered honest-exposure
risk for P3a below).

**A100x4** (e2el 17.30 / ttft 33.18 / tpot 57.73): ttft median **+14.7 % OVER** (30/14)
while tpot median −21.6 % under (11/33) with high-conc OVER blowups where the analytic
fill runs past the ceiling: osworld c320 tpot pred 90.7 vs meas 51.9 (+75 %) → the single
worst e2el cell 77.7; swe c320 tpot 198.4 vs 69.3 (+186 %); terminal c200/c256 95.9/120.0
vs 66.8/68.4. Worst e2el cells: osworld c320 77.7, osworld c256 38.5, osworld c200 27.8,
swe c320 23.2, terminal c200 21.5, terminal c256 21.5.

**A100 tp1** (binding): e2el median −13.9 % (6/38 over), **ttft median −17.2 % under**
(8/36; per-profile medians −1.8/−17.1/−17.9/−20.2) — the same under-TTFT sign the L10 R2
scheduler-truth lever compressed on the 2080Ti. tpot median −2.3 % (healthy).

**L8/H100x4-class diagnosis (confirmed, NAMED SUCCESSOR — not actionable offline):**
A100x2/x4 `decode_grid.status = "missing"` → both run the ANALYTIC decode roofline. The
swe/terminal tpot +95..+186 % over-pricing at high conc on an analytic grid is the same
serving-vs-isolated class L8 resolved for H100x4 with a measured serving grid. The fix
this lane CAN make offline is ceiling ownership (P3, caps the blowups with this GT's own
measured plateaus); the residual mid-pressure fill shape needs a measured A100 serving
decode grid (2 free A100 GPUs, next host window — this lane may not launch).

## Pool truth (P1 evidence, harvested read-only 2026-06-11; verbatim in
`profile_data/engine_logs/a100_Llama-3.1-8B_multi_pool_evidence.txt`)

* **Exact tp2/tp4 engine logs unrecoverable** (same failure mode as L10 tp1): tp2 GT ran
  05-12 port 8089 PID 1527971, tp4 top-up 05-18 port 8090 PID 3078853 (wrapper logs
  `a100:/tmp/bench_Llama-3.1-8B_tp{2,4}_multi_vllm_p808{9,0}.log`, both
  `[sweep] captured engine version: vllm 0.19.0`); `/tmp/vllm_8089.log` and
  `/tmp/vllm_8090.log` were OVERWRITTEN 05-26 by Llama-3.3-70B sglang launches
  (`server_args=ServerArgs(model_path='/data/models/Llama-3.3-70B-Instruct'...`).
* **Same-host same-model tp1 ENGINE POOL LINES survive** (both v0.19.0, util 0.85):
  `a100:/tmp/vllm_mse.log` 05-04 (mml 32768) and `/tmp/vllm_smoke.log` 04-30 (mml 8192):
  `GPU KV cache size: 140,208 tokens` / `Available KV cache memory: 17.12 GiB` = **8763
  blocks** — max_len-INSENSITIVE at fixed util (the L9 3090 finding, reconfirmed).
* **GT-era tp1 anchor is the manifest's measured pool 8458 blocks** ("135328 tokens / 16
  @ gpu_mem=0.85", `configs/deployments/a100_llama31-8b_tp1.json`, status measured). The
  8763 − 8458 = 305-block gap back-solves to **0.6396 GB ≈ the CUDA-graph estimate** the
  GT launcher enables — the exact 1475-vs-1201 structure L10 found on the 2080ti.
* **G6 back-solve** (`configs/kv_pool.py` docstring already carries it): reserve_A100 =
  40 GiB·0.85 − 16.06e9 − 8458·2,097,152 = **2,709,510,400 B (2.710 GB)** vs the generic
  rule mean 3.5 GB. kv-shards scaling (`min(tp, 8 kv_heads)`, per-GPU bytes_per_block
  1,048,576 tp2 / 524,288 tp4):
  * tp2: floor((36,507,222,016 − 8.03e9 − 2,709,510,400)/1,048,576) = **24,574** blocks
  * tp4: floor((36,507,222,016 − 4.015e9 − 2,709,510,400)/524,288) = **56,806** blocks
* **Brackets:** lower = vLLM v1 startup feasibility at GT mml 32768 → **2048 blocks**
  (`vllm/v1/core/kv_cache_utils.py check_enough_kv_cache_memory`); upper = zero-reserve
  host arithmetic **27,157 (tp2) / 61,973 (tp4)**. Corroboration: the surviving zero-CG
  tp1 line (8763 → reserve 2.0699 GB) transfers to tp2 25,184 / tp4 58,026 — the pins sit
  +2.5 % / +2.1 % BELOW those lighter-CG ceilings, as they must (GT ran WITH the CG
  estimate). Residual uncertainty: the tp1→tp2/tp4 transfer of the per-GPU reserve
  (activation peak is max_num_batched_tokens=2048-bound → tp-mild; CG estimate per-rank
  may shrink with tp → pins are floor-leaning within the brackets).

## Scheduler truth (P2/P4 evidence — the L10-named A100 successor, executed read-only)

* **GT server metadata, all three configs** (tp1/tp2/tp4 `chat-multiturn-synth_conc10.json`
  config blocks, verbatim): `"max_model_len": 32768, "gpu_memory_utilization": 0.85,
  "max_num_batched_tokens": null, "max_num_seqs": null` → both scheduler caps ran at vLLM
  resolved defaults. `_engine_version.txt`: `backend=vllm version=0.19.0` (all three).
* **Installed engine on the GT host** (a100, vllm 0.19.0 `g2a69949bd` — same
  version+commit as the 2080ti install L10 cited), `vllm/engine/arg_utils.py
  get_batch_defaults`, **lines 2050–2072**: the explicit A100 carve-out
  (`# NOTE(Kuntai): Setting large max_num_batched_tokens for A100 reduces throughput,
  see PR #17885` … `if device_memory >= 70 * GiB_bytes and "a100" not in device_name:`
  → else `OPENAI_API_SERVER: max_num_batched_tokens=2048, max_num_seqs=256`).
  Device: `NVIDIA A100-SXM4-40GB, 39936 MiB` (nvidia-smi) — BOTH halves of the condition
  route this host to the SMALL defaults (39 GiB < 70 GiB, AND "a100" in the name).
* The sim's module constants are the H100 resolved defaults (8192 budget / 1310 chunk
  threshold / 1024 seqs). Per-config truth for the A100 family:
  `QSimSchedConfig(max_num_batched_tokens=2048, long_prefill_token_threshold=
  int(32768·0.04)=1310 (UNCHANGED vs module — A100 ran the same mml 32768),
  max_num_seqs=256)`. Net effective changes: token budget 4× down, running-set cap 4×
  down. NOTE (unlike the 2080ti case where 256 was inert): at conc 256–320 the cap
  **BINDS** (pool 24,574 blocks holds far more than 256 chat/osworld sessions) — the real
  server also capped running seqs at 256, so this is engine behavior the sim currently
  cannot express for A100.

## Pre-registered lever plan (committed BEFORE any lever; gates after EVERY lever)

Gate triple after every lever, all replay-ON `RAMP_TPOT_REQUIRE_POOLS=1`:
(a) `gate_scoped_rows --gpu-keys A100x2,A100x4`; (b) `--gpu-keys H100,H100x2` → `cmp`
byte-identical vs `/tmp/am_pair_base.predictions.json` (ALWAYS); (c) `--gpu-keys A100` →
byte-identical until P4, then within +0.3 on ALL of e2el/ttft/tpot. Binding keep/revert
rule (pre-registered): a lever that breaks (b), (c), or pushes A100x2 e2el above
min(20, 14.9301) is REVERTED and the conflict documented (campaign protocol; L6 util_bw
precedent). A100x4 ttft_cell MAY worsen under engine truth (its TTFT is already median
OVER) — kept + documented as honest exposure provided A100x4 e2el stays < 20. NO pool or
anchor scanning against gates; NO per-profile gate-picking.

* **P0 (evidence durability, L9 pattern — no code, predictions byte-identical):** commit
  `profile_data/engine_logs/a100_Llama-3.1-8B_multi_pool_evidence.txt` with the verbatim
  excerpts behind every number above: (i) installed `arg_utils.py:2050-2072` carve-out +
  nvidia-smi device line; (ii) `/tmp/vllm_mse.log` + `/tmp/vllm_smoke.log` launch ARGS +
  pool lines; (iii) tp2/tp4 GT wrapper headers (PID/port/date/engine-version) + the
  vllm_8089/8090 overwrite evidence; (iv) GT config blocks (mml 32768, util .85,
  batched/seqs None) + `_engine_version.txt`.
* **P1 (pool truth, engine-anchored reconstruction — expect SMALL moves):** pin
  `available_kv_blocks` A100x2 23820 → **24574**, A100x4 55298 → **56806** in
  `configs/deployments/a100_Llama-3.1-8B_tp{2,4}_vllm.json`; `kv_pool.status` stays
  repo-convention `derived` with source `engine-log-anchored reconstruction` (the exact
  L9-Qwen-tp2/L10 labeling), note carrying: tp1 GT-era anchor 8458 → reserve 2.710 GB
  (kv_pool.py G6 back-solve), kv-shards scaling, brackets [2048, 27157] / [2048, 61973],
  zero-CG corroboration 25,184/58,026, overwrite evidence pointer. Pre-registered
  expectation: +3.2 %/+2.7 % pool → slightly less predicted saturation/eviction → TTFT
  drifts DOWN a touch (direction-worse for x2's under-TTFT, direction-better for x4's
  over-TTFT); |Δ e2el| ≲ 1 pt each; the ceiling anchors are INSENSITIVE to this pool move
  (probed read-only: A100x2 anchors 109.1/122.5 @ old pool → 109.2/121.9 @ new; A100x4
  73.3 → 73.3). No owned-artifact regen needed (A100x2/x4 own NO ceiling yet; tp1/H100
  artifact inputs untouched).
* **P2 (scheduler truth, A100x2 + A100x4 — the L10 QSimSchedConfig pattern):** pin
  `max_model_len: 32768`, `max_num_seqs: 256` in both manifests (+ `data.scheduler`
  provenance note with the verbatim citations; top-level `max_num_batched_tokens` is
  ALREADY the verified 2048). Pre-registered expectation: TTFT predictions RISE at c ≥ 5
  (budget 8192→2048: ~4× slower per-step prefill advancement; seqs cap binds at
  c256/320). A100x2 (TTFT median −12.3 % under) → ttft_cell and e2el IMPROVE. A100x4
  (TTFT median +14.7 % over) → ttft_cell WORSENS (honest exposure, engine truth), e2el
  net effect uncertain (its e2el is median −10.3 % under; composition decides — gate
  records it). TPOT cells EXACTLY unchanged (TTFT-only lever by construction).
* **P3 (own GT-builder ceilings — manifest flips + established `build_saturated_ceiling`):**
  * **P3a A100x2:** flip `data.saturated_ceiling` → measured +
    `profile_data/kernels/saturated_ceiling_A100x2_llama31_8b.json`; pre-probed (read-only,
    at the P1 pool): 143 saturated turns → anchors out28 → **109.2 ms** (n=121), out88 →
    **121.9 ms** (n=22), replacing the inherited H100 243.1/134.9 that never caps an A100x2
    plateau. Expect: swe/terminal tpot over-pricing collapses (tpot_cell 45.08 drops
    SHARPLY); coupled risk: faster predicted drain → TTFT down on the already-under side,
    and the swe compensation (tpot-over masking ttft-under) is removed → e2el may move
    EITHER way. Keep/revert per the binding rule above; if e2el worsens within bounds but
    tpot improves, KEep — e2el goal metric still gates the keep via the
    min(20, baseline+0.3) rule for x2.
  * **P3b A100x4:** same flip → single-cluster artifact out28 → **73.3 ms** (n=27 — THIN,
    flagged in the artifact like L9's 3090x4 n=5 precedent; the out ≥ 50 cluster is empty
    at this GT). Expect: caps the osworld/swe c256–320 blowups (90.7/198.4 → ~73) →
    osworld c320 e2el 77.7 collapses; tpot_cell 57.73 drops; same coupled-TTFT caveat.
  * All OTHER ceiling artifacts must regenerate byte-identical (their pools/GT untouched).
* **P4 (scheduler truth, A100 tp1 — the prompt's "P3c"; the ONLY lever allowed to move
  the binding config):** same manifest pins on `a100_llama31-8b_tp1.json`. tp1 TTFT is
  median −17.2 % UNDER → pre-registered expectation: ttft_cell 22.22 IMPROVES, e2el 15.87
  improves-or-flat, tpot EXACTLY unchanged (14.3728). BINDING: all three within +0.3; on
  violation revert P4 ONLY and document (successor: the tp1 TTFT drain-forecast lever).
* **Finalize:** full pytest; final gate triple re-run at HEAD → `/tmp/am_r1.*`,
  `/tmp/am_a100_r1.*`, pair cmp; results + keep/revert audit appended below; lane summary.

**NOT in scope (named, not guessed):** (1) measured A100 serving decode grid (the
L8-class fill shape behind swe/terminal tpot-over at mid pressure — needs 2 free A100s,
forbidden this round); (2) G7 roofline-utils measurement (runbook already pinned,
DEFERRED — host busy); (3) exact tp2/tp4 engine pool lines (one server start each at GT
flags, next host window — removes the reconstruction brackets); (4) consumer/A100 prefill
law (cached_prefill_grid/fa3_grid are H100-measured; tp1 c1-c10 TTFT residuals live
there).

(Execution results are appended below by the rounds; everything above this line was
committed before any lever was touched.)

## Round 1 execution record (2026-06-11; gates all replay-ON, RAMP_TPOT_REQUIRE_POOLS=1)

**P0 (committed `05c6723`):** evidence file in; no code; predictions byte-identical by
construction (the pre-edit reproduction run at HEAD was already byte-identical).

**P1 (committed `d590044`): KEPT.** Pools 23820 → 24574 / 55298 → 56806. Gates: pair
**byte-identical**; A100 tp1 **byte-identical**; A100x2 e2el 14.6301 → 14.9172 (+0.287,
inside the +0.3 binding bound; ttft 23.88 → 24.10, tpot 45.08 → 44.73), A100x4 e2el
17.3047 → **16.3966** (−0.91; ttft 33.18 → 32.93, tpot 57.73 → 56.18). Both moved in the
pre-registered directions with the pre-registered small magnitudes.

**P2 (scheduler truth tp2/tp4): REVERTED per the pre-registered binding rule —
resolved-as-compensating-fit (the L6 outcome class).** With the engine-true
QSimSchedConfig (budget 2048, seqs 256), A100x2 e2el 14.9172 → 16.2914 (> the
min(20, 14.9301) bound) and A100x4 16.3966 → 20.0168 (> 20); pair byte-identical;
tpot cells EXACTLY unchanged (TTFT-only, as constructed). Diagnosis (per-cell, committed
before the next lever): the budget cut OVERSHOOTS TTFT through zero on A100x2 — signed
ttft median −13.5 % → **+10.4 %** (swe c320 pred 49.7 s → 67.0 s vs meas 57.8 s — the
under-cells flip to bigger over-errors), and on A100x4 amplifies the already-over side
+13.5 % → +26.4 %. Mechanism: the queue sim's DRAIN on these two configs prices decode
with the ANALYTIC roofline + H100-inherited ceiling (out28 anchor 243.1 ms vs measured
A100x2 plateaus 87–122 ms) — the drain is too SLOW, and the H100 8192-token budget was
acting as a compensating fit for it. Removing the compensation alone breaks the binding
gates. Exactly the L4/L6 pairing: the engine truth cannot land until the coupled
mispriced term (the drain) is fixed.

### Pre-registered continuation (committed BEFORE P3): P3 ceilings, then RE-GATE P2 on top

1. **P3a/P3b (own ceilings, as pre-registered above):** flip
   `data.saturated_ceiling` → measured for A100x2 and A100x4, run the established
   `build_saturated_ceiling` (expected artifacts: A100x2 out28→109.2 n=121 /
   out88→121.9 n=22; A100x4 single-cluster out28→73.3 n=27, THIN-flagged). All other
   ceiling artifacts must regenerate byte-identical. Expected: the swe/terminal
   high-conc TPOT over-pricing collapses → tpot_cell drops on both; the queue-sim drain
   speeds up → TTFT predictions DROP broadly (direction-worse for x2's under-TTFT at
   mid conc, direction-better for x4's over-TTFT) — e2el net recorded by the gate.
   Keep/revert per the same binding rule.
2. **P2′ (the L4-precedented re-gate, pre-registered now):** with the drain on own
   measured plateaus, RE-APPLY the unchanged engine-truth scheduler pins (identical
   edits, no new constants) and re-gate. If the composition holds the binding rule
   (A100x2 e2el ≤ 14.9301 absolute bound stays the P1 reference — i.e. ≤
   baseline+0.3 — and A100x4 < 20), the compensating fit retires and engine truth
   lands; if it still breaks, P2 stays reverted and the A100 scheduler truth is
   re-registered as BLOCKED-ON the measured A100 serving decode grid (named successor,
   needs the 2 free A100s).
3. **P4 unchanged** (tp1 scheduler truth, ±0.3 binding): tp1 prices decode with a
   MEASURED grid + its OWN measured ceiling (175.4/125.8) — the compensation mechanism
   above does not apply there; its TTFT is genuinely median-under. Decision by its own
   gate.

## Round 1 results — P3, P2′, P4 (gates replay-ON, RAMP_TPOT_REQUIRE_POOLS=1)

**P3a/P3b (committed `2bf9cfb` with P2′):** `build_saturated_ceiling` produced exactly the
pre-probed artifacts (A100x2 109.2 n=121 / 121.9 n=22; A100x4 single-cluster 73.3 n=27,
THIN-flagged); all eight other ceiling artifacts regenerated byte-identical. TPOT cells
collapsed as pre-registered: A100x2 44.73 → **18.40**, A100x4 56.18 → **28.93**.
**Pre-registered expectation FALSIFIED and recorded:** TTFT cells were EXACTLY unchanged
under P3 (24.0972 / 32.931) — the ceiling clamps `kernel_tpot` only; the queue sim's
drain prices decode through `decode_step_ms` directly (gate_scoped_rows docstring), so
the "faster drain → lower TTFT" coupling does not exist in the composed system. E2EL:
A100x4 16.40 → 15.33 (P3b is independently keepable); A100x2 14.92 → 16.74 — the
swe/terminal TPOT-over compensation removed, VIOLATING x2's binding bound standalone.

**P2′ (the pre-registered L4-style re-gate; same commit):** identical engine-truth pins
re-applied on top of P3. Composition passes every binding rule — and is the ONLY
admissible adopted set: P3a alone breaks x2 (16.74 > 14.93), P2 alone breaks both
(16.29 / 20.02), together they fix opposite signed sides (TPOT-over cut by the own
ceiling, TTFT-under raised by the engine budget):

| config | metric | baseline | final | delta |
|---|---|---|---|---|
| A100x2 | e2el_cell | 14.6301 | **14.2759** | −0.35 (< 20, below baseline) |
| A100x2 | ttft_cell | 23.8786 | 24.3284 | +0.45 (disclosed trade; e2el is the gated goal metric, L3-precedent disclosure) |
| A100x2 | tpot_cell | 45.0845 | **18.3993** | −26.69 |
| A100x4 | e2el_cell | 17.3047 | **17.5061** | +0.20 (goal < 20 MET; P3b-only state was 15.33 — the +2.2 give-back is the price of engine truth on the TTFT side, adopted per the pre-registered composition rule, NOT unbundled per-config) |
| A100x4 | ttft_cell | 33.1784 | 37.3608 | +4.18 (honest exposure: TTFT was already median-OVER (+14.7 %); pre-registered acceptable while e2el < 20) |
| A100x4 | tpot_cell | 57.7320 | **28.9292** | −28.80 |

**P4 (committed `549d49e`): KEPT — improves the BINDING config.** A100 tp1 e2el
15.8653 → **12.2162** (−3.65), ttft 22.2222 → **20.1710** (−2.05), tpot EXACTLY
unchanged 14.3728 (TTFT-only by construction). All three within +0.3 (all improve);
the pre-registered under-TTFT compression landed.

**Finalize gates at HEAD:** pair H100/H100x2 **byte-identical** to
`/tmp/am_pair_base.predictions.json` after EVERY lever including final
(`/tmp/am_pair_r1.*`); A100 tp1 byte-identical until P4 as contracted; final artifacts
`/tmp/am_r1.{predictions,metrics}.json` (A100x2/x4) and
`/tmp/am_a100_r1.{predictions,metrics}.json` (tp1); `/tmp/am_r1.predictions.json` is
byte-identical to the P2′ gate output (P4 touches no x2/x4 input — confirmed by `cmp`).
Full pytest: **389 passed, 1 skipped, 15 subtests passed** (the same counts as the L10
finalize; an initial run WITH `RAMP_TPOT_REQUIRE_POOLS=1` exported showed 3 failures in
tests that deliberately exercise the missing-pool fallback path — the env var escalates
that path to a hard error by design; the suite contract is to run without it).

**Lever audit (keep/revert):** P0 evidence KEPT (no code); P1 pools KEPT (gated);
P2 standalone REVERTED (compensating-fit conflict, L6 class) then ADOPTED as P2′ in the
pre-registered composition with P3; P3a/P3b KEPT (composition / standalone); P4 KEPT
(binding config improves). Nothing else changed.

**Lane outcome: SUCCESS.** A100x4 e2el 17.51 < 20 (goal), A100x2 14.28 < 20 and below
baseline, A100 tp1 12.22 (−3.65 on a binding config), pair byte-identical, TPOT
improvements recorded (−26.7 / −28.8 pts). The A100 family now runs on engine-truth
pools (engine-log-anchored), engine-truth scheduler (vLLM PR #17885 carve-out), and
own-GT ceilings.

**Named successors (the honest remainder):**
1. **Measured A100 serving decode grid** (the L8 class): both multi configs still price
   decode with the ANALYTIC roofline (`decode_grid: missing`); the residual A100x4
   ttft_cell 37.4 and the swe/terminal mid-pressure fill shape live there. Needs 2 free
   A100 GPUs (this lane was forbidden to launch); the L11 serving-grid tooling is the
   vehicle.
2. **Exact tp2/tp4 engine pool lines** at GT flags (one server start each, next host
   window) — removes the reconstruction brackets [2048, 27157] / [2048, 61973].
3. **G7 A100 roofline utils** (placeholders; runbook already pinned, host busy).
4. **A100x4 ceiling is THIN** (n=27, single out28 cluster) — re-anchor when more
   saturated GT lands (L9 3090x4 precedent).
5. **A100 prefill law** is still H100-measured (cached_prefill_grid/fa3_grid inherited);
   tp1 c1–c10 TTFT residuals and the x2/x4 low-conc cells live there.

## Round 2 pre-registration (2026-06-11; committed BEFORE the gate — L10 R3 protocol)

**Prior-round feedback:** the round-1 lane result was not consumed — the orchestrator
report was missing the gate `metrics` payload (fails: `["metrics MISSING"]`; pair=true).
The committed work itself passed every binding gate (record above).

**Lever decision: NO new lever this round (verification-only).** Re-audited the
remainder against the lane constraints: every named successor (1)–(5) above requires a
GPU launch (serving decode grid; tp2/tp4 engine pool lines at GT flags; G7 utils run;
more saturated GT for the THIN x4 ceiling; A100 prefill-law measurement) — all forbidden
while 7/8 a100 GPUs run the benchmark sweep, and no new read-only evidence has appeared
since the round-1 harvest. Applying any further offline constant without such evidence
would be fitting (campaign rule). Exactly the L10-R3 situation: the honest move is a
verification round.

**Pre-registered expectation:** the working tree at HEAD `f409f52` is clean and
untouched since the round-1 finalize gates, so the round-2 gate triple must be
**byte-identical** to the round-1 artifacts on both predictions and metrics:
`/tmp/am_r2.*` ≡ `/tmp/am_r1.*`, `/tmp/am_a100_r2.*` ≡ `/tmp/am_a100_r1.*`, pair ≡
`/tmp/am_pair_base.predictions.json`. Any divergence = environment drift → STOP and
diagnose before reporting. Full pytest at the end (suite contract: WITHOUT
`RAMP_TPOT_REQUIRE_POOLS` exported; the gates themselves run WITH it).
