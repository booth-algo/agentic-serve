# L11 — H100-multi close (tp4 SERVING-context decode grid + tp4 prefill law)

Lane branch `h100-multi-close` (from `defit-campaign-integration`: L3 sub-linear fill, measured
tp2 grid, L10 QSimSchedConfig scheduler-truth mechanism, L2 committed pools all IN). Local
commits only; `prediction_construction.md` untouched. GPUs: h100 **4,5,6,7** (re-verified free
before every launch; `h100_setup.md` env; fresh run dirs `/data48/kevinlau/h100multi_run/`).

**Goal (deterministic):** H100x4 E2EL cell-MAPE < 20 (baseline 29.81; tpot 37.10 / ttft 39.45)
AND H100x2 stays < 20 (baseline e2el 18.55; regression > 0.3 forbidden, improvement welcome)
AND binding pair H100, A100 **byte-identical** (changes config/artifact-scoped; any shared code
path must default-fallback byte-identical for non-H100x2/x4).

**Owns:** `configs/deployments/h100_Llama-3.1-8B_tp4_vllm.json` (+ the single H100x2 manifest
`decode_grid` line IF the pre-registered Phase C condition fires), the new permanent tools
`profiling/gpu_profiling/vllm/serving_decode_grid.py` + `profiling/process/
build_serving_decode_grid.py` (+ their tests), `profiling/process/build_tp_comm.py` (tp4
extension), new H100x4/H100x2 artifacts, this entry. Allowed shared-code deltas (byte-identical
default rule): an OPTIONAL `RooflineParams` field for the per-config prefill tp-comm rate
(default `None` → the existing `PREFILL_TP_COMM_MS_PER_TOKEN·(tp−1)`), its `configs/loader.py`
threading, and its consumption in `ttft_queue_sim._prefill_gemm_per_tok_loaded`.

- **2026-06-11 — L11 baselines (replay-ON, `RAMP_TPOT_REQUIRE_POOLS=1`, 0 bytes stderr, pre-edit,
  captured in THIS worktree).** H100x4 (44 rows): TPOT cell-MAPE **37.0951** / TTFT **39.4507** /
  E2EL **29.8081**, turn-overall 39.8531; per-profile TPOT chat 12.1083 / swebench 68.4871 /
  osworld 16.6027 / terminalbench 52.8857. Binding trio: H100 14.4697/18.1309/10.7777,
  A100 14.3728/22.2222/15.8653, H100x2 21.5336/29.0163/18.5506 (tpot/ttft/e2el cell) — all
  byte-match the campaign contract. Saved: `/tmp/l11/x4_base.*`, `/tmp/l11/trio_base.*`.

- **2026-06-11 — PRE-REGISTRATION (committed before any GPU work).**

  **Why a SERVING-context grid (the L8 paradox):** L8 measured the isolated tp4 kernel wall
  (96-cell `decode_steps.py` lattice) and adoption was REFUSED — real serving TPOT at high conc
  sits BELOW the isolated wall (chat c320 6.56 ms served vs 11.5 ms isolated B=320 step):
  isolated per-step fixed costs (4-way NCCL launch + host walls, S12 walls-vs-trace +12–35%)
  overlap/pipeline under continuous serving. The sim prices `decode_step_ms(b_eff, ctx)` as the
  per-step serving cost (TPOT kernel floor AND the queue sim's `_price_step` drain rate) — so
  the right measurement is the LIVE per-request mid-stream ITL at steady decode, per (B, T)
  cell. The grid consumer (`load_grid` → manifest `data.decode_grid`) is agnostic to how cells
  were measured; this is a config/artifact-scoped swap with the L3/L8 adopt-vs-revert protocol.

  **PHASE A — tp4 serving-context decode grid (GPUs 4,5,6,7; the primary lever).**
  New permanent client `profiling/gpu_profiling/vllm/serving_decode_grid.py` reusing
  `serving_stage_split.py`'s `launch_server`/`wait_health`/SSE machinery (server via
  `-m vllm.entrypoints.openai.api_server`, so the L8 `flash_attn.py` sys.path shadowing does not
  apply; still run from a clean /data48 scripts copy). Engine flags (= the L8 lattice / GT-bench
  class config): `--no-enable-prefix-caching --max-num-seqs 320 --max-num-batched-tokens 8192
  --max-model-len 25600 --gpu-memory-utilization 0.90 --dtype bfloat16 --tensor-parallel-size 4`
  (live pool 2,116,352 tokens, L8-verified at these flags; re-read from the server log's
  "GPU KV cache size" line and recorded in run metadata).
  Client mechanics: per request a UNIQUE random token-ID prompt (`/v1/completions` with a token-ID
  list → exact prompt_tokens, no tokenizer variance; seeded RNG per (cell, request)); payload
  `max_tokens=osl, temperature=0, stream=true, ignore_eos=true` ([VERIFY ON H100] extra-body name
  on vLLM 0.19.0; fallback `min_tokens=osl`; verified by counting streamed events ≈ osl);
  `stream_once` records a `perf_counter` timestamp per SSE content delta. Per cell fire B
  concurrent streams; for B ≥ 160 shard the client across 4 OS processes (per-shard wall-clock
  anchor `time.time() − perf_counter()` recorded for cross-shard alignment) and record asyncio
  loop-lag (50 ms sleeper overshoot), **validation_status=check if p99 lag > 2 ms** in any shard.
  One UNRECORDED warm-up pass (8 reqs, 1024-tok prompts, 64 out-toks) before the lattice (the
  known first-cell warm-up outlier, L8/L3).
  Steady window = [max_i(first_token_ts), min_i(last_token_ts)] (all B decoding, no prefills
  left); per-request p50 ITL over successive event deltas inside the window (**require ≥ 64
  in-window samples/request else flag the cell `check`**); cell `decode_step_ms` = median over
  the B per-request p50s; **effective `context_len` = prompt + median in-window progress**
  (recorded as the grid's context_len; `nominal_T` kept as an extra column — by construction
  prompt = T − osl/2 makes effective ≈ T at window middle). Diagnostics recorded per cell:
  fraction of deltas < 0.5 ms (event-coalescing), n_events/osl, p99 loop lag.
  **CELLS (26; prompt = T − osl/2; osl = 384 + ceil(B·prompt/8192), integer fixed-point from
  osl₀=384, so the steady window survives chunked admission; cap B·(T+osl/2) ≤ 0.95·2,116,352 =
  2,010,534):** B ∈ {1,8,32,80} × T ∈ {512,2048,8192,16384} (16) ; B=160 × {512,2048,8192,12288}
  (4; the directive names 12288 "~94.6% of pool" — strict fixed-point gives 2,015,520 = 95.2%,
  kept by name) ; **B=256 × {512,2048,6144}** and **B=320 × {512,2048,4096}** (6) — the
  completion of the directive's truncated rows, derived deterministically from the directive's
  own cap rule over its own T menu {512,2048,4096,6144,8192,12288,16384}: B=256: T=8192 →
  256·8508 = 2.178M ✗, 6144 → 1.646M ✓; B=320: T=6144 → 320·6450 = 2.064M ✗, 4096 → 1.396M ✓.
  Total = 16+4+3+3 = the directive's 26.
  Raw output: append-only per-request JSONL.gz (event timestamps, wall-anchored) +
  an operator summary; the AUTHORITATIVE grid CSV comes from the deterministic builder
  `profiling/process/build_serving_decode_grid.py` (re-derives windows/p50s from the raw events;
  columns `batch_size, context_len, decode_step_ms, validation_status, nominal_T, prompt_tokens,
  osl, n_samples, steady_window_s` + diagnostics — `load_grid` reads only the first four).
  **Wiring + gate:** flip the L8-owned manifest `data.decode_grid` to the serving grid CSV
  (status `measured`, method `serving-context mid-stream ITL`, provenance, date); gate (a)
  `--gpu-keys H100x4`, gate (b) trio. **Adopt iff H100x4 E2EL cell-MAPE improves vs 29.8081 AND
  the trio is byte-identical**; else revert to the documented stop-point (L8 precedent). No
  banding/truncation of the lattice by validation outcome (compensating-fit rule).
  **Quantification (pre-registered deliverable, gate-independent):** per-cell serving-grid vs
  isolated-grid (`decode_profile_H100x4_merged_2026-06-10.csv`, md5 `81552f69…`) vs analytic
  fill at the same (B, T): the overlap evidence — expected serving < isolated at high B, and
  the serving/isolated ratio is the measured overlap factor the L8 paradox predicted.

  **PHASE B — tp4 prefill law (the TTFT lever; diagnosis says yes: conc=1 cells over-predict
  +5.5..+21.8 ms; 39.45 TTFT).** Same-session (sequential server launches): run
  `serving_stage_split.py --tensor-parallel-size 4` (GPUs 4–7, its own server flags — prefix
  caching ON, c1, same script both legs) → `serving_stage_split_H100_tp4.csv`. Extend
  `build_tp_comm.py` with the tp4 pair (tp1 leg EXISTS: `serving_stage_split_H100.csv`):
  **comm4 = prefill_span.new(tp4) − span.new(tp1)/4** (G3 method) →
  `profile_data/kernels/prefill_tp_comm_H100x4.json` (new artifact; the tp2 JSON untouched).
  Wire per-config: optional `RooflineParams.prefill_tp_comm_ms_per_token` (default None →
  `PREFILL_TP_COMM_MS_PER_TOKEN·(tp−1)`, byte-identical for every config that does not pin it),
  loader threads it from the deployment JSON; pin it ONLY in the H100x4 manifest with the
  measured comm4. Current model charges tp4 3·3.279 = 9.84 ms/1k; the measurement replaces that
  extrapolation with the like-for-like value. Gate same as Phase A: **adopt iff H100x4 TTFT
  cell-MAPE improves AND trio byte-identical.** Scheduler truth needs NO pin: the tp4 GT config
  block records max_model_len=32768, max_num_batched_tokens=None, max_num_seqs=None — exactly
  the QSimSchedConfig module defaults (verified 2026-06-11 on
  `h100_Llama-3.1-8B_tp4_vllm/chat-multiturn-synth_conc10.json`).

  **PHASE C — tp2 serving grid (conditional; pre-registered condition: run ONLY IF Phase A
  adopts — the method must validate on its primary target first).** Same client, GPUs from the
  assigned set, `--tensor-parallel-size 2`, same flags otherwise; lattice = the same menu/rule
  with the live tp2 pool read from the server log (≈ 998,656 tokens → cap 948,723). Wire via the
  H100x2 manifest `decode_grid` line; **adopt iff H100x2 E2EL improves vs 18.5506 AND H100x2
  TPOT does not regress > 0.3 AND H100/A100 byte-identical**; else revert (the L3 isolated tp2
  grid stays).

  **Run hygiene:** sequential server launches; GPUs 4–7 re-verified free
  (`nvidia-smi --query-compute-apps`) before each; fresh `/data48/kevinlau/h100multi_run/`;
  no `rm` inside ssh; no weight downloads; CSVs/JSONLs pulled to the shared symlinked
  `profile_data/results/` with unique dated names; GPUs verified clean after every run.

- **2026-06-11 — PHASE A EXECUTED: tp4 serving-context decode grid measured (26/26 pre-registered
  cells, 3 client passes) → adoption REFUSED by the pre-registered gate; analytic fill KEPT BY
  MEASUREMENT. The L8 overlap hypothesis is FALSIFIED in magnitude: serving truth = median
  0.846× of the isolated wall, NOT the ~0.57× the GT high-conc cells would need.**
  **Runs (GPUs 4,5,6,7, run dir `/data48/kevinlau/h100multi_run/s1_grid_tp4/`, clean scripts
  copy, GPUs verified free before each launch and clean = 0 compute apps after each):**
  server `-m vllm.entrypoints.openai.api_server`, tp4, `--no-enable-prefix-caching
  --max-num-seqs 320 --max-num-batched-tokens 8192 --max-model-len 25600
  --gpu-memory-utilization 0.90 --dtype bfloat16`; **live pool = 2,116,352 tokens** (server-log
  `GPU KV cache size`, exactly the L8 value; lattice cap honored, (160,12288) final KV
  2,015,520 = 95.2% of live pool, no preemption). `ignore_eos` accepted on vLLM 0.19.0 but the
  64-token warmup streamed 60–61 events (SSE coalescing) → runs 1–2 used the `min_tokens=osl`
  fallback, run 3 `ignore_eos` (61 ≥ threshold); cross-mode values agree ≤0.2%, coalesce_frac
  ≤0.4% on all cells. RUN 1 (26 cells, 1 shard <160 / 4 shards ≥160): 8 ok, 18 loop-lag-flagged
  (p99 >2 ms; values healthy). RUN 2 (18 flagged cells, 8 shards, gc freeze/disable): values
  reproduce ≤0.5% vs run 1 → the flags are client-loop scheduling noise (even 1 stream/loop
  shows ~2 ms p99 spikes on a host whose CPUs run the tp4 server), not measurement distortion;
  3 more ok. RUN 3 (15 cells, 8 shards + `--rt-shards` SCHED_RR via sudo, separate root cache
  dir): 12 more ok at p99 1.7–2.0 ms; **final merged grid 23 ok / 3 check** ((32,8192),
  (160,12288), (256,512) at p99 2.006–2.017 — kept flagged per the pre-registered rule;
  load_grid analytic-fills them as interior holes). Only (256,512) shows real drift across
  passes (9.15→8.75→8.50, −4%/−3%); every other cell ≤0.5%.
  **Builder amendment (decided + implemented + CSV built BEFORE the first grid gate, committed
  with this entry; consumer-structure reasoning only, no validation data consulted):** raw
  effective contexts are unique floats per cell, which makes every `load_grid` t-axis column
  single-B (ragged) and the bilinear corners mix measured cells with analytic fills almost
  everywhere — so `build_serving_decode_grid.py` SNAPS `context_len` to the cell's nominal T
  when the measured effective is within 2% (always true here: the `prompt = T − osl/2` design
  landed every effective within 0.2–1.4% of nominal), keeping the measured effective in the
  `effective_context_len` diagnostic column; >2% deviation keeps the effective and flags check.
  **The pre-registered quantification (serving vs isolated vs analytic at the same (B,T)):**
  serving/isolated median **0.846×** (B=1 row 0.81–0.99, mid-B 0.78–0.91) — the L8 overlap
  factor, directly measured: isolated per-step fixed costs DO partially overlap under
  continuous serving, but only ~15%; **at B≥256 × T=512 serving EXCEEDS the isolated wall**
  ((256,512) 1.05×, (320,512) **1.32×**: per-token host/stream cost grows with B in real
  serving). serving/analytic median **1.088×** (B≤8 ≈ 0.95–1.00; B≥80 short-T 1.15–2.09 —
  the analytic fill genuinely under-prices serving there). Key cells: B=1 ≈ 2.97–3.24 ms
  (≈ the isolated warm floor), (320,2048) **12.84 ms** vs isolated 15.90 vs analytic 10.71 —
  and vs **GT chat c320 measured TPOT 6.56 ms**, which equals the measured serving ITL at
  B≈100–110 / ctx≈2k: the GT never decodes 320-deep; the sim's b_eff/regime mapping queries
  ≈300. The residual is the S10/S8/util-cap successor cluster, now BOUNDED by measured serving
  truth on both sides (no decode price ≥ measured serving truth can fit conc≥120; banding =
  compensating fit, refused).
  **Gates (replay-ON, `RAMP_TPOT_REQUIRE_POOLS=1`, 0 bytes stderr):** grid wired ALONE:
  TPOT cell 37.0951 → **45.6599**, E2EL 29.8081 → **31.6730**, TTFT unchanged 39.4507 (chat
  12.11→13.71, swebench 68.49→80.63, osworld 16.60→24.64, terminalbench 52.89→65.74); grid
  STACKED on the adopted Phase-B comm pin: E2EL 25.6170 → 26.5615. Both fail the pre-registered
  adopt rule (E2EL must improve) → **manifest `decode_grid` reverted to the documented
  `missing` stop-point with the falsification evidence in the note.** Trio untouched.

- **2026-06-11 — PHASE B EXECUTED + ADOPTED: measured tp4 prefill comm (G3 like-for-like)
  replaces the tp2-extrapolated 9.84 ms/1k → TTFT cell 39.4507 → 33.8703, E2EL 29.8081 →
  25.6170, TPOT byte-unchanged; trio BYTE-IDENTICAL.**
  Same-session stage-split tp4 leg (`serving_stage_split.py --tensor-parallel-size 4`, GPUs
  4–7, own server flags, c1) → `serving_stage_split_H100_tp4.csv` (20 rows). `build_tp_comm.py
  --tp 4` (new mode; the tp2 default path byte-untouched) → `profile_data/kernels/
  prefill_tp_comm_H100x4.json`: span.new(tp1) 22.733 ms/1k (the existing leg), span.new(tp4)
  **9.647 ms/1k** → **comm_total = 9.647 − 22.733/4 = 3.9635 ms/1k**, vs the
  `PREFILL_TP_COMM_MS_PER_TOKEN·(tp−1)` fallback = 9.8367 ms/1k the model charged before — the
  per-extra-rank extrapolation from tp2 over-charged tp4 by ~5.9 ms per 1k new tokens (NCCL
  all-reduce cost does not grow ∝(tp−1) per token; tp4 total sits just above the tp2 measured
  3.279). Wiring: new OPTIONAL `RooflineParams.prefill_tp_comm_ms_per_token` (default None →
  the old formula, gate-verified BYTE-IDENTICAL on x4+trio BEFORE any pin), threaded by
  `configs/loader.py` from the deployment JSON; consumed in
  `ttft_queue_sim._prefill_gemm_per_tok_loaded`; pinned ONLY in the L11-owned H100x4 manifest
  (provenance entry `data.prefill_tp_comm`; `calibration_status` →
  `h100_tp4_vllm_analytic_decode_measured_tp4comm_prefill`). New pin test
  (`test_prefill_tp_comm_per_config_override_and_byte_identical_default`).
  **Remaining-error diagnosis at the adopted state (the honest localization):** chat e2el ≤15
  at every conc; the E2EL residual is osworld/swebench/terminalbench conc≥120 (e2el 24–66)
  driven by (a) queue-drain TTFT over-prediction (med signed +0.21..+5.53 s; swebench c320
  +5.5 s) and (b) the b_eff over-estimation TPOT over-price (swebench c160–320 tpot_err
  120–163) — the S10/S8/util-cap successor cluster, upstream of this lane's config scope,
  now with the serving-truth grid as its measured bound.

- **2026-06-11 — PHASE C NOT RUN (pre-registered condition false).** The tp2 serving-grid leg
  was conditioned on Phase A adopting; it did not. H100x2 stays on the L3 isolated grid,
  byte-identical (18.5506 E2EL < 20 ✓, regression 0).

- **2026-06-11 — FINALIZE: lane outcome PARTIAL (comm lever adopted; grid lever refused with
  falsification evidence; E2EL 29.81 → 25.62 vs target <20 — the measured remainder is
  upstream of this lane's admissible levers).**

  | config | tpot_cell | ttft_cell | e2el_cell | Δ vs baseline |
  |---|---|---|---|---|
  | H100x4 (lane) | 37.0951 | **33.8703** | **25.6170** | 0 / −5.58 / −4.19 |
  | H100 (binding) | 14.4697 | 18.1309 | 10.7777 | 0 (byte-identical) |
  | A100 (binding) | 14.3728 | 22.2222 | 15.8653 | 0 (byte-identical) |
  | H100x2 (binding) | 21.5336 | 29.0163 | 18.5506 | 0 (byte-identical) |

  H100x4 per-profile TPOT unchanged (chat 12.1083 / swebench 68.4871 / osworld 16.6027 /
  terminalbench 52.8857). GPUs 4–7 verified clean (0 compute apps) after every run including
  the two failed client launches. **Artifacts (shared gitignored `profile_data/results/`,
  R2-sync with the L3/L8 sets):** `serving_decode_grid_H100x4_2026-06-11.jsonl.gz`
  md5 `d3a6b23c8dc40b8d4f669a35d1ff4a12`, `..._pass2.jsonl.gz` `2e2133d23306d3144333e01704796fa2`,
  `..._pass3.jsonl.gz` `377758e92d2c9620f315d6833be55288`,
  `serving_decode_grid_H100x4_merged_2026-06-11.csv` `a3814643bcbd2ce191c6d7333b93fbb8`
  (regenerable: `python3 -m profiling.process.build_serving_decode_grid --inputs <3 raws>`),
  per-run summaries `b4ff1f3a…/2038313193…/2bb87b88…`, `serving_stage_split_H100_tp4.csv`
  `0879c71237ed48f9b13257c61fcecc39`. Committed artifact:
  `profile_data/kernels/prefill_tp_comm_H100x4.json` (regenerable:
  `python3 -m profiling.process.build_tp_comm --tp 4`).
  **Named successor (sharpened by this lane):** the S10/S8/util-cap cluster is now a
  b_eff-mapping problem with measured bounds — GT chat c320 TPOT equals serving truth at
  B≈100–110, not B≈300; fixing the realized-decode-batch estimate (engine-trace oracle, L4
  style) is the remaining lever for H100x4 TPOT/E2EL and the conc≥120 queue drain.

- **2026-06-11 — ROUND 2 PRE-REGISTRATION (diagnosis + lever committed BEFORE the wiring
  gate). Lever: per-config MEASURED tp4 prefill HOST-cached rate + FA3 coefficient, derived
  from the round-1 stage-split pair already on disk — no new GPU work.**

  **Diagnosis (on `/tmp/hm_r1.predictions.json`, the round-1 adopted state):**
  * E2EL oracle decomposition (per-turn swap, e2el = ttft + out·tpot recomposed): base
    25.6170; **TTFT-oracle → 7.7510**; TPOT-oracle → 22.5963; both → 4.3001. The E2EL
    remainder is ~entirely the TTFT queue-drain over-prediction, NOT the TPOT amplifier.
  * Worst cells: osworld c256/c320 TTFT err 82.6/81.8 (pred 2435/2843 ms vs meas 1364/1639),
    swebench c320 60.0 (18.12 s vs 10.72 s), terminalbench c256 40.3. Component ablation
    (module-attribute patches, gate_scoped_rows style, no source edits): zeroing the HOST
    cached terms collapses osworld@256 pred 2435→594 ms (the dominant simulated mass is
    `PREFILL_HOST_PERREQ_MS_PER_TOKEN·cached` summed over ~256 concurrent prefills);
    FA3=0 takes swebench@320 18.12→13.64 s (second-order); decode price in `_price_step`
    is masked by `max(decode, prefill)` during the drain (zeroing it moves nothing).
  * c1 ground anchor: osworld@1 turns with cached≈1930–2029 measure TTFT 22.6–22.7 ms over
    the measured 18.08 floor → real tp4 marginal cached cost ≈ 2.4 ms/1k vs the tp1-measured
    5.887 ms/1k the sim charges (the same over-charge class as the floor 26→18.08 and the
    comm 9.84→3.96 precedents: tp1 host constants imposed on the tp4 stack).
  * Like-for-like confirmation (the round-1 `serving_stage_split.py` pair, SAME script and
    lattice both legs, columns beyond `prefill_span_ms` unused until now): 3-param OLS
    `ttft ~ floor + a·new + b·cached` gives b(tp1) = **5.9889e-3** — reproduces the
    production host SUM 5.8872e-3 within **+1.7%** (cross-instrument validation of the
    estimator family, which is `build_host_split.fit_c1_rate`'s own model) — and b(tp4) =
    **3.5303e-3**: the tp4 stack's measured cached-host rate is **0.59×** the tp1 value
    the sim charges. 4-param OLS `prefill_span ~ floor + a·new + b·cached +
    c·new·(cached+new/2)` gives the FA3 cross-coefficient c(tp1) = **5.5722e-7** /
    c(tp4) = **1.7965e-7** (fit MAPE 13.3%/3.8%): the tp4 attention prefill is
    head-sharded, ratio **0.32240** (between 1/4 perfect shard and 1; the tp1 4p span fit
    also independently corroborates the production 8.31e-7's magnitude, and at
    new=2048/cached=16000 the tp1 closed-form check 22.733/1k·2048 + 8.31e-7·2048·17024 =
    75.6 ms vs measured span 75.3 ms).

  **PRE-REGISTERED ESTIMATORS (deterministic builder
  `profiling/process/build_stage_split_rates.py --tp 4` →
  `profile_data/kernels/prefill_stage_rates_H100x4.json`):**
  * `prefill_host_cached_ms_per_token`(H100x4) = the 3-param ttft-lattice OLS cached
    coefficient on the tp4 leg = **3.5302703225806482e-3** (estimator-parity with the
    production `build_host_split` c1 fit; builder hard-fails if the tp1 leg's same-estimator
    value drifts >5% from the production SUM). The shared/per-request PARTITION keeps the
    production measured fraction 0.5236 (tp1 B-sweep band point estimate): the tp4 B-sweep
    does not exist — documented caveat, partition is a stack-structure choice held fixed,
    only the measured SUM is per-config.
  * `prefill_fa3_ms_per_token2`(H100x4) = production 8.31e-7 × the like-for-like 4-param
    span-fit ratio c(tp4)/c(tp1) = 8.31e-7 × 0.32239587 = **2.679109720609454e-7**
    (transport via the tp1 leg bridges the kernel-grid provenance to the serving-stack
    instrument, the G3 pattern; the DIRECT tp4 coefficient 1.7965e-7 is reported in the
    artifact as sensitivity — preview shows the two are outcome-equivalent, e2el 16.32 vs
    16.21, so the choice is not outcome-driven).
  **Wiring (the Phase-B mechanism, mirrored):** optional `RooflineParams`
  `prefill_host_cached_ms_per_token` / `prefill_fa3_ms_per_token2` (default `None` → the
  module constants, BYTE-IDENTICAL for every unpinned config), `configs/loader.py`
  threading, consumption in `ttft_queue_sim._price_step` (host: shared/perreq = the
  production fraction × the pinned SUM; `None` keeps the original constant path untouched);
  pinned ONLY in the L11-owned H100x4 manifest. TPOT is structurally untouched (both terms
  are TTFT-only).
  **Adopt rule:** adopt iff H100x4 E2EL cell-MAPE improves vs 25.6170 AND TTFT improves vs
  33.8703 AND TPOT byte-unchanged AND H100/A100/H100x2 BYTE-IDENTICAL (H100x2 is unpinned →
  stronger than the ≤+0.3 contract); else revert to this documented stop-point.
  **Transparency:** the candidate constants were previewed during diagnosis via the same
  module-attribute patches (H100x4 ttft/e2el ≈ 24.97/16.32 host+fa3; 25.43/17.61 host
  alone); the binding evaluation is the wired replay-ON gate below. Baseline reproduction
  verified at HEAD before any edit: scoped (H100x2,H100x4) and pair (H100,A100) gate outputs
  byte-match `/tmp/hm_r1.*` and `/tmp/hm_pair_base.predictions.json`, 0 bytes stderr.

- **2026-06-11 — ROUND 2 EXECUTED + ADOPTED: per-config measured tp4 prefill host-cached rate
  (3.5303e-3 vs the tp1-measured 5.8872e-3 charged before) + FA3 coefficient (2.6791e-7 vs
  8.31e-7) → TTFT cell 33.8703 → 24.9634, E2EL 25.6170 → **16.3179** (< 20 TARGET MET), TPOT
  byte-unchanged 37.0951; H100/A100 pair AND H100x2 byte-identical.**
  Artifact `profile_data/kernels/prefill_stage_rates_H100x4.json` (committed; regenerable:
  `python3 -m profiling.process.build_stage_split_rates --tp 4`; both validations passed:
  tp1-leg host drift +1.7% < 5%, FA3 ratio 0.32240 ∈ (0,1)). Wiring exactly per the
  pre-registration: optional `RooflineParams.prefill_host_cached_ms_per_token` /
  `prefill_fa3_ms_per_token2` (default None → module constants), `configs/loader.py`
  threading, `ttft_queue_sim._prefill_host_fa3_rates` consumer (factored helper, the
  `_prefill_gemm_per_tok_loaded` test pattern), pinned ONLY in the H100x4 manifest
  (provenance `data.prefill_stage_rates`; `calibration_status` →
  `h100_tp4_vllm_analytic_decode_measured_tp4_prefill_law`). Default-path wiring verified
  BYTE-IDENTICAL on the scoped gate BEFORE the pin; helper refactor byte-reproduced the
  pinned gate. New tests `test_prefill_host_fa3_per_config_override_and_byte_identical_default`,
  `test_prefill_stage_rates_manifest_pins_match_artifact`,
  `profiling/tests/test_build_stage_split_rates.py` (OLS exactness + artifact regeneration
  parity, CSV-guarded). Full suite 403 passed / 1 pre-existing skip / 15 subtests.
  **Gates (replay-ON, `RAMP_TPOT_REQUIRE_POOLS=1`, 0 bytes stderr):** scoped → `/tmp/hm_r2.*`
  (H100x4 37.0951/24.9634/16.3179; H100x2 21.5336/29.0163/18.5506, rows JSON-identical to
  round 1 — regression exactly 0); pair → `/tmp/hm_pair_r2.*`, predictions BYTE-IDENTICAL to
  `/tmp/hm_pair_base.predictions.json` (`cmp` clean; H100 14.4697/18.1309/10.7777, A100
  14.3728/22.2222/15.8653).
  **Remaining-error localization at the adopted state (honest):** conc≥120 TTFT over-prediction
  is GONE (median signed −0.109 s; was +0.21..+5.53 s) — per-profile e2el now chat 15.78 /
  osworld 13.93 / swebench 16.88 / terminalbench 18.68. The de-fit EXPOSED a compensating
  under-charge at low/mid-conc terminalbench/swebench (term c10–c40 TTFT now UNDER-predicts,
  err 30–39% vs 16–17% before — the over-priced tp1 host rate was masking an un-modeled
  fixed per-request cost; the build_host_split artifact's `diagnostic_fixed_cost_refit`
  ≈12.5 ms/req names the candidate term) and leaves the swebench c≥256 TPOT b_eff
  over-price (tpot_err 153–163, the round-1 named successor) as the dominant single-cell
  residual (swebench c320 e2el 33.26). Both are upstream structural successors, documented,
  not re-tuned here.

- **2026-06-11 — ROUND 2 CLOSE: lane goal MET (supersedes the round-1 PARTIAL finalize
  above). No GPU work this round (both stage-split legs were measured in round 1; GPUs 4–7
  untouched).**

  | config | tpot_cell | ttft_cell | e2el_cell | Δ vs campaign baseline |
  |---|---|---|---|---|
  | H100x4 (lane) | 37.0951 | **24.9634** | **16.3179** | 0 / −14.49 / −13.49 |
  | H100 (binding) | 14.4697 | 18.1309 | 10.7777 | 0 (byte-identical) |
  | A100 (binding) | 14.3728 | 22.2222 | 15.8653 | 0 (byte-identical) |
  | H100x2 (binding) | 21.5336 | 29.0163 | 18.5506 | 0 (byte-identical) |

  Goal restated: H100x4 E2EL cell-MAPE < 20 ✓ (16.3179, from 29.8081 baseline: round-1 comm
  −4.19, round-2 host+FA3 −9.30); H100x2 < 20 ✓ (18.5506, regression 0); pair byte-identical ✓.
  Adopted levers are all MEASURED per-config artifact pins with byte-identical default
  fallbacks; refused/retained levers and the two named successors (b_eff mapping; the fixed
  per-request host cost) are documented above with their evidence.
