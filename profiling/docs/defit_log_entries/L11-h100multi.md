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
