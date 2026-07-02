# agentic-serve — codebase map & rebuild brief

> Hand-off doc for an agent (Cursor) joining mid-rebuild. Describes (A) what the system
> does, (B) the current/legacy codebase and where things actually live, (C) the known
> bloat, (D) data & ground-truth locations, (E) the gotchas that will bite a naive
> reimplementation, and (F) the greenfield target (`simulator_v2/`) with its contracts.
> Facts here were verified by reading source on 2026-06-22/23; file:line refs point at
> the real code — open them, don't trust this doc blindly.

---

## A. What this system does

Predicts **per-turn latency** — TTFT (time-to-first-token), TPOT (time-per-output-token / inter-token
latency), and E2EL (end-to-end latency) — for **multi-turn LLM serving** (vLLM / SGLang) across GPUs,
models, and tensor-parallel degrees. "Per-turn" because the workloads are agentic multi-turn
conversations (chat, swebench, terminalbench, osworld) where each turn's cached/new/output token mix
differs and KV-cache pressure builds across turns.

It exists as **two prediction modes over one shared math engine**:

- **Backtester** — uses the **ncu kernel-composition method**: measured NVIDIA Nsight Compute kernel
  timing grids are composed into a decode-step / prefill cost, then run against a **measured benchmark**
  to score prediction-vs-ground-truth.
- **Forwarder** — **consumes no kernels.** Predicts the same metrics purely from (a) a hardware **config
  file** (analytic roofline: peak FLOPs / bandwidth / memory + utilization factors), (b) a per-turn
  **ISL:OSL distribution**, (c) **cached-token** counts. Its job is generalizing to GPUs/models that were
  never profiled — so it is *expected and allowed to be less accurate* than the backtester. Do **not**
  tune it toward backtester parity; that defeats its purpose.

**Headline accuracy today (Llama-3.1-8B, H100, tp1):** TPOT MAPE **16.48%**, TTFT **33.12%**, E2EL
**19.67%** (cell-MAPE). swebench-plateau TPOT 9.20%.

**Key identity:** `E2EL ≈ TTFT + output_tokens · TPOT` is near-exact (median APE 0.009%). E2EL is **not a
model** — it's a one-line composition. Never give it its own predictor.

---

## B. Current codebase — where things actually live

Five overlapping top-level dirs. The important realization: **the prediction engine is `simulator/`,
and BOTH modes already call into it** — the kernel-composition math is *not* duplicated. The mess is in
the *drivers* around it, not the core.

```
simulator/            ← THE PREDICTION ENGINE (shared by both modes)
profiling/            ← measurement scripts + the driver/orchestration scripts (~10.6k LOC, 45 files)
configs/              ← forwarder's hardware inputs: models/ gpus/ deployments/ + loader.py
profile_data/         ← DATA: kernels/ (committed grids) + results/ (gitignored measured)
inference-benchmark/  ← benchmark HARNESS (produces ground truth) + React/Vite dashboard
simulator_v2/         ← THE GREENFIELD REBUILD IN PROGRESS (see section F)
```

### The shared engine (`simulator/`)

| file | LOC | role |
|---|---|---|
| `simulator/kernel_tpot.py` | 462 | `predict_cell_tpot(kin, params, grid=, ceiling=)` → per-turn **TPOT** (headline) |
| `simulator/ttft_queue_sim.py` | **1374** | `predict_cell_ttft_qsim(turns, profile, conc, params, ...)` → per-turn **TTFT** (headline). Also `_run_sim`, `_aggregate`, `_cohort_from_pool`, `_prefill_floor_for` |
| `simulator/kernel_step_cost.py` | 514 | composes gemm/flash/elementwise kernels → one decode-step cost. Module globals `_default_grid`, `analytic_grid()`, `load_grid()` |
| `simulator/closed_form_tpot.py` | 551 | defines `RooflineParams` (the hardware dataclass: peak_flops/bw, util_flops/bw, scheduler_overhead, kv pool). Most imports want only this dataclass; the closed-form predictor itself is largely vestigial |
| `simulator/ramp_tpot.py` | 454 | ramp/onset logic for TPOT saturation; a dependency of kernel_tpot |
| `simulator/ttft_predict.py` | 175 | the **old static "M0" amplifier** — now only imported *by* qsim as a `_static` comparison baseline |
| `simulator/kernels/{gemm,flash_attn,elementwise,roofline}.py` | — | per-kernel cost models |
| `simulator/cohort_scale.py`, `cached_prefill_lookup.py` | — | qbar cohort scaling; cached-prefill grid lookup |

### Backtester driver — `profiling/process/build_simulator_rows.py` (289 LOC)
Call chain (verified):
```
main()
  CONFIGS = configs.loader.all_deployments()                 # 84 deployment JSONs
  per cfg: swap kernel_step_cost._default_grid (measured decode grid)
           swap kernel_tpot._active_ceiling_json (saturated-ITL ceiling)
  build_row(profile, conc, params, cfg, bench_root)
    build_turns(bench_file)            # reads measured per_request → per-turn MEDIANS (cached/new/output/ttft/tpot/e2el)
    predict_cell_tpot(kin, params)                            # → per-turn TPOT
    predict_cell_ttft_qsim(turns, profile, conc, params, …)   # → per-turn TTFT
    e2el = ttft + output·tpot
  → writes inference-benchmark/dashboard/public/simulator-predictions.json
```
`build_row()` has ~40 lines of inline calibration commentary (Llama-only `gpu_key` gates, cohort
replay on/off, prefill-floor inheritance) tangled into the hot path — this is the main thing the
rebuild should *untangle*.

### Forwarder driver — `simulator/forward.py` (427) + `profiling/process/build_forward_rows.py` (122)
```
build_forward_rows.main()
  import build_simulator_rows as B          # ← forwarder currently DEPENDS ON backtester (B.build_turns, B.BENCH_BASE, B.PROFILES)
  per cfg/profile/conc: _trajectories(bench) → per-session [(cached,new,output), …]
  simulator.forward.predict_forward(gpu, model, tp, engine, concurrency, trajectories=…)
    resolve_hardware()      # measured artifacts if a deployment exists, ELSE analytic roofline (configs.loader.compose_roofline)
    _cohort_from_trajectories() → cohort + per-turn-index medians + qbar
    _run_and_score()        → predict_cell_tpot + _run_sim/_aggregate (qsim) → per-turn tpot/ttft + per-request percentiles
  → writes inference-benchmark/dashboard/public/forward-predictions.json
```
Note: today's forwarder **reuses measured grids** when a calibrated deployment exists — i.e. it is NOT
strictly kernel-free. The rebuild's forwarder is defined as **strictly config-only / no kernels**
(cleaner separation than the current code).

### configs/ (forwarder hardware inputs — clean, recent)
- `configs/gpus/*.json` (4: H100, A100, RTX3090, RTX2080Ti) — peak_flops_per_s, peak_bw_bytes_per_s,
  util_flops, util_bw, scheduler_overhead_ms_per_step, total_memory_bytes.
- `configs/models/*.json` (9) — n_params, bytes_per_param, kv_bytes_per_token, kv_heads, cache_block_size, …
- `configs/deployments/*.json` (84) — (gpu, model, tp, engine) + bench_dir + max_model_len/max_num_seqs +
  pointers to measured decode_grid / saturated_ceiling + a per-input `data` provenance manifest.
- `configs/loader.py` — `all_deployments()`, `compose_roofline(gpu, model, tp, pool)` → `RooflineParams`.
- `configs/kv_pool.py` — `available_kv_blocks(...)` analytic KV-pool sizing.
- `configs/coverage_report.py` — `python3 -m configs.coverage_report` data-coverage tracker.

---

## C. Known bloat (what the rebuild is escaping)

- **3 TPOT modules**: `kernel_tpot` (live) + `closed_form_tpot` (mostly just `RooflineParams` now) +
  `ramp_tpot` (tangled dep). Should collapse to one.
- **2 TTFT modules**: `ttft_queue_sim` (live, 1374 LOC) + `ttft_predict` (old M0 static, only a comparison baseline).
- **~15 feature-builder scripts** in `profiling/process/` (`build_ramp_knees`, `build_saturated_ceiling`,
  `build_host_split`, `build_tp_comm`, `build_stage_split_rates`, `build_sat_sustain`, `build_prefill_floor`,
  `build_prefill_gemm_util`, …) — each emits one small JSON in `profile_data/kernels/` that the predictor reads.
- **Dead/legacy**: `simulator/_legacy/`, `profiling/process/_pool_sensitivity_probe.py`,
  `prefill_stage_split.py`, `build_decode_grid.py`.
- **Parallel older method**: `inference-benchmark/scripts/roofline/` (11 files: `run_ncu`, `parse_ncu`,
  `profile_all_layers`, `plot_roofline`) — an *earlier* ncu-roofline analysis feeding
  `roofline-data.json`/`roofline-quadrant.json`. Separate from the kernel-composition method.
- **Stale / huge dashboard JSONs** in `inference-benchmark/dashboard/public/`: `data.json` **167 MB**,
  `data.synthetic_distributional.json` 91 MB, `data.trace_replay.json` 27 MB, `data.archived.json` 48 MB,
  `simulator-predictions.json` 62 MB, plus dead `simulator-v2-predictions.json` / `simulator-v3-predictions.json`.
- Two root READMEs (`readme.md` + `README.md`).

---

## D. Data & ground-truth locations

- **Ground truth (multi-turn benchmarks)** lives in a **central store, NOT in the repo**:
  `/mnt/100g/agent-bench/results/synthetic_distributional/<bench_dir>/<profile>_conc<conc>.json`
  where `<bench_dir>` ≈ `<gpu>_<model>_tp<N>_<engine>`. Each file's `per_request[]` holds the raw
  `ttft_ms` / `tpot_ms` / `e2el_ms` + token columns. (Defined as `BENCH_BASE` in build_simulator_rows.py:45.)
- **Profiles**: `chat-multiturn-synth`, `osworld-multiturn-synth`, `swebench-multiturn-synth`,
  `terminalbench-multiturn-synth`.
- **Concurrencies**: `[1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]`.
- **Measured kernel grids**: `profile_data/kernels/` (committed) — gemm/flash_attn/elementwise grids,
  decode grids, `saturated_ceiling_*`, `ramp_knees_*`, `roofline_params_*`, `roofline_utils_*`, etc.
  `profile_data/results/` is gitignored regenerable measured output.
- **Measurement scripts** (produce the grids): `profiling/gpu_profiling/vllm/` (CUDA-event + live-server
  sweeps) and `profiling/profile/vllm/engine_trace/serving_engine_steps.py` (2606 LOC vLLM scheduler instrumentation).

---

## E. Gotchas that will bite a naive reimplementation

1. **The utilization factors are a conflated fudge.** `configs/gpus/H100.json` pins
   `util_flops=0.65`, `util_bw=0.93`, `scheduler_overhead_ms_per_step=5.7`. Per `closed_form_tpot.py:68-74`
   and `profile_data/kernels/roofline_utils_H100.json`: a pre-registered re-derivation over 4 measured
   serving traces gives the *honest* values **0.5886 / 0.8111 / 4.5574**. The reproducible `util_bw=0.81`
   was wired and **REJECTED** (made MAPE worse) because **0.93 secretly absorbs host overhead the decode
   path doesn't price separately**. So `util_bw` is "MBU + unmodeled overhead," not real MBU. If you model
   host overhead explicitly, you must drop util_bw back toward 0.81. (The greenfield H100 YAML keeps the
   pinned values for now, flagged for replacement.)
2. **TPOT/TTFT are path-dependent across turns.** Saturation (KV pressure, eviction) and queueing
   accumulate over the turn sequence and across the concurrent cohort. The engine functions take the
   **whole turn list + a cohort**, never one turn at a time. `predict_cell_tpot(kin,…)` and
   `predict_cell_ttft_qsim(turns,…,cohort=)` both do this. Design the rebuild seam list-in/list-out from
   the start even though concurrency=1 doesn't need it.
3. **`util_flops` has a better source for the backtester.** A single scalar is a stand-in; the
   authoritative prefill-util is a *measured curve* (`prefill_gemm_util_H100.json`). That curve is a
   backtester asset (profiled). The forwarder *can't* use it for an unprofiled GPU — a scalar `util_flops`
   from config IS the right forwarder abstraction. Same quantity, two fidelities → reinforces the two-mode split.
4. **Forwarder is allowed to be worse than backtester.** Don't chase parity.
5. **Ground truth is per-request, then medianed per turn.** `build_turns()` takes per-turn medians of
   successful requests. Cell headline = mean over turns. Match this aggregation or MAPEs won't compare.

---

## F. The greenfield target — `simulator_v2/`

**Decision (2026-06-22):** rebuild from scratch, GREENFIELD, **clarity over exact parity** (small MAPE
drift vs 16.48/33.12/19.67 is acceptable). One pure engine, two thin drivers, one combined `main`.

### Layout (current scaffold)
```
simulator_v2/
  config/h100.yaml          # GPU hardware spec (compute/memory/scheduler groups) — DONE
  config_loader.py          # YAML → hardware model object — STUB (empty)
  engine/
    predict.py              # orchestrator: calls decode+prefill, composes E2EL — WRITTEN (see bug note)
    decode.py               # base step cost → per-turn TPOT under KV pressure — TODO
    prefill.py              # base prefill cost → per-turn TTFT under concurrency/queue — TODO (the big one)
  docs/
    CODEBASE_MAP.md         # this file
    notes.md                # migration notes (e.g. "yaml numbers handwavy, need remeasure")
  main.py                   # one CLI, dispatches backtest|forward — PLANNED
```

### The architecture: one engine, two adapters
The engine never knows which mode it's in. The two modes differ in **exactly two places** — which
`hardware` adapter and which `workload` adapter get plugged in:

| | hardware adapter | workload adapter |
|---|---|---|
| **backtest** | `KernelHardware` (measured ncu grids) | `BenchmarkWorkload` (per-turn medians from a `/mnt/100g` GT file) + the GT for scoring |
| **forward** | `RooflineHardware` (config YAML, NO kernels) | `DistributionWorkload` (ISL:OSL distribution + cached tokens) |

### Layering (where each kind of code goes)
- **`hardware` adapter** — cost of *one* step / *one* request. The ONLY layer that knows
  kernels-vs-roofline (the backtest/forward seam). Exposes e.g.
  `decode_step_ms(batch, ctx_tokens)`, `prefill_ms(new, cached, batch)`, plus capacity facts
  `kv_pool_blocks`, `saturated_step_ms`, `sched`.
- **`engine/decode.py` + `engine/prefill.py`** — take that base cost and add saturation / queueing.
  **Hardware-agnostic** (identical whether cost came from kernels or roofline). This is where the bulk
  of the logic (the old kernel_tpot 462 + qsim 1374) lands. Keep them as separate modules — folding
  them into predict.py recreates the god-module bloat.
- **`engine/predict.py`** — orchestration + the 1-line E2EL composition. Stays ~20 lines forever.
- **`main.py`** — builds `hw` + `workload` per mode (dict-dispatch, not `if`-chains), calls `predict()`,
  then aggregates + scores (backtest) or reports (forward). Aggregation/MAPE live HERE, not in the engine.

### Contracts (implement against these)
```python
# Turn — reuse the proven field names from build_turns()
Turn(cached_context_tokens, new_prefill_tokens, output_tokens, scheduled_requests)

# hardware adapter (Protocol)
hw.decode_step_ms(batch: int, ctx_tokens: float) -> float
hw.prefill_ms(new: float, cached: float, batch: int) -> float
hw.kv_pool_blocks: int ; hw.saturated_step_ms: float ; hw.sched: SchedConfig

# engines — list-in / list-out (path-dependent across turns)
decode.predict(hw, turns, concurrency, *, cohort=None) -> list[float]   # per-turn TPOT ms
prefill.predict(hw, turns, concurrency, *, cohort=None) -> list[float]  # per-turn TTFT ms

# orchestrator
predict(hw, turns, concurrency, *, cohort=None) -> list[TurnPrediction]
TurnPrediction(ttft_ms, tpot_ms, e2el_ms)   # e2el_ms = ttft_ms + turn.output_tokens * tpot_ms
```

### Known bug in the current `engine/predict.py`
The E2EL composition was dropped: the comprehension references an undefined `e2el_ms` and omits `turns`
from the `zip`. Correct form:
```python
return [
    TurnPrediction(ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                   e2el_ms=ttft_ms + turn.output_tokens * tpot_ms)
    for turn, tpot_ms, ttft_ms in zip(turns, tpots, ttfts)
]
```
Also `predict.py` imports `decode` and `prefill`, which don't exist yet → ImportError until stubbed.

### Build order (thin vertical slice first; don't go breadth-first)
1. Define the two adapter protocols + `Turn` (empty contracts).
2. **TPOT at concurrency=1**, one cell (H100 Llama-8B chat c1): `decode_step_ms(batch=1, ctx)` IS the
   TPOT when batch=1. Validate vs GT. ← smallest thing that proves the spine.
3. **TTFT at c=1** (`prefill_ms` full+cached). **E2EL is then free.**
4. **Saturation / queue** — the hard part: TPOT ceiling/ramp under KV pressure, TTFT queueing. Port and
   *simplify* the old `kernel_tpot.py` / `ttft_queue_sim.py` math; decide what complexity to keep.
5. **Flip to forwarder** — swap in `RooflineHardware` + `DistributionWorkload`. If the seams are clean,
   the engine code doesn't change at all.

### Reference oracles in the OLD code (port FROM, don't necessarily keep)
- TPOT math: `simulator/kernel_tpot.py:predict_cell_tpot` (+ `kernel_step_cost.py`, `ramp_tpot.py`)
- TTFT math: `simulator/ttft_queue_sim.py:predict_cell_ttft_qsim` / `_run_sim` / `_aggregate`
- Hardware compose: `configs/loader.py:compose_roofline`, `configs/kv_pool.py:available_kv_blocks`
- Backtest driver to mirror: `profiling/process/build_simulator_rows.py:build_row`
- Forward driver to mirror: `simulator/forward.py:predict_forward`
