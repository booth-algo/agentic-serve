# Concurrency-Aware serving_e2e Predictor — Implementation Plan

## Overview

Extend `predict_serving_e2e()` to accept a `concurrency` parameter and predict TTFT, TPOT, and E2EL at arbitrary request concurrency levels (1-500+). Uses Little's Law to derive the effective decode batch size from concurrency, then feeds that batch size into the existing kernel predictor which already handles variable `bs`.

## Phase 1: Core Concurrency Model

### Step 1: Create `concurrency_model.py`
New module with pure functions:
- `effective_decode_bs(concurrency, ttft_ms, tpot_ms, osl) -> float`: Little's Law steady-state bs_eff. `decode_fraction = D*O / (T + D*O)`, `bs_eff = C * decode_fraction`, clamped to `[1, C]`.
- `iterative_bs_eff(concurrency, ttft_ms_fn, tpot_ms_fn, osl, max_iter=5, tol=0.01) -> float`: Iterative solver for TPOT<->bs_eff circular dependency. Start at bs=1, converge in 2-3 iterations. Damping: `bs_new = 0.7 * bs_computed + 0.3 * bs_old`.
- `ttft_queuing_factor(concurrency, ttft_ms, tpot_ms, osl) -> float`: Multiplicative TTFT scaling. Phase 1: `1 + alpha * (C-1) * prefill_fraction`.
- `estimate_kv_memory_gb(bs_eff, max_kv_len, n_layers, kv_heads, head_dim) -> float`
- `is_saturated(kv_memory_gb, gpu_memory_gb, model_memory_gb) -> bool`

### Step 2: Extend `serving_e2e.py`
- Add `concurrency: int = 1` parameter to `predict_serving_e2e()`
- When `concurrency=1`: fast-path, numerically identical to current code
- When `concurrency > 1`:
  - Compute bs=1 baseline (TTFT and TPOT)
  - Define `tpot_at_bs(bs)` closure using existing trapezoidal integration
  - Call `iterative_bs_eff()` to get converged bs_eff
  - `decode_ms = _integrate_decode_ms(pred, cfg, isl, osl, bs=bs_eff) * alpha`
  - `ttft_ms = ttft_1 * ttft_queuing_factor(concurrency, ...)`
  - Return dict with extra keys: `bs_eff`, `saturated`
- Critical: prefill always uses `bs=1` (continuous batching processes one prefill at a time)

### Step 3: Refactor decode integration
Extract 8-point trapezoidal block into `_integrate_decode_ms(pred, cfg, isl, osl, bs, tp, shard_tp) -> float`. Avoids duplication between concurrency=1 path and iterative solver.

## Phase 2: Validation Infrastructure

### Step 4: Multi-concurrency data loading
Modify `_load_measured_rows()` in `validate.py` to accept `concurrency: int | list[int] | None` where None = all concurrencies.

### Step 5: `validate_serving_e2e_conc_gpu()`
New function that:
- Loads measured rows at all concurrency levels (1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500)
- For each (model, concurrency, profile): calls `predict_serving_e2e(concurrency=C)`
- Computes per-concurrency MAPE
- Detects saturation point (MAPE > 30% threshold)
- Report format: `{gpu}_serving_e2e_conc_{profile}.md`

### Step 6: CLI
Add `--mode serving_e2e_conc` and `--concurrencies` args.

## Phase 3: Calibration (iterative, data-driven)

### Step 7: TTFT queuing alpha
Per-GPU `_TTFT_QUEUE_ALPHA` dict (analogous to `_DECODE_CORRECTION`). Fit alpha to minimize TTFT MAPE across concurrency levels.

### Step 8: Concurrency-dependent decode correction
Check if `_DECODE_CORRECTION` generalizes to higher bs_eff. If not, add `alpha(bs_eff) = alpha_base * (1 + beta * log(bs_eff))`.

### Step 9: Validate and iterate
Run sweeps, analyze where model breaks, tune.

## Phase 4: Advanced (optional)

### Step 10: KV cache saturation model
`max_concurrent_requests(cfg, gpu_memory_gb)` — analytical upper bound.

### Step 11: Concurrency curve visualization
Matplotlib plots: predicted vs measured TPOT/E2EL as f(concurrency).

### Step 12: Dashboard integration

## Key Technical Details

### Iterative bs_eff Solver
Circular dependency: bs_eff -> TPOT(bs_eff) -> decode_fraction -> bs_eff. Converges because TPOT is sub-linear in bs (memory-bound at small M). Typically 2-3 iterations. Cap at 5, dampen if needed.

### GEMM at Non-Integer M
XGBoost handles continuous M naturally. Training data at M=1,128,256,512; intermediate values fall into correct leaves because GEMM is memory-bound at M<64 (constant latency) then compute-bound (linear).

### Prefill bs Under Continuous Batching
Prefill bs is always 1 (server processes one prefill at a time, interleaved with decode batch). TTFT inflation comes from queuing delay, not compute.

### DECODE_CORRECTION at Higher bs
Factor calibrated at bs=1 may not hold at bs=100. At high bs, per-step overhead is smaller fraction -> correction should decrease. Check in Phase 3.

## Validation Data
- data.json: concurrency 1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500
- A100: Llama-8B, gpt-oss-20b, Qwen3.5-9B (3 models x 11 concs x 3+ profiles)
- RTX3090: Llama-8B (1 model x several concs x 3 profiles)
- RTX2080Ti: Llama-8B (limited)

## Success Criteria
- concurrency=1 identical to current (no regression)
- TPOT MAPE < 15% for concurrency 1-80 on A100
- E2EL MAPE < 25% pre-saturation across all configs
- Saturation point detected and reported
- Iterative solver converges in <= 5 iterations

## Risks
| Risk | Severity | Mitigation |
|---|---|---|
| Little's Law assumes steady-state | Medium | C >= 10 reaches steady state quickly; C=1 is by construction |
| Solver diverges | Medium | Cap 5 iterations, damping, fallback to single-pass |
| GEMM inaccurate at M=2-64 (training gap) | Medium | Validate at C=10,20; add synthetic GEMM data if needed |
| KV cache saturation at C > 200 | High | Phase 4 detection; report pre-saturation MAPE only |
| TTFT linear model too simple | Medium | Start simple; upgrade to M/G/1 if needed |
| _DECODE_CORRECTION doesn't generalize to high bs | Medium | Phase 3 step 8; add bs-dependent beta |
