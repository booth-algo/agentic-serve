# Predictor State — 2026-05-02

## Current Metrics (H100x4 current scope, 129 rows, TP=4, symmetric error metric)

| Metric | Before session | After session |
|--------|---------------|---------------|
| TTFT p50 | 60.8% | 92.2% |
| TPOT p50 | 217.3% | **6.5%** |
| E2EL p50 | 181.5% | **9.4%** |

A100 current (44 rows): TTFT 143.0%, TPOT 13.5%, E2EL 46.2%
RTX3090 current (69 rows): TTFT 464.0%, TPOT 329.3% (likely GEMM table fidelity issue)

**Error metric**: symmetric `|pred-meas| / min(pred, meas) * 100`. Same % suffix as before, same color thresholds, but honest: a 21× miss shows as 2029%, not 95.3%.

---

## Changes Made This Session

### 1. Prefix-Cache Priors for Single-Turn (`prefix_cache_priors.py`)

File: `llm_predict/prefix_cache_priors.py`, `llm_predict/data/prefix_cache_priors.json`

`coding-singleturn` had zero TTFT predictions (all marked `unknown_prefix_cache`). Built a prior from `coding_agent_prompts.jsonl`:
- All 500 prompts share identical system prompt (~6982 estimated tokens)
- User messages vary (median ~328 tokens)
- Prior: `new_prefill=328`, `cached=6982`, `hit=0.955`
- 322 coding-singleturn rows now get cache-aware predictions

### 2. Per-Turn Concurrency (`cache_aware.py`)

`predict_multiturn_from_per_turn()` was passing global `concurrency=5` to every turn's decode model. Changed to `max(1, feature.successful)` — Turn 1 uses D=10, Turn 8 uses D=1.

### 3. TP-Aware Decode Step (BIGGEST WIN)

**Root cause**: composer modeled `tensor_parallel_size=1` (full single-GPU GEMMs), but serving data is from H100x4 (TP=4). At TP=4, n_heads drops from 64→16, n_kv_heads 8→2, FFN 28672→7168. ~4× less compute per GPU.

**Wired TP through full pipeline**:
- `composer.py`: `predict_ttft_ms()` and `predict_decode_step_us()` accept `tensor_parallel_size`
- `serving.py`: `predict_serving()`, `_integrate_decode_ms()`, `_iterative_bs_eff()` thread `tp`
- `export_serving_predictions.py`: extracts TP from hardware key (`H100x4 → tp=4`)
- `cache_aware.py`: per-turn predictions get correct TP

**Result**: TPOT 217.3% → 6.5%. The composer was modeling 4× too much compute per GPU.

### 4. TTFT Floor + First Decode (`serving.py:142-155`)

```python
# Fixed per-forward-pass overhead:
tp_barrier_us = 5.0 * 5.0 * cfg.n_layers if tensor_parallel_size > 1 else 0.0
# 5 all-reduces per layer × ~5us barrier latency × 80 layers ≈ 2ms for 70B TP4
scheduler_overhead_us = 500.0  # scheduler loop + kernel launch
ttft_floor_ms = (tp_barrier_us + scheduler_overhead_us) / 1000.0

# First decode step after prefill:
ttft_first_decode_ms = composer.predict_decode_step_us(
    cfg, kv_len=total_context, bs=1, tensor_parallel_size=tp) / 1000.0

ttft_ms = ttft_kernel + ttft_floor_ms + ttft_first_decode_ms + ttft_queue_ms
```

### 5. TTFT Queue Model (`serving.py:167-184`)

Closed-form analytical model for N simultaneous arrivals into continuous batching:

```
m = concurrency / 2.0          # median request position in arrival queue
pref_queue = m * ttft_kernel   # median waits for m prefills ahead of it
decode_queue = m * (m + 1) / 2 * decode_step_bs1  # triangular sum of decode interleaving
ttft_queue_ms = pref_queue + decode_queue
```

### 6. Event-Driven Scheduler Simulation (`serving_sim.py`)

Built but **not wired** into the pipeline. Models per-step continuous batching with serial prefill and decode batch draining. Was ~1-3% worse than analytical model for this benchmark (sequential turns, independent prompts). Kept as utility for future use.

---

## Per-Turn Example: Llama-3.1-70B, chat-multiturn, C=5, SGLang, H100x4

| Turn | Reqs | Kernel | Floor | 1stDec | Queue | Pred | Meas | Ratio |
|------|------|--------|-------|--------|-------|------|------|-------|
| T1 | 10 | 16ms | 3ms | 16ms | 318ms | 352ms | 1120ms | 3.2× |
| T2 | 4 | 28ms | 3ms | 16ms | 103ms | 149ms | 686ms | 4.6× |
| T3 | 3 | 46ms | 3ms | 16ms | 98ms | 162ms | 626ms | 3.9× |
| T4 | 2 | 24ms | 3ms | 16ms | 40ms | 82ms | 100ms | 1.2× |
| T5 | 2 | 19ms | 3ms | 16ms | 35ms | 73ms | 100ms | 1.4× |
| T6 | 2 | 19ms | 3ms | 16ms | 35ms | 72ms | 92ms | 1.3× |
| T7 | 2 | 20ms | 3ms | 16ms | 35ms | 73ms | 88ms | 1.2× |
| T8 | 1 | 26ms | 3ms | 16ms | 0ms | 44ms | 58ms | 1.3× |

**Pattern**: queue-free turns (T4-T8, D≤2) are accurate at 1.2-1.4×. Queued turns (T1-T3, D=3-10) are 3-5× off. The kernel + floor + first_decode values are in the right ballpark, but the queue model under-predicts.

---

## Remaining TTFT Gap Analysis

### What works
- **Queue-free TTFT**: T8 (D=1) is 1.3× off. Kernel + floor + first_decode captures ~76% of measured
- **TPOT**: 6.5% — the decode model is now accurate at TP=4
- **Shape decomposition**: composer correctly separates `Q=prefill_tokens, KV=total_context` for cached prefill

### What doesn't work
- **TTFT queue magnitude**: The analytical queue model predicts 3-5× too little queue delay for early turns
- **Turn 2 is worse than Turn 1** relative to prediction (4.6× vs 3.2×) — the model doesn't capture state-dependent arrival effects
- **Fixed overhead estimates**: TP barrier (2ms for 70B) and scheduler overhead (0.5ms) are rough approximations

### Why the queue model under-predicts

The analytical model assumes:
1. **Serialize prefills**: each request's prefill takes exactly one scheduler step. Debatable for SGLang with chunked prefill.
2. **Linear decode interleaving**: decode step at batch j takes `j × decode_step_1`. The TP-aware composer does produce this, but real hardware may have non-linear batch scaling.
3. **No per-step overhead amplification**: the floor_ms is applied once per TTFT, not per scheduler step. Each scheduler iteration has its own overhead.
4. **vLLM assumptions**: SGLang with `--enable-mixed-chunk` may interleave differently.
5. **Cache hit ≠ free**: Turn 2 has 14% cache hit but 516 total context. The prefill still scans over the full KV. The kernel captures this (Q=443, KV=516), but the composer may under-predict attention cost for partial cache hits.

### Possible next steps

1. **Profile SGLang-specific scheduler behavior** — `--enable-mixed-chunk` may add overhead not captured by vLLM-default `max_num_batched_tokens`
2. **Add per-step floor overhead** — currently floor is once per TTFT; each queue step also has scheduler overhead
3. **Profile flash attention at cached-prefill shapes** — currently roofline-only, 7% of predict time. NCU data would reduce this
4. **Non-linear decode batch scaling** — the composer models linear batch scaling. Real hardware shows sublinear (α≈0.36 for D≤10 on 70B). Need decode-shaped NCU profiles to capture this.
5. **Engine-specific calibration** — add per-backend scheduler parameters (max_tokens, chunk_size, continuous_decode_steps)
6. **Trace-level simulation** — for multi-turn traces where turns interleave (not sequential), build a full request-level simulator with arrival timestamps

---

## Files Modified

| File | Changes |
|------|---------|
| `llm_predict/export_serving_predictions.py` | Symmetric error metric; `_serving_tp_size()`; TP passed through; prior routing |
| `llm_predict/cache_aware.py` | Per-turn concurrency; TP parameter; `ttft_floor_ms`/`ttft_first_decode_ms`/`ttft_queue_ms` fields |
| `llm_predict/serving.py` | `tensor_parallel_size` in all functions; floor + first_decode + queue model; `_sim_*` override params |
| `llm_predict/composer.py` | `tensor_parallel_size` kwarg in all predict methods |
| `llm_predict/prefix_cache_priors.py` | **New** — builds/looks up cache priors from prompt structure |
| `llm_predict/data/prefix_cache_priors.json` | **New** — coding-singleturn prior |
| `llm_predict/serving_sim.py` | **New** — event-driven scheduler simulation (not wired) |
| `inference-benchmark/dashboard/src/components/GemmPage.tsx` | Error display unchanged (%, same thresholds) |
| `llm_predict/tests/test_cache_aware.py` | Updated FakeComposer; added prior/simulation tests |
