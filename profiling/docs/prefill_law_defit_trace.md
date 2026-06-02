# TTFT prefill-law de-fit — deep-dive trace findings

**Date:** 2026-06-02. **Method:** deep-dive 3-lane causal trace (GPU per-token / host bookkeeping /
fit-structure) against the raw c1 benchmark + measured grids, plus an independent c1 re-fit. Goal:
understand what the fitted prefill law `ttft = 22.5 + 0.0310·new + 0.006103·cached` absorbs, so each
coefficient can be de-fitted. (External-literature audit via `/deep-research` folds in separately.)

## Verified starting point
The fit reproduces **exactly** from 63 c1 success turns: `floor=22.67, new=0.031025, cached=0.006105,
R²=0.963` (matches the committed constants). The *form* is excellent (R²=0.96); only the coefficients
are fitted-not-derived. **`new` and `cached` are cleanly separable** — `corr=0.042`, `VIF=1.00`, cond
17882→1.04 standardized (the high raw condition number is pure column-scale, not collinearity). So the
NEW-vs-CACHED decomposition is **identifiable** — de-fitting is well-posed.

## Two corrections to the earlier framing (important)
1. **NEW is NOT "~8× host-amplified" — it's 1.24× the realistic roofline.** The "8×" compared `0.0310`
   to `0.0042` (a *measured large-K asymptote* / a util=1 single-pass kernel), not a roofline. Against
   the project's own `util_flops=0.65` GEMM roofline (**0.02498 ms/tok**, from `roofline_params`),
   NEW=0.0310 is only **1.24×**. So **NEW is mostly real GPU prefill GEMM**, plus a small ~0.006 ms/tok
   per-new-token off-GPU residual (python/sampling/return). The per-chunk-launch interpretation is
   **falsified** (it implies a 19–35 ms launch — 3 orders too big — and 79% of c1 reqs are single-chunk
   yet still need the full slope).
2. **The CACHED 4× gap is NOT block-hashing / KV-alloc.** Those are negligible (~0.16 ms/1k: only 62.5
   16-tok blocks per 1k). The 6.1 ms/1k splits as **~1.5 ms/1k measured GPU paged-attention KV-gather**
   (`cached_prefill_v3_H100.csv`, R²=0.9992) + ~1.5 ms/1k tokenize + **~2.5 ms/1k unexplained host**
   (prefix-cache walk / GIL-serialized scheduler bookkeeping ∝ context length).

## Other findings
- **FLOOR (22.5) is genuine** — a highly significant intercept (F=116.5; dropping it inflates residual
  SS +194%), ≈ the smallest observed pure-prefill TTFT (26 ms). Not a turn-0/curvature artifact.
- **HOST shared/perreq split (0.003485 / 0.002618) is UNIDENTIFIABLE from c1** — at 1 req/step the two
  regressors are the identical `cached` column; only their SUM (0.006103) is estimable. The 57/43 split
  is *imported* from the batch CSV, not co-fit at c1 (the code presents it as if co-fit).
- **Leverage fragility:** the slopes are leverage-driven by the long-cached swebench tail — dropping
  `cached>5000` swings NEW 0.0310→0.0213 (−31%), CACHED →0.0073 (+20%). Bootstrap CV ~7%. Any de-fit
  anchor must reproduce the rates at the swebench operating point, not just on average.

## Per-constant de-fit verdict

| Constant | Verdict | How |
|---|---|---|
| **FLOOR** (22.5) | ✅ de-fittable from existing data | measured min pure-prefill (turn-0) TTFT ~22.5–26 ms; mild per-profile spread (terminalbench ~14) → universal floor carries ~8 ms pooled offset, or go per-profile. No new bench. |
| **NEW** (0.0310) | ✅ ~80% de-fittable now + small microbench | GPU part = `util=0.65` GEMM roofline **0.02498 ms/tok** (fit-free, from `roofline_params`). The ~0.006 ms/tok off-GPU residual needs the stage-split microbench. **Fix the "7×/8×" comment → 1.24×.** Validate at the swebench operating point (leverage ±31%). |
| **HOST_shared** (0.003485) | ⚠ unidentifiable at c1 — derive from existing batch B-sweep | Re-fit `per_req(B) = shared/B + perreq` on `cached_prefill_batch_ttft_H100.csv` (B=1,2,4,8,16 — already on disk). Makes the split a real measurement, not an import. No new GPU. |
| **HOST_perreq** (0.002618) | ⚠ same | Same joint fit; **subtract the measured ~1.5 ms/1k GPU paged-attn slope** (`cached_prefill_v3_H100.csv`) first so the split is host-only. The residual ~2.5 ms/1k host still needs the stage-split microbench. |

## Critical unknown + the one probe that resolves it
**Unknown:** the CPU-vs-device split of c1 TTFT — what the off-GPU residuals physically are (~0.006 ms/tok
on NEW, ~2.5 ms/1k on CACHED). No file in `profile_data/` logs a host-vs-device breakdown; every grid
reports only wall-clock TTFT.
**Probe (one microbench):** a c1 sweep over fixed-new × varying-cached with **CUDA-event device timing
around the model forward, separate from host wall-time** (NVTX / `torch.cuda.synchronize()` bracketing),
logging (a) `tokenizer.encode` ms, (b) device forward ms, (c) scheduler+sampling+detok+return ms; regress
each on new and cached. This partitions both residuals into host vs device and finishes the de-fit.

## Net
A lot is de-fittable **from data already on disk**: FLOOR (measured floor), NEW's GPU part (roofline),
CACHED's GPU part (v3 grid), and the HOST shared/perreq split (batch B-sweep). Only two residuals — NEW's
~0.006 ms/tok off-GPU and CACHED's ~2.5 ms/1k host — need **one** new stage-split microbench. The single
highest-value code fix: NEW is **1.24× the realistic roofline, not 7× host-amplified** (the comment is wrong).

Anchors: `simulator/ttft_queue_sim.py:99-120,522-559`, `profile_data/results/cached_prefill_v3_H100.csv`
(GPU cached-hit slope), `profile_data/results/cached_prefill_batch_ttft_H100.csv` (breaks the host-split
identity), `profile_data/kernels/prefill_profile_H100_dense.csv`, `profile_data/kernels/roofline_params_H100_llama31_8b.json`.

---

## /deep-research reconciliation (external literature, 2026-06-02)

The literature confirms the *mechanisms* and the *principled model*, but — its own honest caveat — does
**not** resolve the quantitative magnitudes (the 8× split and the 6-vs-1.5 ms/1k host question were flagged
unresolved). What it adds:

- **NEW's above-GEMM residual is framework dispatch, not per-chunk launch — confirming the trace.**
  *TaxBreak* (ISPASS 2026) decomposes per-kernel host latency into framework/ATen dispatch + CUDA-library
  front-end excess + an irreducible launch floor (~4.7 µs/kernel on H100), with GEMM showing the largest
  software excess. So NEW's ~0.006 ms/tok off-GPU residual is per-token framework-dispatch overhead. (Ignore
  the extracted "44 ms/token launch" figure — almost certainly a µs units misread.)
- **The canonical de-fitted blueprint already exists — *Vidur* (MLSys 2024):** model serving runtime as
  per-operator profiled regressors in 3 buckets — token-level ops **linear in total tokens** (= our NEW GEMM
  term), attention **quadratic in length** (= our FA3 term, already kernel-derived), and communication. This
  is exactly the fit-free structure to converge our prefill law toward: NEW → profiled token-level roofline,
  FA3 → quadratic (done), no free intercept-slope regression.
- **The unexplained ~2.5 ms/1k host residual now has a named lead: SHA-256 prefix-cache block hashing.**
  vLLM hashes each 16-token block with a *chained* hash over token IDs on the host critical path before
  `find_longest_cache_hit` — and the default is **SHA-256 since v0.11** (much slower than the built-in hash).
  The trace's bottom-up budget assumed a cheap hash (~0.13 ms/1k); SHA-256 over ~62 blocks/1k is materially
  larger and is the prime candidate for the residual. **ACTION:** check the benchmark's vLLM version / hash
  config — if SHA-256, model the per-block hash cost (and it would be a *measured/derivable* host term, not a
  fit). Cached/new is resolved at **block (16-tok) granularity**, so our per-token `cached` term is an approximation.
- **Chunked-prefill structure confirmed:** decode-prioritized scheduling, `max_num_batched_tokens` TTFT/ITL
  tradeoff, `long_prefill_token_threshold` per-step cap (8192/1310) — already in the queue sim.

**Net:** external lit gives the mechanism names (dispatch overhead for NEW; SHA-256 block-hash for the CACHED
host residual) and the Vidur per-operator blueprint, but **not magnitudes** → the host-vs-device stage-split
microbench is still required, and it should additionally toggle `hash=builtin` vs `SHA-256` to confirm the
block-hash hypothesis. Sources: TaxBreak arXiv 2603.12465; Vidur arXiv 2405.05465; vLLM anatomy blog +
prefix-caching design docs.
