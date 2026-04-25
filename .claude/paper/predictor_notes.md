# Per-Kernel Predictor — Scoping Notes

Durable decisions for the per-kernel latency predictor training pipeline
(`llm_predict/training/per_kernel/*`). This file is the single source of
truth when a design choice could otherwise get re-litigated.

## The 6 Target Models

The paper's per-kernel predictor evaluates on these 6 dense, standard-attention
models across A100 / RTX 3090 / RTX 2080 Ti:

| # | Short name | Architecture | Params | Attention |
|---|---|---|---|---|
| 1 | Llama-3.1-8B   | dense          | 8B     | GQA (full attention) |
| 2 | Llama-3.1-70B  | dense          | 70B    | GQA (full attention) |
| 3 | Llama-3.3-70B  | dense          | 70B    | GQA (full attention) |
| 4 | Mixtral-8x7B   | MoE (8 × 7B, top-2) | ~47B active | GQA (full attention) |
| 5 | Qwen2.5-72B    | dense          | 72B    | GQA (full attention) |
| 6 | gpt-oss-20b    | dense          | 20B    | standard (full attention) |

These 6 all fit the single predictor family structure: every transformer
layer has the same op composition — `{norm, q_proj, k_proj, v_proj,
flash_attn, o_proj, norm, gate_proj, up_proj, down_proj}` — so an
e2e latency estimate is just `Σ per-op × n_layers + fixed_overhead`.

## Excluded: Qwen3.5 (hybrid attention)

Qwen3.5-9B and Qwen3.5-27B are collected in the raw ncu CSVs (they *are*
in `/root/llm/profiling-data/{GPU}/ncu/Qwen3.5-*/`) but are **deliberately
excluded from the `flash_attn` family training pool**.

**Why.** Qwen3.5 uses interleaved attention: only 1 in every 4 layers is
`full_attention` (emits `flash_fwd_*` kernels that the classifier correctly
tags). The other 3 in every 4 are `linear_attention`, which vLLM implements
via triton's `chunk_gated_delta_rule` kernels. The current
`gpu_kernel_regex.classify_kernel()` table routes those into
`misc/reduce` rather than `flash_attn` — which would be fine, except:

1. A per-layer composer that treats Qwen3.5 as "4 flash_attn layers per
   group" over-predicts flash_attn time by 4×.
2. Adding a new `linear_attn` family to fix that would give us a training
   pool of exactly 2 models (9B + 27B), which makes leave-one-model-out
   validation degenerate and held-out MAPE untrustworthy.
3. The 6 target models above are all full-attention, so Qwen3.5 support
   is not required for the paper's headline numbers.

**What this means concretely:**

- `labeler.py` still processes Qwen3.5 ncu data (GEMMs, elementwise,
  misc still labeled — those are shape-only and don't depend on attention
  type), but Qwen3.5 rows whose `kernel_family == flash_attn` do not
  contribute to the flash_attn pkl's training pool.
- The runtime composer at `llm_predict/predictors/per_kernel/predictor.py`
  already carries the comment "Skip Qwen3.5 from composition validation
  until the composer adds linear-attention support" — keep that comment.
- `validate.py` / paper MAPE tables omit Qwen3.5 from the per-GPU
  aggregate rows.

**Future work.** Add a `linear_attn` family (covering triton
`chunk_gated_delta_rule`, `chunk_scan_fwd`, and the Qwen3.5-specific
`delta_rule_recurrent` kernels) once we collect ncu data from at least
3 hybrid-attention models. Until then, Qwen3.5 stays out of the per-kernel
predictor. Per-op and e2e predictors can still be fit to Qwen3.5 separately
if needed — they don't share this constraint.

## Other Pipeline Decisions (2026-04-21)

**Output format: CSV (not parquet).** Per-family split outputs written as
CSV under `llm_predict/training/per_kernel/data/per_family/{GPU}/{family}.csv`.
Kept human-greppable at the cost of ~5× disk overhead vs parquet. The
combined `kernels_labeled.csv` remains the raw source-of-truth; per-family
CSVs are derived and regenerable.

**Scope boundary.** The data-formatting plan stops at:
- per-family CSVs,
- a frozen `feature_spec.py` (extracted from trainer.py constants),
- LOMO + held-out split iterators (`splits.py`),
- a smoke test.

It does **not** modify `trainer.py` or produce trained pkls. `trainer.py`
continues to read the combined CSV until a subsequent plan rewrites it to
consume the per-family CSVs. This lets us eyeball the per-family outputs
before touching training code.

## GPU + Model × Family Coverage

After the Qwen3.5 exclusion, the flash_attn training pool shrinks; other
families are unaffected:

| Family       | A100 models | RTX3090 models | RTX2080Ti models | Notes |
|--------------|-------------|----------------|------------------|-------|
| gemm         | 6           | 5 (no Llama-3.3-70B) | 1 (Llama-3.1-8B) | Qwen3.5 rows admitted — shape-only |
| flash_attn   | 6           | 5              | 0 (sm_75 pre-FA2) | **Qwen3.5 excluded** |
| elementwise  | 6           | 5              | 1                | Qwen3.5 rows admitted |
| misc         | 6           | 5              | 1                | Qwen3.5 rows admitted |

RTX 2080 Ti training pool remains 1–2 models even with Qwen3.5 included
for the non-flash families — LOMO is skipped there per `HELD_OUT_BY_GPU`
in `model_specs.py`.

## Changelog

- **2026-04-21** — initial file. Locked: drop Qwen3.5 from flash_attn
  family (option (a)); CSV output (not parquet); scope ends before
  trainer.py changes.
- **2026-04-21 (later, 2080ti train run)** — trainer produced 11 pkls
  across 3 GPUs (A100×4, RTX3090×4, RTX2080Ti×3; no 2080Ti flash_attn
  per sm_75 pre-FA2). Headline aggregate held-out error on
  `prefill_seq128_bs1`: **A100 ~50%**, **RTX3090 ~52%** for all 70B/72B
  held-out models — matches predictor_notes' expected pre-roofline
  baseline. Per-family held-out MAPE on A100: gemm 65%, flash_attn 3.5%,
  elementwise 13.7%, misc 15.3%. Qwen3.5 flash_attn exclusion now
  enforced in both `split_by_family.py` and `trainer.py` via
  `feature_spec.FAMILY_EXCLUDED_MODELS`.
- **2026-04-21 (roofline sweep, 2080ti only)** — ran `collect_gemm.sh
  RTX2080Ti fp16` on 2080ti GPU 1 (cuBLAS issued 1038 of expected 1071
  matmuls, 5191 ncu rows). Roofline-appended LOMO MAPE on 2080Ti gemm:
  Llama-8B held-out 168.7% (was 181.8%), Qwen3.5-9B held-out 75.1%
  (was 83.0%). Modest win — 2-model pool + no held-out 70B is
  structurally the limit. A100 and RTX3090 roofline sweeps deferred
  (Phase 4).
- **2026-04-21 (per-op gap documented)** — the only trained per-op pkl
  that exists anywhere is `A100/trained/perop_analytical_v4.pkl`
  (pulled from R2 2026-04-15, 673 KB, 6 models, no heldout_mape logged
  in payload). RTX3090 and RTX2080Ti have no per-op pkls and no per-op
  CUDA-event traces. H100 has raw `*_moe_profile.csv` for Mixtral +
  gpt-oss-20b on R2 but no trained pkl. Per-op productisation planned
  for Phase 2 of the profiling-coverage plan.


## 2026-04-22 -> 2026-04-25 — Wall-clock validation campaign

### TN-vs-NN cuBLAS kernel discovery (the headline fix)

The roofline GEMM sweep was producing measurements ~2.7x higher than
real model prefill kernels at the same `(M, N, K)`. Root cause: vLLM's
`nn.Linear` lands on an `ampere_bf16_s16816gemm_..._tn` kernel (TN
layout), but `sweep_gemm.py`'s `a @ b` dispatch hit the `_nn` variant.
Same tile, dtype, and stage count - but a different memory access
pattern with substantially different latency.

Fix: switched `sweep_gemm.py` to construct a `torch.nn.Linear` module
and call it directly, matching the TN path vLLM actually uses. Also
bumped warmup + measured reps to 10+10 (was 3+3 with no warmup).
ncu-rep kernel names confirm the new sweep emits the `_tn` variant
across all profiled shapes.

### Wall-clock validation methodology (new)

Added `validate.py --vs-measured --target-seq N --seq-tolerance T` mode
to the per-kernel validator. It pulls `summary.median_ttft_ms` from
`inference-benchmark/dashboard/public/data.json` (real vLLM/SGLang bench
runs) and compares to `composer.predict_ttft_ms(model, seq=avg_seq)`,
restricted to rows whose per-request avg input seq lies within
+/-tolerance of `--target-seq`. Apples-to-apples comparison vs the ncu
sweep which is fixed at seq=128.

Companion benchmark profile `fixed-seq128` added to inference-benchmark
(ISL=128, OSL=128, random tokens, stress-test mode -> no prefix cache,
matches the ncu prefill_seq128_bs1 sweep). Chat-template overhead lands
the actual server-observed seq at ~167, hence the seq=167 reports.

### Cross-GPU per-kernel wall-clock results

After the nn.Linear fix, on Llama-3.1-8B fixed-seq128 bs=1 vs vLLM 0.19:

| GPU       | composer pred | measured TTFT p50 | composer MAPE | ncu Sigma | overhead % |
|-----------|--------------:|-------------------:|---------------:|-----------:|-----------:|
| A100-40GB | 37.43 ms      | 36.71 ms           | **1.97%**      | 26.48 ms   | 27.9% |
| RTX3090   | 58.77 ms      | 59.66 ms           | **1.49%**      | 55.06 ms   | 7.7% |
| RTX2080Ti | 49.72 ms      | 81.05 ms           | 38.66%         | 65.56 ms   | 19.1% |

A100's previous wall-clock MAPE of **64.76%** was a vLLM 0.11 artefact
(the `vllm` env that happened to be on `gpu-4` was 0.11.0; flashinfer
sampler was disabled to bypass missing CUDA headers). After upgrading
to vLLM 0.19 the same composer prediction lands at 1.97% - a 32x
improvement just from the version bump, before any sweep changes.

### Cross-GPU per-kernel ncu held-out (cross-scale extrapolation)

| GPU      | Llama-8B in-train | Llama-70B held-out | Llama-3.3-70B held-out | Qwen-72B held-out |
|----------|-------------------:|--------------------:|-------------------------:|-------------------:|
| A100     | 37.59%            | **1.33%**           | 1.53%                    | 1.37%              |
| RTX3090  | 0.08%             | 2.19%               | n/a (not held)           | 2.37%              |

The composer was never trained on any 70B-class rows, yet predicts
their summed kernel time within ~1-2% on both A100 and RTX3090.
Aggregate Sigma-err on the supported subset is **2.22%** on A100 and
**2.14%** on RTX3090 (4 dense models in scope).

### Out-of-scope rows captured for completeness

| GPU / model              | arch        | composer pred (ms) | wall-clock (ms) | err |
|--------------------------|-------------|-------------------:|-----------------:|----:|
| A100 / gpt-oss-20b       | moe         | 12.00              | 33.88            | 64.59% |

`gpt-oss-20b` shows a **negative overhead %** (-78.1%): wall-clock
TTFT is *less* than ncu kernel sum because ncu sums all expert kernels
fired during profiling, but at inference each token routes to only
k=2 of E experts. The composer separately under-predicts MoE because
gate/router compute, scatter/gather, and load imbalance aren't modelled.

### RTX2080Ti caveat (Turing-specific)

2080Ti wall-clock MAPE is 38.66% even after the nn.Linear fix. The new
sweep does emit `_tn` kernels but cuBLAS's heuristic on Turing picks a
different tile (`64x64_sliced1x2` vs A100's `256x128`) for the same
shape - the algorithm catalog is smaller on sm_75. The composer is
trained on a different cuBLAS algorithm than vLLM hits in production
on 2080Ti. Treated as a known-caveat for the paper; mitigations would
need either kernel-level cuBLAS algo pinning or a tile-aware feature.

### Per-op pool expansion (2026-04-24 -> 2026-04-25)

Per-op held-out MAPE was high because LOMO removed arch-similar
anchors. Added two new per-op profiles:

1. **Qwen2.5-7B-Instruct on RTX2080Ti** (3rd pool model, enables LOMO):
   - Pool: Llama-8B + Qwen3.5-9B; held out: Qwen2.5-7B
   - **17.15% MAPE** - first measurable per-op number on 2080Ti
   - Same-family anchor (Qwen3.5-9B) makes Qwen-7B prediction tractable
2. **Llama-3.1-70B-Instruct on A100 + RTX3090** (arch-anchor for 70B):
   - 3090 with thin grid (76 rows): **38.66% -> 13.95%** (47% reduction)
   - A100 with thin grid: **38.66% -> 30.68%** (modest)
   - A100 dense grid (2048 rows): **30.68% -> 26.72%** (small further gain)
   - 3090 dense grid: regressed to 23.86% - Llama-70B at 88% of training
     mass overpowered other 4 pool models which still had 76 rows each.
     Confirmed the issue is data-volume balance, not coverage.
   - Profiler default grid was patched from 19 cells to 512 cells (bs=1,
     seq 1..512) to match v4 density.

### Methodological caveat: per-op torch.profiler overhead

Discovered that per-op training rows on RTX3090 measure ~50-85% higher
than ncu kernel-exec time for the same `(bs, seq, op)`. Example: 3090
Llama-8B seq=128 norm_pre training value = **322 us**, but a real
rmsnorm on 3090 is bandwidth-bound and exec'es in ~1.5 us. The bulk is
torch.profiler sync + dispatch + event-record overhead per op.

Composer composition (4 ops x 32 layers) amplifies this to ~2x wall-
clock. This is the root cause of the 96.92% per-op e2e MAPE on 3090
when initially compared against wall-clock at seq=167 - not a
predictor bug, but a measurement-granularity mismatch between
torch.profiler-based per-op data and ncu-based per-kernel data.

For the paper: per-kernel (ncu) measures pure kernel time -> aligns
with wall-clock; per-op (torch.profiler) measures CPU-observed op
time -> composer = wall-clock + profiler-overhead floor. The two
predictors are answering different questions despite predicting the
same end metric.

### Dashboard surfacing

`inference-benchmark/dashboard/src/components/ProfilingPage.tsx`
gained a new wall-clock sub-section under the existing Predictor MAPE
panel, populated by `publish_profiling_state.py` parsing the
`{gpu}_wallclock_validation_seq167.md` reports. Live once PR #4 merges.

## Changelog (continued)

- **2026-04-22** - `sweep_gemm.py`: switched `a @ b` -> `nn.Linear`,
  bumped warmup+reps to 10+10. Cross-GPU re-sweep on A100 + RTX2080Ti.
  Wall-clock MAPE A100 31.85% -> 1.97% (vs vLLM 0.19), RTX3090
  unchanged at 1.49% (sweep not re-run, didn't need it),
  RTX2080Ti 110.31% -> 38.66%. Committed `c67777b`.
- **2026-04-22** - A100 vLLM env upgraded 0.11.0 -> 0.19.0; the
  flashinfer-sampler-disabled penalty disappeared, A100 wall-clock
  measured TTFT dropped 71.0 ms -> 36.7 ms on Llama-8B (consistent
  with cuBLAS heuristic + sampler change between versions).
- **2026-04-24** - added Qwen2.5-7B-Instruct to RTX2080Ti per-op pool
  (3 models -> enables LOMO). Held-out Qwen-7B -> **17.15%**. Committed
  `8413626`.
- **2026-04-24** - added Llama-3.1-70B-Instruct per-op profiles
  (thin 19-cell grid) on A100 + RTX3090; moved from heldout -> pool.
  3090 dropped 26.58% -> 13.95%; A100 38.66% -> 30.68%. Committed
  `fd5f005`.
- **2026-04-24 (later)** - dense 512-cell re-profile of Llama-70B on
  A100 + 3090 to match v4 grid density. A100 ticked down to 26.72%;
  3090 regressed to 23.86% from data imbalance (Llama-70B's 2048 rows
  vs other pool models' 76 rows each). Need 3090 to also re-profile
  the other 4 pool models densely for parity (in progress).
- **2026-04-25** - appended `wallclock` field to
  `profiling-state.json` schema; ProfilingPage.tsx renders the new
  table. Committed `52b7cef`.
