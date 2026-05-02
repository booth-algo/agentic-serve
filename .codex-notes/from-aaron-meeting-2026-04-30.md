# Key Direction from Aaron Meeting — 2026-04-30

## Core Principle

The serving predictor should work from first-principles physics. The two
dominant kernels are **GEMM** and **flash attention**. If they are composed
properly based on ISL/OSL (which tells you how much prefill vs decode is
happening), the predictor should not need a large calibration dataset or
empirical correction factors to match ground truth.

> "If you aggregate them properly based on ISL, OSL which tells you how much
> prefill or decode there is, then you should be able to get a good prediction
> and I do not have to need a large training set of ground truth to kind of
> tune the model, because it should aggregate properly anyways."

## Action Taken

- Removed the framework TTFT correction (alpha * kernel_ttft + beta_ms) from
  `framework_corrections.py` and `serving.py`.
  - TTFT is now raw kernel + queue factor only.
  - `_DECODE_CORRECTION` neutralized (all alpha_base=1.0, exponent=0.0).
  - No more `ttft_correction_alpha` / `ttft_correction_beta_ms` in export.
- Prefix-cache-affected rows without per-turn data or a trace-derived prior
  are now flagged as `unsupported` instead of silently falling back to
  (wrong) full-prefill prediction.

## What Remaining Error Means Now

Without correction factors, any gap between predicted and measured is a
signal about what's missing from the physics model:

1. Flash attention (QK transpose, KV cache allocation, attention over full
   context) — this is the most likely source of the fixed ~8-18ms TTFT
   overhead gap.
2. CUDA graph overhead and serving framework scheduling.
3. KV cache residency and prefix-cache contention.

## Paper Story

The contribution is not "we ran many benchmarks." The story is:

- We built a physics-based predictor (GEMM + flash attention) that can take
  any workload profile (ISL/OSL/turn distributions) and predict serving
  latency without running GPUs.
- We validated it with a targeted set of benchmark runs across GPU types,
  models, and workloads — enough to prove the predictor works, not a
  brute-force coverage grid.
- The benchmark data is *evidence for the predictor*, not the product itself.

## Scope

- Single-node multi-GPU (TP=1,2,4,8) is the current scope. State it clearly.
- Multi-node is future work.
- Paper should scope to what exists now rather than promising future work.

## Coverage Philosophy

- Enough data to build a good predictor, not exhaustive coverage.
- Missing cells in the grid are fine — the predictor fills them.
- Present as: "we have good empirical models (GEMM + attention roofline) +
  targeted validation → cover many profiles without running them all."
