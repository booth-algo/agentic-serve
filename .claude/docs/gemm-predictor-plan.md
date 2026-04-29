# GEMM Predictor Overhaul Plan

## Problem
XGBoost GEMM predictor has wrong inductive bias — memorizes training shapes,
fails on held-out architectures. A100 held-out MAPE: 396%. GEMM performance
has sharp dispatch/tile/shape regime changes that tree models can't learn.

## Architecture

### 1. Serving-GEMM Shape Table (not generic predictor)
Profile exactly the GEMMs the composer emits:
- QKV, O, gate, up, down, LM head
- For Llama/Qwen/GPT-OSS/Mixtral-like configs
- TP = 1, 2, 4
- M = decode bs grid (1..320) + prefill ISL grid (128..8192)

Interpolate mostly over M. For a given model, N and K are fixed per op.
Much easier than learning arbitrary (M,N,K) -> ms.

### 2. Physics Baseline + Learned Residual
Instead of: `XGBoost(M,N,K) -> raw latency`
Use: `roofline_baseline(M,N,K) -> expected_ms`, then `XGBoost -> correction_factor`
Final = baseline * correction.

Prevents 400% nonsense — model explains residuals, not matrix multiplication.

### 3. Split GEMM by Regime
- prefill GEMM: large M (512..8000)
- decode GEMM: tiny/tall-skinny M (1..80)
- LM head GEMM
- MoE expert GEMM

Different cuBLAS behavior per regime. Should not share one model.

### 4. Shape-Coverage Diagnostics
For every predicted GEMM, compute distance to nearest profiled shapes.
If outside coverage → flag it, fall back to analytical roofline or trigger profiling.
Would have caught the A100 70B failure: "these shapes are outside the trusted manifold."

### 5. Targeted H100 Data Collection
Not "more random rows" but specific high-value shapes:
- H100 Llama-70B/Qwen-72B serving GEMM grid
- H100 decode-attn grid
- H100 end-to-end calibration points
Few hundred targeted shapes > thousands of generic kernels.

## Implementation Path
1. Keep current composer pipeline
2. Replace `predict_gemm` with hybrid/table-backed predictor
3. Generate `gemm_serving_shapes.csv` from `composer.py` (enumerate all shapes it would emit)
4. Profile those shapes on H100/A100 (via ncu or CUDA events)
5. Use exact lookup/interpolation when covered, residual roofline otherwise
6. Re-run serving E2E validation

## Key Files
- `llm_predict/training/per_kernel/composer.py` — emits GEMM calls
- `llm_predict/predictors/per_kernel/predictor.py` — current XGBoost predict_gemm
- `llm_predict/training/per_kernel/gemm_extrapolation_test.py` — shape test
- `inference-benchmark/scripts/roofline/run_ncu.py` — ncu GEMM sweep

## Status
- [ ] Generate gemm_serving_shapes.csv from composer
- [ ] Profile shapes on H100 + A100
- [ ] Build table-backed GEMM predictor with roofline fallback
- [ ] Add coverage diagnostics
- [ ] Validate serving E2E
