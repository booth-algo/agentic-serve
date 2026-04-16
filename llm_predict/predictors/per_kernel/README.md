# per_kernel — True per-CUDA-kernel latency predictor

**Status: WIP (not yet in repo — lives on runpod at `~/per-kernel-rebuild/`)**

## Scope

Finest-grained predictor in the hierarchy:

| Granularity | Class / Module | What it predicts |
|-------------|----------------|------------------|
| per-op | `PerOpPredictor` | 4 ops per layer: attn, ffn, norm_pre, norm_post |
| **per-kernel (this)** | `PerKernelPredictor` | **1 predictor per individual CUDA kernel invocation, keyed on shape** |

## Design

`per_kernel` uses shape-only features predicting latency at the individual
kernel-variant level, trained on ncu ground-truth per
(kernel_name, shape, GPU, backend). One XGBoost model per family
(gemm, flash_attn, elementwise, misc). Paper Step 3 design.

## Required training data

- ncu CSVs with `Kernel Name`, `dram__bytes_read.sum`, `dram__bytes_write.sum`,
  `launch__*`, `gpu__time_duration.avg`
- Back-annotated with (M, N, K, dtype) via iteration-order mapping
  (see `/root/per-kernel-rebuild/extract_shapes.py` on runpod)

## Current WIP state (runpod, not in repo)

- `perkernel_gemm_shape_v1.pkl` — GEMM-only, 224 roofline kernels, 0.69% train MAPE
- Known issues: sparse (only 56 unique shapes), extrapolates poorly to unseen M

## Next steps to graduate into repo

1. Expand training set by back-annotating model-prefill ncu CSVs
   (Llama-8B, Mixtral, etc. at bs=1, seq=128)
2. Validate leave-one-model-out CV ≤10% MAPE per kernel family
3. Add separate predictors for flash_attn, elementwise (not just GEMM)
4. Write a `predictor.py` exposing `load/predict_*` for drop-in use by
   `dispatch.py`
5. Per-GPU pinning: train separate A100 and H100 predictors
6. Move trained pickles to R2 `profiling-data/{GPU}/trained/per_kernel/`

## Blockers

- ncu access on runpod currently fails with `ERR_NVGPUCTRPERM`. Options:
  - Run ncu collection on gpu-4 (A100, works) when available
  - Request RunPod pod variant with `NVreg_RestrictProfilingToAdminUsers=0`
- Need to fix shape back-annotation for `ncu_gemm_sweep_raw.csv` (script origin unknown)
