# training/per_kernel

Offline training pipeline for the per-kernel shape-only XGBoost predictors consumed by `llm_predict/predictors/per_kernel/predictor.py`.

Ported from `runpod:~/per-kernel-rebuild/` (baseline: commit 9483ba9 scrap of per-category RF; the composition method — enumerate ops → predict per op → sum — survived and lives in `llm_predict/predictors/per_kernel/predictor.py`).

## What lives here

| Module | Role | Port source |
|---|---|---|
| `labeler.py` | ncu CSV → shape-back-annotated `kernels_labeled.csv` | `label_all_v3.py` |
| `trainer.py` | per-(GPU × family) XGBoost training + validation report | `train_shape_v2.py` |
| `composer.py` | enumerate ops per layer, call `PerKernelPredictor`, sum | `compose_percat.py` |
| `validate.py` | paper-facing MAPE tables per GPU | new, thin CLI |
| `model_specs.py` | `ModelConfig` dataclass bridging `/root/agentic-serve/model_configs/*.json` | new |
| `gpu_kernel_regex.py` | per-arch tile regex + flash_attn boundary detection | new (consolidates regex from `label_all_v3.py:68,126`) |
| `reports/` | committed Markdown validation outputs | new |

## Input data

- Raw ncu: `/root/llm/profiling-data/{A100,RTX3090,RTX2080Ti}/ncu/{model}/prefill_seq128_bs1.csv` (19 files, 133K rows). Mirrored to R2 at `s3://agent-bench/profiling-data/`.
- Model archs: `/root/agentic-serve/model_configs/*.json`.

## Output artifacts

- `llm_predict/training/per_kernel/data/kernels_labeled.csv` — single combined labeled dataset with `gpu` column.
- `llm_predict/profiling/data/{A100,RTX3090,RTX2080Ti}/trained/per_kernel/perkernel_{gemm,flash_attn,elementwise,misc}_shape_v2.pkl` — 12 pkls (or fewer if 2080Ti misc has <5 rows).
- `llm_predict/training/per_kernel/reports/{gpu}_validation.md` — per-GPU MAPE + aggregate error tables.
- R2 upload via `upload_to_r2.sh` → `s3://agent-bench/profiling-data/{GPU}/trained/per_kernel/`.

## Reproduction

```bash
# Phase A1 — back-annotate
python -m llm_predict.training.per_kernel.labeler \
    --ncu-root /root/llm/profiling-data \
    --out llm_predict/training/per_kernel/data/kernels_labeled.csv

# Phase A2 + A3 — train + validate
python -m llm_predict.training.per_kernel.trainer \
    --data llm_predict/training/per_kernel/data/kernels_labeled.csv \
    --out-dir llm_predict/profiling/data

# Phase 4 — composition validation
python -m llm_predict.training.per_kernel.validate --gpu A100
python -m llm_predict.training.per_kernel.validate --gpu RTX3090
python -m llm_predict.training.per_kernel.validate --gpu RTX2080Ti
```

## Roofline GEMM sweep (for shape extrapolation)

Model-prefill ncu alone gives sparse shape coverage (a handful of fixed GEMM
shapes per model), so held-out 70B-class aggregate MAPE is ~65%. The roofline
sweep fills the shape space with 1,071 matmuls per GPU (21 M values × 17 dense-
model N/K templates × 3 reps) covering every QKV/FFN/LM-head shape for
Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, and Qwen-72B.

```bash
# 1) Run on each GPU host (takes ~15–25 min per host, ~1 kernel/sec under ncu):
#    gpu-4 (A100)
NCU=/usr/local/cuda-12.6/bin/ncu \
PY=/data/kevinlau/miniconda3/bin/python \
bash collect_gemm.sh A100

#    3090
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu \
PY=/home/kevinlau/miniconda3/envs/vllm/bin/python \
bash collect_gemm.sh RTX3090

#    2080Ti (fp16 — Turing can't bf16 tensor-core)
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu \
PY=/home/kevinlau/miniconda3/envs/vllm/bin/python \
bash collect_gemm.sh RTX2080Ti fp16

# 2) Copy /tmp/ncu_gemm_sweep_{gpu}.{csv,manifest.json} back to Hetzner.

# 3) Merge into labeled CSV (one call per GPU):
for GPU in A100 RTX3090 RTX2080Ti; do
  python -m llm_predict.training.per_kernel.roofline_labeler \
      --sweep-prefix /tmp/ncu_gemm_sweep_${GPU} \
      --gpu ${GPU} \
      --append-to llm_predict/training/per_kernel/data/kernels_labeled.csv
done

# 4) Retrain + re-validate (pkls overwrite in place).
python -m llm_predict.training.per_kernel.trainer
python -m llm_predict.training.per_kernel.validate
```

Expected post-sweep improvement: held-out 70B-class aggregate MAPE should drop
from ~65% to the runpod baseline (~4%) as the predictor sees K=28672/29568 and
M=16384 during training instead of extrapolating.

## GPU coverage

| GPU | Models | Notes |
|---|---|---|
| A100 | Llama-3.1-{8B,70B}, Llama-3.3-70B, Mixtral-8x7B, Qwen2.5-72B, Qwen3.5-{9B,27B}, gpt-oss-20b | layer-walker via flash_attn boundaries |
| RTX3090 | same minus Llama-3.3-70B (7 models) | layer-walker works (1936 flash_fwd kernels across 6 of 7 models) |
| RTX2080Ti | Llama-3.1-8B, Qwen3.5-9B (2 models) | **no flash kernels** (sm_75 pre-FA2); uses v2-style tile + candidate-set matching only; LOMO skipped (2 models degenerate) |

## Known bugs fixed during port

- `dispatch.py:18` default `gpu='H100'` → `'A100'` (H100 had no pkls).
- `label_all_v3.py:68` classifier regex `r'(ampere|hopper|sm\d+).*gemm'` extended with `turing` to stop dropping RTX2080Ti kernels into `'other'`.
- Hash-suffixed pkl naming (`gemm_171c4dbfdd45f87d.pkl`) retired; runtime loader's `perkernel_{family}_shape_v2.pkl` is the single convention.

## Out of scope

- Composition strategy redesign (option 1 locked at 9483ba9).
- per_op predictor changes.
- Decode-phase kernels.
- H100 data (blocked by runpod container perms).
- FP8 / MXFP4 model coverage.
