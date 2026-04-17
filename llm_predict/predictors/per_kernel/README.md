# per_kernel — per-CUDA-kernel latency predictor (inference API)

Runtime-side predictor consumed by `PredictorDispatch`. One XGBoost model per kernel family, shape-only features, trained on ncu ground-truth `gpu__time_duration`.

**Hierarchy:**

| Granularity | Class / Module | What it predicts |
|---|---|---|
| **per-kernel (this)** | `PerKernelPredictor` | Single-kernel latency per op, keyed on shape |
| per-op | `PerOpPredictor` | 4 ops per layer: attn, ffn, norm_pre, norm_post |
| e2e | analytical composition | Σ per-op × n_layers + overhead (no ML) |

## API

```python
from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

p = PerKernelPredictor(gpu='A100')   # or 'RTX3090', 'RTX2080Ti'
if p.load():
    t_ms = p.predict_gemm(M=512, N=4096, K=4096, dtype='bf16')
    t_ms = p.predict_attention_prefill(bs=1, seq=512, n_heads=32, head_dim=128)
    t_ms = p.predict_elementwise('rmsnorm', numel=512*4096)
    t_ms = p.predict_misc('reduce', numel=65536)
```

All predict methods return ms, or `-1.0` if the family pkl isn't on disk. `load()` is idempotent and returns `True` if any family loaded.

## Pkl layout

```
llm_predict/profiling/data/{A100,RTX3090,RTX2080Ti}/trained/per_kernel/
    perkernel_gemm_shape_v2.pkl
    perkernel_flash_attn_shape_v2.pkl
    perkernel_elementwise_shape_v2.pkl
    perkernel_misc_shape_v2.pkl
```

Each pkl payload: `{model, feature_cols, target, kernel_family, gpu, dtype, n_training, heldout_mape, version}`. `target` is always `log_gpu_time_duration_ms`.

## Training

See `llm_predict/training/per_kernel/README.md` for the labeling + training + validation pipeline. Pkls are produced there and either committed under `llm_predict/profiling/data/…` or pulled from R2 via `llm_predict/training/per_kernel/pull_pkls.sh`.

## Composition

`PerKernelPredictor` alone is a point-estimate per op. To predict end-to-end TTFT, callers enumerate ops per layer, call the right `predict_*` per op, and sum. Reference implementation: `llm_predict/training/per_kernel/composer.py` (ported from `compose_percat.py`).

## Known limitations

- **flash_attn** training pool is small (tens of rows per GPU). Per-kernel MAPE on held-out GQA configs is high; aggregate-layer error stays low because flash_attn is a small fraction of total time.
- **RTX 2080Ti** lacks FA2 (sm_75); `flash_attn` family may be absent — `load()` skips it silently and `predict_attention_prefill()` returns `-1.0`.
- **H100** recognized but no pkls yet (blocked by runpod container profiling perms).
- Decode-phase kernels are not modeled — training data is prefill-only.
