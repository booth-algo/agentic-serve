# Per-Op vs Per-Kernel: the arch-coverage tradeoff

Experimental grid run 2026-04-21 on A100-only data comparing per-op
(26-feature analytical layer predictor) and per-kernel (per-family
shape-feature XGBoost) on the same train/test configurations. All
e2e numbers are absolute error on prefill_seq128_bs1 summed across
the model's 80-layer (Llama-70B/Llama-3.3-70B/Qwen-72B) or 32-layer
(Llama-8B/Mixtral) transformer stack.

## Headline

| experiment | per-op e2e err | per-kernel e2e err |
|---|---:|---:|
| Llama-8B → Llama-70B (10× pure scale, no sweep aux) | 28.62% | 58.50% (w/o roofline) |
| Llama-8B + **roofline sweep** → Llama-70B | — | **6.21%** |
| Llama-8B + Mixtral + roofline → Llama-70B | 28.62% | 6.35% |
| full 5-model pool + sweeps → Llama-70B | — | 7.63% |
| in-arch: Llama-family (incl. 70B) → Llama-3.3-70B | **0.80%** | **0.16%** |

**The synthetic-GEMM roofline sweep is the load-bearing ingredient
for per-kernel cross-scale**: without it per-kernel collapses to 58%.
With it, per-kernel trained on only Llama-8B + roofline hits 6.21% on
Llama-70B — a 10× parameter extrapolation with no 70B-class training
model.

**Per-op does NOT benefit from the roofline sweep** (its features are
layer-level, not kernel-level). Per-op's path to low e2e error is
through architecture coverage: it hits 0.80% only when an
architecturally-identical anchor model (Llama-70B in this case) is
present in training.

## Per-op cross-family matrix

Train on one family's models, predict a model in a different family.

| train pool | test | full MAPE (all bs/seq/kv) | e2e bs=1 seq=128 |
|---|---|---:|---:|
| Llama-family {8B, 70B} | Qwen-72B | 35.3% | n/a (row missing) |
| Qwen-only {Qwen-72B} | Llama-70B | 41.0% | **1.25%** |
| non-Llama {Qwen, Mixtral, gpt-oss-20b} | Llama-70B | 37.2% | **4.69%** |

Qwen-72B → Llama-70B e2e is 1.25% despite high full-grid MAPE. Reason:
the two share identical transformer dims (d=8192, h=64, kv=8) and
only differ on ffn (29568 vs 28672); per-op's architecture features
(d, h, kv, ffn) map this as a near-identity transform. The high
full-grid MAPE (41%) comes from (bs, seq, kv_cache) cells outside
the prefill_seq128_bs1 point.

## Per-op cross-scale matrix

Holding out every 70B-class model, asking per-op to extrapolate from
smaller models.

| train pool | test | full MAPE | e2e |
|---|---|---:|---:|
| Llama-8B only | Llama-70B | 36.4% | **28.62%** |
| Llama-8B + Mixtral | Llama-70B | 36.4% | 28.62% |
| Llama-8B + gpt-oss-20b | Llama-70B | 41.3% | 38.06% |
| no 70B-class {Llama-8B, Mixtral, gpt-oss-20b, gpt-oss-120b} | Llama-70B | 47.4% | 59.37% |
| Llama-8B only | Llama-3.3-70B | 35.5% | 28.58% |

Per-op cannot bridge a 10× parameter scale without a same-arch anchor.
Adding more architecturally-distant models to the pool (gpt-oss-20b,
Mixtral, 120B MoE) makes predictions WORSE, not better, because the
model learns conflicting scale-latency patterns.

## Per-op cross-MoE matrix

Can dense predict MoE, or vice-versa?

| train pool | test | full MAPE | e2e |
|---|---|---:|---:|
| MoE only {Mixtral, gpt-oss-20b} | Llama-70B (dense) | 209.6% | **358.31%** |

Catastrophic. When only MoE-style architectures are in training, the
model's E and k feature one-hots dominate and it cannot extrapolate
to dense (E=0, k=0) at all.

## Per-kernel cross-scale matrix

Direct mirror of per-op cross-scale, but at the per-kernel level.

| train pool | test | per-kernel e2e |
|---|---|---:|
| Llama-8B only (no sweep) | Llama-70B | 58.50% |
| **Llama-8B + roofline** | **Llama-70B** | **6.21%** |
| Llama-8B + Mixtral + roofline | Llama-70B | 6.35% |
| full 5-model pool + roofline + misc sweeps | Llama-70B | 7.63% |
| Llama-8B + Llama-70B + roofline (in-arch) | Llama-3.3-70B | **0.16%** |

Per-kernel's secret is the roofline GEMM sweep (~1000 synthetic matmuls
covering the full shape space from M=1 to M=16384). Without the sweep,
per-kernel is just as bad as per-op at cross-scale. **With** the sweep,
per-kernel generalises from 8B to 70B at 6.21% using only Llama-8B
kernels as the model-specific signal.

## Why per-kernel wins cross-scale

Transformer inference kernels are shape-parameterised but
model-agnostic: the same `ampere_bf16_s16816gemm_256x128_...` tile
family runs on Llama-8B's o_proj (M=128, N=4096, K=4096) and
Llama-70B's o_proj (M=128, N=8192, K=8192). The roofline sweep
densely samples the (M, N, K) shape space at bf16 on A100, giving
the per-kernel predictor direct measurements of every shape it
will see at inference time — including shapes that only appear in
70B-class models.

Per-op features are coarser: attention latency for a whole layer
as a function of (d, h, kv, ffn, n_tokens). Scale extrapolation
requires the model to learn the hidden "how does latency grow
with d" function; with only 8B-scale training points (d=4096),
this is an extrapolation well outside training support (d=8192).

## Recommendation for the paper

| scenario | recommended predictor |
|---|---|
| Have ncu access, want to predict novel model scales | **per-kernel + roofline sweep** |
| Have ncu access, want to predict novel attention variants | **per-kernel** (if kernels have been seen) or per-op fallback |
| Cannot run ncu (cloud, H100 without perms, etc.) | per-op, but requires arch-matching training model |
| Have arch anchor in training, want faster inference | per-op (simpler composer, 10× faster predict) |

The honest framing: per-kernel is the workhorse for
architecture-transfer, per-op is a capable fallback when arch
coverage is available or ncu access isn't.

## Caveats

- All numbers are A100-only. Cross-hardware transfer is a separate
  experiment.
- Per-op results use the raw-us fit (no log transform), n_estimators=200,
  lr=0.1, max_depth=6 — exactly matching the v4 (perop_analytical_v4)
  scheme. Log-space fit was tried and consistently under-predicted by
  ~30% on prefill workloads.
- gpt-oss-120b rows were added manually (arch: d=2880, h=64, kv=8,
  ffn=2880, E=128, k=4) because model_specs.py doesn't yet carry it.
- Per-kernel numbers use existing kernels_labeled.csv with roofline
  + misc sweeps ingested and flash_sweep stripped (the flash sweep
  hurt due to multi-kernel dispatch mislabeling — documented in the
  commit log).
- The held-out metric is e2e at (bs=1, seq=128) specifically.
  Broader validation on decode rows (kv_cache_len > 0) is an
  open follow-up.

## Reproducing

Scripts used (on 2080ti clone, kev/predictor branch):

    /tmp/crossmatrix.py      # per-op grid (this doc)
    /tmp/perkernel_cross.py  # per-kernel grid (this doc)
    /tmp/v5_with_120b.py     # gpt-oss-120b backfill + v5 parity check
    /tmp/v5_fair.py          # v4-like held-out split for v5 comparison

Training data on 2080ti:
    ~/agentic-serve/llm_predict/training/per_op/data/per_op_labeled.csv   (12624 A100 rows incl. v4 backfill)
    ~/agentic-serve/llm_predict/training/per_kernel/data/kernels_labeled.csv  (128932 rows)
