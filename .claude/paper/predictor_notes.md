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
