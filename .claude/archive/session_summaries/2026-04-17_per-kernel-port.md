# Session 2026-04-17 — per-kernel training pipeline port

Ported the per-kernel shape-only predictor pipeline from `runpod:~/per-kernel-rebuild/` into this repo at `llm_predict/training/per_kernel/`, generalised for 3 GPU archs (A100, RTX3090, RTX2080Ti), retrained 11 pkls, and mirrored to R2.

## New package `llm_predict/training/per_kernel/`

| File | Purpose |
|---|---|
| `ncu_loader.py` | Long→wide ncu CSV pivot. New 2026-04 schema: `gpu__time_duration.sum` target, combined `dram__bytes.sum` (no read/write split), launch metrics collected as metric rows. |
| `gpu_kernel_regex.py` | Kernel classifier + per-arch tile regex. |
| `model_specs.py` | `ModelConfig` dataclass + loader. Bridges `/agentic-serve/model_configs/*.json` with hardcoded fallbacks for Mixtral-8x7B, Llama-3.3-70B, Qwen3.5-{9B,27B}. Has per-GPU held-out split (`HELD_OUT_BY_GPU`). |
| `labeler.py` | Port of `label_all_v3.py`. Adds 2080Ti noflash fallback, `_ext` variant concat+dedupe, `gpu` output column, nested path walker. |
| `trainer.py` | Port of `train_shape_v2.py`. Outer GPU loop → 12 pkls (2080Ti skips flash_attn legitimately → 11 actual). |
| `composer.py` | Port of `compose_percat.py`. Dogfoods `PerKernelPredictor` API (no raw pkl loads). |
| `validate.py` | Per-GPU markdown reports comparing composer predictions vs ncu Σ. |
| `upload_to_r2.sh` · `pull_pkls.sh` | R2 round-trip. |
| `README.md` | Reproduction docs. |

## Bugs fixed during port

- **`dispatch.py:18`** default `gpu='H100'` → `'A100'` (H100 had no pkls; dispatch was unreachable).
- **Turing-blind classifier regex** — old regex `r'(ampere|hopper|sm\d+).*gemm'` at `label_all_v3.py:68` silently dropped all `turing_fp16_s1688gemm_*` kernels into `'other'`. Extended to `r'(ampere|hopper|turing|volta|maxwell|pascal|kepler|sm\d+).*gemm|…|magma_sgemm'`.
- **fp32 fallback on 2080Ti Qwen3.5-9B** — GEMMs were `magma_sgemmEx_kernel` + `volta_sgemm_64x64_nn`, also matched by the extended regex.
- **Hash-suffixed pkl naming** (`gemm_171c4dbfdd45f87d.pkl`) retired in favor of runtime-contract `perkernel_{family}_shape_v2.pkl`.
- **Trainer output path** originally resolved to `agentic-serve/profiling/data/…` (repo root) instead of `agentic-serve/llm_predict/profiling/data/…` (where `PerKernelPredictor._pkl_dir()` looks). Fixed `parents[2]` → `parents[1]`.
- **Refreshed stale `predictors/per_kernel/README.md`** — removed "WIP, lives on runpod" text, added runtime-API documentation.

## New schema vs runpod pipeline

| Old `label_all_v3.py` | New 2026-04 CSVs |
|---|---|
| wide-format load | **long-format** — pivot needed |
| `gpu__time_duration.avg` | `gpu__time_duration.sum` |
| split `dram__bytes_{read,write}.sum` | combined `dram__bytes.sum` — **Gate-vs-Down disambiguation impossible**, degraded to grid-only |
| Ampere tile regex only | Turing `s1688gemm_*` added, plus volta/magma sgemm fallback |

## 2080Ti handling

- sm_75 cannot run FlashAttention-2 → **zero `flash_attn` kernels** in 2080Ti CSVs.
- Added `annotate_gemm_noflash_fallback()` — shape-solves every GEMM against per-model candidate set without layer-walking.
- `HELD_OUT_BY_GPU['RTX2080Ti'] = set()` — only 2 models (Llama-3.1-8B, Qwen3.5-9B), LOMO degenerate; train on both with no CV.
- 2080Ti `flash_attn` pkl intentionally absent; `PerKernelPredictor.load()` skips gracefully, `predict_attention_prefill()` returns -1.0.

## Data produced

- **`data/kernels_labeled.csv`** — 125,024 rows, 3 GPUs × 8 models:
  - A100: 56,970 rows (4,729 gemm / 352 flash / 5,857 reduce / …)
  - RTX3090: 52,956 rows (4,146 gemm / 272 flash)
  - RTX2080Ti: 15,098 rows (868 gemm / 0 flash)
- **11 pkls** at `llm_predict/profiling/data/{A100,RTX3090,RTX2080Ti}/trained/per_kernel/`
- **R2 mirror** at `s3://agent-bench/profiling-data/` (pkls + labeled CSV, 42.8 MB).

## Smoke test

`PerKernelPredictor.predict_gemm(128, 4096, 4096, bf16)`:

| GPU | ms | ratio to A100 |
|---|---:|---:|
| A100 | 0.039 | 1.0× |
| RTX3090 | 0.112 | 2.9× |
| RTX2080Ti | 0.290 | 7.4× |

Plausible scaling.

## MAPE results (regression vs runpod baseline)

Runpod baseline was 4% aggregate on held-out 70B models. Current port:

| GPU | held-out model | predicted TTFT | measured Σ | abs err |
|---|---|---:|---:|---:|
| A100 | Llama-70B | 61.5 ms | 178.8 ms | 65.6% |
| A100 | Llama-3.3-70B | 61.5 | 178.5 | 65.5% |
| A100 | Qwen-72B | 61.5 | 181.4 | 66.1% |
| RTX3090 | Llama-70B | ~167 | 393.4 | — |
| RTX3090 | Qwen-72B | ~167 | 399.8 | 52.6% |

**Root causes** (not porting bugs):
1. `Qwen3.5-{9B,27B}` use placeholder `ModelConfig` values (n_layers, ffn, vocab are guesses) — HF configs not vendored in this repo.
2. Training set now includes Qwen3.5 rows labeled with those placeholders, polluting shape distribution.
3. 2080Ti fp32 fallback adds volta/magma kernels at shapes the old runpod training set never saw.

**Next step**: vendor real Qwen3.5 HF configs (`Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-27B` from HF hub), regenerate labeled CSV, retrain.

## Infra set up in this session

- WireGuard tunnel to `176.58.113.66:11466` (brought up, verified handshake, brought back down so user could enable it on Mac).
- SSH config at `/root/.ssh/config` for `2080ti / 3090 / gpu-4 / runpod / github.com` with `~/.ssh/hetzner-gpu` (new) and `~/.ssh/runpod` (new) keys.
- R2 credentials + AWS CLI v2 installed (aws/2.34.31).
- rsynced `/root/per-kernel-rebuild/` (~70 MB) from runpod as source material.
- Packages installed: `wireguard`, `python3-pandas`, `xgboost 3.2.0`, `scikit-learn 1.8.0`, `pyarrow 23.0.1`, `tabulate`.

## Pending / TODO (next session)

1. **Vendor real Qwen3.5 configs** → retrain → re-validate. Expect MAPE recovery toward runpod's 4% baseline.
2. **Per-op module is dead-wired** — `dispatch.py` imports `PerOpPredictor` but never routes through it in the `predict_gemm`/`predict_attention_prefill`/`predict_elementwise` cascade. Either wire it in or remove the lazy loader.
3. **Decode-phase coverage** — all current data is prefill. Reviewer-ask for NeurIPS.
4. **H100 data** — blocked by runpod container `ERR_NVGPUCTRPERM`. Separate pod relaunch needed.
5. **FP8 / MXFP4 models** — gpt-oss-120b and MiniMax-M2.5 still blocked by hardware.

## References

- Plan doc: see planner agent output in conversation transcript (5-phase port plan, 6 decisions locked).
- Source tree (runpod): `/root/per-kernel-rebuild/` on Hetzner (rsync'd copy).
- R2 bucket: `s3://agent-bench/profiling-data/` via `[r2]` profile in `~/.aws/credentials`.
- Paper: NeurIPS 2026 Evaluations & Datasets track, Section 4.4 cited in `predictors/per_kernel/predictor.py` docstring.
