# Predictor Pipeline Context

## Architecture
Per-kernel XGBoost models trained on ncu profiling data predict CUDA kernel
execution time. Four kernel families: gemm, flash_attn, elementwise, misc.
One model per (GPU, family). Predictions compose into serving_e2e: TTFT + TPOT + E2EL.

Concurrency model: steady-state effective decode batch size (bs_eff) with
iterative solver. Power-law decode correction per GPU. Piecewise TTFT queuing.

## Key Files
- `llm_predict/training/per_kernel/serving_e2e.py` — core predictor
- `llm_predict/training/per_kernel/concurrency_model.py` — bs_eff + TTFT queuing
- `llm_predict/training/per_kernel/composer.py` — per-layer kernel composition
- `llm_predict/training/per_kernel/labeler.py` — ncu CSV → kernels_labeled.csv
- `llm_predict/training/per_kernel/trainer.py` — XGBoost training
- `llm_predict/training/per_kernel/validate.py` — validation reports
- `llm_predict/training/per_kernel/flash_labeler.py` — flash attention labeling
- `llm_predict/training/per_kernel/ensure_data.py` — auto-download from R2

## Training Data (kernels_labeled.csv)
- 46MB, gitignored, stored on Cloudflare R2
- ncu_root on 2080ti: `/home/kevinlau/llm/profiling-data`
- Layout: `{ncu_root}/{gpu}/ncu/{model_dir}/prefill_seq128_bs1.csv`
- Auto-download via ensure_data.py when missing

## Per-GPU Decode Correction
| GPU | alpha_base | exponent | alpha_floor |
|-----|-----------|----------|-------------|
| A100 | 1.497 | -0.13 | — |
| RTX3090 | 0.832 | -0.02 | — |
| RTX2080Ti | 0.605 | -0.25 | 0.32 |
| H100 | 1.84 | -0.15 | — |

## Training Data Coverage
| GPU | Real models | Rows | Quality |
|-----|------------|------|---------|
| A100 | 8 + sweeps | 61K | Good |
| RTX3090 | 7 + sweeps | 55K | Good |
| RTX2080Ti | 2 + roofline | 19K | Sparse |
| H100 | 2 + roofline | 5K | Very sparse |

## H100 ncu Profiling
- ncu: `/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu`
- Use `vllm` env (not `predictor`) — predictor torch needs CUDA 13, driver is 12.4
- After any reboot: perf_event_paranoid=1, nvidia module reload, fabric manager restart, zpool import
- sudo is passwordless
- Models on H100: Llama-8B, gpt-oss-20b, Qwen3.5-9B, gemma-2-9b-it, granite-3.0-8b-instruct, Mixtral-8x7B
- Only Llama-8B + gpt-oss-20b ncu-profiled so far

## Dashboard
- Deployed to GitHub Pages via CI on push to main
- Build requires Node.js (only on Hetzner, NOT on 2080ti)
- Build check: clone to /tmp on Hetzner, copy changed files, run tsc + vite build
- CI config: tsconfig.app.json has noUnusedLocals:true — always verify

## 15 Canonical Profiles
chat-short, chat-medium, chat-long, chat-multiturn-short, chat-multiturn-medium,
chat-multiturn-long, coding-agent, prefill-heavy, decode-heavy,
terminalbench-multiturn-short, terminalbench-multiturn-medium,
swebench-multiturn-short, swebench-multiturn-medium,
osworld-multiturn-short, osworld-multiturn-medium

## Known Issues
- H100 TTFT queuing c_thresh too low (fires at C=30, should be ~100)
- H100 MoE efficiency 0.585 wrong for Hopper (predictions 10x off)
- H100/2080Ti accuracy limited by sparse training data (2 models each)
- 3090 roofline nn.Linear sweep broke predictions (reverted, root cause unknown)
- Qwen3.5-9B hybrid attention produces 55K kernels (stripped as out of scope)
