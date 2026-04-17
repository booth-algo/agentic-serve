# Session 2026-04-17 — Benchmark Sweeps + Dashboard Plumbing

Scope: populate the public dashboard (`booth-algo.github.io/agentic-serve`) with A100 / RTX 3090 / RTX 2080Ti results using the repo's own `inference-benchmark/` tool, and build a roadmap for remaining sweeps. Parallel track to the per-kernel predictor work in `2026-04-17_per-kernel-port.md`.

## Why results were missing from the live dashboard

1. **GH Action-driven pipeline**: `.github/workflows/rebuild-data.yml` syncs `s3://agent-bench/results/` → runs `dashboard/scripts/build-data.ts` → uploads `data.json` → GitHub Pages deploys.
2. **Root causes for each missing GPU** (before this session's fixes):
   - **A100 TP=1 (Llama-8B)**: results *did not exist* in R2 yet; no benchmark had been run.
   - **RTX 3090 (all)**: results *did not exist* in R2; `inference-benchmark/` had never been run on the 3090 host.
   - **RTX 2080Ti (all)**: results didn't exist AND `detectHardware()` in `build-data.ts` had no rule for the `2080ti_` dir prefix — it would've fallen through to `'Unknown'` and been filtered.

## Fixes committed (this session)

- **`e00cf8a` Dashboard: detect RTX 2080Ti results + focused benchmark runners** — adds detection branch for `2080ti_` / `rtx2080_` dir prefixes in `build-data.ts` (returns `2080Ti` / `2080Tix{2,4,8}`); adds `scripts/run_one_bench.sh` (atomic vLLM launch-bench-teardown) and `scripts/sweep_all_profiles.sh` (idempotent profile × concurrency sweep against a single vLLM server).
- **`367e94c` Port per-kernel predictor training pipeline** — separate work, see `2026-04-17_per-kernel-port.md`.

**Push command (permissions-gated for agent, user must run)**:
```bash
cd /root/agentic-serve && git push origin main
```

## Host inventory

| Host | GPUs | Model weights path | Python/vLLM |
|---|---|---|---|
| gpu-4 | 8× A100-SXM4-40GB (sm_80) | `/data/models/` | `/data/kevinlau/miniconda3/bin/python` · vllm 0.19.0 |
| 3090  | 8× RTX 3090 24GB (sm_86) | `/home/kevinlau/models/` | `/home/kevinlau/miniconda3/envs/vllm/bin/python` · vllm 0.19.0 |
| 2080ti | 8× RTX 2080 Ti 22GB (sm_75) | `/home/kevinlau/models/` | same conda env · vllm 0.19.0; needs `ninja` + `datasets` pip-installed (done), `CUDA_VISIBLE_DEVICES=5` (GPU 0 has 8 GB resident from another user) |

Known quirks:
- **sm_75 (2080Ti)**: no FA2 (falls back to FLASHINFER JIT — requires `ninja` on PATH). bf16 tensor cores unavailable → vLLM casts to fp16. Max practical context ≤8K.
- Reachability: only via WireGuard tunnel to `176.58.113.66:11466`. SSH identities in `/root/.ssh/config` as `gpu-4 / 3090 / 2080ti` using `~/.ssh/hetzner-gpu`.

## Benchmark tooling

`inference-benchmark/` tool from the agentic-serve repo:
- `src.benchmark.runner` CLI — POSTs to OpenAI-compatible endpoint (vLLM / sglang).
- Profiles relevant to the dashboard:
  - ✅ `chat-short` (ShareGPT, ISL~500/OSL~300) — works everywhere
  - ✅ `chat-medium` (ISL~4K/OSL~1K) — works everywhere
  - ✅ `chat-long` (ISL~8K/OSL~2K) — works on A100 + 3090 with `max_model_len ≥ 32K`; too tight for 2080Ti
  - ❌ `prefill-heavy` / `decode-heavy` / `random-1k` — silently failed per-bench in this round (no JSON saved). Need runner-level debugging; likely per-request timeout or tokenizer mismatch.
  - ⚠️ `coding-agent` (ISL~17K) — requires `max_model_len ≥ 17K`, not feasible on 2080Ti's 8K cap.

Sweep helper:
```bash
bash scripts/sweep_all_profiles.sh \
     MODEL_PATH TP SHORT_NAME vllm OUT_DIR \
     PY GPU_MEM MAX_LEN CONC_LIST PROFILE_LIST NREQ
```
Idempotent: skips existing non-empty JSONs. Re-runnable.

## Results already in R2 (as of end-of-session)

Under `s3://agent-bench/results/`:

| Dir prefix | Source | Files |
|---|---|---:|
| `a100_Llama-3.1-8B_tp1_vllm/` | this session | 33 (chat-short/medium/long × 11) |
| `3090_Llama-3.1-8B_tp1_vllm/` | this session | 30 (chat-short/medium/long × 10) |
| `2080ti_Llama-3.1-8B_tp1_vllm/` | this session | 12 (chat-short/medium × 6) |
| `a100_Mixtral-8x7B_tp4_vllm/` | in flight | expected 33 |
| `3090_Qwen3.5-9B_tp1_vllm/` | in flight | expected 30 |
| `2080ti_Qwen3.5-9B_tp1_vllm/` | in flight | expected 12 |

Pre-existing (older runs, untouched):
- `a100_Llama-70B_tp{2,4}_vllm/`
- `a100_Qwen2.5-72B_tp4_vllm/`, `a100_Mixtral-8x7B_tp4_vllm/` (will overwrite when new sweep completes)
- `a100_gpt-oss-20b_tp{1,2}_vllm/`, `a100_gpt-oss-120b_tp{2,8}_vllm/`
- Extensive H100 data (non-prefixed dirs like `Llama-3.1-8B_tp1_vllm/`, etc.)

## Outstanding sweep plan

Once A100 TP=1/TP=4 + 3090 TP=1 + 2080Ti TP=1 land, the full dashboard matrix still needs the following to match the H100 baseline coverage.

### gpu-4 (A100 8×40GB = 320GB VRAM)

| Model | TP | VRAM fit | Status | Recommended profiles |
|---|---|---|---|---|
| **Llama-3.1-8B** | 1 | ✅ 16GB | **DONE** | chat-s/m/l done, decode/prefill/random broken |
| Llama-3.1-8B | 2 | ✅ | **TODO** | full sweep |
| **Llama-3.1-70B** | 4 | ✅ 140GB/160GB | old data exists | verify freshness, re-sweep if stale |
| Llama-3.1-70B | 2 | tight (140GB/80GB, FP8 only) | **TODO (FP8 variant)** | chat-short/medium |
| Llama-3.3-70B | 4 | ✅ | **TODO** | chat-s/m/l, full conc sweep |
| **Mixtral-8x7B** | 4 | ✅ 90GB | **IN FLIGHT** | chat-s/m/l |
| Mixtral-8x7B | 2 | tight | skip |
| **Qwen2.5-72B** | 4 | ✅ 144GB | old data exists | verify, re-sweep |
| Qwen3.5-9B | 1 | ✅ 18GB | **TODO** | chat-s/m/l (hybrid attn — note composer caveat) |
| Qwen3.5-27B | 2 | ✅ 52GB | **TODO** | chat-s/m |
| **gpt-oss-20b** | 1 | ✅ 40GB (MXFP4) | old data exists | verify |
| gpt-oss-20b | 2 | ✅ | old data exists | verify |
| gpt-oss-120b | 2 | ✅ 80GB (MXFP4) | old data exists | verify |
| gpt-oss-120b | 8 | ✅ | old data exists | verify |
| MiniMax-M2.5 | 4 | 215GB/160GB — won't fit BF16; needs FP8 | **BLOCKED** | needs H100 or FP8 re-quant |
| GLM-4.6-FP8 | 4 | 361GB/160GB — doesn't fit | **BLOCKED** | needs more H100 nodes |

### 3090 host (8× RTX 3090 24GB = 192GB total)

| Model | TP | VRAM fit | Status | Recommended profiles |
|---|---|---|---|---|
| **Llama-3.1-8B** | 1 | ✅ 16GB/24GB | **DONE** | chat-s/m/l done |
| **Qwen3.5-9B** | 1 | ✅ 18GB/24GB | **IN FLIGHT** | chat-s/m/l |
| Qwen3.5-9B | 2 | ✅ | **TODO** | higher-conc saturation |
| Qwen3.5-27B | 4 | ✅ 52GB/96GB | **TODO** | chat-s/m |
| Mixtral-8x7B | 4 | tight 90GB/96GB | **TODO, risky** | chat-short only to avoid OOM |
| Qwen2.5-72B | 8 | tight 144GB/192GB | **TODO, risky** | chat-short only |
| Llama-3.1-70B | 8 | tight 140GB/192GB | **TODO, risky** | chat-short only |
| gpt-oss-20b | 1 | ⚠️ MXFP4 → bf16 produces wrong kernels (same issue as 2080Ti profiling; may run but slow) | **TRY** | chat-s/m |

### 2080Ti host (8× RTX 2080 Ti 22GB = 176GB total, sm_75 fp16-only)

| Model | TP | VRAM fit | Status | Notes |
|---|---|---|---|---|
| **Llama-3.1-8B** | 1 | ✅ fp16 | **DONE** | chat-s/m × 6 concs |
| **Qwen3.5-9B** | 1 | tight 18GB/22GB | **IN FLIGHT** | fp16 fallback |
| gpt-oss-20b | 2 | unknown (MXFP4 fallback behavior) | **SKIP or try** | likely broken per ncu session |
| Anything 70B+ | 8 | might fit BF16 mathematically but sm_75 can't run | **SKIP** | not representative for paper |

### Profile coverage to add (same matrix across all hosts)

For every (host, model, TP) that currently has only chat-s/m/l, add:
- `prefill-heavy` (synthetic random 1K/256) — **needs runner debug**
- `decode-heavy` (random 256/4K) — **needs runner debug**
- `random-1k` (ISL=1K OSL=1K random, for cross-validation with InferenceX) — **needs runner debug**

Root-cause these three before spending more sweep time on them. Likely suspects:
1. Runner's random-token generator requires a tokenizer load path that may fail for Qwen3.5 hybrid / MoE fused kernels.
2. Per-request timeout (300s default) too short at high concurrency on decode-heavy (OSL=4K tokens).
3. `--ignore-eos` not auto-set for synthetic profiles → OSL not hit → runner gets stuck.

## Session execution log

- **Dashboard diagnosis** via R2 listing + `build-data.ts` source review → root cause (missing data + missing 2080Ti detector).
- **Dashboard patch** to `build-data.ts` (committed in `e00cf8a`).
- **`run_one_bench.sh` + `sweep_all_profiles.sh`** written + pushed to each GPU host via rsync.
- **Initial smoke test** (Llama-3.1-8B chat-short conc 1/10/40) on all 3 hosts — succeeded after:
  - 2080Ti GPU-0 had 8GB resident → switched to GPU-5 via `CUDA_VISIBLE_DEVICES=5`.
  - 2080Ti `ninja` missing for FLASHINFER JIT → `pip install ninja`.
  - 2080Ti `datasets` missing for ShareGPT → `pip install datasets`.
- **Full sweeps** (chat-short/medium/long × conc-up-to-500) fired on all 3 hosts — all completed; 33 + 30 + 12 JSONs uploaded to R2.
- **Additional model sweeps** fired (Mixtral TP=4 on A100; Qwen3.5-9B TP=1 on 3090 + 2080Ti) — in flight at session end.

## Headline numbers (observed)

Llama-3.1-8B TP=1 vllm chat-short @ conc=10:
- A100:   E2EL p50 2.7s / p99 4.7s
- 3090:   E2EL p50 3.0s / p99 5.7s
- 2080Ti: E2EL p50 9.0s / p99 14.6s  (fp16 fallback, FLASHINFER JIT backend)

## Next-session checklist

1. **Push the 2 commits** (`367e94c`, `e00cf8a`) from this repo to `origin/main`.
2. **Trigger `Rebuild Dashboard Data` workflow** in GitHub Actions — will pick up the new A100/3090/2080Ti dirs and render them with the patched detector.
3. **Wait for 3 in-flight sweeps** (Mixtral A100, Qwen3.5-9B 3090 + 2080Ti) to land, rsync + s3 sync them.
4. **Debug prefill-heavy / decode-heavy / random-1k** — investigate runner-level timeout / tokenizer issues. Likely a 30-min fix once reproduced.
5. **Fire the TODO matrix above** per available GPU budget. Suggested priority order:
   - Qwen3.5-27B TP=2 A100 · Qwen3.5-27B TP=4 3090 (fills dense 27B row)
   - Llama-3.3-70B TP=4 A100 (fills dense 70B row)
   - Llama-3.1-8B TP=2 A100 (completes dashed A100 TP-axis)
   - Mixtral-8x7B TP=4 3090 (tight memory, chat-short only)

## References

- Per-kernel predictor work this session: `.claude/session_summaries/2026-04-17_per-kernel-port.md`
- ncu session: `/root/llm/session-context-2026-04-17.md` (moved to `/root/llm/`)
- Benchmark source: `inference-benchmark/src/benchmark/runner.py`
- Sweep helper: `inference-benchmark/scripts/sweep_all_profiles.sh`
- Dashboard detector: `inference-benchmark/dashboard/scripts/build-data.ts:detectHardware`
- Rebuild action: `.github/workflows/rebuild-data.yml`
