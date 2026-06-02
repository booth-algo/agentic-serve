# Plan — TTFT prefill-law de-fit + A100×1 simulator support

Date: 2026-06-02. Written before pivoting GPU work from the (saturated) H100s to the free A100.

## Part A — TTFT prefill-law de-fit (in progress)

**Goal:** remove the fitted constants from the prefill law `ttft = FLOOR + NEW·new + HOST·cached`,
replacing each with a measured/derived quantity. Audits: `prefill_law_defit_trace.md`,
`fitted_constants.md`, `fitted_constants_audit.md`.

**Done this session:**
- TPOT ceiling (`SATURATED_BASE/OVERHEAD`) → measured plateau anchors (gate held 15.91→15.89).
- tp2 decode KV sharding (`kv_bytes_per_token/min(tp,kv_heads)`): tp2 TPOT 57→39%.
- **NEW (0.0310) → DERIVED tp-aware GEMM roofline (`_prefill_gemm_per_tok`, 0.02498 tp1) + explicit
  dispatch residual (0.00602).** tp1 byte-identical (TTFT 33.01 / E2EL 19.79 / TPOT 15.89, 148 tests);
  **tp2 TTFT 43.3→31.7% bonus** (GEMM tp-scales). Corrected the wrong "7×" comment → 1.24× roofline.

**Still fitted — pending the host-vs-device stage-split microbench:**
- `FLOOR` (22.5): the optimal c1 intercept; can't move offline without regressing → measure as the
  `new=1, cached=0` TTFT.
- `NEW` dispatch residual (0.00602 ms/tok): framework dispatch (TaxBreak) → host slope vs `new`.
- `CACHED` host residual (~4.6 ms/1k of the 6.1; ~1.5 is measured GPU paged-attn): lead is **SHA-256
  prefix-cache block hashing** (vLLM ≥0.11; confirmed 0.19.1 default) → host slope vs `cached`, and the
  `sha256 − builtin` delta. The shared/perreq split is unidentifiable at c1 → from the batch B-sweep.

**Microbench:** `profiling/gpu_profiling/vllm/prefill_stage_split.py` — offline `LLM` + `torch.profiler`,
B=1, sweeps new×cached, splits each request into tokenize / device-forward / host; toggles `--hash`
and `--eager`. Resolves FLOOR + both residuals + the SHA-256 question in one run.

**Status:** all 8 H100s saturated (80–100% util). A detached **watcher is running on the h100**
(`/data48/kevinlau/tmp/cpbatch_run/stage_split_watcher.sh`, PID 3992619) that waits for an idle H100
(>65 GB free, <15% util) then auto-runs the microbench for `sha256` + `builtin` → `stage_split_*.csv`.
12h deadline. Check `stage_split_watcher.log`.

## Part B — A100×1 simulator support (new, pivot target)

**Why:** H100s are full; one A100 is free to experiment with. Adding A100×1 is the mechanical
"add a CONFIGS row" path the unified generator was built for (`build_simulator_rows.py`), but with real
A100 profiling.

**Environment (a100 host, verified):**
- Free GPU: **index 2**, A100-SXM4-**40GB** (0% util, 39.4 GB free). Others busy.
- vLLM: **`/home/kevinlau/miniconda3/envs/vllm/bin/python` → vLLM 0.19.0** (NOT the default python3).
- Model local: `/data/models/Llama-3.1-8B-Instruct`. Repo: `/data/kevinlau/agentic-serve`. Run dir:
  `/data/kevinlau/tmp`.
- Existing A100 data: `/data/kevinlau/per_op_traces_decode/A100/Llama-3.1-8B-Instruct/tp1/phase1_bsseq_profiles.csv`
  (legacy per-op decode profile — possible seed/cross-check).

**A100-40GB hardware constants (for `RooflineParams`):**
- `peak_flops_per_s` = 312e12 (BF16 dense, A100 SXM). `peak_bw_bytes_per_s` = 1.555e12 (HBM2e 40GB).
- `kv_bytes_per_token` = 131072 (same model/GQA). `kv_heads` = 8, `tensor_parallel` = 1.
- `available_kv_blocks`: **MEASURE** (40 GB ≪ H100 80 GB → far smaller than 27250; expect very low KV
  → the saturation/eviction regime hits at low concurrency, interesting for the sim).
- util_flops / util_bw: re-anchor from A100 measurements (don't assume the H100 0.65/0.93).

**Steps (on the free A100 GPU 2, conda vLLM):**
1. **Decode grid** — `…/envs/vllm/bin/python profiling/gpu_profiling/vllm/cuda_events/decode_steps.py
   --tensor-parallel-size 1 --output profile_data/results/decode_profile_A100_<date>.csv`
   (CUDA_VISIBLE_DEVICES=2). Triangular grid; A100-40GB OOMs earlier than H100.
2. **KV pool** — read `available_kv_blocks` from a vLLM init on the A100 (or the decode profiler's log).
3. **Prefill grid** — `sweep/prefill_steps.py --tensor-parallel-size 1` (for the TTFT roofline anchor).
4. **A100 RooflineParams JSON** — `profile_data/kernels/roofline_params_A100_llama31_8b.json` with the
   specs above + measured util factors + KV pool.
5. **Add a CONFIGS row** in `build_simulator_rows.py`: `Config("A100", 1, <bench_dir?>, <kv>,
   decode_grid=A100 csv, …)`. The dashboard self-selects the new GPU key (no UI change — per
   `simulator-dashboard-tp2-firstcut`).
6. **Ground truth — RESOLVED (correction 2026-06-02):** A100 ground truth ALREADY EXISTS in the central
   store at `/mnt/100g/.../synthetic_distributional/a100_Llama-3.1-8B_tp1_vllm/` (all 4 multiturn-synth
   profiles × conc 1–320, gpu_mem=0.85 — same as the decode profiling, so KV pool 8458 is consistent).
   My earlier "no A100 data" was a WRONG-PLACE error: I checked the a100 HOST's stale local git checkout
   (April, synth_count=0) instead of the central result store, which is keyed by GPU and written by the
   orchestrator regardless of the host checkout. **NO RERUN NEEDED** — the A100 config points at the
   existing data with `ground_truth=True`. Real MAPE: TPOT 25.4% / TTFT 46.4% / E2EL 25.1% (TTFT +
   saturated ceiling still H100-anchored → first-cut; re-anchor on A100 next).
7. **Bonus:** run `prefill_stage_split.py` on the free A100 too — gives A100 host/device split (the host
   terms — tokenize, SHA-256 hash — are GPU-independent, so the A100 run cross-checks the H100 de-fit's
   host residuals while we wait for an idle H100).

**Gates:** A100 config must not touch H100/H100x2 numbers (per-config grid swap is local). If an A100
ground-truth benchmark exists, report A100 TPOT/TTFT/E2EL MAPE; else label predictions-only first-cut.

**Open decisions for the user:** (1) predictions-only A100 first-cut vs run the A100 multi-turn benchmark
for a real gate; (2) whether to also run the prefill stage-split on the A100 now (host terms transfer).
