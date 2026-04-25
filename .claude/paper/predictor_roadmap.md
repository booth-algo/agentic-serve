# Predictor-track roadmap (post-2026-04-25)

Coordinated plan covering three open questions:
1. **Post-bench profiling campaign** — what to fire when GPUs free up
2. **Validation set model expansion** — what to download overnight
3. **ISL/OSL parameterization** — replacing fixed-seq128 with `(model, gpu, isl, osl, bs)` predictor signature (professor request)

Companion to `predictor_notes.md` (campaign log) and `per_op_vs_per_kernel_tradeoff.md` (analysis).
Open tasks tracker: `~/.claude/projects/-root/memory/project_predictor_open_tasks.md`.

---

## (1) Post-bench profiling campaign — HIGH impact

**Goal.** Close per-op cross-scale convergence (A100 26.72%, 3090 23.86% regressed, 2080Ti 17.15%); bring H100 from zero to parity; merge tactical fixes (#3 GEMM re-sweep, #5 flash_labeler).

**Wall-clock estimate.** ~10h end-to-end if 3090/A100/H100/2080Ti run in parallel; ~24h serialised.

### Steps

1. **3090 dense per-op (#1, HIGH)** — resume cmd in `project_predictor_open_tasks.md:40`. 5 models × ~512 cells, n_layers=2. Output: `RTX3090/{model}/tp1/phase1_bsseq_profiles.csv`. ~6h (Llama-70B 4h; small models 20-40min each).
2. **A100 dense per-op (#2, HIGH)** — same five models on gpu-4 GPU0 sglang env. Run **parallel** with step 1. ~4h.
3. **H100 prefill ncu sweep (NEW)** — once env verified:
   - `bash llm_predict/training/per_kernel/collect_gemm.sh H100 bf16` → 1071 matmuls, ~25min ncu wall.
   - `python -m llm_predict.training.per_kernel.profiler --gpu H100` for 6 target models (Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Mixtral-8x7B, Qwen2.5-72B, gpt-oss-20b). ncu prefill ~10-15min per small model, 30-45min per 70B. ~4h total.
4. **H100 dense per-op** — same 6 models on H100 GPU1 (different from step 3 to overlap). ~4h. Reuses existing `DEFAULT_GRID` (decode extension lands as item 3 below).
5. **flash_labeler NVTX fix (#5)** — code-only on 2080Ti while GPUs run. Patch `flash_labeler.py:67` to match by NVTX range name not position. Re-label A100 flash, retrain `perkernel_flash_attn_shape_v2.pkl` for A100 + 3090. Unblocks ~252 rows; expected 3090 flash MAPE 36% → <10%. ~3h.
6. **3090 GEMM re-sweep (#3)** — once 3090 frees from step 1: `bash collect_gemm.sh RTX3090 bf16`. Strip + relabel + retrain on 2080Ti. ~1h GPU + 30min relabel. Expected wall-clock 1.49% → ~0.5%.
7. **Centralised retrain on 2080Ti** — scp CSVs to `2080ti:/home/kevinlau/agentic-serve/llm_predict/profiling/data/{GPU}/`, run `labeler.py` then `trainer.py` per GPU, then `validate.py --vs-measured`. Push pkls to R2. ~2h.

### Ordering
Steps 1+2+5 in parallel after bench frees A100+3090. H100 (3+4) independent any time after env verified. Step 6 must wait on step 1 (same host). Step 7 is funnel — gates dashboard republish.

### Risks
- 3090 OOM repeat from stale processes — preflight `nvidia-smi` + `fuser /dev/nvidia*`, pin gpu2.
- H100 `accelerate`/`transformers` skew with 70B configs — fall back to n_layers=2.
- gpt-oss-20b MXFP4 needs special `--dtype` path on H100; smoke-test first model first.
- ncu permissions on H100 runpod container — verify `ncu --version` + 1-kernel smoke before queueing.

### Expected outcome
Per-op: A100 → ~12-15%, 3090 → ~10-13%, H100 first numbers within 5% per-kernel (Hopper has cleanest GEMM tiles).

---

## (2) Validation set model expansion — MED impact

**Goal.** Strengthen architecture/size diversity for cross-family generalisation. Current pool over-indexes on Llama-derived dense + GQA. Reviewers will ask for non-Llama dense, alternative MoE, sub-7B point.

**Constraint.** RULE #1 from `feedback_no_concurrent_downloads.md`: strictly sequential HF/rsync/scp. Binding constraint, not bandwidth.

**Window.** ~12h overnight, 100Mbps home → ~35 GB/hr realistic. Budget ~400 GB.

### Tier-1 targets (sequential, this order)

| # | Model | bf16 GB | Why | Target host |
|---|---|---|---|---|
| 1 | `microsoft/Phi-3.5-mini-instruct` | 7.6 | Sub-7B size gap, different attn layout | 3090 |
| 2 | `google/gemma-2-9b-it` | 18 | Sliding-window attention variant | 3090 |
| 3 | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 31 | 2nd MoE family + MLA novelty | A100 `/data/models` |
| 4 | `01-ai/Yi-1.5-34B-Chat` | 68 | 30-50B size gap (we have 8/20/27/47/70/72) | A100 `/data/models` |
| 5 | `ibm-granite/granite-3.0-8b-instruct` | 16 | Clean dense control vs Llama-8B | 3090 |

Total ~140 GB sequential ≈ 4h wall-clock — fits overnight.

**Skipped (justification):**
- DeepSeek-V3 (671B) — infeasible at 100Mbps.
- Mamba/SSM — breaks per-kernel `flash_attn`/`linear_attn` family separation per `predictor_notes.md`.
- GLM, Phi-4 — defer until tier-1 profiled and we know if more architecture diversity is needed.

### Run pattern
Single tmux on each host:
```
huggingface-cli download <repo> --local-dir /home/kevinlau/models/<basename>
```
Sequential `for repo in …; do …; done`. **No `&` anywhere.**

### Pre-flight
- HF token validity (refresh if needed).
- Yi-1.5-34B may be gated — accept license before bed.
- `df -h` on `/data` (shared with 9 users on A100); abort Yi if <100GB free margin.
- DeepSeek-V2-Lite uses MLA → needs vLLM 0.7+; that's a profiling-time concern, not a download-time concern.

### Risks
- HF rate limits / token expiry — pre-validate.
- Yi-34B gated — license click required.
- `/data` shared disk could fill mid-download.

---

## (3) ISL/OSL parameterization for predictors — HIGH impact

**Goal.** Replace `(model, gpu, seq=128, prefill-only)` with `(model, gpu, isl, osl, bs)` returning prefill TTFT **and** decode TPOT/E2EL. Required for professor's ask. Validates against real benchmark profiles in `inference-benchmark/src/workloads/profiles.py:305-417`:
- chat-short ISL=200/OSL=100
- chat-medium ISL=2000/OSL=1000
- chat-long ISL=8000/OSL=2000
- agent multi-turn — variable per turn

### Naming convention (LOCKED IN 2026-04-25)

The two predictor tracks are named explicitly in code, pkls, reports, dashboard, and paper sections:

| Track | Input | Output | Validates against | Status |
|---|---|---|---|---|
| **`microbench_ttft`** | `(model, gpu, seq=128, bs=1)` | TTFT only (prefill kernel sum + overhead correction) | `summary.median_ttft_ms` from `data.json` | Done — A100 1.97%, 3090 1.49% |
| **`serving_e2e`** | `(model, gpu, isl, osl, bs)` | TTFT + TPOT + E2EL | `median_ttft_ms` + `median_tpot_ms` + `median_e2el_ms` per benchmark profile | Phase 1-4 below |

**Concrete mapping across the codebase:**

```
llm_predict/predictors/per_kernel/
├── predictor.py              # shared XGBoost feature extraction (unchanged)
├── composer.py               # shared composition logic (extended, not split)
├── microbench_ttft.py        # NEW: predict_microbench_ttft(model, gpu, seq=128)
└── serving_e2e.py            # NEW: predict_serving_e2e(model, gpu, isl, osl, bs)
```

**pkls** rename next sweep cycle:
- `perkernel_microbench_ttft_<gpu>_<family>_v1.pkl` — current sweep, no kv_cache features
- `perkernel_serving_e2e_<gpu>_<family>_v1.pkl` — adds `phase` + `kv_cache_len` features

**validate.py modes:**
```
validate.py --mode microbench_ttft --target-seq 128       # current behavior, renamed flag
validate.py --mode serving_e2e --profile chat-medium      # new — looks up ISL/OSL from profiles.py
```

**Reports:**
```
reports/{gpu}_microbench_ttft_seq128.md       # rename existing wallclock_validation_seq167.md
reports/{gpu}_serving_e2e_<profile>.md        # new, one per profile
```

**Dashboard ProfilingPage** Predictor MAPE panel splits into two sub-panels:
- "Microbench TTFT" — current content (per-kernel internal + per-op held-out + wallclock fixed-seq128)
- "Serving E2E" — new (per-profile TTFT, TPOT, E2EL MAPE)

**Paper section titles:**
- §X "`microbench_ttft`: cross-scale prefill kernel composition" — current contributions
- §Y "`serving_e2e`: ISL/OSL-aware end-to-end latency"

### Phased rollout

**Phase 1 — backwards-compatible shim (4-6h, code-only)**
- New `serving_e2e.predict_serving_e2e(predictor, cfg, isl, osl, bs=1) -> dict` returning `{ttft_ms, tpot_ms, e2el_ms, decode_ms}`.
- When `osl == 0`, returns `ttft_ms` identical to current `composer.predict_ttft_ms` and `tpot_ms = e2el_ms - ttft_ms = 0`.
- Existing `composer.predict_ttft_ms(pred, cfg, seq=128)` is **renamed but not split** — alias `predict_microbench_ttft = predict_ttft_ms`. Backwards-compatible re-export.
- All current callers (`validate.py`, `dashboard`) keep working via the alias; rename callers in a follow-up commit.
- New `predict_attention_decode(bs, kv_cache_len, n_heads, head_dim, kv_heads)` on `PerKernelPredictor` — initially uses memory-bandwidth approximation (`kv_cache_len * n_kv_heads * head_dim * 2 bytes / HBM_bw`); replaced by trained pkl in Phase 2.
- Files: `composer.py` (alias), `serving_e2e.py` (new), `predictors/per_kernel/predictor.py` (new method), `feature_spec.py` (add `phase` column default `prefill`).
- **Unit test:** for any (model, gpu) in `kernels_labeled.csv`, `predict_serving_e2e(isl=128, osl=0).ttft_ms == predict_microbench_ttft(seq=128)` to ≥6 sig-figs.

**Phase 2 — decode per-op profiler grid + retrain (1 day, incl. 3h GPU)**
- Extend `per_op/profiler.py:DEFAULT_GRID` with decode cells:
  - `bs ∈ {1, 2, 4, 8}`
  - `kv_cache_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}`
  - = 32 new cells
- Critical change: `use_cache=True` + pre-populate `past_key_values` of right size. Currently profiler is `use_cache=False` — that's load-bearing.
- Re-run profiler on Llama-8B + Qwen3.5-9B on 2080Ti first (smallest, fastest validation). Then promote to A100/3090/H100 dense pool.
- Re-train per-op pkls into `serving_e2e_<gpu>_<family>_v1.pkl`. Schema already has `kv_cache_len` column; admit `kv_cache_len > 0` rows in `labeler.py`.
- Old `microbench_ttft` per-op pkls remain pinned; serving_e2e is a parallel pkl set, not a replacement.
- **Coordinate with item (1)**: this grid update must land BEFORE (1) per-op kicks so re-profiles capture decode in one pass.

**Phase 3 — serving_e2e composer wires decode (1 day)**
- `serving_e2e.predict_serving_e2e`: `ttft = prefill_ms(isl, bs)`; `decode_total = ∫ decode_step_ms(kv=isl+t, bs) dt over t∈[0, osl]`.
- Compute analytically as 8-point trapezoidal over kv_len (NOT 2000-call loop for long OSLs).
- `tpot = decode_total / max(osl, 1)`; `e2el = ttft + decode_total`.
- `validate.py`: add `--mode serving_e2e --profile <name>`. Read benchmark `results/*.json`, look up profile from `_metadata.profile_name`, fetch ISL/OSL/bs from `profiles.py`, compare `median_e2el_ms` + `median_tpot_ms` + `median_ttft_ms` separately.
- New report: `reports/{gpu}_serving_e2e_<profile>.md` (one per profile).
- Existing `--vs-measured` flag becomes alias for `--mode microbench_ttft --target-seq 128` for backwards compat.

**Phase 4 — dashboard surface (4-6h)**
- `publish_profiling_state.py` parses both `microbench_ttft_seq128.md` and `serving_e2e_<profile>.md` reports into separate JSON keys.
- `ProfilingPage.tsx` Predictor MAPE panel splits into "Microbench TTFT" sub-panel (current content) + "Serving E2E" sub-panel (per-profile TTFT/TPOT/E2EL MAPE).
- Existing fixed-seq128 table stays under microbench. New per-profile table under serving.

### Risks
- Decode attn pkl trained on too-narrow kv_len range explodes beyond support → profile to 16K, clip predictor input at training max.
- torch.profiler decode overhead worse than prefill (per-token sync amplifies the 50-85% overhead caveat). May need multiplicative correction term per-GPU.
- MoE decode activates only top-k experts but profiler measures all-expert pass → predictor will overestimate. Out-of-scope flag for v1 (matches current per-op limitations).

---

## Recommended timeline

| T+ | Action |
|----|--------|
| 0h (tonight) | Kick (2) sequential downloads on 3090 then A100. Unattended. |
| 8h (tomorrow AM) | Land (3) Phase 1 code on 2080Ti SSH (per `feedback_ssh_edits_on_host.md`). PR vs `kev/validate-vs-measured`. ~5h. |
| 13h (tomorrow PM) | Land (3) Phase 2 grid extension (code-only). Bench should be wrapping. ~3h. |
| 16h (tomorrow eve) | Bench done — fire (1) steps 1-4 parallel across 3090/A100/H100. (1) step 5 (flash_labeler) + step 6 (3090 GEMM) on 2080Ti / 3090 idle slots. ~10h overnight. |
| 30h (day after) | (1) step 7 retrain funnel; (3) Phase 3 composer wiring; (3) Phase 4 dashboard. ~1.5d. |
| 50h | Re-profile cheapest tier-1 downloads (Phi-3.5, Granite-8B ~30min each) with new decode grid. |

### Cross-links
- (3) Phase 2 must land **before** (1) per-op kicks → captures decode in one re-profile pass.
- (3) Phase 1 shim must land **before** (1)'s `validate.py` reruns → existing seq128 reports keep producing identical numbers.
- (2) is zero-contention (downloads only, no GPU/code conflict) → fire first.

### Paper-impact summary
- **(1) HIGH** — closes per-op gap, adds H100 (obvious reviewer ask for 2026 paper). Strengthens `microbench_ttft` headline numbers.
- **(2) MED** — doesn't move headline MAPE; answers "tested outside Llama/Qwen?" for reviewers. Stays MED until profiled.
- **(3) HIGH** — `serving_e2e` is the predictor's serving value prop. Without it: `microbench_ttft` is a kernel benchmark only. With it: claims "predicts vLLM TTFT + TPOT + E2EL across realistic ISL/OSL workloads".

### Naming convention summary (locked 2026-04-25)
- `microbench_ttft` = current track. Inputs: `(model, gpu, seq=128, bs=1)`. Outputs: TTFT only. Validates against `median_ttft_ms`.
- `serving_e2e` = new track per item (3). Inputs: `(model, gpu, isl, osl, bs)`. Outputs: TTFT + TPOT + E2EL. Validates against `median_ttft_ms` + `median_tpot_ms` + `median_e2el_ms` per profile.
- Reports, pkls, dashboard panels, paper sections all use these exact strings — no aliasing in user-visible surfaces.
