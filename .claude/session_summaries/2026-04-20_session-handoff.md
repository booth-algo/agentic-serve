# Session handoff — 2026-04-17 → 2026-04-20

Three-day multi-thread session covering: (1) per-kernel predictor training pipeline port, (2) inference-benchmark dashboard plumbing + live sweeps across A100/3090/2080Ti, (3) cron orchestrator for continuous sweeps, (4) incident response on WG outage and disk exhaustion. This doc is the single source of truth for picking up the work from a fresh session.

---

## TL;DR

- **Per-kernel predictor**: Full training + labeling + validation pipeline ported from `runpod:~/per-kernel-rebuild/` into `llm_predict/training/per_kernel/`. 11 XGBoost pkls trained across 3 GPU archs. Held-out RTX3090 Llama-70B aggregate MAPE **0.74%**, A100 4-5%. Code + data committed locally, pending push.
- **Live dashboard sweeps**: 270+ new benchmark JSONs across 7 (GPU × model) dirs uploaded to `s3://agent-bench/results/`. `build-data.ts` patched to detect 2080Ti. Dashboard NOT yet rebuilt — requires `git push` + manual GH Action trigger.
- **Cron orchestrator**: `bench_orchestrator.sh` + `bench_jobs.txt` matrix running `*/30 * * * *` on Hetzner. Patched `setsid`+`</dev/null` dispatch. 9 jobs done, 3 abandoned (hybrid-attn Qwen3.5-27B), 5 re-queued pending.
- **Currently blocked on**: (a) WG tunnel data-plane failure — handshake OK but no IP forwarding; (b) gpu-4 root fs 100% full (~820 GB root-owned data, needs sudo prune).
- **5 git commits pending push** on branch `main`.

---

## Git state

Local `main` is ahead of `origin/main` by 5 commits:

```
8cf08dd Cron-driven benchmark orchestrator across all 3 GPU hosts
c07234c Add multi-turn focused sweep helper
be77aeb Session summary: benchmark sweeps + dashboard plumbing plan
e00cf8a Dashboard: detect RTX 2080Ti results + focused benchmark runners
367e94c Port per-kernel predictor training pipeline (runpod -> in-repo)
69ae729 Dashboard: strip -Instruct-v0.1 suffix from Mixtral model name  (pre-session)
```

**Push command** (agent permissions-gated; user must run):
```bash
cd /root/agentic-serve && git push origin main
```

Git author configured inline per-commit as `Kevin Lau <quantvengers@gmail.com>` (no global config change).

---

## Package: `llm_predict/training/per_kernel/`

Ported from runpod `~/per-kernel-rebuild/`. See `2026-04-17_per-kernel-port.md` for the historical port narrative. Quick-reference:

| File | Role |
|---|---|
| `ncu_loader.py` | Long→wide pivot of 2026-04 ncu CSV format (`.sum` target, combined `dram_bytes`) |
| `gpu_kernel_regex.py` | Kernel classifier + per-arch tile regex (ampere/turing/volta/hopper + magma_sgemm) |
| `model_specs.py` | `ModelConfig` bridging `model_configs/*.json` + fallbacks (Qwen3.5 hybrid-attn flagged) |
| `labeler.py` | Port of `label_all_v3.py`; 2080Ti noflash fallback, `_ext` merge |
| `trainer.py` | Per-(GPU × family) XGBoost; per-subfamily misc option (currently disabled, monolithic model used) |
| `composer.py` | Dogfoods `PerKernelPredictor`; enumerates ops × n_layers for TTFT |
| `validate.py` | Per-GPU MAPE reports |
| `sweep_gemm.py` + `roofline_labeler.py` | 1,071 matmul dense-model GEMM sweep; iteration-order labeled |
| `sweep_misc.py` + `misc_labeler.py` | Synthetic reduce/cast/copy/softmax at d=4096..16384 |
| `sweep_flash.py` + `flash_labeler.py` | FA2 sweep (fragile — FA2 dispatches multi-kernel; labeler iter-order breaks) |
| `upload_to_r2.sh` · `pull_pkls.sh` | R2 round-trip for pkls + labeled CSV |
| `reports/*.md` | Training + validation report artifacts |

### Training data (R2)
- **`s3://agent-bench/profiling-data/kernels_labeled.csv`** — 45 MB, **128,204 rows** across A100 / RTX3090 / RTX2080Ti × 8 models + roofline GEMM + synthetic misc.
- **`s3://agent-bench/profiling-data/{A100,RTX3090,RTX2080Ti}/trained/per_kernel/perkernel_{gemm,flash_attn,elementwise,misc}_shape_v2.pkl`** — 11 pkls total (2080Ti lacks flash, legitimately skipped).

### MAPE headline (held-out dense)
| GPU | Llama-70B | Llama-3.3-70B | Qwen-72B | Llama-8B |
|---|---:|---:|---:|---:|
| A100 | 4.39% | 4.21% | 3.36% | 0.76% |
| RTX3090 | **0.74%** | — | **0.95%** | 3.95% |
| RTX2080Ti | no heldout (only 2 models available) | | | |

Per-family held-out MAPE ≤15% on A100, ≤20% on RTX3090 for all families except 3090 flash (~27%, training-pool-bound) and 3090 misc (~18%, after d=8192 synthetic sweep).

---

## Package: `inference-benchmark/` (dashboard sweeps)

### Dashboard pipeline
GH Action `.github/workflows/rebuild-data.yml` is `workflow_dispatch` only:
1. `aws s3 sync s3://agent-bench/results/ results/`
2. `npx tsx dashboard/scripts/build-data.ts` → `public/data.json`
3. Upload `data.json` + `roofline-quadrant.json` to R2
4. Deploy job publishes to GitHub Pages

**`build-data.ts` patch** (commit `e00cf8a`): adds 2080Ti detection branch (mirrors 3090). Returns `2080Ti` / `2080Tix{2,4,8}`. Dirs matching `2080ti_` or `rtx2080_` prefix now show on the dashboard.

**Next step to light up dashboard**:
```bash
gh workflow run "Rebuild Dashboard Data" -R booth-algo/agentic-serve
```

### New benchmark helpers (commits `e00cf8a`, `c07234c`, `8cf08dd`)

| Script | Role |
|---|---|
| `scripts/run_one_bench.sh` | Atomic launch-vllm + ONE benchmark + teardown |
| `scripts/sweep_all_profiles.sh` | One vLLM server, sweep ALL profiles × concs, teardown. Idempotent (skips existing non-empty JSONs) |
| `scripts/sweep_multiturn_profiles.sh` | Same as above with `--mode multi-turn` and multi-turn profiles |
| `scripts/bench_jobs.txt` | Priority-ordered job matrix (host × model × TP × mode) — 20 jobs |
| `scripts/bench_orchestrator.sh` | Cron-callable dispatcher: idempotent state, OOM auto-retry, setsid-detached remote exec |

Host-side requirements discovered during this work (installed on each):
- `ninja` (pip) — FLASHINFER JIT on 2080Ti (sm_75 has no FA2)
- `datasets` (pip) — ShareGPT loading in benchmark runner
- 2080Ti specifically needs `CUDA_VISIBLE_DEVICES=5` or 5,6 (GPU 0 has another user's 8 GB resident) + `PATH=/home/kevinlau/miniconda3/envs/vllm/bin:$PATH`

### Cron
Hetzner crontab installed:
```cron
*/30 * * * * bash /root/agentic-serve/inference-benchmark/scripts/bench_orchestrator.sh >> /tmp/bench_orchestrator.cron.log 2>&1
```
Every 30 min: ssh-polls each host, dispatches next `pending` job if host idle, finalizes completed ones (rsync → s3 sync → mark done), retries once on OOM with halved `max_model_len`, else marks `abandoned`.

State dir `/tmp/bench_jobs/state/`:
- `<job_id>.status` (pending|running|done|failed|abandoned)
- `<job_id>.attempt` (retry count int)
- `<job_id>.max_len_override` (reduced max_len after OOM)

---

## Host inventory

| Host | Arch | Memory/GPU | Python / vLLM | Model weights | SSH config |
|---|---|---|---|---|---|
| gpu-4 | 8× A100-SXM4-40GB sm_80 | 38.5 GiB free/GPU | `/data/kevinlau/miniconda3/bin/python` · vllm 0.19.0 | `/data/models/` | User=kevinlau, IdentityFile=~/.ssh/hetzner-gpu |
| 3090 | 8× RTX 3090 24GB sm_86 | 23.6 GiB | `/home/kevinlau/miniconda3/envs/vllm/bin/python` | `/home/kevinlau/models/` | same |
| 2080ti | 8× RTX 2080 Ti 22GB sm_75 | 21.7 GiB (GPU 0 has 8 GB resident) | same | `/home/kevinlau/models/` | same; needs CUDA_VISIBLE_DEVICES=5 or 5,6 |

All reachable only via WireGuard tunnel (see below for outage).

---

## R2 state — `s3://agent-bench/`

### `results/` (dashboard data)

Produced by **this session's** sweeps:

| Dir | Files | Profiles × Concurrencies |
|---|---:|---|
| `a100_Llama-3.1-8B_tp1_vllm` | **75** | chat-s/m/l × 11 (1→500) + multi-turn × 7 |
| `a100_Mixtral-8x7B_tp4_vllm` | **63** | 33 new chat-s/m/l × 11 + 30 pre-existing |
| `a100_Qwen3.5-9B_tp1_vllm` | **27** | chat-s/m/l × 9 |
| `3090_Llama-3.1-8B_tp1_vllm` | **42** | chat-s/m/l × 10 (1→320) + multi-turn × 12 |
| `3090_Qwen3.5-9B_tp2_vllm` | **30** | chat-s/m/l × 10 |
| `2080ti_Llama-3.1-8B_tp1_vllm` | **18** | chat-s/m × 6 (1→120) + multi-turn × 6 |
| `2080ti_Qwen3.5-9B_tp2_vllm` | **12** | chat-s/m × 6 |

**Pre-existing** (older runs, untouched):
- `a100_Llama-{70B_tp{2,4},8B_tp{1,2}}_vllm/`
- `a100_Qwen2.5-72B_tp4_vllm/` (30 legacy files)
- `a100_gpt-oss-{20b_tp{1,2},120b_tp{2,8}}_vllm/`
- Extensive H100 data under non-prefixed dirs (`Llama-3.1-8B_tp1_vllm/`, `Mixtral-8x7B_tp2_sglang/`, etc.)

### `profiling-data/` (per-kernel predictor artifacts)

- `kernels_labeled.csv` — 45 MB, 128K rows
- `{A100,RTX3090,RTX2080Ti}/trained/per_kernel/perkernel_{gemm,flash_attn,elementwise,misc}_shape_v2.pkl` — 11 pkls
- `A100/ncu/{model}/prefill_seq128_bs1.csv` + `_ext` variants — raw ncu
- `RTX3090/ncu/{model}/…` — raw ncu
- `RTX2080Ti/ncu/{model}/…` — raw ncu

---

## Orchestrator state at end-of-session

From `/tmp/bench_jobs/state/*.status`:

| Job ID | Status |
|---|---|
| `gpu-4_Llama-3.1-8B_tp1_{single,multi}` | done |
| `gpu-4_Mixtral-8x7B_tp4_single` | done |
| `gpu-4_Mixtral-8x7B_tp4_multi` | **re-queued (abandoned prior, will re-fire on tunnel-up)** |
| `gpu-4_Llama-3.1-70B_tp4_single` | **re-queued** |
| `gpu-4_Llama-3.3-70B_tp4_single` | **re-queued** |
| `gpu-4_Qwen2.5-72B_tp4_single` | pending (no state file — will dispatch fresh) |
| `gpu-4_Qwen3.5-9B_tp1_single` | done |
| `gpu-4_Qwen3.5-27B_tp2_single` | abandoned (OOM — needs TP=4) |
| `gpu-4_Qwen3.5-27B_tp4_single` | abandoned (hybrid-attn triton warmup issue) |
| `gpu-4_gpt-oss-20b_tp1_single` | pending |
| `3090_Llama-3.1-8B_tp1_{single,multi}` | done |
| `3090_Qwen3.5-9B_tp2_single` | done |
| `3090_Qwen3.5-9B_tp2_multi` | **re-queued** |
| `3090_Qwen3.5-27B_tp4_single` | **re-queued** |
| `3090_gpt-oss-20b_tp1_single` | **running** (orchestrator thinks, may be stale) |
| `3090_Mixtral-8x7B_tp4_single` | pending |
| `2080ti_Llama-3.1-8B_tp1_{single,multi}` | done |
| `2080ti_Qwen3.5-9B_tp2_single` | done |

---

## Known failure patterns

### 1. Silent benchmark failures (profile-specific)
`prefill-heavy` / `decode-heavy` / `random-1k` silently exit without writing JSON on ALL tested (GPU × model) combos. Runner doesn't crash — just produces empty output. Most likely causes:
- Per-request timeout of 300s insufficient at high OSL
- `--ignore-eos` not auto-set for these synthetic profiles → runner hangs on unfilled output
- Tokenizer load path breaks on hybrid/MoE models

**Untested fixes**: add `--ignore-eos` to sweep_*.sh for these profiles, bump `--timeout` to 600s.

### 2. Qwen3.5-27B TP=2/TP=4 on A100 — hybrid-attn triton
Qwen3.5 uses hybrid attention (75% GDN linear_attention + 25% full_attention, head_dim=256). At TP=2: per-GPU needs >40 GB for 27B weights + KV + graph compile → OOM at 37.86/38.52 GiB. At TP=4: triton `chunk_gated_delta_rule` warmup fails with apparent silent kernel issue. `composer.py` already warns that hybrid-attn composition is inaccurate for this model.

### 3. gpu-4 disk full (100% of 876 GB root fs)
Discovered when Qwen2.5-72B manual sweep hit `No space left on device` on `mkdir /tmp/results/...`. Only ~14 GB is under `kevinlau` home; the remaining ~820 GB is root-owned (likely `/var/lib/docker` images from other users). Cleanable only with sudo:
```bash
ssh gpu-4 'sudo docker system prune -af --volumes'
ssh gpu-4 'sudo du -sh /var/lib/docker /var/lib/containerd'
```
User-accessible cleanup available (≈4 GB from `/tmp/torchinductor_kevinlau`, `~/.cache/vllm`, `~/.cache/huggingface/datasets`, old ncu artifacts) — won't materially help the root-fs situation.

### 4. WireGuard tunnel data-plane failure (current, as of end-of-session)
Symptom: handshake succeeds (1-2 min ago per `wg show`), endpoint `176.58.113.66` pings, counters show no transfer growth across multiple polls; `ping 10.250.0.1` / `ping 10.250.30.36` / `ssh gpu-4` all 100% packet loss.

Our side config OK (`AllowedIPs = 10.250.0.0/16`, route `10.250.0.0/16 dev wg0`, peer pubkey `qTvtq1qF…`). Our pubkey `RLlc/88Xw4MO2xN8hdlzILLCV4kYgbFCqaVPzUQJBWM=`. Concentrator at `176.58.113.66:11466` accepting handshakes but not forwarding.

**User must investigate on the concentrator / VPN gateway** — our peer entry may have been dropped, concentrator rebooted without loading routes, or iptables FORWARD rule removed.

Until tunnel is restored, cron orchestrator is firing but every `ssh $HOST` times out; job state won't advance.

### 5. FA2 flash-sweep labeling fragility
`sweep_flash.py` via `F.scaled_dot_product_attention` dispatches 1 primary `flash_fwd_kernel` + 0-2 auxiliary `flash_fwd_splitkv_*` kernels per call on A100/3090. Iteration-order labeler in `flash_labeler.py` assumes 1:1 and mis-aligns. Fix: export ncu with NVTX range names (needs `--page raw --section RangeNvtx` or XML export) OR bypass torch wrapper and call `flash_attn.flash_attn_func` directly. Not yet implemented.

---

## Command crib sheet

```bash
# Verify R2 access
aws --profile r2 --endpoint-url "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com" s3 ls s3://agent-bench/

# Push pending commits
cd /root/agentic-serve && git push origin main

# Trigger dashboard rebuild
gh workflow run "Rebuild Dashboard Data" -R booth-algo/agentic-serve

# Test WG tunnel + SSH
wg show wg0
ping -c 3 10.250.30.36   # gpu-4
ssh -o ConnectTimeout=5 gpu-4 'hostname'

# Manually run one benchmark (when tunnel healthy)
ssh gpu-4 'bash /tmp/inference-benchmark/scripts/sweep_all_profiles.sh \
  /data/models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm \
  /tmp/results/a100_Llama-3.1-8B_tp1_vllm \
  /data/kevinlau/miniconda3/bin/python 0.85 32768 \
  "1 10 20 40 80 120 160 200 256 320 500" \
  "chat-short chat-medium chat-long" \
  100'

# Check orchestrator activity
tail -50 /tmp/bench_orchestrator.log
ls /tmp/bench_jobs/state/

# Re-queue a stuck job
rm -f /tmp/bench_jobs/state/<job_id>.{status,attempt,max_len_override}

# Retrain per-kernel predictor
cd /root/agentic-serve && python -m llm_predict.training.per_kernel.trainer
python -m llm_predict.training.per_kernel.validate
```

---

## Critical environment

- **Hetzner box**: `hetzner-server`, OS Ubuntu 24.04, Python 3.13 (system), no conda.
- **Pip-installed system-wide** (via `--break-system-packages`): `awscli v2`, `xgboost 3.2.0`, `scikit-learn 1.8.0`, `pyarrow 23.0.1`, `pandas 2.1.4`, `tabulate`.
- **SSH config** at `/root/.ssh/config`: `gpu-4 / 3090 / 2080ti / runpod / github.com`. Keys: `~/.ssh/hetzner-gpu` (GPU cluster), `~/.ssh/runpod` (runpod), `~/.ssh/github`.
- **WG**: `/etc/wireguard/wg0.conf`, interface IP `10.250.201.35/32`, peer `qTvtq1qF…@176.58.113.66:11466`, AllowedIPs `10.250.0.0/16`.
- **R2 credentials**: `~/.aws/credentials` `[r2]` profile.
- **Cron**: `crontab -l` shows the 30-min orchestrator line.

---

## Next-session checklist

**Before any sweep work:**
1. Restore WG tunnel data-plane (concentrator-side fix, see §Known failure #4).
2. Clear gpu-4 disk space with sudo (see §Known failure #3).
3. `git push origin main` to publish the 5 pending commits.
4. Trigger `Rebuild Dashboard Data` GH Action to light up A100/3090/2080Ti on the live dashboard.

**Then continue sweep coverage:**
5. Let cron handle re-queued jobs; monitor `/tmp/bench_orchestrator.log`.
6. Manually fire Qwen2.5-72B TP=4 on A100 (disk-full was the only blocker last time; should work once space cleared).
7. Fill gpt-oss-20b sweep on A100 (pending in queue).
8. Investigate silent-failure profiles (`prefill-heavy`, `decode-heavy`, `random-1k`) — add `--ignore-eos` + longer timeout, rerun on Llama-3.1-8B as canary.

**Outstanding per-kernel improvements (lower priority):**
- NVTX-aware flash labeler rewrite (§Known failure #5) → should close the 27% 3090 flash gap.
- Vendor real Qwen3.5 configs; composer doesn't support hybrid attn, so TTFT prediction for Qwen3.5 models is structurally wrong regardless of predictor quality.

---

## References

- Prior session docs: `.claude/session_summaries/2026-04-17_per-kernel-port.md` · `2026-04-17_benchmark-sweeps.md` · `2026-04-15_h100-pdd-restructure.md`
- ncu data: `s3://agent-bench/profiling-data/kernels_labeled.csv`
- Source of truth for benchmark tooling: `inference-benchmark/src/benchmark/runner.py`
- Dashboard detector: `inference-benchmark/dashboard/scripts/build-data.ts:detectHardware`
- Rebuild workflow: `.github/workflows/rebuild-data.yml`
- Orchestrator: `inference-benchmark/scripts/bench_orchestrator.sh` + `bench_jobs.txt`
- Host sweep helpers on each GPU host at `/tmp/inference-benchmark/scripts/`
