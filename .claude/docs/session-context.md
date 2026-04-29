# Session Context for Predictor Work

## SSH Hosts (from Hetzner orchestrator)

| Host | SSH alias | GPUs | Python (predictor) | Python (CUDA/vllm) | sudo |
|------|-----------|------|-------------------|--------------------|------|
| Hetzner | (local) | none | no | no | n/a |
| 2080ti | `ssh 2080ti` | 8x RTX 2080Ti | `/home/kevinlau/miniconda3/envs/predictor/bin/python` | `/home/kevinlau/miniconda3/envs/vllm/bin/python` | unknown |
| gpu-4 (A100) | `ssh gpu-4` | 8x A100-40GB | `/data/kevinlau/miniconda3/bin/python` | same | unknown |
| 3090 | `ssh 3090` | 8x RTX 3090 | no predictor env | `/home/kevinlau/miniconda3/envs/vllm/bin/python` | unknown |
| h100 | `ssh h100` | 8x H100-80GB | no predictor env (torch too new) | `/home/kevinlau/miniconda3/envs/vllm/bin/python` | **passwordless** |

## Critical Rules
- **Never download models concurrently** — strictly sequential
- **Never push directly to main** — always use PRs via `gh` CLI
- **SSH-wrapped git push bypasses deny rules** — user pushes manually
- **Dashboard builds need Node.js** — only on Hetzner, NOT on 2080ti
- **All predictor code edits via SSH on 2080ti** — another CC session on Hetzner manages benchmarks

## Repo Location
- 2080ti: `~/agentic-serve/`
- H100: `~/agentic-serve/` (synced manually, no git)
- Hetzner: `/root/agentic-serve/` (other session's working tree — don't touch)

## Predictor Pipeline
1. **Per-kernel XGBoost** — one model per (GPU, kernel_family). Trained on ncu data in `kernels_labeled.csv`
2. **GEMM table predictor** — `gemm_table_predictor.py`, uses `gemm_serving_ncu_{GPU}.csv`
3. **Composer** — `composer.py` walks transformer layers, calls predict_gemm/predict_attn/etc
4. **serving_e2e** — TTFT + decode integration + concurrency model
5. **Validation** — `validate.py --mode serving_e2e_conc`

## Key Paths on 2080ti
- Training data: `llm_predict/training/per_kernel/data/kernels_labeled.csv` (gitignored, on R2)
- GEMM table: `llm_predict/training/per_kernel/data/gemm_serving_ncu_H100.csv`
- Serving shapes: `llm_predict/training/per_kernel/data/gemm_serving_shapes.csv`
- Trained pkls: `llm_predict/profiling/data/{GPU}/trained/per_kernel/`
- Validation reports: `llm_predict/training/per_kernel/reports/`
- ncu root: `/home/kevinlau/llm/profiling-data/{GPU}/ncu/{model}/prefill_seq128_bs1.csv`
- Benchmark data: `inference-benchmark/dashboard/public/data.json`

## H100 ncu Setup (after any reboot)
```bash
ssh h100 'sudo sysctl -w kernel.perf_event_paranoid=1'
ssh h100 'sudo systemctl stop nvidia-fabricmanager; sudo systemctl stop nvidia-persistenced; sudo fuser -k /dev/nvidia* 2>/dev/null; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia_uvm; sudo rmmod nvidia; sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0; sudo modprobe nvidia_uvm; sudo modprobe nvidia_modeset; sudo modprobe nvidia_drm; sudo systemctl restart nvidia-fabricmanager; sudo nvidia-smi -pm 1'
```
Also: `sudo zpool import data48` for ZFS data pool.

ncu path: `/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu`

H100 torch issue: `predictor` env has torch 2.11+cu130 (needs CUDA 13, driver is 12.4). Use `vllm` env for CUDA work.

## GEMM Sweep on H100
```bash
SHAPES_CSV=/tmp/gemm_serving_shapes.csv REPS=10 \
NCU=/usr/local/cuda-12.1/nsight-compute-2023.1.1/ncu \
PY=~/miniconda3/envs/vllm/bin/python \
bash /tmp/sweep_dir/collect_gemm.sh H100 bf16

# Must include nvjet in kernel filter for H100:
--kernel-name "regex:gemm|GEMM|sgemm|hgemm|fmha|flash|nvjet"
```

## Dashboard Build (on Hetzner only)
```bash
cd /tmp && git clone --depth 1 file:///root/agentic-serve profiling-build-check
scp 2080ti:~/agentic-serve/<changed-files> /tmp/profiling-build-check/<same-paths>
cd /tmp/profiling-build-check/inference-benchmark/dashboard
npm install && npx tsc -p tsconfig.app.json --noEmit && npx vite build
```

## Current Decode Correction (serving_e2e.py)
| GPU | alpha_base | exponent | alpha_floor |
|-----|-----------|----------|-------------|
| A100 | 1.497 | -0.13 | — |
| RTX3090 | 0.832 | -0.02 | — |
| RTX2080Ti | 0.605 | -0.25 | 0.32 |
| H100 | 1.48 | -0.08 | — |

## Current TTFT Queuing (concurrency_model.py)
| GPU | K (saturation) | plateau_base |
|-----|---------------|-------------|
| A100 | 1.8 | 8.0 |
| RTX3090 | 1.8 | 8.0 |
| RTX2080Ti | 1.8 | 8.0 |
| H100 | 0.6 | 4.0 |

## MoE Efficiency (composer.py)
| GPU | moe_eff |
|-----|---------|
| A100/RTX3090/RTX2080Ti | 0.585 |
| H100 | 0.15 |

## Current Best Numbers (H100, 3 dense models)
| Profile | TPOT | E2EL |
|---------|------|------|
| chat-short | 21.0% | 18.6% |
| chat-medium | 23.3% | 23.7% |
| chat-long | 23.9% | 21.6% |

C=1 Llama-8B: TPOT 0.4%, E2EL 2.4%

## Known Issues
- Gemma-9B C=1 TPOT 53% — nearest_nk fallback bad for vocab=256K
- Table M<=2 vs XGBoost decode correction tradeoff not fully resolved
- A100 GEMM held-out MAPE 396% — needs serving shape sweep (partially done, stopped)
- Mixtral profiling fails (transformers 5.6.2 weight conversion)
- Gemma needs `--chat-template /tmp/gemma_chat_template.jinja` for vLLM benchmarks

## Open Branches
- `kev/table-gemm-wip` — all uncommitted work (this session)
- `kev/validate-vs-measured` — old open PR (#4)

## Session PRs Merged: #35-#45 (11 total)
