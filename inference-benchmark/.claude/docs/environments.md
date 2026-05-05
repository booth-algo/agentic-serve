# Host Environments & Model Inventory

Last updated: 2026-05-04

## Hosts

### h100 (gpu-13) — 4× H100 80GB

| Resource | Path |
|----------|------|
| vllm python | `/home/kevinlau/miniconda3/envs/vllm/bin/python` (vllm 0.19.0) |
| sglang python | `/home/kevinlau/miniconda3/envs/sglang/bin/python` |
| Models | `/data/models/` |
| Models (alt) | `/data48/kevinlau/models/` |

Models:
- `/data/models/Llama-3.1-8B-Instruct`
- `/data/models/gemma-2-9b-it`
- `/data/models/gpt-oss-20b`
- `/data/models/granite-3.0-8b-instruct`
- `/data/models/Mixtral-8x7B-Instruct`
- `/data/models/Qwen3.5-9B`
- `/data48/kevinlau/models/Llama-3.1-8B-Instruct` (dup)

### h100-2 — 4× H100 80GB

| Resource | Path |
|----------|------|
| vllm python | **NOT INSTALLED** |
| sglang python | **NOT INSTALLED** |
| Models | `/data/models/` |

Models:
- `/data/models/deepseek-math-7b-base`
- `/data/models/deepseek-math-7b-rl`

Needs vllm/sglang env setup and Llama model download.

### gpu-4 (a100) — 4× A100-SXM4-40GB

| Resource | Path |
|----------|------|
| vllm python | `/data/kevinlau/miniconda3/bin/python` (vllm 0.19.0) |
| sglang python | `/data/kevinlau/miniconda3/envs/sglang/bin/python` |
| Models | `/data/models/` |

Models:
- `/data/models/Llama-3.1-8B-Instruct`
- `/data/models/Llama-3.1-70B-Instruct`
- `/data/models/Llama-3.3-70B-Instruct`
- `/data/models/DeepSeek-V2-Lite-Chat`
- `/data/models/GLM-4.6-FP8`
- `/data/models/gpt-oss-20b`
- `/data/models/gpt-oss-120b`
- `/data/models/MiniMax-M2.5`
- `/data/models/Mixtral-8x7B-Instruct`
- `/data/models/Qwen2.5-72B-Instruct`

### 2080ti — RTX 2080 Ti

| Resource | Path |
|----------|------|
| vllm python | conda env `vllm` (vllm 0.19.0) |
| sglang python | conda env `sglang` (sglang 0.5.9) |
| Models | (none locally — benchmarks not run here) |

## Launch Patterns

**h100 single-turn/multi-turn:**
```bash
cd /tmp/inference-benchmark
DASHBOARD_SCOPE=fixed PORT=8089 \
bash scripts/sweep_multiturn_profiles.sh \
  /data48/kevinlau/models/Llama-3.1-8B-Instruct \
  1 Llama-3.1-8B vllm /tmp/results/fixed/h100_Llama-3.1-8B_tp1_vllm \
  /home/kevinlau/miniconda3/envs/vllm/bin/python 0.75 32768 "20 40" \
  "swebench-multiturn-mse terminalbench-multiturn-mse"
```

**gpu-4 single-turn/multi-turn:**
```bash
cd /tmp/inference-benchmark
DASHBOARD_SCOPE=fixed PORT=8089 \
bash scripts/sweep_multiturn_profiles.sh \
  /data/models/Llama-3.1-8B-Instruct \
  1 Llama-3.1-8B vllm /tmp/results/fixed/a100_Llama-3.1-8B_tp1_vllm \
  /data/kevinlau/miniconda3/bin/python 0.75 32768 "20 40" \
  "swebench-multiturn-mse terminalbench-multiturn-mse"
```

**h100-2:** Python and model setup needed first.

## Sweep Result Directories

| Host | Path |
|------|------|
| h100 | `/tmp/results/mse/h100_Llama-3.1-8B_tp1_vllm/` |
| h100-2 | `/tmp/results/mse/h100-2_Llama-3.1-8B_tp1_vllm/` |
| gpu-4 | `/tmp/results/mse/a100_Llama-3.1-8B_tp1_vllm/` |
