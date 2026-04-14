# Model Registry

## Downloaded Models

Stored at `/workspace/models/` on the RunPod network volume.

### Dense Models

| Model | HF Repo | Params | Size on Disk | TP Configs | Notes |
|-------|---------|--------|:------------:|:----------:|-------|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 30GB | 1, 2 | Gated (Llama license) |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 9B | 19GB | 1, 2 | `--trust-remote-code --disable-overlap-schedule` (SGLang) |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | 27B | 52GB | 2, 4 | `--trust-remote-code --disable-overlap-schedule` (SGLang) |
| Llama-3.1-70B-Instruct | `meta-llama/Llama-3.1-70B-Instruct` | 70B | 263GB | 2, 4 | Gated (Llama license) |
| Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | 70B | 263GB | 2, 4 | Gated (separate gate from 3.1) |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | 72B | 136GB | 2, 4 | `--trust-remote-code` |

### MoE Models

| Model | HF Repo | Total / Active Params | Size on Disk | TP Configs | Notes |
|-------|---------|----------------------|:------------:|:----------:|-------|
| gpt-oss-20b | `openai/gpt-oss-20b` | 21B / 3.6B | ~40GB | 1 | MXFP4 quantized, fits 16GB |
| gpt-oss-120b | `openai/gpt-oss-120b` | 117B / 5.1B | ~80GB | 1 | MXFP4 quantized, fits single H100 |
| MiniMax-M2.5 | `MiniMaxAI/MiniMax-M2.5` | 230B / 10B | 215GB | 4 | Already quantized weights |
| GLM-4.6-FP8 | `zai-org/GLM-4.6-FP8` | 355B / 32B | ~361GB | 4 | Tight on 4x H100 (320GB VRAM), may need `--gpu-mem 0.98` + short `--max-model-len` |

## Hardware: 4x H100 SXM5 80GB (320GB total VRAM)

## VRAM Planning

| Model | Weight Size | Fits 4x H100? | KV Budget | Notes |
|-------|:----------:|:--------------:|:---------:|-------|
| 8B TP=1 | ~16GB | Yes | ~64GB | Single GPU, plenty of room |
| 9B TP=1 | ~18GB | Yes | ~62GB | Single GPU |
| 20B MoE TP=1 | ~40GB | Yes | ~40GB | Single GPU (MXFP4) |
| 27B TP=2 | ~52GB | Yes | ~108GB | Comfortable |
| 70B TP=2 | ~140GB | Yes (BF16) | ~20GB | Tight KV; use FP8 for more headroom |
| 70B TP=4 | ~140GB | Yes (BF16) | ~180GB | Comfortable with TP=4 |
| 72B TP=2 | ~136GB | Yes (BF16) | ~24GB | Tight KV; use FP8 for more headroom |
| 120B MoE TP=1 | ~80GB | Yes | ~0GB | Single GPU, minimal KV |
| 120B MoE TP=2 | ~80GB | Yes | ~80GB | Better KV budget |
| MiniMax-M2.5 TP=4 | ~215GB | Yes | ~105GB | Comfortable |
| GLM-4.6-FP8 TP=4 | ~361GB | Tight | ~0GB | 41GB over 320GB — may need offloading or very short context |

## Access Requirements

| Model Family | Gated? | Notes |
|-------------|--------|-------|
| Llama 3.1 | Yes | HF license acceptance required |
| Llama 3.3 | Yes | Separate gate from 3.1 |
| Qwen (all) | No | Open access |
| OpenAI gpt-oss | No | Apache 2.0 |
| MiniMax | No | Open access |
| GLM / zai-org | No | Open access |

## Benchmark Priority

### Tier 1 — Must benchmark (production-relevant)
1. **Llama-3.1-70B** — baseline dense model, TP=2 and TP=4
2. **Llama-3.3-70B** — newer Llama, compare vs 3.1
3. **MiniMax-M2.5** — large MoE, TP=4
4. **gpt-oss-120b** — OpenAI MoE, single GPU

### Tier 2 — Important
5. **Qwen2.5-72B** — finish partial runs
6. **GLM-4.6-FP8** — if it fits, valuable MoE comparison
7. **gpt-oss-20b** — small MoE baseline

### Tier 3 — Fill in
8. **Llama-3.1-8B** — vLLM runs (SGLang already done)
9. **Qwen3.5-27B** — redo empty results
10. **Qwen3.5-9B** — blocked on transformers arch support, recheck
