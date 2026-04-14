# Server Management — vLLM & SGLang

## Launching vLLM

Use `scripts/launch_server.sh` for mode-aware launches:
```bash
# Single-turn mode (prefix caching ON)
CUDA_VISIBLE_DEVICES=0 ./scripts/launch_server.sh single-turn \
  --model /workspace/models/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1

# Stress-test mode (prefix caching OFF)
CUDA_VISIBLE_DEVICES=0 ./scripts/launch_server.sh stress-test \
  --model /workspace/models/Llama-3.1-8B-Instruct
```

### Defaults
- Port: 8000
- API key: `test`
- dtype: bfloat16
- GPU memory utilization: 0.75
- Max model len: 32768
- Logs: `/tmp/vllm_<mode>.log`

### Flags by Mode
| Mode | `--enable-prefix-caching` | `--enable-chunked-prefill` |
|------|:------------------------:|:--------------------------:|
| stress-test | NO | YES |
| single-turn | YES | YES |

## Launching SGLang

```bash
python -m sglang.launch_server \
  --model /workspace/models/Llama-3.1-8B-Instruct \
  --port 8000 \
  --tp 1 \
  --host 0.0.0.0
```

### Qwen3.5 Special Flags
Qwen3.5 (hybrid Gated Delta Net architecture) requires:
```bash
python -m sglang.launch_server \
  --model /workspace/models/Qwen3.5-9B \
  --trust-remote-code \
  --disable-overlap-schedule \
  ...
```

### CuDNN Fix for Qwen3.5
SGLang on Qwen3.5 requires CuDNN 9.15+:
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
# OR
export SGLANG_DISABLE_CUDNN_CHECK=1
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Killing Servers

```bash
# Graceful
pkill -f "sglang.launch_server"
pkill -f "vllm.entrypoints"

# Force (if graceful fails)
pkill -9 -f "sglang.launch_server"
pkill -9 -f "vllm.entrypoints"
```

`run_all_benchmarks.sh` handles server lifecycle automatically.

## Large Model VRAM Notes

### Current hardware: 4x H100 SXM5 80GB (320GB total)

| Model | TP | max_model_len | gpu_mem | Notes |
|-------|:--:|:------------:|:-------:|-------|
| Llama 8B / Qwen 9B | 1 | 32768 | 0.90 | Single GPU, plenty of room |
| gpt-oss-20b (MoE) | 1 | 32768 | 0.90 | MXFP4, fits single GPU |
| Qwen3.5-27B | 2 | 16384 | 0.92 | Comfortable |
| Llama 70B / Qwen 72B (BF16) | 2 or 4 | 4096-32768 | 0.95 | TP=4 gives much more KV headroom |
| gpt-oss-120b (MoE) | 1 or 2 | 32768 | 0.90 | MXFP4, single GPU possible |
| MiniMax-M2.5 (MoE) | 4 | TBD | 0.90 | 215GB weights, comfortable on 4 GPUs |
| GLM-4.6-FP8 (MoE) | 4 | TBD | 0.98 | 337GB weights, tight on 320GB VRAM |

See `.claude/tools/model-registry.md` for full VRAM planning table.
