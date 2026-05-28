# Profiling workspace

This folder is for clean, reproducible profiling experiments that validate the
serving simulator against production-like measurements.

## Current goal

Rebuild flash/decode evidence without using the old PyTorch SDPA NCU path as
serving truth. The old `llm_predict/sweep/sweep_flash_serving.py` path (now deleted) can
overstate decode attention cost because it:

- calls `torch.nn.functional.scaled_dot_product_attention`, not vLLM's serving
  kernel;
- expands GQA K/V heads before attention while still labeling rows as
  `n_kv_heads=8`;
- can include NVTX-range setup kernels in post-processing when NVTX survives the
  NCU CSV export.

## Layout

Use `profiling/profile/scripts/run_vllm_profile.py` for orchestration.
The profiling entrypoints live below `profile/vllm/`:

- `profile/vllm/cuda_events/` — entrypoints for both CUDA-event timing and NCU
  (``--source ncu`` wraps the same script with ``ncu`` and auto-appends
  ``--cuda-profiler-start-stop``).
- `profile/vllm/workloads/` — shared workload construction and timing loops.
- `profile/vllm/sweep/` — vLLM-based step profiling (prefill).
- `profile/vllm/engine_trace/` — instrumented vLLM API server for scheduler
  step tracing.

The script runner accepts ``--source cuda-events``, ``--source ncu``, or
``--source engine-trace`` and forwards workload arguments after ``--``.

## Scripts

### `profile/vllm/cuda_events/decode_steps.py`

Profiles production-like vLLM generation wall time with CUDA events and exports
per-step TPOT-style decode timing by `(batch_size, context_len)`.

Example on the H100 profiling host:

```bash
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
CUDA_VISIBLE_DEVICES=6 \
~/miniconda3/envs/vllm/bin/python profiling/profile/scripts/run_vllm_profile.py \
  --source cuda-events \
  --target decode-steps \
  -- \
  --model /data48/kevinlau/models/Llama-3.1-8B-Instruct \
  --output llm_predict/data/decode_profile_H100.csv
```

### `profile/vllm/cuda_events/flash_attn.py`

Sweeps vLLM's vendored `flash_attn_varlen_func` directly for decode-shaped
GQA attention. This is an isolated-kernel sanity check: it does not run the
full engine, does not use PyTorch SDPA, and does not expand KV heads.

The script can optionally compare `flash_ms_median * layers` against a vLLM
decode profile to report physically interpretable flash-attention percentage
of the full decode step.

```bash
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
CUDA_VISIBLE_DEVICES=6 \
~/miniconda3/envs/vllm/bin/python profiling/profile/scripts/run_vllm_profile.py \
  --source cuda-events \
  --target flash-attn \
  -- \
  --decode-profile profiling/results/decode_profile_H100_large_2026-05-17.csv \
  --output profiling/results/flash_attention_sweep_H100.csv
```

On H100, the direct vLLM call should use `--fa-version 3`. `--fa-version 2`
can fail in this environment with unsupported PTX even though full vLLM serving
works through its normal backend selection.

### `profile/vllm/cuda_events/decode_kernel_trace.py`

Profiles actual vLLM generation with `torch.profiler` and buckets the CUDA
kernels that make up an incremental decode step. It runs a baseline
`max_tokens=1` profile and a longer `max_tokens=N` profile, then attributes
`(N - 1)` decode steps by subtraction.

vLLM V1 normally runs the GPU engine in an `EngineCore` subprocess, which the
client-process `torch.profiler` cannot see. The script therefore defaults to
`VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing vLLM. CUDA graphs and normal
compile-time fusions remain enabled in the primary `production` mode.

Small-grid example on the H100 profiling host:

```bash
TMPDIR=/data48/kevinlau/tmp \
XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
CUDA_VISIBLE_DEVICES=6 \
~/miniconda3/envs/vllm/bin/python profiling/profile/scripts/run_vllm_profile.py \
  --source cuda-events \
  --target decode-kernel-trace \
  -- \
  --model /data/kevinlau/models/Llama-3.1-8B-Instruct \
  --shapes 1:512 8:4096 32:8192 128:2048 \
  --flash-sweep profiling/results/flash_attention_sweep_H100_2026-05-17.csv \
  --output-prefix profiling/results/decode_kernel_trace_H100_small_2026-05-17
```

Outputs:

- `*_raw_events.csv` — raw CUDA kernel events from the baseline and full runs.
- `*_delta_attribution.csv` — per-bucket decode-step latency and percent.
- `*_wide_summary.csv` — one row per `(batch_size, context_len)` with bucket
  percentages and flash-sweep comparison columns.

Raw profiler output can include synthetic `## Call CompiledFxGraph ... ##`
CUDA rows whose device time overlaps child kernels. The bucketed attribution
keeps those rows in the raw CSV for auditability but excludes them from bucket
sums to avoid double counting.

Use `--fusion-mode fusion-minimized` only as a diagnostic ablation. For BF16
single-GPU Llama-3.1-8B, many documented vLLM fusion flags are already inactive
or quantization-specific.

### `profile/scripts/collect_decode_kernels.sh`

Runs the decode GEMM and fused-kernel NCU target entrypoints. The combined
collector supports `smoke`, `full`, `gemm_smoke`, `gemm_full`, `fused_smoke`,
and `fused_full`. It also runs CUDA-event sanity sweeps before the NCU passes.

```bash
GPU_ID=6 \
TMP_ROOT=/data48/kevinlau/tmp \
PYTHON_BIN=$HOME/miniconda3/envs/vllm/bin/python \
bash profiling/profile/scripts/collect_decode_kernels.sh gemm_full
```

For one-off source selection without a full collector:

```bash
GPU_ID=6 \
TMP_ROOT=/data48/kevinlau/tmp \
~/miniconda3/envs/vllm/bin/python profiling/profile/scripts/run_vllm_profile.py \
  --source ncu \
  --target decode-gemm \
  --ncu-output profiling/results/manual_ncu/gemm_qkv_B1 \
  -- \
  --output profiling/results/manual_ncu/gemm_qkv_B1_cuda_events.csv \
  --batch-sizes 1 \
  --ops qkv_fused \
  --warmups 1 \
  --repeats 1 \
  --inner-iters 1
```

The old flat profiling entrypoints and old `vllm/ncu/collect_*.sh` collector
paths have been removed.

### `process/predict_llama31_8b_h100_tpot_from_kernels.py`

Composes TPOT from kernel-profile components:

```text
TPOT = attention_ms(B,T) + gemm_linear_ms(B) + small_kernel_ms(B,T) + residual
```

The script uses measured decode TPOT only as a validation target.

### `process/compare_flash_to_decode_profile.py`

Audits isolated flash-attention CSV values against the vLLM decode profile. It
reports the old per-layer flash value and the full-model `layers * flash`
percentage of measured TPOT.

```bash
python3 profiling/process/compare_flash_to_decode_profile.py \
  --flash llm_predict/data/flash_attn/H100.csv \
  --decode-profile llm_predict/data/decode_profile_H100.csv
```

## Interpretation

Use `decode_profile_H100.csv` as full-step ground truth. Do not treat isolated
NCU flash values as serving TPOT unless the collection target is the real
serving kernel and the post-processor proves it is counting only the intended
kernel launches.
