# D-Sweep: Decode Step Overhead Calibration (Outstanding)

Date: 2026-05-02

## Status

**Not yet run.** Planned, calibration script exists (`llm_predict/calibrate_step_overhead.py`), but no GPU profiling has been executed.

## Why We Need It

The composer predicts decode step wall-clock time from kernel profiles (GEMM, flash attention, elementwise). At D=1, the prediction is reasonable (1.3× off for queue-free turns on H100x4). At D>1, the prediction is 3-20× too low because the composer only models **kernel compute time**, not:

- Paged-attention block table indirection
- KV cache block allocation/management per active request
- Scheduler loop overhead (Python dispatch, CUDA graph management)
- TP all-reduce latency per forward pass
- Inter-GPU communication (NVLink/PCIe bandwidth contention at higher batch)

These overheads scale with active decode count D and constitute the dominant fraction of decode step wall-clock at D>1. Without them, TTFT queue model (which uses decode step time between prefills) also under-predicts by 3-5×.

## Calibration Approach

Run a lightweight benchmark on each GPU, measuring `median_tpot_ms` at known D values:

```
D ∈ {1, 2, 4, 8, 16, 32}
Model: Llama-3.1-8B (dense, fast, fits all GPUs)
Workload: chat-singleturn (simple, short context)
Backend: vLLM with --enable-prefix-caching
```

For each D:
```
step_wall_meas = median_tpot_ms × D
step_wall_pred = composer.predict_decode_step_us(bs=D) / 1000
overhead(D) = step_wall_meas - step_wall_pred
```

Fit: `overhead = step_overhead_base_us/1000 + step_overhead_per_req_us/1000 × D`

### Existing Data

The `calibrate_step_overhead.py` script already computes these from `data.json` benchmark data (62K data points). It produces `per_req_us` values of 15-64ms per active request with r² of 0.3-0.9. However, these values capture the **full** composer-vs-reality gap (kernel model errors + overhead), not just step overhead. They also include VRAM-constrained rows and engine-specific policy differences.

A targeted D-sweep with a single model on a single GPU would isolate the step overhead from kernel model errors.

## Required Per GPU

| GPU       | TP | Available? | Notes |
|-----------|----|-----------|-------|
| H100      | 1  | Busy with benchmark sweeps | Also need torch installed |
| A100      | 1  | gpu-4, may be free | Already used for flash/GEMM sweeps |
| RTX3090   | 1  | GPUs 5-7 free | 24GB, fits 8B |
| RTX2080Ti | 1  | Available | 11GB, fits 8B |

## Command

```bash
# On each GPU host:
cd ~/agentic-serve/inference-benchmark
for D in 1 2 4 8 16 32; do
  python3 -m src.benchmark.runner \
    --profile chat-singleturn \
    --concurrency $D \
    --model Llama-3.1-8B \
    --backend vllm \
    --output results/d-sweep_D${D}.json
done
```

## Integration

Once `step_overhead_base_us` and `step_overhead_per_req_us` are calibrated, update `gpu_specs.py`:

```python
"H100": GpuSpec(
    ...
    step_overhead_base_us=<calibrated>,
    step_overhead_per_req_us=<calibrated>,
),
```

These flow into `composer.predict_decode_step_us()` and automatically improve both TPOT predictions and the TTFT queue model (which uses decode step time between prefills).

## Priority

High. This addresses the largest remaining gap — batch-scaling of decode step time — which affects both TPOT and TTFT for all models and all GPUs. The current metrics (H100x4 TPOT 6.5%) are already good at low D (per-turn), but the gap grows with D (swebench D=7-10 shows TPOT 230-283% on gpt-oss).
