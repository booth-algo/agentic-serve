# KV Cache Overflow Calibration Plan

Date: 2026-05-02

## Problem

The RTX3090 (24GB VRAM) shows catastrophic TTFT errors at modest concurrency with long-context workloads. C=1 predictions are accurate (27.9% error), but C=5 at 12K context hits 1254% — predicted 624ms, measured 8.5 seconds. The queue model assumes infinite VRAM; the 3090 at C=5×12K needs ~38GB of KV cache against 24GB physical RAM.

## Physics

When `total_kv_bytes > available_vram`, the scheduler swaps KV blocks between GPU↔CPU over PCIe. Each prefill writes new KV (must free space first). Each attention read may need swap-in. The overhead is bounded by PCIe bandwidth, but scheduler thrashing / memory fragmentation / preemption adds non-linear penalties near the VRAM limit.

## Model Sketch (to implement)

### GPU spec additions (`gpu_specs.py`)
```python
vram_gb: float         # H100=80, A100=40, RTX3090=24, RTX2080Ti=11
pcie_bw_gb_s: float    # NVLink ~600, PCIe 3.0 x16 ~12, PCIe 4.0 x16 ~25
```

### KV budget
```python
kv_per_token = 2 * n_kv_heads * head_dim * n_layers * 2  # fp16 bytes
kv_per_request = total_context_tokens * kv_per_token

# Effective budget (lower than theoretical due to allocator/fragmentation)
params_bytes = model parameter bytes (arch-derived)
kv_budget_bytes = vram_bytes - params_bytes - cuda_graph_overhead - activation_scratch
# cuda_graph_overhead ≈ 2GB, activation_scratch ≈ concurrency * 100MB
```

### Overflow overhead
```python
kv_deficit = max(0, total_kv - kv_budget_bytes)

# Prefill: must free deficit bytes for new KV
swap_out_ms = max(0, new_prefill_bytes - free_bytes) / pcie_bw_bytes_per_ms

# Steady-state: if KV production rate exceeds PCIe swap rate, diverge
kv_growth_rate = D_active * kv_per_token  # bytes/step
step_time_ms = predict_decode_step_us / 1000
kv_production_rate = kv_growth_rate / step_time_ms  # bytes/ms
swap_rate = pcie_bw_bytes_per_ms

if kv_production_rate > swap_rate:
    # Unsustainable: scheduler falls behind continuously
    overflow_diverged = True
    ttft_overflow_ms = wall_clock_divergence_estimate
else:
    overflow_diverged = False
    ttft_overflow_ms = kv_deficit / swap_rate + swap_latency_per_block
```

### Unknowns requiring calibration
1. **Effective kv_budget** — where does the cliff actually start? Theory says ~5.5GB for 8B on 24GB, but allocator/fragmentation may shrink usable budget to 3-4GB.
2. **Cliff shape** — linear overflow penalty or superlinear? Profiling data across the VRAM utilization range answers this.
3. **TPOT impact** — does oversubscription degrade decode steps too (steady-state swapping) or only prefill (one-time swap)?

## Calibration Sweep

Run on **RTX3090 single GPU**, Llama-3.1-8B, vLLM with `--enable-prefix-caching`:

### Phase 1: Latency cliffs (14 runs, ~30 min)
```
chat-singleturn:    C ∈ {1, 5, 10, 15, 20, 30, 40, 60, 80, 100}
coding-singleturn:  C ∈ {1, 5, 10, 15, 20, 30, 40}
```

### Phase 2: Long-context (6 runs, ~15 min)
```
swebench-multiturn-short: C ∈ {1, 2, 5, 10, 20, 40}
terminalbench-multiturn-short: C ∈ {1, 2, 5, 10, 20, 40}
```

### Per-run telemetry needed
- `median_ttft_ms`, `median_tpot_ms`
- `gpu_mem_used_bytes` (if exposed by vLLM metrics)
- `num_blocks_swapped` or `gpu_cache_usage` (vLLM `/metrics` endpoint)
- `server_args`: max_num_batched_tokens, max_num_seqs, gpu_memory_utilization

### Quick diagnostic (single shot)
```bash
ssh 3090 "cd ~/agentic-serve/inference-benchmark && \
  bash scripts/sweep_single_profile.sh coding-singleturn Llama-3.1-8B 2080Ti 20 vllm"
```
Verify that measured TTFT shows the expected divergence at oversubscription.

## Implementation Notes

The overflow model plugs into `predict_serving()` in `serving.py`:
```python
ttft_overflow_ms = compute_kv_overflow(
    gpu_spec, cfg, prefill_tokens, total_context, concurrency
)
ttft_ms = ttft_kernel + ttft_floor_ms + ttft_first_decode_ms + ttft_queue_ms + ttft_overflow_ms
```

The overflow term is zero when `total_kv ≤ kv_budget`. The cost function (swap time + steady-state penalty) is empirically calibrated from the sweep data.

## Estimating model parameters without profiling

For 8B on RTX3090:
```
params: ~16GB fp16
vram: 24GB
kv_budget_theoretical: 24 - 16 - 2(cuda) - 0.5(activation) ≈ 5.5GB
kv_per_token: 2 * 8 * 128 * 32 * 2 = 131KB
max_context_budget: 5.5GB / 131KB ≈ 43K tokens total
```

So at C=5 with 8K context each → 40K tokens → borderline. At C=5 with 12K → 60K → 1.4× oversubscribed. This matches the observed cliff pattern.

For gpt-oss-20b (MoE, larger params, more experts loaded):
```
params: ~40GB fp16 (with expert offloading?) — doesn't fit on 24GB at all
```
This explains why gpt-oss-20b on RTX3090 is at 545% TTFT even at C=1 — the model itself doesn't fit in VRAM, requiring permanent CPU offloading.
