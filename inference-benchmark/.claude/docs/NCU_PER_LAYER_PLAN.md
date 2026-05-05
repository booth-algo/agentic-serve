# Per-Layer NCU Roofline Profiling Plan

## Goal

Profile Llama-3.1-8B on H100 (TP=1, vLLM v0.19, bf16, prefix caching ON, C=1–80) and produce:

1. **Per-layer roofline table**: for each of the 32 transformer layers, determine whether memory-bound, compute-bound, or at the ridge — with execution time, OI, FLOPs, bytes
2. **Per-layer OI/CF plot**: same style as `roofline_multiturn_8b.py` (combined all-layers OI+CF panels) but with per-layer points overlaid on the roofline

## Context from Meeting

Aaron wants: "each layer and whether it's memory bound or compute bound... tabulate for each layer... from your NCU profile... show for one model as an example."

## Key Insight

The existing paper OI/CF plots combine ALL layers into a single throughput number. Per-layer analysis decomposes this: at any given concurrency, some layers are memory-bound (attention, layernorm, elementwise ops with low arithmetic intensity) and some are compute-bound (large GEMMs like the FFN up/down projections at high batch size).

## Architecture Reference

Llama-3.1-8B: d=4096, h=32, kv_h=8, n_layers=32, intermediate=14336, bf16 (2 bytes/elem)

Per layer components and their OI:
| Component | FLOPs (b=1) | Bytes (b=1) | OI (b=1) | FLOPs (b=80) | OI (b=80) |
|-----------|-------------|-------------|----------|--------------|-----------|
| Q/K/V proj | 2×4096×4096×3 = 100.7M | 2×(4096×4096×3)×2 = 201.3MB | 0.50 | 8.05G | 15.9 |
| Attention (flash) | ~4×b×h×seq×kv_len×head_dim | ~2×b×h×head_dim×(q+2kv+o) | varies | varies | varies |
| O proj | 2×4096×4096 = 33.6M | 2×(4096²+4096²)×2 = 134.2MB | 0.25 | 2.68G | 7.95 |
| Gate/Up proj | 2×4096×14336×2 = 235.0M | 2×(4096×14336×2)×2 = 470.0MB | 0.50 | 18.8G | 15.9 |
| Down proj | 2×14336×4096 = 117.5M | 2×(14336×4096+4096²)×2 = 302.0MB | 0.39 | 9.4G | 12.4 |

H100 ridge point (bf16): 989.4 TFLOP/s / 3.35 TB/s = 295.4 FLOP/byte

## Phase 1: Profile vLLM with NCU at Two Concurrencies

Profile vLLM server under load at C=1 (memory-bound end) and C=80 (compute-bound end).

### Approach
1. Launch vLLM: `vllm serve /data/models/Llama-3.1-8B-Instruct --dtype bfloat16 --enable-prefix-caching --gpu-memory-utilization 0.75 --max-model-len 32768 --port 8089`
2. Attach NCU to vLLM worker process: `ncu --set full --csv --profile-from-start no --target-processes all -o ncu_output`
3. Send workload (e.g., a chat-multiturn-medium profile at C=1 and C=80)
4. Trigger `cudaProfilerStart()` via a small helper process that signals the vLLM worker
5. Export .ncu-rep to CSV for parsing

### Script: `scripts/roofline/profile_vllm_ncu.py`
- Launches vLLM as subprocess under NCU
- Sends benchmark requests via the existing `src/benchmark/runner.py`
- Captures per-kernel traces with layer-level attribution

### Important detail
vLLM's CUDA kernel naming conventions include layer-identifying information:
- GEMM kernels often reference weight parameter names (e.g., `model.layers.5.self_attn.q_proj`)
- Flash attention kernels may not distinguish layers directly
- We can use PyTorch's CUDA stream correlation via NVTX ranges

Alternative: use `CUDA_LAUNCH_BLOCKING=1` and insert NVTX markers before each layer's forward pass, which NCU will capture.

## Phase 2: Parse NCU Output by Layer

### Script: `scripts/roofline/parse_ncu_per_layer.py`

Parse the NCU CSV output and group kernels by layer:

1. **Layer attribution via kernel name**: Match `model.layers.N.` patterns in kernel names where available
2. **Layer attribution via NVTX ranges**: If NVTX ranges are available, use them to attribute unclassified kernels
3. **Fallback**: If layer attribution is impossible from NCU alone, use PyTorch Profiler (which preserves module hierarchy) as the primary profiling tool

For each layer, compute:
- `total_cuda_time_us`: sum of all kernel times in that layer
- `total_flops`: sum of analytical FLOPs (from model architecture, not NCU's FLOP counter)
- `total_bytes`: sum of estimated DRAM bytes (from tensor shapes)
- `OI = total_flops / total_bytes`
- `achieved_tflops = total_flops / total_cuda_time_us / 1e6 / 1e12`
- `bound`: "memory" if OI < ridge_point, "compute" if OI > ridge_point

## Phase 3: Per-Layer Roofline Plot

### Script: `scripts/roofline/plot_roofline_per_layer.py`

Two-panel figure matching the paper's OI/CF style:

### Panel (a): Per-layer OI roofline
- H100 HW roofline (same as paper)
- Ridge point annotation (295 FLOP/byte)
- Each of the 32 layers as a labeled point:
  - x = OI (FLOP/byte)
  - y = achieved TFLOP/s
  - Color = bound type (memory=blue, compute=red)
  - Size = execution time fraction
- Overlay C=1 and C=80 combined OI_eff from the paper's existing plots for reference

### Panel (b): Per-layer time breakdown
- Stacked bar chart: execution time per layer, colored by kernel category
- Or: table below the plot showing per-layer data

## Phase 4: Tabulation

Generate a table for the paper:
| Layer | Time (μs) | Time % | FLOPs (M) | Bytes (MB) | OI | Bound | Dominant Kernel |
|-------|-----------|--------|-----------|------------|-----|-------|-----------------|
| 0 | ... | ... | ... | ... | ... | memory/compute | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Practical Decisions

### Profiling tool choice: PyTorch Profiler vs NCU

**Recommend PyTorch Profiler for per-layer, NCU for kernel-level detail.**

- PyTorch Profiler naturally preserves module hierarchy → easy layer attribution
- NCU is better for precise FLOP counting and memory bandwidth measurement
- Best approach: use PyTorch Profiler for per-layer time/FLOP breakdown, cross-validate with NCU for a few sample layers

### Implementation path

1. **First**: Extend `_ncu_target.py` to profile ALL 32 layers (not just 2), keeping the module hierarchy for layer attribution
2. **Second**: Write `parse_ncu_per_layer.py` that groups kernel records by layer index from the Torch profiler output
3. **Third**: Write `plot_roofline_per_layer.py` that generates the OI/CF per-layer plot
4. **Fourth**: Run at C=1 and C=80 on h100-2

### Scripts to create

```
scripts/roofline/
├── profile_all_layers.py        # Profiles all 32 layers with Torch Profiler
├── parse_per_layer.py           # Groups kernel records by layer from profiler JSON
├── plot_per_layer_roofline.py   # Per-layer OI/CF roofline figure
└── table_per_layer.py           # Generates per-layer table (LaTeX/CSV)
```

### Dependencies
- h100-2 accessible via SSH
- `/data/models/Llama-3.1-8B-Instruct` on h100-2
- `torch.profiler` (already available)
- Python env with `vllm` installed
- NCU at `/usr/local/cuda/bin/ncu`

### Fallback if NCU on vLLM is impractical
If NCU profiling of a running vLLM server proves too complex:
1. Use the existing `_ncu_target.py` but profile all 32 layers
2. This gives per-layer kernel data under PyTorch (matching vLLM's kernel selection)
3. Validate against the combined OI/CF plot from the paper
4. The per-layer OI values from PyTorch should match vLLM's kernel OI values (same model, same batch sizes, same GPU)
