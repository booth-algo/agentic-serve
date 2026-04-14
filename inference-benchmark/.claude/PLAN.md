# Per-Kernel Roofline Research Plan

## Goal
Publication-quality per-kernel roofline analysis for LLM inference, covering:
1. Measured hardware counters via ncu (not PyTorch Profiler estimates)
2. Virtual GPU scaling model for extrapolating to larger models
3. Optimal parallelism framework (TP/DP tradeoff analysis)

Target models: GLM, MiniMax, DeepSeek + existing models (Llama, Qwen, GPT-OSS)
Target metrics: memory bandwidth, memory latency, compute latency per kernel

---

## Step 1: ncu Hardware Counter Profiling

### 1.1 Bare-Metal Access
- **Problem**: RunPod containers block ncu (`RmProfilingAdminOnly: 1`, no `CAP_SYS_ADMIN`)
- **Solution**: Lambda Labs or equivalent bare-metal GPU rental
- **Action**: Rent H100 instance, verify `ncu --set full` works, run profiling sweep
- **Setup script needed**: install miniconda, clone repo, symlink models (or download)
- **Estimated cost**: 2-4 hours × $2-4/hr = ~$10

### 1.2 ncu Profiling Pipeline (already built, needs permissions)
- `scripts/roofline/run_ncu.py` — orchestrator (ready, tested, blocked only by permissions)
- `scripts/roofline/_ncu_target.py` — model loading + forward pass (working)
- **Key ncu metrics to collect**:
  - `dram__bytes_read.sum` / `dram__bytes_write.sum` — measured HBM bandwidth
  - `flop_count_hp` — measured half-precision FLOPs
  - `gpu__time_duration.sum` — kernel duration
  - `sm__throughput.avg_pct_of_peak_sustained_elapsed` — SM utilization
  - `l2_cache_hit_rate` — cache effectiveness (important for KV cache reuse)

### 1.3 Multi-GPU TP/DP Profiling
- **Current state**: Profiles use `device_map="auto"` (layer split, not true TP)
- **Needed**: Profile under actual TP=2,4,8 using SGLang/vLLM's `/start_profile` → `/stop_profile` endpoints
  - Launch server with `--tp N`
  - Send warmup requests
  - Hit `/start_profile`, send profiling requests, hit `/stop_profile`
  - Parse the resulting Chrome trace JSON
- **This captures**: NCCL allreduce kernels, actual TP communication overhead, scheduling latency
- **New script needed**: `scripts/roofline/profile_serving.py` — profiles through the serving stack

### 1.4 Target Models (new)
| Model | Params | Architecture | Min GPUs | Status |
|-------|--------|-------------|----------|--------|
| GLM-4.6 | 355B | MoE | 5× H100 (FP8) | Needs multi-node |
| GLM-5 | 744B | MoE | 5-6× H100 (FP8) | Needs multi-node |
| MiniMax-M2.5 | 230B | MoE | 3× H100 (FP8) | Needs multi-node |
| DeepSeek-V3.2 | 671B | MoE | 5× H100 (FP8) | Needs multi-node |

All are MoE — roofline analysis will show expert routing overhead vs dense compute.

### 1.5 Deliverables
- Per-kernel roofline plots with **measured** (not estimated) FLOPs and bandwidth
- Breakdown: memory bandwidth utilization, memory latency, compute latency per kernel category
- Comparison: dense (Llama-70B) vs MoE (DeepSeek, GLM, MiniMax) kernel profiles
- Comparison: TP=1 vs TP=2 vs TP=4 vs TP=8 communication overhead scaling

---

## Step 2: Virtual GPU / Scaling Model

### 2.1 Concept
Given a profiled model (e.g., 80-layer Llama-70B), extrapolate what happens at 1000 layers:
- **Per-layer kernel time is constant** (same GEMM shapes, same attention dimensions)
- **KV cache grows linearly** with layers → memory pressure increases
- **Communication scales** with TP degree and model size
- Predict: total bytes/sec, compute utilization, bottleneck kernel at scale

### 2.2 Approach
- Use measured per-layer kernel profiles from Step 1 as building blocks
- Model the full forward pass as: `N_layers × per_layer_time + communication_overhead + KV_cache_pressure`
- For attention: KV cache size grows with layers, so later layers have more cache pressure
- For TP: allreduce volume scales with hidden_size × TP_degree
- Output: theoretical throughput (tokens/sec) as a function of:
  - Number of layers
  - Hidden size
  - Number of GPUs
  - TP/DP/PP configuration
  - Batch size

### 2.3 Implementation
- **Input**: Per-kernel profile JSON from Step 1 + model architecture config
- **New module**: `scripts/roofline/virtual_gpu.py`
  - `scale_model(profile, target_layers, target_hidden, tp, dp, pp)` → predicted throughput
  - Accounts for: compute time, memory bandwidth, communication latency, KV cache
- **Validation**: Compare prediction vs actual measurement for models we have (8B, 32B, 70B)
- **Waiting on**: User to provide new codebase for this component

### 2.4 Deliverables
- Scaling curves: predicted throughput vs model size (layers)
- Scaling curves: predicted throughput vs GPU count
- Validation: prediction accuracy vs measured (should be within 10-15%)

---

## Step 3: Optimal Parallelism Framework

### 3.1 Problem
Given a model and a GPU cluster, what is the optimal TP/DP/PP split?

**The tradeoff**:
- **More TP** (tensor parallelism): Each GPU holds less model → faster per-GPU compute, BUT more allreduce communication after every layer
- **More DP** (data parallelism): Independent replicas → linear throughput scaling, BUT each replica must hold the full model
- **More PP** (pipeline parallelism): Split layers across GPUs → less memory per GPU, BUT pipeline bubbles reduce utilization

### 3.2 Cost Model
For each configuration (TP_t, DP_d, PP_p) where t×d×p = total_GPUs:

```
per_layer_compute = measured_kernel_time / TP_t          # TP splits GEMM
per_layer_allreduce = 2 * hidden_size * bytes_per_elem / bandwidth * (TP_t - 1) / TP_t
per_layer_total = per_layer_compute + per_layer_allreduce

pipeline_stages = num_layers / PP_p
pipeline_bubble = (PP_p - 1) / (microbatches + PP_p - 1)  # bubble fraction

effective_throughput = DP_d * batch_per_replica / (pipeline_stages * per_layer_total) * (1 - pipeline_bubble)
```

### 3.3 Key Variables
- **Measured from Step 1**: per_layer_compute, per_kernel_breakdown, memory_bandwidth_utilization
- **Measured from Step 1.3**: allreduce latency at different TP sizes (alpha + beta model)
- **From Step 2**: scaling predictions for layers/hidden_size not directly measured
- **Hardware constants**: NVLink bandwidth (900 GB/s H100), inter-node bandwidth (varies)

### 3.4 Search Algorithm
For a given model + cluster:
1. Enumerate all valid (TP, DP, PP) configurations
2. For each: compute predicted throughput using cost model
3. Constraints: memory per GPU ≤ 80GB, TP ≤ 8 (NVLink domain)
4. Output: Pareto frontier of throughput vs latency

### 3.5 Implementation
- **New module**: `scripts/roofline/optimal_parallelism.py`
  - `find_optimal(model_config, num_gpus, interconnect_topology)` → ranked configurations
  - Outputs: throughput, latency, memory usage, communication overhead per config
- **Visualization**: Heatmap of throughput across TP×DP grid, annotated with bottleneck type

### 3.6 Deliverables
- Optimal parallelism recommendations for each model × cluster size
- Heatmaps showing throughput landscape across TP/DP/PP configurations
- Analysis: at what model size does TP=2→TP=4 become beneficial?
- Analysis: when does PP become necessary (model doesn't fit in TP×DP)?

---

## Timeline & Dependencies

```
Step 1.1 (bare-metal access)  ──→  Step 1.2 (ncu profiling)  ──→  Step 1.5 (results)
                                        │                              │
Step 1.3 (TP profiling)  ──────────────→┤                              │
                                        ↓                              ↓
                                   Step 2 (virtual GPU)  ──→  Step 3 (optimal parallelism)
                                   (+ user's new codebase)
```

- **Now**: Step 1.2 pipeline is built and tested with PyTorch Profiler
- **Next**: Rent bare-metal for ncu (Step 1.1), or start Step 1.3 with SGLang profiling endpoints on current RunPod
- **Blocked**: Step 2 waiting on user's codebase, Step 1.4 waiting on multi-node GPU access

## Current Assets

### Completed
- Per-kernel roofline pipeline: profile → parse → plot (scripts/roofline/)
- 48 profiles across 5 models (Llama-8B, Qwen3-32B, Llama-70B, Qwen2.5-72B, gpt-oss-20b)
- Interactive dashboard with roofline tab (dashboard/src/components/RooflinePage.tsx)
- Cross-model comparison plots and GEMM scaling charts
- 170+ publication-quality PDF/PNG figures
- All data on GCS (gs://sequrity-experiments/inference-benchmark/results/)

### Blocked
- ncu hardware counters (RunPod `RmProfilingAdminOnly`)
- Qwen3.5-9B/27B profiling (transformers doesn't support `qwen3_5` arch)
- gpt-oss-120b profiling (OOM on single GPU, needs TP=2 LayerSlice)
- GLM/MiniMax/DeepSeek (need multi-node GPU access, models not downloaded)

## Notes
- PyTorch Profiler gives accurate GEMM FLOPs (formula-based from tensor shapes) but estimates for attention/elementwise
- ncu would give measured FLOPs and bandwidth for ALL kernels — needed for publication-quality claims
- For bare-metal: Lambda Labs H100 ~$2/hr, or CoreWeave/AWS p5 bare-metal
- Consider RunPod bare-metal tier (not container) for ncu access
