# LLMCompass Simulator — Next Steps Plan

## Status: Per-kernel accuracy, multi-node comm model, and MLA complete
- ML hybrid profiler: 18/18 GOOD (0.83-1.14x)
- Serving validation: Llama-70B TP=2 E2E 0.99x, PP=2 1.12x
- Multi-model validation: 7/8 configs GOOD/OK across Llama-8B/70B, Qwen3-32B, Qwen3.5-27B
- Hierarchical AllReduce + PP bubbles wired in
- MLA support implemented (TransformerBlockMLATP) for DeepSeek-V3/GLM-5

---

## Step 1: MLA Support (Multi-head Latent Attention) — DONE

Implemented `TransformerBlockMLATP` in `transformer.py`:
- KV down-projection to d_latent (compressed cache, 5-10x smaller)
- KV up-projections for K/V decompression
- Optional Q LoRA path (q_lora_rank)
- Partial RoPE (qk_rope_head_dim)
- kv_cache_memory_bytes() uses d_latent
- Validation pending (DeepSeek-V3 needs 5+ H100s)

---

## Step 2: MoE Validation

### Why
`TransformerBlockMoETP` exists but has never been validated against real measurements.
All target models (MiniMax-M2.5, GLM-4.6/5, DeepSeek-V3.2, Qwen3 MoE, Llama 4) are MoE.

### What to Validate
- Gate/router GEMM latency (small: [b*s, d] × [d, n_experts])
- Expert dispatch overhead (token routing to different experts)
- Per-expert FFN GEMMs (same shape as dense FFN but on fewer tokens)
- Top-k expert selection (softmax + topk)

### Plan
1. Find a MoE model that fits on 2xH100:
   - `gpt-oss-20b` (already downloaded, MoE) — check if SGLang supports it
   - `Qwen3-30B-A3B` (MoE, 30B total / 3B active) — might need download
   - `DeepSeek-V2-Lite` (if available)

2. Benchmark on SGLang with TP=2:
   - Run same TTFT/TPOT/E2E benchmark as before
   - Compare with LLMCompass MoE prediction

3. Profile MoE-specific kernels:
   - Gate GEMM, expert dispatch, per-expert FFN
   - Add to kernel profiler if needed

### Effort: ~2-3 hours
### Blocked: Need to check if gpt-oss-20b works with SGLang

---

## Step 3: Multi-Node Validation

### Why
Hierarchical AllReduce model is built but only validated analytically (6.12x speedup
over flat). Need real inter-node measurements to validate.

### What's Needed
- 2+ node H100 cluster with InfiniBand
- RunPod multi-node pods, Lambda Labs, or university cluster
- Run SGLang with TP=4 across 2 nodes (2 GPUs per node) vs TP=4 single node
- Measure actual inter-node allreduce overhead

### Plan
1. Rent 2-node cluster (Lambda Labs ~$4/hr per node, ~2 hours needed)
2. Deploy SGLang with TP=4 across 2 nodes
3. Benchmark Llama-70B: same TTFT/TPOT/E2E
4. Compare: single-node TP=4 vs cross-node TP=4
5. Validate hierarchical AllReduce prediction accuracy

### Effort: ~$20 + 3-4 hours
### Blocked: Need multi-node hardware access

---

## Step 4: Bare-Metal ncu Profiling

### Why
Current per-kernel profiler uses `torch.mm` + CUDA events (accurate but no hardware
counters). ncu gives measured FLOPs, bandwidth, cache hit rates — needed for
publication-quality claims about hardware utilization.

### What's Needed
- Bare-metal GPU access (no container `RmProfilingAdminOnly` restriction)
- Lambda Labs H100 (~$2/hr) or CoreWeave bare-metal
- Run `ncu --set full` on the profiling scripts

### Plan
1. Rent bare-metal H100 instance
2. Run `scripts/roofline/run_ncu.py` with `--set full`
3. Collect: `dram__bytes_read`, `flop_count_hp`, `sm__throughput`, `l2_cache_hit_rate`
4. Update roofline plots with measured (not estimated) data points
5. Compare RF-predicted vs ncu-measured per-kernel metrics

### Effort: ~$10 + 2-3 hours
### Blocked: Need bare-metal GPU rental

---

## Priority Order

```
Step 1 (MLA)  ──→  Step 2 (MoE validation)  ──→  Step 3 (multi-node)
                                                        │
                                              Step 4 (ncu profiling)
```

Steps 1-2 can be done on current RunPod 2xH100.
Steps 3-4 need new hardware.
