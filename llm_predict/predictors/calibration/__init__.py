"""Calibration layer: cross-cutting corrections applied to any predictor tier.

Unlike the per-tier predictors (per_op, per_category, per_kernel) which estimate
raw kernel/op latencies, the calibration layer models *second-order* effects that
apply regardless of tier:

- **MoE dispatch overhead**: fused_moe kernel cost not captured by per-expert GEMM sums.
- **All-reduce**: NCCL ring allreduce latency for TP>1 (per-layer, 2× per layer).
- **Framework overhead**: vLLM/SGLang tokenizer + scheduler + kernel launch costs
  that dominate TTFT at short ISL and conc=1.

Callers compose: `calibrated = predict(...) + framework() + 2·n_layers·allreduce(...)`
(or multiply by MoE correction factor for MoE models).
"""
