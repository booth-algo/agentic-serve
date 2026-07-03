# What do I need measured for backtest mode?

Checklist for taking a (GPU, model, tp, engine) deployment from `v2_roofline_firstcut`
to calibrated kernel composition. Ordered by accuracy leverage. Everything lands in a
`configs/deployments/<name>.yaml` manifest entry with `status: measured` + a path —
`build_simulator_v2_predictions.py` and the loaders consume it from there.

## 0. Ground truth (prerequisite)

A bench_dir under the central store (`<gpu>_<model>_tp<N>_<engine>`) with the four
multi-turn profiles. Without GT there is nothing to backtest — and no ceiling (see §3).

## 1. Config facts — one server launch, free

| input | how | why it matters |
| --- | --- | --- |
| `available_kv_blocks` | vLLM startup log / engine trace | pins eviction onset; the analytic pool (`configs/kv_pool.py`) is ±10-30% |
| scheduler settings | deployment flags (`max_num_batched_tokens`, `max_num_seqs`) | chunked-prefill budget = the TTFT drain rate |

## 2. Kernel leaf tables — per GPU × dtype, NCU/microbench session

These compose the decode/prefill floors (`kernel_floor/`). The GEMM table is
**shape-indexed and reusable** across models and tp on the same GPU (sharded tp
shapes are just different queries). Attention grids are **head-config-specific** —
one per (GPU, model-head-config, tp-shard).

| table | path pattern | fallback when missing |
| --- | --- | --- |
| GEMM (M,N,K → µs) | `profile_data/kernels/forward_pass/gemm/{GPU}.csv` | analytic roofline (util_flops/util_bw handwavy) |
| FA decode grid (kv×batch) | `profile_data/kernels/flash_attn/{GPU}.csv` | analytic roofline (auto, on head mismatch too) |
| FA prefill grid (causal N) | `profile_data/kernels/fa3_prefill_{GPU}.csv` | analytic causal roofline |
| elementwise | `profile_data/kernels/elementwise/{GPU}.json` | analytic |
| FA cached-prefill grid (U×P) | not yet in profile_data | `cross_attn_ms_per_token_pair` constant (GPU YAML) |

## 3. Saturated TPOT ceiling — derived from THIS deployment's GT

`build_saturated_ceiling.py` on cells at pressure ≥ ~2.5 →
`profile_data/kernels/saturated_ceiling/{...}.json`. Measured-plateau anchor, the only
legitimate GT-derived input (plateau only, never per-cell fitting). No saturated GT →
no ceiling → TPOT stays roofline-anchored (known to scale wrongly with tp).

## 4. Serving-stack terms — live server probes (per GPU host)

| input | probe | lands in |
| --- | --- | --- |
| frontend f / lanes / load-mult | `serving_herd_scaling.py` (idle c1–c160 + decoy-loaded sweeps), client-referenced | GPU YAML `frontend:` section |
| pool capacity check | `pool_capacity_probe.py` (APC hit/miss boundary) | confirms §1 pool |

Absent `frontend:` section = stage disabled → sub-saturation TTFT under-predicts
(~40–60% at c5–c20), exactly like pre-2026-07-03 H100.

## 5. tp>1 additions (kernel-ADJUSTED composition)

| input | status | note |
| --- | --- | --- |
| prefill TP-comm rate | MEASURED (H100/x4, 3090x2/x4: `prefill_tp_comm_*.json`) | per-token all-reduce, like-for-like tp pair method |
| decode all-reduce | missing | latency-bound small payloads; analytic term until profiled |
| sharded-head FA grids | missing | until profiled, head-mismatch → roofline fallback (automatic) |

## Never

- **Step-level decode grids (B×T whole-step tables) as cost models.** Policy
  2026-07-03: they are forward-pass measurements, not kernels — and v1 refused them
  empirically (H100x4: isolated-kernel wall sits above serving GT at conc≥120; see
  that deployment manifest). Existing step CSVs are validation data only.
- **In-situ fits to backtest GT** (per-cell tuning, compensating knobs). The ceiling
  plateau anchor (§3) is the one sanctioned GT-derived input.
