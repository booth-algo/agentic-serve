# L9-consumer: RTX 2080Ti / RTX 3090 calibration de-fit (2026-06-10)

**Lane contract:** branch `consumer-gpu-calibration` (off the campaign integration branch).
Owns `configs/gpus/{RTX3090,RTX2080Ti}.json`, `configs/deployments/{3090,2080ti}_*.json`, their
new artifacts. NO GPU work — all evidence from existing GT (`/mnt/100g` bench dirs), host
state/engine logs read off the hosts (no compute), committed sweep config, vendor specs, and
engine source. NO FITTING: every changed number is hardware truth, engine config read from
logs/source, or a measured-GT artifact via the established builders. Gate runs with
`RAMP_TPOT_REQUIRE_POOLS=1` (replay-ON, pools committed).

## Hypothesis verification (the diagnosis, checked before touching anything)

* **"2080Ti steady-state decode is off because the analytic roofline is mispriced (no measured
  kernels)" — FALSIFIED at the kernel level.** The pure analytic roofline (launch floor +
  max(weights/tp/bw, flops) + KV/bw) reproduces c1 TPOT well on all six configs (pred/meas:
  30.6/27.4, 16.1/15.8, 8.9/12.2, 20.8/20.2, 11.2/12.1, 6.5/7.8 ms). The actual mechanism for
  the +337%/+533% 2080Ti tp1/tp2 bias was the **`kv_pool=1` sentinel**: pressure =
  `scheduled*blocks/1` saturates permanently, so every turn was priced at the H100-inherited
  saturated ceiling (243.1 ms @ out28) even at c1.
* **"Prefill (TTFT) badly mispredicted on both consumer GPUs, recollection says
  UNDER-predicting" — SIGN WRONG for half the configs.** Measured baseline cell MAPEs
  (replay-ON, pools-required): RTX2080Ti ttft_cell **540,044%**, RTX2080Tix2 **2,056,162%**
  (c1 median signed error +1.15e6 ms / +3.13e6 ms — OVER-prediction by 4 orders of magnitude,
  again the pool-sentinel queue blow-up, not the prefill law). RTX3090 ttft_cell 157.3%
  (over-tail concentrated in big-ctx cells), RTX3090x2 38.2%, RTX3090x4 45.8%.

## Hardware truth: the 2080Ti host has 22 GiB-modded cards

`nvidia-smi --query-gpu=name,memory.total` on the `2080ti` host: **all 8 GPUs report
22528 MiB** (= 23,622,320,128 bytes), not the stock 11 GiB. Corroborated by committed config
(`inference-benchmark/scripts/sweep.yaml` `hosts.2080ti.vram_gb_per_gpu: 22`) and the
orchestrator gpu-state snapshots. These are the known GDDR6-density-mod cards: same TU102 die,
same 352-bit bus at 14 Gbps, so **peak_bw stays the datasheet 616 GB/s**; only
`total_memory_bytes` changes (11811160064 → 23622320128 in `configs/gpus/RTX2080Ti.json`).
This single wrong constant produced the `kv_pool=1` sentinels (11 GiB × 0.85 < 16.06 GB fp16
weights → negative budget) that drove the dominant 2080Ti biases.

## Levers applied

### 1. peak_flops accumulate-mode verification (values UNCHANGED, provenance upgraded)

* **RTX2080Ti 53.8 TF kept.** Engine logs prove the GT runs were fp16 (`Casting
  torch.bfloat16 to torch.float16`, `dtype=torch.float16` — sm75 has no bf16); PyTorch/cuBLAS
  HGEMM accumulates in FP32 (`CUBLAS_COMPUTE_32F`), and GeForce Turing runs FP16-with-FP32-
  accumulate at half the 107.6 TF FP16-accumulate rate (NVIDIA Turing whitepaper).
* **RTX3090 71 TF kept.** Engine logs show `dtype=torch.bfloat16`; BF16 tensor ops always
  accumulate in FP32, and GeForce GA102 runs that at half the 142 TF FP16-accumulate rate
  (NVIDIA GA102 whitepaper).
* Note the diagnosis already showed `util_flops` never binds in the analyzed regime
  (bandwidth-bound for 100% of the 1733 low-pressure turns; crossover ≈ batch 60), so no
  prediction moves from this lever — it retires the "FP16-accumulate ambiguity" caveat.

### 2. True KV pools (engine-config truth)

The launcher (`sweep_multiturn_profiles.sh`) starts a stock vLLM/sglang server; the engine
computes its own pool at startup and logs it. Surviving server logs on the hosts (read-only;
matched to GT via the GT files' own `config.max_model_len`/`gpu_memory_utilization` metadata
and the bench wrapper logs) give exact pools — blocks = engine tokens / 16:

| deployment | old blocks | new blocks | provenance |
|---|---|---|---|
| 2080ti Llama tp2 vllm | **1** (sentinel) | **8352** | `2080ti:/tmp/vllm_8093.log` 05-11: `GPU KV cache size: 133,632 tokens` (32768/0.85, fp16) |
| 2080ti Llama tp4 vllm | 4815 | **24570** | `2080ti:/tmp/vllm_8090.log` 05-22: `393,120 tokens` |
| 2080ti Qwen3.5-9B tp4 vllm | 13068 | **22011** | `2080ti:/tmp/vllm_8089.log` 05-22: `352,176 tokens` (kv_pool.py rule gives 89,663 — hybrid page padding unmodeled, 4.1x over) |
| 3090 Llama tp1 vllm | 1117 | **2008** | `3090:/tmp/vllm_8093.log` 06-02: `32,128 tokens`; the matching bench wrapper wrote THIS GT dir (16384/0.85 era; 9 older files say 32768 — pool is max_len-insensitive at fixed util) |
| 3090 Llama tp1 sglang | 1117 | **2421** | `3090:/tmp/vllm_8092.log` 06-04: `max_total_num_tokens=38738` |
| 3090 Llama tp2 sglang | 9893 | **12408** | `3090:/tmp/vllm_8091.log` 06-03: `#tokens: 198541` |
| 3090 gpt-oss-120b tp4 vllm | 15830 | **8194** | `3090:/tmp/vllm_8089.log` 06-04: `131,104 tokens` (util 0.95, matches GT metadata) |
| 3090 gpt-oss-120b tp4 sglang | 7092 | **12356** | `3090:/tmp/vllm_8090.log` 06-04: `#tokens: 197702` |

Where no engine log survives (vllm_PORT.log is truncated per launch), the pinned value is an
**engine-feasibility bound or engine-anchored reconstruction**, never a fit:

| deployment | old | new | basis |
|---|---|---|---|
| 2080ti Llama tp1 vllm | **1** | **512** | vLLM v1 refuses to start unless the pool holds `max_model_len` tokens (`vllm/v1/core/kv_cache_utils.py::check_enough_kv_cache_memory`); GT ran at max_model_len=8192 (GT config metadata + orchestrator `max_len_override` state file) → ≥ 512 blocks; longest measured session context 8181 tokens corroborates. Host arithmetic brackets ≤ 1916. The kv_pool.py rule gives 247 (< the startup bound → provably too small). |
| 3090 Qwen3.5-9B tp1 vllm | 1 | 256 | same startup bound at max_model_len=4096 |
| 3090 Qwen3.5-9B tp1 sglang | 1 | 256 | measured-GT bound: longest measured session context 4087 tokens demonstrably fit |
| 2080ti Qwen3.5-9B tp2 vllm | 1 | 6407 | engine-anchored reconstruction from the SAME host+model tp4 engine log (back-solved 3.70 GB/GPU non-KV overhead; per-GPU per-token bytes scaled from kv-shards 4 → 2); the rule's 26,420 is falsified 4.1x by the tp4 sibling log |

Remaining `derived` pools (3090 Llama tp2/tp4 vllm, etc.) keep the established kv_pool.py
rule; their notes now carry the measured caveat (the 3.5 GB reserve back-solves to ~1.63 GB on
the 3090 host from the tp1 engine log → derived pools likely UNDER-estimate 10–20%).

Hand-pinned measured/bounded pools are flagged in each manifest: a naive
`generate_deployments.py` regeneration would revert them to rule values (the generator has no
skip-list for these — see successors).

### 3. Prefill floors (already measured; ownership made explicit)

`profile_data/kernels/prefill_floor_llama31_8b.json` already carries per-config measured
consumer floors (min clean-c1 TTFT: RTX2080Ti 77.67, x2 54.10, x4 79.11; RTX3090 44.56,
x2 51.69, x4 48.84 ms). Regeneration via `build_prefill_floor` is **byte-identical** — no
change. Added `data.prefill_floor` manifest entries (status measured, path, resolver note) to
the six consumer Llama deployments so ownership is auditable from the manifest like the other
artifacts. The catastrophic TTFT bias was NOT the floor — it was the pool sentinel (above).

### 4. Own saturated ceilings (measured plateau anchors from THIS GT)

Flipped `data.saturated_ceiling` to measured + per-config path and ran the established
`build_saturated_ceiling` (pressure ≥ 2.5 with the NEW pools):

| config | saturated turns | anchors (out→ms) |
|---|---|---|
| RTX2080Ti | 759 | 25→51.8, 95→42.1 |
| RTX2080Tix2 | 321 | 26→671.9, 86→421.0 |
| RTX2080Tix4 | **0 → ceiling NOT owned** (stays H100-inherited, documented in manifest) |
| RTX3090 | 658 | 26→151.4, 89→68.7 |
| RTX3090x2 | 370 | 25→529.9, 86→439.8 |
| RTX3090x4 | 5 | 88→745.0 (THIN: n=5, single cluster — flagged) |

This replaces the H100-inherited anchors (243.1/134.9 ms) that simultaneously OVER-priced the
2080Ti tp1 low-conc regime (via the pool sentinel) and UNDER-priced the real consumer plateaus
at conc ≥ 200 (measured 410–670 ms). H100/H100x2/A100 ceiling artifacts regenerate
**byte-identical** (their pools/GT untouched).

## Gate results (replay-ON, RAMP_TPOT_REQUIRE_POOLS=1)

Binding scope (H100/A100/H100x2, must be ≤ baseline + 0.3): **byte-flat** — the predictions
JSON is byte-identical to the lane's own pre-edit baseline (all tpot/ttft/e2el cell deltas
exactly +0.0000; H100 14.47/18.13/10.78, H100x2 21.53/29.02/18.55, A100 14.37/22.22/15.87).
Expected: no shared code touched, the regenerated shared artifacts (prefill floor, H100/A100
ceilings) are byte-identical, and the binding configs read none of the consumer JSONs.

Consumer scope (not binding; the lane's target) — cell MAPE %, before → after:

| config | tpot_cell | ttft_cell | e2el_cell |
|---|---|---|---|
| RTX2080Ti | 312.20 → **26.40** | 540,044.40 → **68.42** | 176,777.55 → **35.47** |
| RTX2080Tix2 | 242.29 → **22.13** | 2,056,162.18 → **37.39** | 480,476.90 → **28.63** |
| RTX2080Tix4 | 74.95 → **57.10** | 127.23 → **43.56** | 78.55 → **52.97** |
| RTX3090 | 71.44 → **20.15** | 157.29 → **28.67** | 116.97 → **16.00** |
| RTX3090x2 | 50.40 → **35.16** | 38.21 → 38.21 | 40.04 → **34.34** |
| RTX3090x4 | 59.60 → **51.65** | 45.79 → 45.76 | 56.45 → **52.34** |

Every cell improves or stays flat; nothing regresses. The biggest residuals left (tp≥2 TPOT
51–57%) are exactly the documented unpriced tp-collective decode term + H100 prefill law
(successors 1 and 4 below).

pytest: full suite green (389 passed, 1 skipped, 15 subtests).

## NOT fixable offline — named successors

1. **tp≥2 decode collective term (the real tp4/x2 under-bias).** `_decode_roofline_full`
   (`simulator/kernel_step_cost.py`) prices NO tensor-parallel all-reduce (≈64 collectives/step
   for 32 Llama layers) and `launch_floor_for` borrows the NVLink H100x2 grid floor (1.82 ms)
   for PCIe consumer pairs. Residuals (c1): +3.3 ms at 2080Tix4, +0.9 ms at 3090x2
   (NVLink-paired). Needs a measured per-host tp-comm probe / decode grid on the consumer
   hosts (GPU work) — the established `build_tp_comm.py` / decode-grid builders are the
   vehicle.
2. **Consumer roofline utils are still H100 placeholders** (`util_flops/util_bw/
   scheduler_overhead` in both gpu JSONs). Successor: run `build_roofline_utils.py` (pinned L6
   recipe) on a serving wall trace from each host.
3. **Missing engine logs** for 2080ti Llama tp1, 2080ti Qwen tp2, 3090 Qwen tp1 (exact pools;
   currently bounds/reconstruction). One server start per config with the GT flags captures
   the line — next host window.
4. **H100-inherited prefill law** (`cached_prefill_grid`/`fa3_grid`) still mispriced for
   consumer cards (3090 big-ctx TTFT over-tail +52..+146 ms at c5–20 survives the pool fix).
   Needs measured prefill grids on the consumer hosts (and note sm75 has no FA3 at all — the
   2080Ti prefill path is a different kernel family).
5. **G6 RESERVE rule vs consumer hosts.** Back-solved reserves: 1.63 GB (3090 tp1 engine log),
   ≤2.97 GB (2080ti tp1 feasibility) vs the pinned 3.5 GB mean — the documented small-pool
   amplification is now measured, not hypothetical. Hand-off to the G6 owner; rule itself
   unchanged here.
6. **`generate_deployments.py` regeneration parity** (file owned by L5, not this lane): it
   would revert the hand-pinned measured pools. Successor: teach it a measured-pool skip-list
   or read pools from a committed engine-log artifact.
7. **RTX3090x4 ceiling is anchored on n=5 turns** (single out≈88 cluster). Honest but thin;
   re-anchor when more saturated GT lands.
