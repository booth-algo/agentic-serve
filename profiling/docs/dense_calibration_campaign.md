# Dense-model calibration campaign (2026-06-13 → 06-15)

Goal: close every dense-model cell on the A100 / H100 / RTX3090 families to <20% E2EL cell-MAPE
(the Llama-3.1-8B playbook generalized to the other models). Outcome: **partial — bounded by GPU
access and a model-architecture discovery.** This doc is the honest record.

## The 19 target cells split into two very different problems

**Qwen3.5-9B / Qwen3.5-27B (13 cells) are NOT dense transformers.** `config.json`: `model_type:
qwen3_5`, `Qwen3_5ForConditionalGeneration` (multimodal VLM), `layer_types` = 3×`linear_attention` :
1×`full_attention` (`full_attention_interval: 4`). Only 1-in-4 layers has a growing KV cache; the rest
carry a constant gated-delta-net (gdn) recurrent state. The predictor models full-attention dense
transformers, so its pool/pressure, TTFT-prefill, and decode-cost models are all structurally off.

**Genuinely dense (6 cells):** Llama-3.1-70B, Llama-3.3-70B, Qwen2.5-72B — all tp4, on H100x4 + A100x4.

## What blocked the GPU work (both dense sub-tracks)

- **H100x4 dense** — needs the 70B/72B weights on h100. All transfer paths failed: Hetzner relay
  ~207 KB/s (136GB ≈ 7 days), h100 has no HF internet (DNS fails), and adding an h100→a100 LAN SSH key
  was policy-denied. **Skipped** (user decision).
- **A100x4 dense** — weights present, but the a100 host ran **5 persistent serving processes** pinned
  at 100% for 6.7h straight (not a draining sweep); only 3 of 8 GPUs free, and tp4 needs 4. The decode
  grids these cells need to reach <20% cannot run. **Blocked pending a freed GPU.**

## Hybrid decode grids are a dead end in this environment

GT ran the hybrids in CUDA-graph mode (3090 9B tp1 GT conc1 TPOT = 22.4ms = graph-speed), despite the
bench wrapper's misleading "enforce eager: enabled" label. But `decode_steps.py` only *starts* the
hybrid with `--enforce-eager` (eager B=1 = 116ms, ~5× GT) — wiring that grid explodes the cell through
the queue sim (3090 9B tp1 716%→2651%, H100 9B 40.5%→354%). Measuring *with* graphs (to match GT)
fails at engine init — a broken `triton_kernels` import in the installed vLLM breaks the non-eager
path. So hybrid decode grids cannot be measured GT-faithfully here, and the hybrids are TTFT-
structurally-capped regardless (the full-attention prefill law caps even the best cell, H100 9B with a
correct pool, at ~40%). **Hybrid deliverable = the offline pool fix only.**

## What shipped: reserve-corrected KV-pool fixes

`configs/kv_pool.py` uses a single fixed `RESERVE_BYTES = 3.5GB` (back-solved from Llama-3.1-8B on
H100/A100). It is wrong in two regimes, and the wrong pools were pinned into the manifests:
- consumer GPUs (RTX3090): measured reserve ~1.63GB (tp1) / ~0.68GB (tp2/tp4) — the 3.5GB drove pools
  below zero into hand-pinned floors;
- A100 70B/72B tp4 knife-edge: 141-145GB weights barely fit 4×40GB, so 3.5GB over-subtracts a tiny
  budget.

Fix = recompute each affected pin with `kv_pool.available_kv_blocks(..., reserve_bytes=<measured>)`.
Gate (project rule): no metric regresses >0.5pt. Offline, no GPU (`_pool_sensitivity_probe.py`).

| cell | old → new pool | reserve | E2EL old → new | verdict |
|---|---|---|---|---|
| 3090 Qwen3.5-9B tp1 | 256 → 1846 | 1.63GB | **716 → 138** | ADOPT (config-bug floor) |
| A100x4 Llama-3.1-70B | 1545 → 2147 | 2.71GB | **238 → 156** | ADOPT |
| A100x4 Llama-3.3-70B | 1545 → 2147 | 2.71GB | **243 → 160** | ADOPT |
| A100x4 Qwen2.5-72B | 724 → 1326 | 2.71GB | **126 → 82** | ADOPT |
| 3090 Qwen3.5-9B tp4 | 103590 → 125105 | 0.68GB | 66.96 → 66.66 | ADOPT (neutral) |
| 3090 Qwen3.5-9B tp2 | 33383 (kept) | — | 62.35 (vs 63.58) | REVERT: corrected 44140 regresses +1.2pt → compensating fit |
| 3090 Qwen3.5-27B tp4 | 17218 (kept) | — | 70.02 (vs 72.09) | REVERT: corrected 27975 regresses +2.1pt → compensating fit |

The two reverts are the project's documented honest fallback: the physically-correct reserve regresses
MAPE because the cell is structurally TTFT-saturated and the prior generic-reserve pin incidentally
compensates — so the gate-passing pin is retained and labelled a compensating fit, with the physical
value recorded in the manifest note (awaits the hybrid prefill/pressure fix).

No binding cell (Llama-3.1-8B on H100/A100/H100x2) is touched — per-cell pins, zero cross-cell effect.

## Tooling banked
- `profiling/process/_pool_sensitivity_probe.py` — offline pool / grid sensitivity (DECODE_GRID_OVERRIDE env).
- `profiling/gpu_profiling/vllm/cuda_events/decode_steps.py` — `--enforce-eager`, `--trust-remote-code`,
  `--gdn-prefill-backend` flags (the last unblocks hybrid model start on Hopper; saved for a future env
  where graph-mode measurement works).

## Open backlog (needs an unlock)
1. **A100x4 dense → <20%** (the realizable win): free one a100 GPU → measure the 3 decode grids (graphs,
   standard transformers will measure fine) + ceilings. Pool fix already banked.
2. **H100x4 dense**: get 70B/72B weights onto h100 (LAN key or manual copy), then same as Llama-8B.
3. **Hybrid (13 cells) → <20%**: structural — needs hybrid-aware prefill (linear-attention O(n)),
   hybrid pool/pressure accounting, and a graph-mode-capable measurement env. Closer to the MoE gap
   than to calibration. See `project_qwen35_hybrid_discovery` memory.
