# simulator_v2 — handoff

> Session snapshot for whoever picks this up next. Where things stand, what changed, what's
> open, and how to run it. Deep refs: `CODEBASE_MAP.md` (architecture), `ttft.md` (TTFT model
> + findings), `knobs.md` (every tunable). Last updated 2026-07-01.

## Status

v2 predicts per-turn TTFT / TPOT / E2EL via **kernel composition** (backtest) — no in-situ
fits. Backtest + the "Simulator v2" dashboard tab are wired and deployed. Forward mode is
still a stub.

Headline TTFT (H100 / Llama-3.1-8B tp1, cell-MAPE, mean of per-turn APE):

| aggregate | chat | osworld | swebench | terminalbench |
|---|---|---|---|---|
| **26.9%** | 32.6 | 17.5 | 28.1 | 29.3 |

## What changed this session

1. **Removed the `response_resident_fraction` (ρ) knob** — it double-counted. The ground
   truth accounts prefill as `cache_estimate_source="previous_prompt_tokens"`, so the prev
   response is already inside `new_prefill`; ρ credited it again. Deleted from `GpuConfig`,
   the `Hardware` protocol, `queue_sim`, the loader, and the YAML. Aggregate 28.6→26.9%.
   (`ttft.md` / `knobs.md` "Finding: response-resident".)
2. **Confirmed the host rate is correct** — re-probed on the live H100
   (`serving_stage_split_H100_reprobe.csv`): cached 6.13e-3 vs production 5.887e-3 (+4%,
   within tolerance), and real prompts are *lighter* per token (~0.83×). So the earlier
   host×1.3–1.5 "fix" was a fudge; **do not scale the host rate.**
3. **Identified the residual sub-saturation gap = vLLM API-server frontend serialization**
   (not GPU, not scheduler). A live multi-concurrency probe (`serving_herd_scaling.py`) shows
   the server's own frontend growing 7–10× c1→c20 while queue≈0 and GPU prefill stays flat.
   Measured `f(new,cached) = 6.5 + 0.0046·(new+cached)` ms, ~serial. **Characterized but not
   modeled** — see below. Housed in `engine/serving_frontend.py`.

## Key finding: sub-saturation TTFT is frontend serialization

- At **c1 the engine model is already accurate** (0.9–1.0); the gap switches on the instant a
  herd forms (c5+: ~0.56) and recovers once recompute dominates. So it's a *herd* effect.
- It is **serving-harness overhead** (HTTP/tokenize/IPC/stream), **not kernel physics**. The
  kernel sim faithfully predicts *engine* TTFT; the benchmark's measured TTFT additionally
  carries this frontend cost.
- **Not shipped as a term.** The `barrier_stagger_epochs` prototype lifts the high
  sub-saturation band but leaves the c5–c20 dip and over-predicts at the saturation transition
  (full-serial `f` over-extrapolates; the real frontend parallelizes by c80). A shippable term
  needs load-dependent parallelism + engine pipelining — a fixed lane count is a regime fudge.
  Full detail in `ttft.md`.

## Open threads (ranked)

1. **Frontend-serialization term** (if deemed in-scope — it's harness overhead). Needs: a
   probe sweep to c40–c160 for the parallelism curve, then a *pipelined* frontend/engine
   resource (delay hidden under recompute via `max`). Model + measured `f` live in
   `engine/serving_frontend.py`; probe in `serving_herd_scaling.py`. Honest option: just keep
   it documented as a scope boundary and don't model it.
2. **Forward mode** — `getters/workload.load_distribution` raises `NotImplementedError`;
   forward ceiling `saturated_step_ms` is a 200ms placeholder. The unsolved no-ground-truth path.
3. **Agentic residuals** — swebench / terminalbench ~28–29%, not yet dug into.
4. **Handwavy YAML** — `util_flops` / `util_bw`, `request_overhead_ms` (flat 25ms). See
   `knobs.md` "accuracy levers".

## How to run

```bash
# backtest MAPE / diagnostics
python -m simulator_v2.engine.kv_pressure                       # KV-pressure per cell
python -m simulator_v2.engine.step_trace chat-multiturn-synth 120   # per-step TTFT decomposition

# dashboard: regenerate the v2 predictions JSON, then deploy (copy into dist/)
python3 inference-benchmark/scripts/build_simulator_v2_predictions.py
cp inference-benchmark/dashboard/public/simulator-v2sim-predictions.json \
   inference-benchmark/dashboard/dist/simulator-v2sim-predictions.json
# (served same-origin from dist/; the ?v= hash is a cache-buster, no bundle rebuild needed)
```

**H100 probes** (this box is CPU-only; the GPU is remote):
```bash
ssh h100                       # 10.250.30.45 over wg0; GPUs 0-3,7 usually free (avoid 4-6)
# env: CUDA_VISIBLE_DEVICES=7  PYTHON=~/miniconda3/envs/vllm/bin/python
#      MODEL=/data48/kevinlau/models/Llama-3.1-8B-Instruct
# the h100 repo checkout is STALE — rsync the probe script over before running:
rsync -az profiling/gpu_profiling/vllm/serving_herd_scaling.py h100:/home/kevinlau/agentic-serve/profiling/gpu_profiling/vllm/
# then self-launching sweep (writes CSV):
#   CUDA_VISIBLE_DEVICES=7 <PYTHON> profiling/gpu_profiling/vllm/serving_herd_scaling.py --news 128,2048 --cacheds 0,8000 --concs 1,5,10,20,40,80
```

## Artifacts

- Dashboard JSON: `inference-benchmark/dashboard/public/simulator-v2sim-predictions.json`
  (44 cells, 4 multi-turn profiles; deployed to `dist/`).
- Probe data: `profile_data/results/serving_stage_split_H100_reprobe.csv` (c1 stage split),
  `serving_herd_scaling_H100.csv` (concurrency × (new,cached)).
- Diagnostics/probes: `engine/step_trace.py`, `engine/serving_frontend.py`,
  `profiling/gpu_profiling/vllm/serving_herd_scaling.py`.

## Principles (held this session)

- **No compensating fudges.** We deleted ρ, rejected host-rate scaling, and deferred the
  frontend term rather than tune a lane count per regime. Prefer measured provenance or an
  honest scope boundary over a knob that only fits by coincidence of shape.
- **Provenance over parity.** Every constant should trace to a datasheet, arch, or a probe
  (`knobs.md`). Don't tune forward toward backtest.
