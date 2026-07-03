# simulator_v2 — handoff

> Session snapshot for whoever picks this up next. Where things stand, what changed, what's
> open, and how to run it. Deep refs: `CODEBASE_MAP.md` (architecture), `ttft.md` (TTFT model
> + findings), `knobs.md` (every tunable). Last updated 2026-07-02.

## Status

v2 predicts per-turn TTFT / TPOT / E2EL via **kernel composition** (backtest) — no in-situ
fits. Backtest + the "Simulator v2" dashboard tab are wired and deployed. Forward mode is
still a stub. The 2026-07-02 landing (shared-prefix pool dedup + tail step-cost terms,
commits `9392b59` + `866da21`) took the headline from 26.9 to **23.6**.

Headline TTFT (H100 / Llama-3.1-8B tp1, cell-MAPE, mean of per-turn APE), after the
2026-07-03 frontend landing (open thread 1 below):

| aggregate | chat | osworld | swebench | terminalbench |
|---|---|---|---|---|
| **16.7%** | 19.6 | 16.9 | 12.6 | 17.5 |

(2026-07-02 pool-dedup + step-cost landing: 26.9 → 23.6; 2026-07-03 frontend stage:
23.6 → 16.7.) swebench c80–c320 sits at 6–15; chat c160–c256 at 10–13. Largest
remaining cells: osworld c80 (37), chat c5 (33), chat c1 (29), osworld c40 (27).

## What changed this session (2026-07-02)

1. **Shared-prefix POOL dedup** — the transition-band (c40–c160 onset) overshoot was
   phantom KV demand: the session-granular cache charged every session its own copy of the
   cross-session shared prefix, which vLLM's APC stores ONCE (256×1024 ≈ 60% of the pool at
   swebench c256) → the eviction cascade fired 2–3 turns early. Reservation (`_schedule`)
   and decode-growth (`_on_step`) are now net of `shared_prefix_tokens`. **Both paths must
   be net** — an earlier `_schedule`-only attempt silently failed (decode growth re-claimed
   the span every turn).
2. **Tail step-cost terms** — deep saturation was a −25% *slope* error: mixed steps priced
   ~197ms vs the measured ~250ms saturated step. Landed (a) cross-context chunk attention
   `rate·U·P` (measured FA3-cached slope, `h100.yaml compute.cross_attn_ms_per_token_pair`
   = 7.29e-7) and (b) additive GPU composition (`decode + prefill + cross`, host still
   pipelined via max) in `_price_step`.
3. **Falsified with data (keep dead)**: the global barrier is EXACT (GT timestamps, 0% turn
   interleave; `runner.py:334` gather); cohort taper is faithful; flat-LRU eviction is a
   no-op; incremental block allocation regresses tails. With 60% phantom demand, any
   eviction policy cascades — the demand was wrong, not the policy.
4. **GT-inference technique** (no engine cache telemetry exists; GT cache fields are
   harness estimates): real recompute per turn = drain-window(p95−p5 TTFT) ×
   (budget−herd)/drain-ITL ÷ beyond-shared cached. Validated at saturation (92–101% vs sim
   99–101%); at onset it was the smoking gun (GT 0–30% where sim said 73–86%).
5. **Docs/dashboard**: ttft.md + knobs.md updated for the new model; dashboard JSON
   regenerated + deployed to `dist/`; repo pushed (the whole `simulator_v2/` tree is now
   committed for the first time).

Accepted regressions in the landing: osworld c160 +5.6, c80 +3.3 (a residual mid-conc hump,
unexplained); terminalbench c120 t17–19 still over (~160–180%).

## Open threads (ranked)

1. **Frontend stage LANDED 2026-07-03 → aggregate 16.7** (from 23.6). Client-referenced
   measured model (f_cli floor+slopes, lanes curve, streaming-load mult curve) wired via
   `engine/serving_frontend.py` + `frontend:` YAML; `prefill_host` and
   `request_overhead_ms` retired (double-count — proven by the V1 A/B breaking every c1
   cell). Probe CSVs: `serving_herd_scaling_H100_{c160,loaded,smallD}.csv`. Remaining
   residuals from this band: **chat c1/c5 (~29/33)** — small-TTFT turns where the f_cli
   floor/slopes over-shoot; likely needs the real-prompt chars/token correction (~0.83×,
   see corollary above) or a chat-prompt re-probe.
2. **Moderate-pressure over-churn — DUG, NOT LANDED** (see ttft.md 2026-07-03 finding):
   terminalbench c80–c160 (~20–22) over-predicts because the sim churns ~4.7× the
   arithmetic shortfall vs GT's ~2.6× (amplifier = admit-recompute's up-front full
   reservation stealing the live herd's caches). A lazy/hybrid/flat/cliff variant ladder
   fixes tb (→~17) and holds every tail, but osworld c200 breaks 9.4→27 under every lazy
   variant: **churn is bistable** (hit-aristocracy vs full-rotation equilibria) and
   allocation timing selects the equilibrium — our session-granular cache can't resolve
   the block-flow timing that picks it in vLLM. Kept eager. Future pass: occupancy-based
   parked-cache survival. Scratch: `/tmp/wf-hint/regime{7,8,9}*.py`. Related, exonerated:
   pool constant (back-solve 432–470k ≈ pinned 436k; confirmatory `pool_capacity_probe.py`
   still armed behind the GPU watcher on the h100, result → `pool_capacity_H100.txt`).
   Separated out: **osworld c40/c80 (~27/37) never evicts** — its hump is frontend
   over-shoot at huge cached contexts (t2 ratios up to 322), same family as chat c1/c5.
3. **tp>1 port (ACTIVE 2026-07-03)** — dashboard now carries the full v1 config matrix
   (21 gpu_keys / 9 models / 2974 rows) as labeled `v2_roofline_firstcut`; only
   H100-8B-tp1 is calibrated. **POLICY: step-level decode grids are BANNED as cost
   models** (they are whole-forward-pass measurements, not kernels; v1 empirically
   refused them for H100x4 — isolated-kernel walls sit above serving GT). tp>1 decode
   = kernel-ADJUSTED composition: tp1 GEMM table queried at sharded shapes (free —
   table is shape-indexed), sharded-head attention (FA grids head-mismatch -> roofline
   fallback until tp2-head grids are profiled), explicit comm terms (prefill all-reduce
   MEASURED at 3.28e-3 ms/tok, `prefill_tp_comm_H100.json`; decode all-reduce needs a
   latency-bound analytic or measurement). Measured per-deployment facts stay: tp2
   pool 62,416 blocks, tp2 GT-derived ceiling. Current first-cut: H100x2 TTFT 36 /
   TPOT 137; H100x4 42 / 340 (anchored-analytic ceiling scales the wrong way with tp).
4. **Forward mode** — `getters/workload.load_distribution` raises `NotImplementedError`;
   forward ceiling `saturated_step_ms` is a 200ms placeholder. The unsolved
   no-ground-truth path, and the strategic reason v2 exists.
4. **FA3-cached grid re-profile** — upgrade the 7.29e-7 cross-attn constant to grid interp
   (the original grid CSV never made it into `profile_data/`; producer candidates in
   `profiling/gpu_profiling/vllm/cuda_events/cached_prefill_steps_v3.py`).
5. **Handwavy YAML** — `util_flops` / `util_bw`, `request_overhead_ms` (flat 25ms). See
   `knobs.md` "accuracy levers". Also `sum_kernels.fused_step_ms`'s mixed `max()` docstring
   is stale (path now only corner-called; the claim "cheaper phase rides free" is
   known-wrong for mixed steps).

## How to run

```bash
# backtest MAPE / diagnostics
python -m simulator_v2.engine.kv_pressure                       # KV-pressure per cell
python -m simulator_v2.engine.step_trace chat-multiturn-synth 120   # per-step TTFT decomposition
python3 /tmp/wf-hint/verify_landed.py                           # 44-cell gate (expects 23.56)

# dashboard: regenerate the v2 predictions JSON, then deploy (copy into dist/)
python3 inference-benchmark/scripts/build_simulator_v2_predictions.py
cp inference-benchmark/dashboard/public/simulator-v2sim-predictions.json \
   inference-benchmark/dashboard/dist/simulator-v2sim-predictions.json
# (served same-origin from dist/; the ?v= hash is a cache-buster, no bundle rebuild needed)
```

**H100 probes** (this box is CPU-only; the GPU is remote):
```bash
ssh h100                       # 10.250.30.45 over wg0; GPUs 0-3 usually free (7 was BUSY 2026-07-02; avoid 4-6)
# env: CUDA_VISIBLE_DEVICES=<free>  PYTHON=~/miniconda3/envs/vllm/bin/python
#      MODEL=/data48/kevinlau/models/Llama-3.1-8B-Instruct
# the h100 repo checkout is STALE — rsync the probe script over before running:
rsync -az profiling/gpu_profiling/vllm/serving_herd_scaling.py h100:/home/kevinlau/agentic-serve/profiling/gpu_profiling/vllm/
# then self-launching sweep (writes CSV incrementally):
#   CUDA_VISIBLE_DEVICES=3 <PYTHON> profiling/gpu_profiling/vllm/serving_herd_scaling.py \
#     --port 8793 --news 128,2048 --cacheds 0,8000 --concs 1,5,10,20,40,80,160 --trials 3 \
#     --out profile_data/results/serving_herd_scaling_H100_c160.csv
```

## Artifacts

- Dashboard JSON: `inference-benchmark/dashboard/public/simulator-v2sim-predictions.json`
  (44 cells, 4 multi-turn profiles; deployed to `dist/`, committed).
- Probe data: `profile_data/results/serving_stage_split_H100_reprobe.csv` (c1 stage split),
  `serving_herd_scaling_H100.csv` (c1–c20), `serving_herd_scaling_H100_c160.csv` (c1–c160,
  in flight — on the h100 checkout, scp back when done).
- Session probes/A-Bs (scratch, this box): `/tmp/wf-hint/regime1_*.py` (volume/steps/price),
  `regime2*_*.py` (evict/alloc/dedup), `regime3_gate.py`, `verify_landed.py`.
- Diagnostics: `engine/step_trace.py`, `engine/serving_frontend.py`,
  `profiling/gpu_profiling/vllm/serving_herd_scaling.py`.

## Principles (held this session)

- **No compensating fudges.** Both step-cost terms are measured physics; the dedup is
  vLLM's actual storage semantics. Three plausible mechanism fixes (flat-LRU, incremental
  allocation, and earlier the fixed-serial frontend) were tested and rejected on data
  rather than tuned into place.
- **Falsify with ground truth before landing.** The barrier/taper/eviction-policy suspects
  each got a direct GT measurement; the one that survived (phantom shared-prefix demand)
  also had exact demand-crossing arithmetic behind it.
