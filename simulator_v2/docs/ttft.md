# Per-turn TTFT

TTFT (time to first token) is **not** a quasi-static per-step quantity like TPOT —
it is the wall-clock **queue wait** a request sees before its first token. It comes
from an event-driven **queue simulation** (`engine/queue_sim.py`) that replays the
cohort through vLLM's scheduler over a **session-persistent KV prefix cache**, and
records `first_token_epoch − arrival_epoch` per request (median per turn-index).

```
TTFT[turn] = queue wait (chunked-prefill scheduling, KV eviction → recompute)
           + request_overhead_ms        (per-request host floor, added at first token)
```

Per-step cost is `max(gpu, host)` where **gpu = decode + prefill + cross-context
attention, additive** (one fused forward pass; FLOPs add — the old `max(decode,
prefill)` piggyback under-priced heavily-mixed steps ~25% at the saturated tail),
and `host` is a **measured** serving term that pipelines with the GPU (see The
model and the 2026-07-02 finding).

## The model

- **Cohort**: `concurrency` sessions, replayed from the cell's real per-session
  trajectories (varied sizes/turn counts de-synchronize the herd). Barrier
  round-robin: turn `t`'s whole herd arrives together; turn `t+1` releases only
  after every turn-`t` request departs (matches the benchmark's `asyncio.gather`).
- **Scheduler**: decode-priority chunked prefill — per-step token budget
  (`max_num_batched_tokens`), per-request chunk cap (`long_prefill_token_threshold`
  = `max_model_len·0.04`), running-set cap (`max_num_seqs`).
- **KV prefix cache** (`PrefixLRUCache`): each session's cached prefix persists
  across turns; under pressure the globally-LRU-oldest prefixes are trimmed from
  their **tail** (dead/departed residents first, then idle herd). Hit/miss is
  decided live from resident blocks. **Pool accounting dedups the cross-session
  shared prefix** (vLLM's APC stores it once, by block content hash): sessions
  reserve only their beyond-shared context, in both the admission reservation and
  the decode-growth path — charging every session its own copy is phantom demand
  (concurrency × shared tokens) that fires the eviction cascade 2–3 turns early
  (see the 2026-07-02 finding).
- **Host serving cost** (`prefill_host` rates, ms/token): re-tokenize/parse/IPC of
  the re-sent prompt each turn (shared once/step + per-request, frac-spread over the
  prefill steps; measured from v1's stage-split, not fit). It **pipelines** with the
  GPU, so the step is `max(gpu, host)`: host-bound for a cheap cache-hit prefill
  (the sub-saturation lift), hidden under a big recompute (the saturated regime is
  unchanged). 0 rates → byte-identical GPU-only step.
- **Cross-context chunk attention** (`cross_attn_ms_per_token_pair`, measured
  FA3-cached slope): a prefill chunk of U tokens attends everything already
  resident for its request (hit prefix + previously completed chunks), costing
  +rate·U·P on top of the full-causal chunk grid. Negligible sub-saturation
  (small U·P); ~+35 ms/step at the recompute tail where 13k-token contexts
  re-prefill in 1310-token chunks.

## The climb mechanism (KV pressure)

A session's KV persists across its turns, so the cohort's cumulative working set
grows. When the **active herd's resident KV exceeds the pool**, the engine must
trim active prefixes; an evicted session **re-prefills its lost tail** next turn
(recompute), on top of its new tokens. That recompute congests the chunked-prefill
budget and delays new arrivals' first token — the high-concurrency TTFT climb,
which compounds because cached context keeps growing.

Define **demand pressure** = peak active-herd resident KV / pool:

```
pressure = max_t  Σ_{sessions reaching turn t} (isl_t + osl_t)  /  kv_pool_tokens
```

H100/8B pool = `27,537 blocks × 16 = 440,592 tokens` (small 8B weights → large KV
pool; see below). `pressure > 1.0` ⇔ the herd alone overflows ⇔ tier-2 eviction ⇔
recompute. Crossover concurrency scales as `pool / context_per_session`, so
long-context agentic workloads overflow far earlier than chat.

## KV-pressure readout (finding)

`python -m simulator_v2.engine.kv_pressure` reports, per cell, the demand pressure,
the realized recompute (`recomp/pool` = re-prefilled evicted-prefix tokens / pool,
from the sim's `recompute_tokens` stat), and the TTFT pred/meas ratio. Two crisp
findings (H100/8B, measured cells):

**1. Recompute turns on exactly at pressure = 1.0.** `recomp/pool` is literally
`0.00` for every cell below 1.0 and non-zero the instant it crosses — the analytic
pool-overflow point and the simulated eviction onset coincide.

| profile | ctx | crosses 1.0 at | pressure → |
|---|---|---|---|
| swebench-multiturn | 8–12k | **c40** | 0.61 → 1.24 |
| terminalbench-multiturn | 4–8k | **c80** | 0.68 → 1.40 |
| osworld-multiturn | 2–9k | **c80** | 0.60 → 1.03 |
| chat-multiturn | ~2k | **c256** | 0.87 → 1.11 |
| chat-singleturn | small | never (max 0.51 @ c500) | — |

**2. Prediction accuracy flips at pressure 1.0.** In the recompute regime
(pressure > 1) the queue sim is accurate; in the sub-saturation regime (pressure <
1) it under-predicts — there is no recompute to carry the TTFT, but the real server
still has contention/host wait we don't yet model. Example ladder (swebench):

| conc | pressure | recomp/pool | pred | meas | ratio |
|---|---|---|---|---|---|
| 10 | 0.33 | 0.00 | 116.8 | 463.4 | **0.25** |
| 20 | 0.61 | 0.00 | 213.3 | 679.9 | **0.31** |
| 40 | 1.24 | 6.09 | 1311.7 | 1364.5 | **0.96** |
| 80 | 2.35 | 33.2 | 7547.0 | 7572.7 | **1.00** |
| 200 | 6.10 | 99.5 | 19726 | 21057 | **0.94** |
| 320 | 9.85 | 163.9 | 32234 | 35970 | **0.90** |

Across profiles: **pressure > 1 → ratio ≈ 0.90–1.15** (good); **pressure < 1 →
ratio ≈ 0.25–0.57** (under). So the low-concurrency under-prediction is precisely
the entire pressure-below-1.0 band.

> The `pred`/`ratio` columns above are the **pre-host-term snapshot** that motivated
> the sub-saturation host term; `pressure`/`recomp/pool` are intrinsic and unchanged.
> With the host term in, the pressure<1 ratios lift to ~0.6–1.0 (see Accuracy), while
> the pressure>1 ratios are unchanged (`max(gpu, host)` hides host under recompute).

Two caveats the readout also surfaces:
- **chat-multiturn stays under even above 1.0** (c256 pressure 1.11 → ratio 0.51):
  crossing 1.0 is necessary but not sufficient — chat's short contexts make the
  recompute volume tiny (`recomp/pool` 0.27), so the climb is weak. Pressure ≫ 1
  (deep recompute) is where the model truly locks on.
- **chat-singleturn over-predicts at mid-conc** (pressure ~0.02): a separate
  prefill-attention *lumping* artifact (`prefill_us(Σchunk)` prices concurrent
  small prefills as one big quadratic sequence), not a pressure effect.

## Finding: sub-saturation under-prediction is erased prefill, not contention

A per-step decomposition (`python -m simulator_v2.engine.step_trace chat-multiturn-synth
120`) traced *where* the sim drains too fast in the pressure<1 band. Result: step cost is
the kernel floor (prefill branch dominates every step) and total drain is ~invariant to the
chunk budget (prefill is priced ~linearly in tokens), so it was **never** "too few steps."
The gap is that the sim **credited away prefill work the server actually did**.

The ground truth accounts prefill with `cache_estimate_source = "previous_prompt_tokens"`:
per request `new_prefill = total_context − cached_context`, and `cached_context[t] =
total_context[t−1]` (the previous **prompt**) — which excludes the previous **response**.
So the response (`asst[t−1]`, ~230 tok/turn) is re-prefilled every turn and is already in
the `new_prefill` we feed the sim. A short-lived `response_resident_fraction` (ρ) knob
removed it *again* (double-credit): turn-0 measured median 362 ms ≈ the sim's throughput on
the full 15 000-tok herd (394 ms), yet ρ=1 predicted 240 ms. Because that credit is a
structural double-count for this ground truth (correct only at 0), **the knob was removed**
and the sim now takes `new_prefill` at face value — chat conc120 39.5→27.1%, aggregate
28.6→26.9%, no residual-service term. This is the "missing contention" the sections above
anticipated: it was measured re-prefill all along. (Caveat: whether vLLM's APC *truly*
re-prefills the response is unconfirmable from a harness estimate; taking `new_prefill` as
given keeps the backtest consistent with its scored inputs.)

## Finding: the residual sub-saturation gap is API-server frontend serialization

After ρ, a residual under-prediction remains in the pressure<1 band. The concurrency ladder
localizes it: **sim/measured TTFT ≈ 0.9–1.0 at c1** (per-request model is right) but **drops
to ~0.56–0.72 the instant a herd forms (c5+)** and recovers only once recompute dominates
(osworld/swebench → ~1.0 at c40; chat/terminal stay low, they saturate far later). So it is a
*herd* effect, not a per-request floor.

A live multi-concurrency probe on the H100 (`serving_herd_scaling.py`; a barrier burst of N
identical requests over a shared primed prefix, so GPU stays flat) pins the stage. At
new=128, cached=2000 (`profile_data/results/serving_herd_scaling_H100.csv`):

| conc | ttft_med | scheduler queue | GPU prefill | **frontend (host)** |
|---|---|---|---|---|
| 1  | 39  | 0.03 | 15.7 | **23** |
| 5  | 134 | 0.03 | 39.8 | **94** |
| 10 | 195 | 0.05 | 43.9 | **151** |
| 20 | 325 | 0.10 | 77.7 | **247** |
| 40 | 360 | 0.03 | 37.4 | **323** |

Scheduler queue ≈ 0 and GPU prefill stays flat; **the entire TTFT growth is the frontend**
(HTTP recv/parse, chat-template, tokenize, IPC, SSE stream) processing the concurrent herd
one-at-a-time. This is **serving-harness overhead, not GPU physics** — which is exactly why
the kernel-composition sim misses it (it prices kernels + a flat, *parallel* 25 ms/request,
never a *serial* frontend) and why scaling the host *rate* fit by coincidence of shape
(∝ tokens·herd) but is a fudge — the true effect is a serial per-request frontend *service*.

Corollary rules, from the same probe campaign: the generic host rate is **confirmed correct**
(`serving_stage_split_H100_reprobe.csv`: cached 6.13e-3 vs production 5.887e-3, +4%, within
tolerance), and real prompts are **lighter** per token (~3.6 chars/tok vs the synthetic 4.44,
~0.83×) — so "under-probed host rates" is falsified; do **not** scale the host rate.

**Verified server-side.** The client wall could be inflated by the probe's own single
asyncio client serializing N SSE reads, so a second sweep also scraped vLLM's own
`time_to_first_token_seconds` histogram (server-side, client-loop-immune) across a (new,cached)
× conc grid (`serving_herd_scaling_H100.csv`, the extended run). The *server* frontend
(`server_ttft − queue − prefill`) grows **7–10× from c1→c20** at every work-point — it is
genuinely the server frontend, not a client artifact.

**Characterized.** Per-request frontend service `f ≈ 6.5 + 0.0049·new + 0.0043·cached` ms —
a ~6.5 ms floor + ~4.6e-3 ms per *total prompt token* (≈ the host rate; the frontend
tokenizes/parses the full re-sent prompt). Serialization `s(N)=frontend(N)/f` tracks the
barrier-serial `(N+1)/2` at small N (3.2 / 5.1 at N=5/10) then goes **sub-linear** at N=20
(7.4–8.8 vs 10.5) — the frontend gains parallelism/GPU-overlap under load.

**SHIPPED 2026-07-03** — the blockers above were resolved by measurement, not tuning.
Three probe campaigns on the live H100 (`serving_herd_scaling.py`, extended to c160 +
decoy-loaded variants; CSVs in `profile_data/results/serving_herd_scaling_H100_{c160,loaded,smallD}.csv`):
(1) the **lanes curve** to c160 (≈1 through c10 → ~2.5 at c160; the frontend never gets
very parallel, it just stops being strictly serial); (2) the **streaming-load multiplier**
on f (×1.26–1.59 when peers pump SSE, ramping gently from ×1.05 at 2 streams); (3) the
**client-side reference**: the benchmark clocks TTFT on a single-process asyncio client
whose own loop adds ~1.3× over the server-side span — so the shipped model is
client-referenced (f_cli = 9.8 + 6.0e-3·new + 6.1e-3·cached; its floor IS the send+return
path, retiring `request_overhead_ms`). Mechanism in `engine/serving_frontend.py`
(`herd_arrival_epochs`: FIFO drain at fractional measured lanes, ARRIVAL delayed into the
engine, TTFT clocked from release — engine-hiding falls out structurally); constants in
the `frontend:` YAML section. The `prefill_host` in-engine proxy is retired with it (same
cost measured twice — keeping both broke every c1 cell in the V1 A/B, the double-count
proof). Gate: aggregate 23.56 → **16.7**; chat c160–c256 29–33 → 10–13; the c5–c20 band
34–44 → 8–33. Residuals: chat c1/c5 (~29/33), osworld c40/c80 (~27/37).

**Scope note.** This is **serving-harness overhead** (API server HTTP/tokenize/IPC/stream),
not GPU/kernel physics. The kernel-composition sim faithfully predicts the *engine* TTFT
(queue+prefill); the benchmark's measured TTFT additionally carries this frontend
serialization. Most of the sub-saturation gap is that boundary, not a kernel-floor error.

## Finding (2026-07-02): the three-band fix — shared-prefix pool dedup + tail step cost

The per-turn signed grids split the residual error into three bands with independent
causes; two were fixed this session (aggregate 26.86 → **23.56**), the third
(frontend, above) is in measurement.

**Band 1 — deep saturation (c200+, tails): −25% *slope*, a step-pricing error.**
The sim's recompute *volume* was exonerated first: at the c256/c320 tail it already
re-prefills 91–93% of every cached token (miss% = 100), and the per-request
distribution shows sim p5 ≈ GT p5 with the whole gap in the drain window (sim 91.5 s
vs GT 119.9 s at swebench c320 t29). The step audit found the sim charging ~197 ms
for a ~7.5k-token mixed step the real engine takes ~250 ms over (our own measured
TPOT plateau). Two missing kernel terms close it almost exactly (197+35+21 ≈ 253):
cross-context chunk attention (rate·U·P, the measured FA3-cached slope — the
"separate future grid" the old `fused_step_ms` docstring promised) and decode no
longer riding free under `max()` (one fused pass; FLOPs add).

**Band 2 — transition (c40–c160 onset): +80–150%, a phantom-demand error.**
With no engine cache telemetry in the GT (all cache fields are harness estimates),
real recompute was inferred per turn as `drain-window(p95−p5 TTFT) ×
(budget−herd)/drain-ITL`, ÷ beyond-shared cached tokens — validated at saturation
(GT-implied 92–101% vs sim 99–101%). At onset it read **GT 12/0/0/30/69% (swebench
c256 t1–5) vs sim 40/73/81/86/98%**: the sim's cascade fired 2–3 turns early. Root
cause: the session-granular pool charged **every session its own copy of the
cross-session shared prefix**, which vLLM's APC stores **once** — 256×1024 ≈ 262k
phantom tokens = 60% of the pool at swebench c256. The demand-crossing arithmetic
matches exactly: real (deduped) demand crosses the pool at ctx ≈ 2.7k ≈ t4–5, where
GT's recompute takes off; the sim's gross demand crossed at t1–2, where its cascade
fired. Post-dedup the sim's onset is 0/0/0/37/76/100% at t1–7 — the real cascade.
The phantom fraction shrinks from 60% of the pool at onset to ~7% of demand at t29,
which is why the error was confined to the transition band.

**Falsified along the way (keep these dead):**
- **Global barrier arrivals are EXACT** — GT `dispatch/completed` timestamps show
  0% turn interleave at every cell checked, next herd fires +0.0 s after the last
  straggler (`runner.py:334` `asyncio.gather`). Not a modeling gap.
- **Cohort taper is faithful** — sim herd size tracks GT per turn to within the few
  failed requests (osworld c200 collapses 200→55 by t15 and the trajectory-cycled
  cohort reproduces it).
- **Flat-LRU eviction** (dropping the departed-first tier): no-op. **Incremental
  block allocation**: negative (wrecks tails). With 60% phantom demand, *any*
  eviction policy cascades — the demand was wrong, not the policy.
- **Dedup leak warning**: an earlier `_schedule`-only dedup attempt silently failed
  because the `_on_step` decode-growth path still claimed blocks gross-of-shared,
  re-inflating the phantom demand every turn. Both paths must be net-of-shared.

**Gate (44 cells, cell-MAPE mean)**: 26.86 → 23.56. swebench c80–c320: 24–29% →
7–15%; terminalbench c200/c256 halved; osworld c320 12.3 → 6.6. Accepted
regressions: osworld c160 +5.6, c80 +3.3 (a residual mid-conc hump, unexplained).
TPOT byte-identical.

## Why an 8B on an H100 preempts at all

The small model is exactly *why* the pool is large: ~16 GB of weights leaves ~57 GB
for KV = **~441k tokens**. But preemption is a *workload* property —
`demand ≈ concurrency × resident_context`. Long agentic contexts (8–9k), multi-turn
KV persistence (context grows every turn), and the high concurrencies an 8B is run
at for throughput multiply past 441k easily (swebench c200 demands ~1.6M tokens,
3.6× the pool). vLLM v1 preemption is **recompute** (free LRU blocks, re-prefill),
not swap.

## Status

| Piece | Where | State |
|---|---|---|
| Event-driven queue sim (barrier herd, chunked prefill, capacity gate) | `engine/queue_sim.py` | ✅ done |
| Per-step cost — additive `decode + prefill + cross` (2026-07-02; was `max` piggyback) | `engine/queue_sim.py` `_price_step` | ✅ done (`fused_step_ms`'s mixed `max()` path is now only corner-called; docstring stale) |
| Cross-context chunk attention (measured FA3-cached slope) | `_price_step` + `cross_attn_ms_per_token_pair` (YAML) | ✅ done 2026-07-02 (constant; upgrade to grid interp when the cached grid is re-profiled) |
| Session-persistent KV + LRU tail-trim eviction + recompute | `engine/queue_sim.py` (`PrefixLRUCache`) | ✅ done |
| Shared-prefix POOL dedup (reservation + decode-growth net of shared) | `engine/queue_sim.py` `_schedule`/`_on_step` | ✅ done 2026-07-02 — fixes the transition-band cascade onset |
| Per-request host floor — `request_overhead_ms` | `getters/hardware.py` (config) | ✅ done (flat 25 ms; handwavy) |
| Real per-session trajectories | `getters/workload.py` (`load_benchmark`) | ✅ done (incl. single-turn `input_tokens` fallback) |
| KV-pressure readout (`pressure`, `recompute_tokens`) | `engine/kv_pressure.py` | ✅ done |
| Sub-saturation host term (re-tokenize the re-sent prompt, `max(gpu, host)`) | `engine/queue_sim.py` + `prefill_host` config | ✅ done — lifts the pressure<1 band (per-turn 54.9→33.7%) |
| Cached-prefix attention (FA3 re-encode) | — | superseded 2026-07-02 — the earlier c1-band attempt was reverted (not the lever there); the *recompute-tail* form landed as the cross-context term above |
| Per-request prefill attention (fix single-turn lumping) | — | ❌ open follow-up (single-turn over-prediction) |
| Re-probe host rates on replayed chat prompts | — | ❌ open — measured probe rate leaves c5–c20 mildly under (v1's caveat) |

## Accuracy (H100/8B, 55 cells)

Cumulative, per lever:

| stage | per-turn MAPE | cell-headline MAPE |
|---|---|---|
| pre-recompute (queue only) | 71.0% | 66.5% |
| + recompute (`PrefixLRUCache`) | 54.9% | 50.9% |
| + sub-saturation host term | 33.7% | 38.3% |

On the 44-cell multi-turn dashboard aggregate (mean of cell-MAPEs, the HANDOFF
headline metric): ρ-removal 28.6→26.9, **+ pool dedup + step-cost terms
(2026-07-02) 26.9→23.6** — per profile: chat 32.4, osworld 17.9, swebench 18.2,
terminalbench 25.8. The swebench high-conc column (c80–c320) sits at 7–15%.

- **Recompute regime (pressure > 1): TTFT ratio ≈ 0.9–1.15** — the climb is captured,
  and unchanged by the host term (`max(gpu, host)` hides host under the recompute).
- **Sub-saturation (pressure < 1)** is now lifted from ~0.25–0.45 to ~0.6–1.0; the
  host term closed most of the gap.
- Residual: (a) chat/terminal/swebench c5–c20 still mildly under (~0.6–0.8) — the
  measured *probe* host rate is lighter than real chat prompts (v1's caveat; re-probe
  to close); (b) chat-singleturn over-predicts (the separate prefill *lumping* bug).

## Reference (v1)

`simulator/ttft_queue_sim.py` (the closed-loop sim with recompute preemption + the
block-level `PrefixLRUCache`), `simulator/ttft_predict.py` (the static closed-form
predictor, ~61% MAPE — superseded by the sim). v1's `_price_step` also itemizes the
measured host/FA3/TP-comm prefill terms v2 currently folds into the kernel floor +
a flat host overhead.
