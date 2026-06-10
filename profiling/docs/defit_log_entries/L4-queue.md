# L4 — queue/eviction re-derivation (S7–S9): trace-oracle capture log

Lane: `queue-eviction-rederive` / `agentic-serve-queue`. Scope: `simulator/ttft_queue_sim.py`
internals (queue/eviction/cohort — NOT the pinned PREFILL_* pricing constants),
sim-behavior tests, `profiling/profile/vllm/engine_trace/*`. Engine traces are
VALIDATION ORACLES only, never predictor inputs.

## 2026-06-10 — baseline + oracle inventory + fresh churn captures

### Replay-ON baseline (before any sim edit)

`python3 -m profiling.process.gate_scoped_rows --out /tmp/l4_base.predictions.json
--metrics-out /tmp/l4_base.metrics.json` in this worktree (realized pools symlinked;
no replay-OFF warning; 132 rows). Binding gate for any future L4 change:
H100/A100 `ttft_cell`/`e2el_cell`/`tpot_cell`/`chat` <= baseline + 0.3,
H100 swe-plateau <= +0.3.

| gpu | tpot_cell | ttft_cell | e2el_cell | chat (tpot_profile) | swe plateau |
|---|---|---|---|---|---|
| H100 | 14.3182 | 18.1328 | 10.7586 | 5.4631 | 8.8219 |
| A100 | 14.3417 | 22.2221 | 15.8638 | 17.2226 | 8.9727 |
| H100x2 | 28.7486 | 29.0163 | 21.8316 | 17.2432 | 4.5659 |

### Tooling recovered into the worktree

`profiling/profile/vllm/engine_trace/{serving_engine_steps.py,run_instrumented_api_server.py,sitecustomize.py}`
were referenced by `profiling/README.md` and `h100_setup.md` but existed ONLY on the
H100 host (`/data48/kevinlau/agentic-serve/...`, never committed). Recovered verbatim
into the lane-owned path so the capture method is reproducible.

What the tracer logs (per scheduler step, vLLM V1 class hook on `Scheduler.schedule()`
plus a KV-cache truth hook): decode batch + request ids, prefill seqs/tokens/request ids,
waiting/running queues + request ids, `free_kv_blocks`, preemption/swap/recompute counts +
preempted request ids, and `engine_cache_truth.kv_events` with per-request
`get_computed_blocks` (LIVE computed_tokens at scheduling time) and `allocate_slots`
(new_block_count, allocation_failed). Two capture modes: offline full-cell replay
(`--trace-scope full-cell --prompt-shape benchmark-cache-faithful`, one vLLM process,
prefix cache evolves across turns, turn barrier = herd) and live server mode
(`run_instrumented_api_server.py` + benchmark client).

### Archived-trace sufficiency review (`profile_data/_archive/vllm_engine_step_trace_*`)

Server-mode captures exist for swe c40/c80 t12, swe c320 t2, terminal c80 t16 (+ smoke,
+ full-cell cache_truth for swe c320 t2 / terminal c80 t16). All have per-step
kv_events. Churn audit (per-request live hit vs the session's prior history):

| trace | turn>0 lookups full/partial/total-miss | evicted tokens | verdict |
|---|---|---|---|
| terminal_c80_t16 serving | 945 / 315 / 0 | 2.30M | heavy natural churn |
| swe_c320_t2 serving | 380 / 260 / 0 | 0.33M | moderate churn, only 2 turns |
| swe_c80_t12 serving | 949 / 11 / 0 | 0.04M | near-zero churn |
| swe_c40_t12 serving | 480 / 0 / 0 | 0 | zero-churn control |

First S7 observation falling out of the audit: under natural pressure the engine
produces ONLY partial prefix losses (315 partial, 0 whole-session misses in terminal
c80 t16) — tail blocks evicted first, head-of-prefix survives. That is the
LRU-oldest-block-by-block signature (v1 frees a finished request's blocks tail-first
onto the free queue). The sim's tier-2 whole-session MRU preemption would produce
whole-history misses, which never occur in the archived serving traces. Preemptions
and allocation failures are 0 everywhere in the archive — eviction in v1 is silent
free-queue reuse, NOT preemption.

Gap: no osworld-like long-context capture, and no capture with pool pressure high
enough to expose eviction ORDER while some prefixes still survive.

### Fresh captures (H100 GPU 5, full-cell offline mode, ~12 min GPU total, GPU left clean)

All from `/data48/kevinlau/queue_trace_run/` (fresh dir), benchmark-turns CSV cells,
`--include-diagnostic` so every turn of the trajectory replays, Llama-3.1-8B-Instruct,
max-model-len 16384, temperature 0, local copies under
`profile_data/_archive/l4_queue_trace_run/` (gitignored archive; unique `l4_` names):

| run id | cell | gpu-mem-util (pool) | regime captured |
|---|---|---|---|
| `l4_osworld_c40_t29_full_cell` | osworld c40, turns 0–29 | 0.90 (27650 blocks) | no-pressure live-hit control: min_free 6374, hit_frac ≈0.99 late turns, 0 preemptions |
| `l4_osworld_c40_t29_pool050_full_cell` | osworld c40, turns 0–29 | 0.50 | long-context churn→recovery arc: min_free 1, 1 preemption, hit_frac 0.00–0.36 turns 3–10, recovers to 0.99 |
| `l4_swe_c40_t12_pool060_full_cell` | swe c40, turns 0–12 | 0.60 | partial-churn sweet spot for eviction-ORDER adjudication: hit_frac 0.76–0.94 turns 1–10, collapse at t11–12, min_free 178 |
| `l4_swe_c40_t12_pool035_full_cell` | swe c40, turns 0–12 | 0.35 | thrash extreme: min_free 7, hit_frac 0.000 from turn 4, still 0 preemptions (engine recomputes, never tier-2 preempts idle sessions) |

Each run also produced a `*_token_history.json` (exact prompt/output token ids per
session/turn) for sim-side replay parity, and the step jsonl/csv carry per-request
`get_computed_blocks` live-hit truth (S8 oracle) plus queue contents per step
(S9 herd-order oracle).

Reduced-pool runs shrink TOTAL pool capacity to reach the eviction regime on one GPU;
they adjudicate ORDERING/protection semantics (S7/S9) and live-vs-frozen hit logic
(S8), not absolute capacity calibration — capacity stays governed by the deployment
configs.

### Verdict on sufficiency

Archived serving traces ALONE were insufficient (no osworld-like load, churn only in
terminal c80 t16, no controlled pressure sweep). Combined set (archive serving-mode +
4 fresh full-cell pressure points) is sufficient to adjudicate S7 (eviction order:
whole-session MRU vs LRU-oldest-block), S8 (barrier-frozen vs live hit/miss), and S9
(herd_pending protection vs engine order). Re-derivation work proceeds against these
oracles next.

## 2026-06-10 — S7/S8/S9 ADJUDICATION (analysis only, no sim edits yet)

Tool: `profiling/profile/vllm/engine_trace/compare_eviction_semantics.py` — a faithful
re-implementation of the vLLM v1 BlockPool (block-hash prefix cache, single LRU free
queue, tail-first frees, popleft eviction) driven by each trace's own kv_events stream
and VALIDATED against the trace's live `get_computed_blocks` truth, plus counterfactual
replays of the production `PrefixLRUCache` rule variants under the engine's own
admission/finish order. Engine request id `<n>-<hash>` maps to token_history by global
submission counter (verified 100% by prompt-token match on every trace).

**Replica validation (the S7 order+granularity verdict in one number):** the LRU-oldest
block-by-block replica reproduces the engine's live computed_tokens EXACTLY on
100% (osworld no-pressure), 92.7% (osworld pool050), 98.1-98.6% (swe pool035/060) of
lookups. Flipping ONLY the eviction order to MRU-newest degrades exactness to 80.9-93.3%
and grows absolute token error 3.2-26x (pool050: 136k -> 1115k tok; pool060 final
deterministic run: 14.8k -> 384k). vLLM v1 evicts
free-queue blocks LRU-OLDEST, block-by-block — the sim's tier-2 whole-session MRU has no
engine analog.

**S7 granularity:** all four ARCHIVE serving traces (natural load): 100% of turn>0
prefix losses are PARTIAL (tail-trim, head survives; zero whole-prefix misses in 602
lossy lookups; terminal c80 median loss frac 0.997 yet never 1.0). Full-cell barrier
replays: partial losses appear at the pressure edge (swe pool060 t11: 16 full / 9
partial / 15 zero) and 100%-zero turns occur ONLY when the herd working set exceeds the
whole pool (then LRU recycling kills every prefix before reuse — an emergent outcome,
not victim selection). The sim's whole-session eviction is a victim-selection primitive
the engine does not have.

**S8 frozen-vs-barrier vs live:** freeze over-credit (replica barrier snapshot minus
live truth at scheduling): osworld pool050 977k tok = 27.0% of frozen credit (135 reqs
lose prefix MID-TURN); swe pool060 360k = 20.2%; swe pool035 826k = 85.4%. No-pressure
control: 0 (freeze is harmless when nothing evicts). Live erosion is visible even within
a SINGLE scheduling step (pool060 step 287: five same-step lookups descend 353/313/273/
233/193 blocks as each peer's allocation eats the next prefix).

**S9 herd protection:** owner status of evicted blocks at eviction time: WAITING herd
members supply 69.6% (pool060), 50.2% (pool050), 43.0% (pool035) of all evicted blocks —
the sim's herd_pending forbids exactly these. The engine also evicts just-finished
sessions' blocks LAST (newest in free queue) while the sim trims them FIRST
(depart -> herd_pending.discard -> tier-1 victim): the order inversion is real on both
ends.

**Counterfactual re-prefill tokens (engine admission/finish order, trace pool):**

| trace | engine truth | sim CURRENT (frozen+tail+whole) | live+lru+PARTIAL |
|---|---|---|---|
| osworld pool050 | 1,850,012 | 882,728 (-52.3%) | 1,617,408 (-12.6%) |
| swe pool060 | 669,064 | 307,024 (-54.1%) | 660,360 (-1.3%) |
| swe pool035 | 1,952,856 | 1,126,512 (-42.3%) | 1,924,488 (-1.5%) |
| osworld full pool (control) | 395,660 | 405,616 (+2.5%) | 405,616 (+2.5%) |

Same story on the FA3-quadratic proxy sum M*(R+M/2): current -35% to -51% vs truth;
live+lru+partial -0.9% to -11.8%. The current cluster UNDER-counts re-prefill work
under pressure in BOTH linear and quadratic terms; engine-faithful partial LRU + live
hits IS the missing mechanism (the S8 freeze + S9 protection were compensating for the
cascade that whole-session eviction itself caused). Residual osworld pool050 gap
(-12.6%, concentrated t8/t10 drain handoff) is the session-granular cache's inability
to represent block interleaving — documented as the honest stop-point for the cache
model itself.

Caveats: Spearman(finish-order, loss) per-turn correlations are CONFOUNDED (drain +
context-size correlate with finish order; signs mixed) — the LRU-vs-MRU replica
falsification supersedes them. Counterfactuals use the engine's admission order, not
the sim's emergent queueing; production gate impact still requires the replay-ON gate
after any sim edit (baseline pinned above).

## 2026-06-10 — RE-DERIVATION ROUND 1: S7 LANDED (engine-faithful eviction), S8 freeze GATE-RETAINED

### What changed in `simulator/ttft_queue_sim.py` (queue/eviction internals only; PREFILL_* pins untouched)

**S7 LANDED — tier-2 partial LRU trims replace whole-session MRU preemption.** `_evict` tier 2
now trims idle herd residents PARTIALLY (tail blocks, exactly `need`) in global-recency
LRU-oldest order; `preempt_policy` default flipped `'tail'` → `'lru'` everywhere
(`_ServerState`, `_run_sim`, `predict_cell_ttft_qsim`). `'tail'` (MRU-first) and
`_trim_tail(whole=True)` survive ONLY as the adjudication tool's counterfactual/falsification
seams (`compare_eviction_semantics.py`), never the production path. Evidence (previous entry):
LRU-oldest replica exact on 92.7–100% of live lookups, MRU flip 80.9–93.3% with 3.2–26x token
error; 100% of natural-load prefix losses partial (zero whole-session misses in 602 lossy
lookups). Whole-prefix loss remains an EMERGENT outcome (`need` >= the victim's whole residual
under extreme pressure), not a victim-selection primitive.

**Gate (replay-ON, 0 warnings): S7-only is prediction-BYTE-IDENTICAL to the pinned baseline**
(`cmp` equal on `/tmp/l4_base.predictions.json` vs `/tmp/l4_r1b.predictions.json`; all gate
metrics equal to 4 decimals) — exactly the adjudication's prediction ("the freeze hides
eviction from hit accounting; all frozen variants produced byte-identical re-prefill totals").
NOT dead code: an over-subscribed swe c200 cell exercises 2380 tier-2 victim trims (81%
partial). So the falsified rule retires at zero prediction risk.

**S8 unfreeze BUILT AND GATE-REJECTED (the honest stop-point of this round).** Implemented the
engine-true LIVE hit/miss at admission (`cache.cached_blocks(sid)` instead of
`resident_at_barrier`) bundled with S7 — the trace-validated `live+lru+partial` combination
(counterfactual within −1.3…−12.6% of engine re-prefill truth vs −42…−54% for the current
rules). Replay-ON gate vs baseline (`/tmp/l4_r1.metrics.json`):

| gpu | ttft_cell | e2el_cell | tpot_cell | chat |
|---|---|---|---|---|
| H100 | 18.1328 → 21.6005 (**+3.47, FAIL**) | 10.7586 → 10.4291 (−0.33) | identical | identical |
| A100 | 22.2221 → 21.0000 (−1.22, improves) | 15.8638 → 14.7126 (−1.15, improves) | identical | identical |
| H100x2 (advisory) | 29.0163 → 33.9071 (+4.89) | 21.8316 → 24.1462 (+2.31) | identical | identical |

Per-cell: the regression is concentrated where the analysis predicted — swe/terminal (and
osworld) high-conc cells flip from negative/near-zero signed error to large OVER-prediction
(e.g. H100 terminal c160 signed mean +375 → +1818 ms; swe c80 −650 → +1118 ms). Reading: the
re-prefill VOLUME is now trace-faithful, so the freeze was NOT compensating eviction semantics
— it compensates the volume→TTFT over-amplification that lives outside the eviction cluster
(pricing/queue-amplifier interaction, audit-v2 S10 territory / the pinned PREFILL_* cluster,
out of this lane's scope and not derivable from these traces). Per the honesty rule ("the
mechanism that should replace a compensating rule's compensation must come from the traces,
not a knob") and the no-cherry-picking precedent (A100 improves, H100 regresses — same shape
as the util-curve rejection), the freeze is RETAINED ON THE GATE and is now LABELLED as a
compensating rule at every site (`_ServerState.resident_at_barrier`, `_schedule`,
`_release_herd`, module docstring), pinned by
`test_hit_miss_frozen_at_barrier_retained_compensating_rule` so it cannot drift silently.

**S9 (herd tier structure) kept, residual divergence documented.** Tier 1 (free residents)
before tier 2 (idle herd) is the counterfactual-VALIDATED combination; the engine itself draws
no herd distinction (waiting herd members supplied 43.0–69.6% of evicted blocks; just-finished
sessions' blocks are evicted LAST, not first). The session-granular tier ordering remains the
documented honest stop-point of the cache model (it cannot represent block interleaving).

**Tests** (`test_ttft_queue_sim.py`, sim-behavior only; constant-pin test untouched): added
`test_tier2_trim_is_partial_lru_oldest`, `test_tier1_free_residents_still_reclaimed_before_herd`,
`test_preempt_policy_default_is_engine_faithful_lru`,
`test_hit_miss_frozen_at_barrier_retained_compensating_rule` (replaces nothing — the freeze had
no behavior pin before). Full suite green.
