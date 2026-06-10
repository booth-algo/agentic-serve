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
