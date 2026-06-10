# TTFT prefill-pricing de-fit plan: host SUM (R2) → GEMM util cap (R1/S6) → TP comm (G3)

**Goal:** close the fitted values that audit v2 (`fitted_constants_audit_v2.md`) confirmed in the
TTFT step pricing — the host cached-token **sum** 6.103e-3 (R2, RED), the prefill-GEMM saturation
cap `PREFILL_GEMM_UTIL_SAT = 1.0` + its ramp shape (R1/S6, RED), and the tensor-parallel comm
remainder `PREFILL_TP_COMM = 0.00585` (G3, GRAY) — by replacing each with a regenerable
measurement, gated under the **production replay-ON configuration** (the fidelity rule learned
2026-06-10: a gate without the per-GPU realized pools measures a non-production cohort and has
already flipped one verdict). Every rule below is pre-registered; "measurement rejected on the
gate → keep value + honest label + documented disagreement" is a SUCCESS outcome (knee precedent).

**Shared gate protocol (all three items):**
- Baseline: `python3 -m profiling.process.gate_scoped_rows` at the current branch state, run in a
  checkout WITH the realized pools (the tool now warns if they are absent). Record the full
  metrics JSON.
- Binding (vs baseline, MAPE points): H100 & A100 `ttft_cell`, `e2el_cell`, `chat` ≤ +0.3;
  `tpot_cell` must be **byte-identical** (none of these constants touches the TPOT path — any
  TPOT delta means a wiring bug, not a judgment call).
- H100x2: advisory for R2/R1; for G3 see its own rule (it is the tp2 term being remeasured).
- Tests: pin updates per outcome; full `pytest simulator/tests/ profiling/tests/` green.
- Docs per outcome: De-fit log entry in `prediction_construction.md` (+ debt/audit-v2 rows),
  artifact `_notes`, memory one-liner.

---

## Item 1 — R2: host cached-token SUM 6.103e-3 → the live measurement (OFFLINE, do first)

**What it is:** both `PREFILL_HOST_*` constants are `partition × SUM`. The partition (0.5236) is
the measured pooled-OLS point estimate (de-fit 2026-06-10). The SUM is still the cached
coefficient of the c1 *benchmark* regression (commit `760d9bd`) — fitted to cells inside the
scored validation payload. The regenerable live measurement exists on disk:
`build_host_split.py`'s lstsq over `prefill_live_ttft_H100.csv` → **5.8872e-3** (−3.5%).

**Pre-registered step:**
1. Extend `build_host_split.py` to emit the candidate constants from the LIVE sum
   (`shared = 0.5236 × 5.8872e-3`, `perreq = 0.4764 × 5.8872e-3`) alongside the current ones —
   one artifact, both sums recorded, provenance per value.
2. **Gap decomposition (before gating, evidence not a gate):** explain the 0.216e-3/tok
   difference. Known candidates: (a) the live probe's fixed ~12.5 ms/req cost (excluded P=2000
   plane) being absorbed differently — reconcile against the measured per-request prefill floor
   (production 25.86 ms vs the live lstsq floor 16.03 ms: does floor + fixed account for the
   sum gap?); (b) benchmark prompts exercise heavier chat-template/tokenizer paths than the
   probe's synthetic prompts. Write the reconciliation into the artifact `_notes`.
3. Wire candidate → gate per the shared protocol. Expected risk: the fitted sum may be
   compensating for unmodeled host cost, so TTFT may rise ~0.1–0.3pt; the gate decides.
4. Outcome A (PASS): adopt; pins move to the live-sum constants; R2 closed-measured.
   Outcome B (FAIL): keep 6.103e-3 + the existing honest label; record the gate numbers AND the
   gap decomposition — if (2) attributes the gap to a real unmodeled term, file that term as the
   actual de-fit target (the sum is then a documented proxy, not an open mystery).

No GPU. ~1 hour. Tooling: existing.

## Item 2 — R1 + S6: `PREFILL_GEMM_UTIL_SAT = 1.0` and the util ramp shape

**What it is:** `_prefill_gemm_per_tok_loaded` ramps util from `util_flops` (0.65) to 1.0 as the
per-step batch fills the chunked-prefill budget. Endpoints are engine config (verified); the cap
and the linear-ramp *shape* are validation-anchored. Two measurements disagree about what
happens at large batch: the GT turn-0 cohort rate implies util ≈ 1.05 (> 1, impossible — broken
accounting) while the repo's only in-domain microbench (`profile_data/kernels/prefill_gemm_H100.csv`)
shows util **plateauing at 0.655–0.672** with no ramp at all.

**Phase A — resolve the accounting contradiction (OFFLINE, prerequisite):**
1. Re-derive the "GT cohort ≈ 15.5 ms/1k" number from bench data with explicit accounting:
   per-step wall vs per-request sums under CONCURRENT chunked prefills (multiple requests share
   a step's token budget — wall-clock per token across overlapping prefills can legitimately
   beat the single-stream roofline *per request* without violating physics *per step*). State
   which quantity the sim's `_price_step` actually prices (total chunk tokens per step) and
   compute the like-for-like measured rate for THAT quantity.
2. Audit `prefill_gemm_H100.csv` provenance: what did it measure (offline kernel? serving
   step?), at which m, with/without CUDA-graph + chunked-prefill overheads. Decide whether it is
   the right anchor for serving-step pricing or needs a serving-context re-measurement.

**Phase B — measure util(m) in the serving context (H100, only if Phase A says the existing
artifact is not serving-faithful).** Check the H100 host for running jobs FIRST (standing rule).
Offline-engine run (lane-B pattern — `cuda_events` around `execute_model`, mp=0 tp1): prompts
sized so chunk steps sweep m ∈ {512, 1310, 2048, 4096, 6144, 8192} (long prompts chunk at the
budget); per-step device-ms vs tokens-in-step → `util(m) = roofline_ms(m) / device_ms(m)`.
Artifact: `profile_data/kernels/prefill_gemm_util_H100.csv` + builder, deterministic.

**Phase C — wire + gate:** replace `util_flops → UTIL_SAT` linear ramp with the measured
`util(m)` lookup (interpolated, decode-grid pattern; falls back to the current ramp when the
artifact is absent so non-H100 configs are unchanged). Gate per the shared protocol.
**Pre-registered expectation:** if the measured plateau (~0.67) regresses TTFT badly, that is
evidence the cohort-rate accounting (Phase A) — not the cap — was carrying real overlap physics;
fallback = keep the ramp + cap with the honest label, and file the per-step overlap model as the
structural successor (it pairs with S10).

## Item 3 — G3: `PREFILL_TP_COMM = 0.00585` (tp>1 only)

**What it is:** a backed-out remainder (tp2 ttft.new 18.5 − GEMM/2 12.65) from an
instrumentation-inconsistent pair (tp2 multiprocess vs tp1 in-process); physics puts the NVLink
all-reduce at ~1–3 ms/1k, so the term plausibly absorbs ~2.5 ms/1k of multiprocess ZMQ-IPC host
overhead under a comm label.

**Pre-registered step (needs 2 free H100 GPUs — check the host first):**
1. Like-for-like pair: run `serving_stage_split.py` (api_server, multiprocess, stats ON — the
   SAME stack both times) at tp1 and tp2 on the same model. Per-tp decomposition:
   `frontend.new`, `prefill_span.new`. The tp2 comm+overhead term = `prefill_span.new(tp2) −
   prefill_span.new(tp1)/2`, with the host frontend separated out by construction (the original
   conflation).
2. Optional physics cross-check: NCCL all-reduce microbench at the relevant message sizes
   (hidden-size × chunk) → expected ms/1k band; the measured term should land in it once the
   host share is removed.
3. Wire the measured term; gate: tp1 **byte-identical** (binding), H100x2 ≤ baseline + 0.5
   (advisory-binding hybrid: this term exists only for tp≥2, so H100x2 is the target — if the
   honest measurement regresses H100x2 >0.5, keep 0.00585 + label, and record that the old
   remainder was compensating for the tp2 sub-linearity bug (the deferred spec), not comm).

## Order & ownership

R2 (offline, ~1h) → R1 Phase A (offline) → R1 Phase B/C + G3 together (one H100 session, both
need the live server; GPU check first; sequential model loads — never concurrent downloads).
Each item lands as its own commit with its own gate record; none blocks PR #74 (this plan
executes on top of it).

## Out of scope (named so they aren't absorbed silently)

S10 (prefill floor split — wants the same serving-step telemetry; fold into R1 Phase B only if
free), G7 (A100 util placeholders — separate planned A100 campaign), the eviction-cluster
structural items S7–S9 (trace-validation campaign, different tooling), tp2 decode sub-linearity
(separate spec).

## Execution record

**Item 1 (R2) — EXECUTED 2026-06-10: ADOPTED.** Builder emits the live-sum constants as the
artifact's primary `constants` block (benchmark sum retired to `benchmark_sum_reference`;
regenerates byte-identically). Gap decomposition completed pre-gate: benchmark fit > probe on all
three regression parameters (floor 22.5/16.03, new 31.0/29.4, cached 6.103/5.887) → real
workload-dependent host cost is the leading explanation; refinement filed (re-probe with replayed
benchmark prompts). Gate (replay-ON): H100 TTFT +0.06 / E2EL −0.10, A100 +0.16/+0.10, TPOT
byte-identical, H100x2 advisory TTFT 33.06→31.76 / E2EL 25.04→24.01 → **PASS**, adopted
(SHARED 3.0824e-3 / PERREQ 2.8048e-3). Note: the pre-gate prediction (workload-gap → rejection)
was WRONG — the gap is real but small enough that the honest value holds tp1 within noise and
helps tp2. 187 tests + 12 subtests green.

**Items 2 (R1/S6) and 3 (G3):** _pending — Item 2 Phase A is offline and next; Phases B/C and
Item 3 need the H100 host (GPU check first)._
