"""Forward, closed-loop, event-driven multi-turn TTFT **queue** simulator with
**session-persistent KV + RECOMPUTE preemption** (vLLM v1 model).

TTFT is not a quasi-static per-turn quantity like TPOT — it is the wall-clock **queue
wait** a request sees before its first token (client_queue_wait ~= 0, so all server-side
queueing is folded into the measured ``ttft_ms``). The per-turn TTFT is a
*saturate-ramp-RECOVER* curve that emerges from the cohort's trajectory.

THE CLIMB MECHANISM (measured + traced against /root/vllm v1 scheduler + the L4
engine-trace oracles, 2026-06-10): a multi-turn session's KV **persists across its turns**
(prefix reuse), growing each turn. Under load the cohort's cumulative KV vastly exceeds
the pool, so the engine's free-queue recycling **trims** resident prefixes from their
TAIL, LRU-OLDEST first (vllm/v1/core/block_pool.py: tail-first frees + popleft eviction;
whole-prefix loss only emerges when the herd working set exceeds the whole pool). A
trimmed session's next turn must **re-prefill the lost tail of its context**, not just the
new tokens. That re-prefill congests the chunked-prefill token budget and
head-of-line-blocks new arrivals' first token. Because cached context grows every turn,
the re-prefill cost compounds — TTFT climbs unboundedly for full-staying cohorts
(swebench/terminal, flat survival) and saturates/recovers as the cohort drains (osworld).
A turn whose session KV is still **resident** is a cache HIT and prefills only its new
tokens (cheap) — so the hit/miss fraction self-adjusts to KV pressure and sets the
magnitude.

MODEL (fit-free — NO new fitted constants):

* Cohort: ``C = round(concurrency)`` sessions; each session's turn-count drawn FORWARD from
  the profile turn-count survival histogram (``ramp_tpot.forward_survival`` / ``PROFILE_DIST``,
  the REALIZED success-filtered distribution) DETERMINISTICALLY by quantile ``q_k=(k+0.5)/C``
  (reproducible; no RNG, no wall-clock reads). Each session ALSO gets a measured per-session
  context-size SCALE (``ramp_tpot.context_scale_quantiles``) applied to the median trajectory,
  so the cohort's KV working set has the real SPREAD — small sessions stay cache-resident
  (hits) while the large minority is evicted, keeping the MEDIAN session a hit near the pool
  cliff (the osworld saturate-RECOVER). Survival + scale are measured WORKLOAD properties.
* Shared GPU continuous-batching steps; admission gated by KV blocks
  (``PrefixLRUCache(available_kv_blocks=27250, 16)``) + ``max_num_seqs`` + ``MAX_NUM_BATCHED_TOKENS``
  (vLLM serving defaults, documented config — not MAPE knobs).
* **Block-level prefix cache (``PrefixLRUCache``) with ENGINE-FAITHFUL two-tier eviction
  (re-derived 2026-06-10 against the vLLM v1 engine-trace oracles — full evidence in
  profiling/docs/defit_log_entries/L4-queue.md).** A session's cached PREFIX persists across
  turns AND across eviction. Tier 1 reclaims FREE residents (departed/dead sessions' blocks —
  the LRU buffer that shields active sessions). Tier 2, only under GENUINE over-subscription
  (the cohort KV exceeds the pool so tier 1 can't satisfy admission), trims idle herd
  residents' prefixes PARTIALLY in global-recency LRU order (oldest-touched first). That is
  the real engine's semantics: a BlockPool replica (block-hash prefix cache, single LRU free
  queue, tail-first frees, popleft eviction) reproduces the traces' live computed_tokens
  EXACTLY on 92.7-100% of lookups; flipping ONLY the order to MRU-newest degrades exactness
  to 80.9-93.3% and grows absolute token error 3.2-26x. In all four archived serving traces
  100% of turn>0 prefix losses are PARTIAL (tail-trim, head survives; zero whole-session
  misses in 602 lossy lookups) — the retired tier-2 whole-session MRU preemption (audit-v2
  S7) was a victim-selection primitive the engine does not have. In-flight KV (a req
  prefilling/decoding THIS step) is never evicted. Without tier 2 the sim DEADLOCKS at high
  concurrency (whole herd protected → no admission → silent fallback to the static formula).
  A turn's hit/miss is FROZEN at barrier release (``resident_at_barrier``) — a COMPENSATING
  RULE, RETAINED ON THE GATE (audit-v2 S8, adjudicated 2026-06-10): the engine computes hits
  LIVE at scheduling (erosion is real even within one scheduling step), and the freeze
  over-credits 20.2-85.4% of its frozen prefix credit under pressure (frozen rule variants
  under-count re-prefill tokens 42-54% vs engine truth; live+lru+partial lands within
  1.3-12.6%). UNFREEZING WAS BUILT AND GATE-REJECTED (2026-06-10, together with the S7
  partial-LRU landing): H100 ttft_cell 18.13→21.60 (+3.47), H100x2 advisory 29.02→33.91,
  while A100 IMPROVED (TTFT −1.22, E2EL −1.15) and TPOT stayed byte-identical — i.e. the
  freeze is compensating an over-amplification of re-prefill volume into TTFT that lives
  OUTSIDE the eviction cluster (volume is now trace-faithful; the overshoot is the
  pricing/queue-amplifier interaction), so per the no-cherry-picking precedent it stays
  until that structural successor lands. Light over-subscription → cheap hits → TTFT
  RECOVERS; heavy → deep re-prefills → the PEAK.
* Barrier round-robin (matches the harness ``run_multi_turn_benchmark``: all sessions' turn-N
  requests dispatched together, ``asyncio.gather`` between turns): turn ``t``'s ENTIRE herd of
  surviving sessions arrives at the SAME epoch; turn ``t+1`` is released only after EVERY
  turn-``t`` request departs. Per-turn TTFT is the queue wait of C contemporaneous arrivals.
  HIT (session resident, KV covers cached) → prefill only ``new``; MISS → prefill ``cached+new``.
* Per-step wall-time from MEASURED kernels: decode = ``decode_step_ms(batch, ctx)``; one fused
  prefill pass = ``cached_prefill_step_ms`` (roofline above the grid edge). ``TTFT[turn] =
  first_token_epoch - arrival_epoch``; aggregate per (profile, concurrency, turn_index) -> MEDIAN.

Output is the ADDITIVE column ``ttft_pred_qsim`` (+ ``e2el_pred_qsim``). The ``ttft_pred`` /
``tpot_pred`` / ``tpot_err`` / ``tpot_pred_kernel`` columns are never repointed (M0 + kernel
headline stay byte-identical).
"""

from __future__ import annotations

import heapq
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulator.cached_prefill_lookup import cached_prefill_step_ms
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator.ramp_tpot import (
    PROFILE_DIST, _gpu_slug, context_scale_quantiles, survival_for, trajectory_pool,
)
from simulator.ttft_predict import _prefill_per_token_ms, predict_turn_ttft

__all__ = ["predict_cell_ttft_qsim", "predict_cell_e2el_qsim", "PROFILE_DIST", "QSimSchedConfig"]

# --- vLLM serving defaults (documented runtime config, NOT fitted) ------------
# Resolved by vLLM EngineArgs for H100 + OPENAI_API_SERVER (arg_utils._set_default_args,
# device_memory>=70GiB & not-A100): max_num_batched_tokens=8192, max_num_seqs=1024. The
# benchmark launched with these unset (server metadata: both null) so these resolved
# defaults are what actually ran.
MAX_NUM_SEQS = 1024         # vLLM H100 OPENAI_API_SERVER resolved default (running-set cap)
MAX_NUM_BATCHED_TOKENS = 8192  # vLLM H100 OPENAI_API_SERVER resolved default (per-step token budget)
# vLLM v1 caps EACH prefill (fresh OR resumed RECOMPUTE re-prefill) at
# ``long_prefill_token_threshold`` tokens per step when chunked prefill is on; with the
# threshold unset, SchedulerConfig sets it to ``int(max_model_len * 0.04)``
# (vllm/config/scheduler.py). The benchmark ran max_model_len=32768 (recorded in server
# metadata) -> 1310. So many prefills advance CONCURRENTLY (~budget/threshold of them),
# each by a bounded chunk, rather than one big re-prefill monopolizing the budget. This is
# the de-serialization that keeps a long re-prefill from head-of-line-blocking cheap turns.
# Config-derived (max_model_len x vLLM's 0.04), NOT a fitted constant.
MAX_MODEL_LEN = 32768
LONG_PREFILL_TOKEN_THRESHOLD = int(MAX_MODEL_LEN * 0.04)  # = 1310


@dataclass(frozen=True)
class QSimSchedConfig:
    """Per-deployment vLLM scheduler truth for the queue sim's ADMISSION arithmetic
    (``_schedule``/``_price_step`` token budget, per-request chunk cap, running-set cap).

    The module constants above are the H100 OPENAI_API_SERVER resolved defaults; they are
    NOT engine truth for <70 GiB consumer devices: vllm 0.19.0 (g2a69949bd, read off the
    2080ti host install 2026-06-11) ``vllm/engine/arg_utils.py get_batch_defaults``:
    ``device_memory < 70 GiB (or "a100" in device_name) -> OPENAI_API_SERVER:
    max_num_batched_tokens=2048, max_num_seqs=256``. The kernel-TPOT side already prices
    per-deployment (``RooflineParams.max_num_batched_tokens``); this struct brings the
    TTFT sim to the same per-config truth. Threaded by the emitter
    (``build_simulator_rows``) ONLY for deployments whose manifest pins
    ``max_model_len``/``max_num_seqs`` (verified GT server metadata + resolved engine
    defaults); every ``None`` field — and a ``None`` sched — resolves to the module
    constants (BYTE-IDENTICAL default; the ``prefill_floor_ms`` threading precedent).
    ``long_prefill_token_threshold`` keeps the module's established
    ``int(max_model_len*0.04)`` rule with the config's OWN max_model_len (engine-source
    caveat recorded in defit_log_entries/L10-tp1sub20.md round 2: upstream the 0.04 rule
    is conditioned on ``max_num_partial_prefills>1``; the sim's adoption of it is a
    pre-existing gated structural choice, not relitigated here). NOT fitted constants —
    engine-config truth with verbatim citations in the deployment manifest. NOTE: the
    ``_prefill_gemm_per_tok_loaded`` util-ramp endpoints stay on the module constants by
    design — that ramp is a retained compensating fit inside the prefill-law pricing
    stack (audit-v2 R1/S6), part of the consumer-prefill-law successor, NOT scheduler
    admission arithmetic."""
    max_num_batched_tokens: int | None = None
    long_prefill_token_threshold: int | None = None
    max_num_seqs: int | None = None

# --- PREFILL COST: measured-serving anchors + pipeline FA3 kernel -------------
# TTFT prefill cost has three measured parts (the H100 HIT-vs-MISS profile settled the
# physics: a cache HIT SKIPS the GPU prefill of cached tokens, so the per-cached-token cost
# is HOST work — re-tokenize/hash the re-sent conversation — NOT a GPU kernel; the GPU
# prefills only the new/re-prefilled tokens):
#
# 1. NEW (serving per-(re)prefilled-token, 0.0310 ms/tok measured at c1): SPLIT into a DERIVED,
#    tensor-parallel-aware GEMM roofline (``_prefill_gemm_per_tok`` = 2·(params/tp)/tok at
#    util_flops=0.65 → 0.02498 ms/tok on tp1) PLUS a small per-new-token HOST serving-stack term
#    (0.005745 ms/tok). DE-FIT 2026-06-05: this term was a backed-out remainder (fitted 0.0310 −
#    roofline) mis-labelled "off-GPU framework dispatch"; the c1 live-server stage-split microbench
#    (serving_stage_split.py → profile_data/results/serving_stage_split_H100.csv) MEASURED it directly
#    as frontend.new = tokenize + chat-template/parse + ZMQ-IPC per new token = 5.745 ms/1k. It is NOT
#    GPU framework dispatch: the GPU forward window is roofline-clean (prefill_span.new 22.7 ≈ the
#    util-ramped GEMM, no above-roofline excess). The GEMM part is fit-free + tp-scales; this host term
#    does NOT shard with tp. See profiling/docs/prefill_stage_split_results.md + prefill_law_defit_trace.md.
# 2. FA3 (pipeline attention kernel, 8.31e-7 ms/token^2): from fa3_prefill_H100.csv
#    (FA3(8192)=27.9ms / (8192²/2)). Adds the SUPER-LINEAR attention growth — negligible for a
#    HIT (Q=new small), the quadratic re-encode for a MISS. Extra physical grounding at ~no
#    accuracy cost (the serving re-prefill is ~linear; FA3 is small vs chunked+host overhead).
# 3. HOST (re-tokenize the re-sent cached context, 0.006103 ms/1k total): the dominant HIT
#    cost. The SUM is measured; the shared/per-request PARTITION is a within-measured-band
#    [0.40,0.54] engineering choice (50/50; measured point estimate 0.5236 was gate-rejected —
#    see the PREFILL_HOST_* block below).
# All three RATES are MEASURED (c1 + controlled serving sweeps + the pipeline FA3 grid) — held out
# from the multi-turn data we report; the HOST shared/per-request partition alone is a
# within-band choice, not a measurement.
PREFILL_FLOOR_MS = 26.0                           # DE-FITTED 2026-06-03: measured min pure-prefill TTFT (c1 turn-0, cached~=0) = 26.07 ms across the synth profiles (chat-singleturn new=0/cached=0 min 27.4). Replaces the fitted c1 regression intercept 22.5 — the linear law extrapolated ~4 ms BELOW the real floor. Gate: TTFT 33.01->32.89% (improves; the measured anchor is consistent with the data). See profiling/docs/prefill_law_defit_trace.md. NOTE: 26.0 is the H100-TP1 floor; it is a FALLBACK only — the per-config measured floor (below) supersedes it when present (a single tp1 constant wrongly imposed 26 on tp2/tp4, whose real floors are lower: H100x2=14, the dominant tp2 low-conc over-prediction).

# Per-config MEASURED prefill floor (ms), keyed by gpu_key slug — the SAME measured-anchor method
# as the 26.0 above ("min pure-prefill TTFT, c1"), computed PER deployment so tp2/tp4 stop
# inheriting the tp1 floor. Generated by ``profiling/process/build_prefill_floor.py``; resolved
# below by ``_prefill_floor_for(gpu_key)`` with 26.0 as the fallback (gpu_key=None or no entry ->
# byte-identical to the legacy constant). Fit-free (regenerable like the decode grid / ceiling).
_PREFILL_FLOOR_JSON = Path(__file__).resolve().parents[1] / "profile_data/kernels/prefill_floor_llama31_8b.json"
_PREFILL_FLOOR_CACHE: dict[str, float] | None = None


def _prefill_floor_for(gpu_key: str | None) -> float:
    """Measured per-config prefill floor (ms) for ``gpu_key``; ``PREFILL_FLOOR_MS`` (26.0) when the
    artifact is absent, the key is unknown, or ``gpu_key`` is None (default -> byte-identical)."""
    global _PREFILL_FLOOR_CACHE
    if gpu_key is None:
        return PREFILL_FLOOR_MS
    if _PREFILL_FLOOR_CACHE is None:
        try:
            raw = json.loads(_PREFILL_FLOOR_JSON.read_text())
            _PREFILL_FLOOR_CACHE = {k: float(v["floor_ms"]) for k, v in raw.items()}
        except Exception:
            _PREFILL_FLOOR_CACHE = {}
    return _PREFILL_FLOOR_CACHE.get(_gpu_slug(gpu_key), PREFILL_FLOOR_MS)
# NEW = DERIVED tp-aware GEMM roofline (``_prefill_gemm_per_tok``, below) + this per-new-token HOST
# serving-stack term. DE-FIT 2026-06-05: MEASURED = frontend.new 5.745 ms/1k from the c1 live-server
# stage-split (serving_stage_split_H100.csv), replacing the backed-out remainder (fitted 0.0310 −
# roofline = 0.00602) that was mis-labelled GPU "dispatch". Name retained for diff-minimality; it is a
# HOST term (tokenize + parse + ZMQ-IPC per new token) and does NOT shard with tp.
PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN = 0.005745   # MEASURED frontend.new (host serving-stack / new tok)
PREFILL_FA3_MS_PER_TOKEN2 = 8.31e-7             # pipeline FA3 attention kernel, ms per token^2
# MEASURED SPLIT (de-fit 2026-06-10; band live-measured 2026-06-03 via the vLLM-server concurrency sweep
# live_split_probe.py, commit 9dce1dc). The cached host cost is per-request serving-stack work (HTTP body
# parse + chat-template + tokenize + ZMQ IPC) that PARTLY amortizes across a batch: the B-sweep's
# per-added-request slope (prefill_live_split_H100.csv) vs the c1 rate (5.887 ms/1k from
# prefill_live_ttft_H100.csv — reproduces the fitted 6.103) bounds the shared fraction to [0.40, 0.54]
# (P=8000 → 0.402, P=16000 → 0.546; P=2000 excluded: a fixed ~12.5 ms/req cost misattributed as
# per-token). Shipped values = the measured POINT ESTIMATE of that band, shared fraction 0.5236 (pooled
# OLS over the band planes; regenerate: python3 -m profiling.process.build_host_split →
# profile_data/kernels/prefill_host_split_H100.json, which pins these literals via test_ttft_queue_sim).
# Gate history: an initial 2026-06-09 worktree gate REJECTED the split by +0.44pt H100 TTFT — but that
# gate ran with trajectory replay OFF (the per-GPU realized pools are gitignored and were absent from the
# fresh worktree, i.e. a non-production cohort). Re-gated 2026-06-10 under the production replay-ON
# config: H100 TTFT-cell 18.20→18.07, E2EL 11.33→11.20 (improves), A100 +0.12/+0.10 (within ±0.3),
# H100x2 advisory 34.71→33.06 → ADOPTED. Replaces the gate-tuned 50/50 (and the earlier imported 57/43;
# the offline batch-CSV's 12/88 was wrong — lacked the serving stack). SUM de-fit (R2 CLOSED
# 2026-06-10, ttft_pricing_defit_plan.md Item 1): the sum is now the LIVE regenerable measurement
# 5.8872e-3 (build_host_split's own c1 lstsq over prefill_live_ttft_H100.csv), replacing the
# benchmark-fitted 6.103e-3 (the 760d9bd c1 benchmark-regression coefficient — audit-v2 R2).
# Gate (replay-ON): H100 TTFT +0.06 / E2EL −0.10, A100 +0.16/+0.10, TPOT byte-identical, H100x2
# advisory TTFT 33.06→31.76 → PASS. Workload caveat (documented, the future refinement): the
# benchmark fit exceeded the probe on all three regression params (floor/new/cached) — real
# chat-templated prompts exercise heavier host paths than the probe's synthetic text; re-probe
# with replayed benchmark prompts to close that gap. The artifact pins these literals.
PREFILL_HOST_SHARED_MS_PER_TOKEN = 0.0030824476411757708  # MEASURED 0.5236×5.8872e-3 — amortized once per step
PREFILL_HOST_PERREQ_MS_PER_TOKEN = 0.002804790423340364   # MEASURED 0.4764×5.8872e-3 — per request, summed

# COMPENSATING FIT, RETAINED ON THE GATE (audit-v2 R1/S6; measurement built and adoption
# REJECTED 2026-06-10 — ttft_pricing_defit_plan.md Item 2). The true per-step prefill-GEMM
# utilization IS measured: prefill_util_sweep.py (h100 GPU 6, CUDA events at exact full-budget
# chunk sizes, zero-prefix GEMM intercepts via OLS against the sim's own FA3 regressor — the
# slopes independently re-measure FA3 ≈ 8.9e-7 vs production 8.31e-7) →
# profile_data/kernels/prefill_gemm_util_H100.json: util_sim 0.640 (m=512) → 0.744 (2048) →
# 0.754 (8192). It confirms util_flops≈0.65 at small m and REFUTES saturation at 1.0 (the old
# "15.5 ms/1k GT cohort" anchor was a shared-prefix double-count — De-fit log 2026-06-10).
# WIRING THE MEASURED CURVE FAILED THE GATE: H100 TTFT-cell 18.13→21.28 (+3.15), H100x2
# advisory 31.76→34.88, while A100 IMPROVED (TTFT −0.38, E2EL −1.79) and TPOT stayed
# byte-identical — i.e. the util→1.0 ramp under-prices saturated steps to compensate a
# structural error in the H100 deep-cohort queue interaction (the audit-v2 S7–S10 cluster).
# Per-config adoption would be metric cherry-picking; the ramp+cap stays until the structural
# successor (saturated-step/queue re-derivation) lands. NOT a measurement — a tuned cap.
PREFILL_GEMM_UTIL_SAT = 1.0                      # compensating-fit cap (measured plateau: 0.754)
# MEASURED like-for-like (G3 de-fit 2026-06-10, ttft_pricing_defit_plan.md Item 3): the tp1/tp2
# stage-split pair on the SAME multiprocess api_server stack (serving_stage_split.py
# --tensor-parallel-size {1,2}, h100 GPUs 6+7) → comm = prefill_span.new(tp2) − span.new(tp1)/2 =
# 14.645 − 22.733/2 = 3.279 ms/1k (build_tp_comm.py → prefill_tp_comm_H100.json, which pins this
# literal). Lands at the top of the NCCL all-reduce physics band (~1–3 ms/1k) — the retired 5.85
# was a backed-out remainder from an instrumentation-INCONSISTENT pair (tp2 multiprocess vs tp1
# in-process) that absorbed ~2.5 ms/1k of host IPC under a comm label (audit-v2 G3, confirmed).
# tp>1 only; tp1 → 0 (tp1 predictions byte-identical by construction).
PREFILL_TP_COMM_MS_PER_TOKEN = 0.003278887802709865


def _prefill_gemm_per_tok(p: RooflineParams) -> float:
    """DERIVED compute-bound prefill GEMM time per (re)prefilled token: 2·(n_params/tp) FLOPs
    per token at ``peak_flops·util_flops``, tensor-parallel sharded. The fit-free dominant part
    of the serving NEW rate — 0.02498 ms/tok on tp1, halving per added TP rank. No fitted constant."""
    tp = max(1, int(getattr(p, "tensor_parallel", 1)))
    return 2.0 * (float(p.prefill_n_params) / tp) / (p.peak_flops_per_s * p.util_flops) * 1e3


def _prefill_gemm_per_tok_loaded(p: RooflineParams, batch_tokens: float) -> float:
    """Batch-aware prefill GEMM rate: util ramps util_flops→PREFILL_GEMM_UTIL_SAT with the
    per-step batch over [long_prefill_threshold, max_num_batched_tokens], plus the
    per-extra-rank tensor-parallel all-reduce. Reduces to ``_prefill_gemm_per_tok`` at small
    batch, tp=1. PROVENANCE (audit-v2 R1/S6, 2026-06-10): the ramp endpoints are engine config
    (1310/8192, verified); the ramp SHAPE and the 1.0 cap are a COMPENSATING FIT retained on
    the gate — the measured per-step curve (prefill_gemm_util_H100.json: 0.640→0.754, plateau
    by m≈2048) was wired and gate-REJECTED (H100 TTFT +3.15pt; A100 improved; see the De-fit
    log). TP_COMM is a backed-out remainder (audit-v2 G3, pending its like-for-like
    measurement)."""
    tp = max(1, int(getattr(p, "tensor_parallel", 1)))
    lo, hi = float(LONG_PREFILL_TOKEN_THRESHOLD), float(MAX_NUM_BATCHED_TOKENS)
    frac = 0.0 if hi <= lo else min(1.0, max(0.0, (batch_tokens - lo) / (hi - lo)))
    util = p.util_flops + (PREFILL_GEMM_UTIL_SAT - p.util_flops) * frac
    gemm = 2.0 * (float(p.prefill_n_params) / tp) / (p.peak_flops_per_s * util) * 1e3
    # Comm term: the per-config MEASURED total (RooflineParams.prefill_tp_comm_ms_per_token, G3
    # like-for-like at the config's OWN tp degree — L11 de-fit) when the deployment pins it; else
    # the tp2-measured comm scaled to this tp by the RING all-reduce law 2·(tp−1)/tp (the per-rank
    # volume a ring all-reduce moves): byte-identical at tp1 (→0) and tp2 (2·1/2 = 1, the measured
    # anchor), and SUB-linear above — tp4 = 1.5× the tp2 rate, not the old linear (tp−1) = 3×. The
    # linear extrapolation over-charged tp>2 prefill comm (ablation: a100 8B tp4 c320 +5949 ms, ~70%
    # of it this term; h100 8B tp4 comm_off under-shot → the rate is also GPU-specific). GPU
    # follow-up: measure comm directly at tp4 per GPU (build_tp_comm.py stage-split) and pin
    # prefill_tp_comm_ms_per_token (NVLink vs PCIe differ); this physics law is the zero-GPU interim.
    comm = getattr(p, "prefill_tp_comm_ms_per_token", None)
    if comm is None:
        comm = PREFILL_TP_COMM_MS_PER_TOKEN * (2.0 * (tp - 1) / tp)
    # Batched-drain amortization (L13 S7, 2026-06-12): the c1 stage-split comm rate is measured at
    # single-request chunks; the engine's BARRIER DRAIN (many requests' chunks batched per step)
    # amortizes the PCIe all-reduce well below it (S7: replayed GT ladder, engine-side computed
    # tokens from /metrics vs measured drain makespans). A deployment that pins the MEASURED
    # saturated rate interpolates c1 -> saturated over the SAME chunk-fill fraction as the util
    # ramp above (small/single chunks keep the exact c1 rate -> the c1 stage-split anchors are
    # untouched). None -> flat c1 comm, BYTE-IDENTICAL for every config that does not pin it.
    comm_sat = getattr(p, "prefill_tp_comm_saturated_ms_per_token", None)
    if comm_sat is not None:
        comm = comm + (comm_sat - comm) * frac
    return gemm + comm


# Event kinds; ordering is (epoch, seq, kind) — deterministic FIFO at equal epochs.
_ARRIVAL = 0
_STEP = 1
_FIRST_TOKEN = 2
_DEPART = 3

# Largest new-prefill the measured cached-prefill grid covers; above it the prefill-compute
# roofline (continuous with the grid edge) prices a fused multi-session pass. No new constant.
_GRID_U_MAX = 1024.0


def _prefill_pass_ms(total_chunk_tokens: float, mean_prefix: float, params: RooflineParams) -> float:
    """Wall-time of ONE fused prefill forward pass over ``total_chunk_tokens`` tokens at
    ``mean_prefix`` cached prefix — measured cached-prefill grid up to its U edge, the
    prefill-compute roofline above it (continuous, the large-batch extrapolant)."""
    u = max(1.0, float(total_chunk_tokens))
    if u <= _GRID_U_MAX:
        return cached_prefill_step_ms(u, max(1.0, float(mean_prefix)))
    return u * _prefill_per_token_ms(params) + params.scheduler_overhead_ms_per_step


# ------------------------------------------------------- block-level prefix cache


class PrefixLRUCache:
    """Block-level KV prefix cache with global LRU eviction (vLLM v1 BlockPool + APC,
    modeled at session granularity).

    Each session's cached PREFIX (a number of contiguous blocks from the start of its
    context) persists across turns AND across eviction. Making room for a new allocation
    reuses the globally-LRU-OLDEST cached blocks, trimming a victim session's prefix from its
    TAIL (the most-recent, least-shared end). A session reclaims whatever prefix SURVIVED on
    its next turn (a cache HIT) and re-prefills only the trimmed tail — never its whole
    context unless every block was physically reused.

    This is what makes the measured saturate-ramp-RECOVER emerge with the right magnitude: a
    MILDLY over-subscribed cohort (the drained tail) reuses few blocks per turn, so sessions
    keep almost all their prefix → cheap hits → TTFT recovers; a HEAVILY over-subscribed one
    (the peak) churns the whole pool → deep re-prefills → TTFT peaks. Partial tail-trim in
    LRU-oldest order is the TRACE-VALIDATED engine behaviour on BOTH tiers (S7 re-derivation
    2026-06-10: 100% of turn>0 prefix losses in the archived serving traces are partial;
    LRU-oldest replica exact on 92.7-100% of live lookups, MRU falsified at 3.2-26x the token
    error — see defit_log_entries/L4-queue.md). Whole-prefix loss still EMERGES under extreme
    pressure (herd working set > whole pool recycles every block before reuse), it is just no
    longer a victim-selection primitive. Capacity and block size are vLLM config
    (``available_kv_blocks`` / ``cache_block_size``); NO fitted constants."""

    def __init__(self, capacity_blocks: int, block_size: int = 16) -> None:
        self.capacity = int(capacity_blocks)
        self.block_size = int(block_size)
        self.cached: dict[int, int] = {}   # session_id -> resident prefix blocks
        self.recency: dict[int, int] = {}  # session_id -> last-touch tick (LRU key)
        self._tick = 0
        self.evictions = 0
        # Sticky over-subscription marker (L13 S8 round 3): flips True the first time
        # tier-2 eviction (PARTIAL trims of idle HERD residents — genuine
        # over-subscription) is reached, and stays True (the engine's free queue then
        # recycles blocks continuously for the rest of the run). Consumed ONLY by the
        # duplicate-session credit (``qsim_duplicate_session_fraction``): a duplicate's
        # hit lives in its TWIN's blocks, and once the pool recycles, the twin's content
        # no longer survives between turns (S8: tb tp4 cell computed/bench_new 0.40-0.52
        # at no-eviction c10-40 but 3.94 at c80 where the cohort exceeds the pool — the
        # dedup DIES under eviction). No pricing effect for unpinned configs.
        self.pressure_seen = False

    def tokens_to_blocks(self, num_tokens: float) -> int:
        return int(math.ceil(max(0.0, float(num_tokens)) / self.block_size))

    def cached_blocks(self, sid: int) -> int:
        return self.cached.get(sid, 0)

    def used(self) -> int:
        return sum(self.cached.values())

    def free(self) -> int:
        return self.capacity - self.used()

    def touch(self, sid: int) -> None:
        """Mark ``sid`` most-recently-used (so it is evicted LAST)."""
        self.recency[sid] = self._tick
        self._tick += 1

    def _trim_tail(self, sids: list[int], need: int, whole: bool = False) -> None:
        """Free blocks by evicting ``sids`` (in the given order) until ``need`` blocks are free.
        ``whole=False`` (the production path) trims only the marginal tail — the engine's
        block-granular free-queue recycling (S7: 100% of natural-load prefix losses in the
        archived serving traces are PARTIAL tail-trims). ``whole=True`` evicts each victim's
        ENTIRE resident prefix; it is RETIRED from production (trace-falsified: zero
        whole-session misses in 602 lossy lookups) and kept only as the adjudication tool's
        counterfactual seam (compare_eviction_semantics.py)."""
        for v in sids:
            if self.free() >= need:
                break
            trim = self.cached[v] if whole else min(self.cached[v], need - self.free())
            self.cached[v] -= trim
            self.evictions += 1
            if self.cached[v] <= 0:
                del self.cached[v]

    def _evict(
        self, need: int, hard_protect: set[int], soft_protect: set[int], policy: str = "lru"
    ) -> bool:
        """Free ``need`` physical blocks in two tiers:

        1. **Reclaim free residents** — sessions in NEITHER protect set (departed/dead
           sessions' residual prefix), trimmed LRU-oldest-first. This is the rotation buffer.
        2. **Trim under genuine over-subscription** — if the cohort's persistent KV fills
           the pool so tier 1 can't satisfy ``need``, PARTIALLY trim ``soft_protect`` (herd
           members not in-flight = idle resident hits-to-be) in global-recency LRU order,
           oldest-touched first. A trimmed session re-prefills its lost tail on its turn (a
           partial MISS) — the climb.

        ENGINE-FAITHFUL (S7 re-derivation 2026-06-10, defit_log_entries/L4-queue.md): vLLM v1
        evicts free-queue blocks LRU-OLDEST block-by-block (block_pool.py popleft; tail-first
        frees), so victims lose their prefix TAIL partially and oldest-touched sessions lose
        first. The trace replica validated this exactly (92.7-100% of live lookups; MRU order
        falsified at 3.2-26x the token error; 100% of natural-load prefix losses are partial).
        The retired rule here — ``'tail'`` (MRU-first) victims trimmed WHOLE — had no engine
        analog; ``policy='tail'`` is retained ONLY as the adjudication tool's falsification
        seam (compare_eviction_semantics.py counterfactuals), never the production path.

        ``hard_protect`` (a req in-flight THIS step, KV pinned) is never evicted. Returns True
        once ``need`` is free, False only if even trimming every idle resident is not enough
        (a single over-large head behind pinned in-flight KV — deferred, retried on completion)."""
        if need <= self.free():
            return True
        free_residents = sorted(
            (s for s in self.cached
             if s not in hard_protect and s not in soft_protect and self.cached[s] > 0),
            key=lambda s: (self.recency.get(s, -1), s),  # oldest first; sid tiebreak (determinism)
        )
        self._trim_tail(free_residents, need)
        if self.free() >= need:
            return True
        # tier 2: genuine over-subscription -> PARTIAL tail-trims of idle herd residents in
        # LRU-oldest order — the engine's free-queue recycling (S7: whole-session MRU
        # preemption falsified against the trace oracles; whole-prefix loss only EMERGES when
        # the herd working set exceeds the whole pool). The freed amount is exactly ``need``:
        # the engine never frees more than the allocation requires.
        self.pressure_seen = True  # tier 2 reached = genuine over-subscription (sticky; see __init__)
        soft = [
            s for s in self.cached
            if s in soft_protect and s not in hard_protect and self.cached[s] > 0
        ]
        soft.sort(key=lambda s: (self.recency.get(s, -1), s), reverse=(policy == "tail"))
        self._trim_tail(soft, need)
        return self.free() >= need

    def grow_to(
        self,
        sid: int,
        target_blocks: int,
        hard_protect: set[int],
        soft_protect: set[int],
        policy: str = "lru",
    ) -> bool:
        """Make ``sid`` resident up to ``target_blocks``, RECLAIMING its surviving prefix and
        allocating only the delta (reclaiming free residents, then partially trimming idle
        herd residents LRU-oldest under over-subscription — see ``_evict``). Touches ``sid``
        (MRU). Returns
        False (HOL block) only if the delta cannot be freed even after preemption. Context only
        grows, so a target below the current residency just keeps the larger residency."""
        cur = self.cached.get(sid, 0)
        if target_blocks <= cur:
            self.touch(sid)
            return True
        delta = target_blocks - cur
        if not self._evict(delta, hard_protect | {sid}, soft_protect - {sid}, policy):
            return False
        self.cached[sid] = target_blocks
        self.touch(sid)
        return True


# ---------------------------------------------------------------- session model


@dataclass
class TurnSpec:
    turn_index: int
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float


@dataclass
class Session:
    session_id: int
    turn_count: int
    turns: list[TurnSpec]
    next_turn_idx: int = 0
    # KV residency now lives in the shared PrefixLRUCache (keyed by session_id); eviction
    # protection for the current herd lives in _ServerState.herd_pending.


# ------------------------------------------------------------------ in-flight reqs


@dataclass
class _Req:
    """One (session, turn) request as it moves waiting -> prefilling -> running."""

    rid: int
    session_id: int
    turn_index: int
    arrival_epoch: float
    cached: float
    new_prefill: float
    output: float
    remaining_prefill: float
    output_left: int
    kv_tokens: float           # resident KV after this turn's prefill (cached+new), grows with decode
    is_miss: bool = False      # cache miss (session was evicted) -> re-prefilled full context
    resident_prefix: float = 0.0  # cached tokens that are a HIT (attended, not re-prefilled); set at admission
    prefill_total: float = 0.0    # total tokens to (re-)prefill this turn; set at admission (for chunk fraction)
    prev_output: float = 0.0      # previous turn's output tokens (the response inside this turn's
                                  # ``new_prefill``) — the S7 response-resident credit basis; 0 at turn 0


@dataclass
class _ServerState:
    params: RooflineParams
    cache: PrefixLRUCache
    sessions: list[Session]
    clock: float = 0.0
    seq: int = 0  # monotone tiebreak for the event heap
    heap: list[tuple[float, int, int, Any]] = field(default_factory=list)
    waiting: list[_Req] = field(default_factory=list)        # FIFO
    prefilling: dict[int, _Req] = field(default_factory=dict)
    running: dict[int, _Req] = field(default_factory=dict)
    herd_pending: set[int] = field(default_factory=set)      # current-herd sessions not yet departed (evict-protected)
    results: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    step_scheduled: bool = False
    # --- barrier round-robin state (matches the benchmark harness, see _release_herd) ---
    current_turn: int = 0       # turn_index of the herd currently in flight
    # Tier-2 victim order: 'lru' (oldest-touched first, partial trims) is the ENGINE-FAITHFUL
    # production value (S7 re-derivation 2026-06-10); 'tail' (MRU-first) is retained only as
    # the adjudication tool's falsification seam.
    herd_remaining: int = 0     # requests of the current herd not yet departed; 0 -> barrier
    preempt_policy: str = "lru"
    # sid -> resident blocks at herd release: the S8 hit/miss FREEZE — a compensating rule
    # RETAINED ON THE GATE (engine truth is LIVE lookups; unfreeze gate-rejected 2026-06-10,
    # H100 +3.47pt — see the module docstring and defit_log_entries/L4-queue.md).
    resident_at_barrier: dict[int, int] = field(default_factory=dict)
    # --- shared cross-session APC prefix (prefix_aware_synthetic workloads) ---
    # ``shared_prefix_tokens`` (S) is the profile-constant prefix that EVERY session's prompt
    # carries at the FRONT of its context (system-prompt-level, generated from a per-PROFILE label
    # -> identical token blocks across sessions). vLLM's APC dedups it: the FIRST session to prefill
    # it pays once, all others HIT. The benchmark's per-session cache estimate records cached=0 at
    # turn-0 (it tracks only intra-session history), so without this the sim would re-prefill S for
    # ALL C sessions (C-fold over-count -> the turn-0 over-prediction). ``shared_primed`` flips True
    # on the first SUCCESSFUL admission (global, monotone) so exactly one session pays S; thereafter
    # the shared prefix is MRU-resident (touched every turn by every session -> never the LRU victim)
    # and credited to all peers. S=0 -> feature off, behaviour byte-identical. NOT a fitted constant
    # (a per-cell DATA input read from request_metadata.shared_prefix_actual_tokens).
    shared_prefix_tokens: float = 0.0
    shared_primed: bool = False
    # First SUCCESSFUL admission of the run (global, monotone) — the duplicate-session
    # credit's priming gate (L13 S8): a duplicate's hit lives in its TWIN's blocks, so an
    # EMPTY cache cannot be hit; after any admission the twin content is this-run MRU
    # resident. Independent of ``shared_primed`` (which is the S>0 APC-prefix feature).
    dup_primed: bool = False
    # Per-config measured prefill floor (ms); resolved from gpu_key at predict_cell_ttft_qsim and
    # threaded here. Default = the legacy H100-tp1 constant (gpu_key=None -> byte-identical).
    prefill_floor_ms: float = PREFILL_FLOOR_MS
    # Per-config scheduler truth (QSimSchedConfig), resolved at _run_sim; defaults = the
    # module-level H100 constants (sched=None -> byte-identical). See QSimSchedConfig.
    sched_max_num_batched_tokens: float = float(MAX_NUM_BATCHED_TOKENS)
    sched_long_prefill_threshold: float = float(LONG_PREFILL_TOKEN_THRESHOLD)
    sched_max_num_seqs: int = MAX_NUM_SEQS

    def push(self, epoch: float, kind: int, payload: Any) -> None:
        heapq.heappush(self.heap, (epoch, self.seq, kind, payload))
        self.seq += 1


def _encode_rid(session_id: int, turn_index: int) -> int:
    """Stable per-(session, turn) request id. 4096 turns max per session — ample."""
    return session_id * 4096 + turn_index


# ----------------------------------------------------------------- cohort builder


def _draw_turn_count(survival: list[float], quantile: float) -> int:
    """Inverse-survival, deterministic. ``survival[t]=S(t)`` = fraction alive AT turn t."""
    if not survival:
        return 1
    reached = 0
    for s in survival:
        if s >= quantile:
            reached += 1
        else:
            break
    return max(1, reached)


def _cohort_from_pool(pool: list, c: int) -> list[Session]:
    """Trajectory-REPLAY cohort: build ``c`` sessions by deterministically cycling the per-GPU pool of
    REAL session trajectories (each a list of ``[cached, new, output]`` per turn). This is the JOINT
    cohort (survival + context-scale + their correlation) that reaches the oracle floor — vs the
    survival/scale marginals which lose the joint structure (feasibility 2026-06-04)."""
    n = len(pool)
    sessions: list[Session] = []
    for k in range(c):
        traj = pool[k % n]
        specs = [
            TurnSpec(
                turn_index=i,
                cached_context_tokens=float(t[0]),
                new_prefill_tokens=float(t[1]),
                output_tokens=max(1.0, float(t[2])),
            )
            for i, t in enumerate(traj)
        ]
        if specs:
            sessions.append(Session(session_id=k, turn_count=len(specs), turns=specs))
    return sessions


def _build_cohort(
    turns: list[dict[str, Any]], profile: str, concurrency: float,
    gpu_key: str | None = None,
    survival_override: list[float] | None = None,
    scale_override: list[float] | None = None,
) -> list[Session]:
    """Forward cohort. PREFERRED: trajectory REPLAY from the per-GPU real-session pool (the joint
    cohort that reaches the oracle floor). FALLBACK (no pool, or LOCO override): deterministic
    survival-quantile + context-scale marginals resolved per-(profile, concurrency, gpu) via
    ``ramp_tpot`` (pooled when ``gpu_key=None`` → byte-identical to legacy). ``survival_override``/
    ``scale_override`` (LOCO test seam) force the marginal path with caller-supplied curves."""
    c = max(1, int(round(float(concurrency))))
    # Trajectory-replay when a per-GPU pool is available and no LOCO override is forcing the marginals.
    if survival_override is None and scale_override is None:
        pool = trajectory_pool(profile, concurrency, gpu_key)
        if pool:
            return _cohort_from_pool(pool, c)
    survival = (
        survival_override if survival_override is not None
        else survival_for(profile, concurrency, gpu_key)
    )

    spec_by_idx: dict[int, TurnSpec] = {}
    max_turn_idx = 0
    for t in turns:
        ti = int(t.get("turn_index", 0))
        spec_by_idx[ti] = TurnSpec(
            turn_index=ti,
            cached_context_tokens=float(t.get("cached_context_tokens") or 0.0),
            new_prefill_tokens=float(t.get("new_prefill_tokens") or 0.0),
            output_tokens=max(1.0, float(t.get("output_tokens") or 1.0)),
        )
        max_turn_idx = max(max_turn_idx, ti)
    n_turn_slots = max_turn_idx + 1

    def spec_for(idx: int) -> TurnSpec:
        if idx in spec_by_idx:
            return spec_by_idx[idx]
        lower = [i for i in spec_by_idx if i <= idx]
        if lower:
            return spec_by_idx[max(lower)]
        return spec_by_idx[min(spec_by_idx)] if spec_by_idx else TurnSpec(idx, 0.0, 1.0, 1.0)

    # Per-session context-size SCALE (measured workload spread): each cohort session runs
    # systematically larger/smaller contexts than the median, so the KV working set has the
    # real spread — small sessions stay resident (hits) while the large minority is evicted,
    # keeping the MEDIAN session a hit near the pool cliff (the osworld saturate-RECOVER). The
    # per-(conc,turn) MEDIAN trajectory is preserved; only the per-session spread is added.
    scale_q = (
        scale_override if scale_override is not None
        else context_scale_quantiles(profile, concurrency, gpu_key)
    )

    def session_scale(qk: float) -> float:
        if not scale_q:
            return 1.0
        nq = len(scale_q)
        return scale_q[min(nq - 1, max(0, int(round(qk * (nq - 1)))))]

    def scaled_spec(idx: int, f: float) -> TurnSpec:
        s = spec_for(idx)
        if f == 1.0:
            return s
        return TurnSpec(
            turn_index=s.turn_index,
            cached_context_tokens=s.cached_context_tokens * f,
            new_prefill_tokens=s.new_prefill_tokens * f,
            output_tokens=s.output_tokens,
        )

    sessions: list[Session] = []
    for k in range(c):
        q = (k + 0.5) / c
        if survival:
            tc = min(_draw_turn_count(survival, q), n_turn_slots) if n_turn_slots > 0 else _draw_turn_count(survival, q)
        else:
            tc = n_turn_slots if n_turn_slots > 0 else 1
        tc = max(1, tc)
        f = session_scale(q)
        sessions.append(Session(session_id=k, turn_count=tc, turns=[scaled_spec(i, f) for i in range(tc)]))
    return sessions


# ------------------------------------------------------------------- the sim core


def _running_ctx_mean(state: _ServerState) -> float:
    if not state.running:
        return 1.0
    return statistics.fmean(r.kv_tokens for r in state.running.values())


def _schedule(state: _ServerState) -> None:
    """Admission for the current herd. The hit/miss decision is made from the block-level
    prefix cache: the session's SURVIVING resident prefix (per the barrier snapshot — the
    retained S8 freeze, see below) is a HIT (re-prefill only the evicted tail + new tokens);
    a fully-evicted session is a full MISS. Reserve the full context blocks, RECLAIMING the
    surviving prefix and evicting free residents first (dead sessions or sessions that
    already completed THIS round), then — only under genuine over-subscription — partially
    trimming idle herd residents LRU-oldest (see ``_evict``). A head that can't get blocks
    yet is DEFERRED (skipped, stays waiting) and retried once a completion frees blocks — so
    hits run while misses wait, and the resident set ROTATES (the saturate-ramp-RECOVER).
    Also gated by the per-step token budget + max_seqs."""
    if not state.waiting:
        return
    decode_slots = len(state.running)
    budget = state.sched_max_num_batched_tokens - decode_slots
    for r in state.prefilling.values():
        budget -= min(r.remaining_prefill, state.sched_long_prefill_threshold)
    cache = state.cache
    # Hard-protect = sessions with a req in-flight THIS step (KV pinned). Grows as we admit, so
    # a head admitted earlier in this pass is never preempted to make room for a later one.
    in_flight = {r.session_id for r in state.prefilling.values()}
    in_flight |= {r.session_id for r in state.running.values()}
    deferred: list[_Req] = []
    for head in state.waiting:
        if budget <= 0 or len(state.running) + len(state.prefilling) >= state.sched_max_num_seqs:
            deferred.append(head)
            continue
        sid = head.session_id
        # Block-level prefix-cache hit: the surviving resident prefix covers up to
        # ``cached_blocks * block_size`` tokens of this turn's cached context; only the
        # EVICTED tail of that prefix plus the new tokens must be (re-)prefilled. Resident
        # blocks are read from the BARRIER SNAPSHOT (frozen at herd release), not live —
        # a COMPENSATING RULE, RETAINED ON THE GATE (audit-v2 S8, adjudicated against the
        # engine-trace oracles 2026-06-10, defit_log_entries/L4-queue.md): vLLM computes
        # hits LIVE at scheduling, and live erosion is real even WITHIN one scheduling step
        # (trace pool060 step 287: five same-step lookups descend 353/313/273/233/193 blocks
        # as each peer's allocation eats the next prefix); the freeze over-credits
        # 20.2-85.4% of its frozen prefix credit under pressure (frozen variants
        # under-count re-prefill 42-54% vs engine truth). The LIVE lookup was implemented
        # and GATE-REJECTED (H100 ttft_cell 18.13→21.60, H100x2 29.02→33.91, A100 IMPROVED
        # −1.22): with re-prefill VOLUME now trace-faithful, the freeze is compensating the
        # volume→TTFT over-amplification elsewhere (pricing/queue interaction), not
        # eviction semantics — it stays until that structural successor lands (the util-cap
        # precedent), NOT because the engine freezes anything.
        snap_blocks = state.resident_at_barrier.get(sid, cache.cached_blocks(sid))
        # Resident-credit basis (L13 S7 engine truth, 2026-06-12): the engine's prefix cache
        # retains the session's DECODED output blocks, and the next turn's re-tokenized prompt
        # MATCHES those generated tokens up to a per-session divergence point — so a resident
        # turn re-prefills only the un-matched remainder of its context, NOT the benchmark's
        # prompt-basis ``new_prefill`` view (cache_estimate_source='previous_prompt_tokens'
        # excludes output blocks). The S7 /metrics counters measured aggregate hits/req =
        # prev-prompt + rho·prev-output with a BIMODAL per-request split (the measured per-turn
        # TTFT distribution: a ~rho majority of sessions hit their whole previous response, the
        # rest re-prefill it). ``qsim_response_resident_fraction`` (RooflineParams optional
        # field, the L11 pin mechanism) applies that as a DETERMINISTIC session quantile —
        # sessions with (sid+0.5)/C <= rho extend their credit basis by the previous turn's
        # output; the credit stays capped by the LIVE resident-block snapshot, so eviction
        # semantics are untouched. None (default) and 0.0 are the legacy prompt-basis cap,
        # BYTE-IDENTICAL for every config that does not pin it.
        rho = getattr(state.params, "qsim_response_resident_fraction", None)
        credit_basis = head.cached
        if rho is not None and head.prev_output > 0.0 and state.sessions:
            if (sid + 0.5) / len(state.sessions) <= float(rho):
                credit_basis = head.cached + min(head.prev_output, head.new_prefill)
        resident_prefix = min(credit_basis, snap_blocks * cache.block_size)
        # Shared cross-session APC prefix: once primed (some session has prefilled it), the front
        # S tokens of THIS session's context are resident too — a HIT — even at turn-0 where the
        # per-session ``cached`` is 0 and the shared block lives inside ``new_prefill``. Credited
        # via MAX with the per-session resident prefix (both describe front-of-context resident
        # tokens; a SUM would double-count at turn>=1, where the session's own prefix already
        # covers the shared block). The first session to admit this run gets shared_resident=0
        # (``shared_primed`` is still False when computed here) so it PAYS the shared prefill once.
        shared_resident = (
            min(state.shared_prefix_tokens, head.cached + head.new_prefill)
            if state.shared_prefix_tokens > 0.0 and state.shared_primed
            else 0.0
        )
        # Duplicate-session cross-session dedup (L13 S8 engine truth, 2026-06-12): the
        # trace-replay profiles draw sessions from a FINITE trace set, so cohorts contain
        # repeated traces whose ENTIRE turn content (history + injected delta + response)
        # the engine's prefix cache hits via the TWIN session's blocks — S8 /metrics
        # cell-level computed/bench_new on tb tp4: 1.02 (c1) -> 0.82 (c5) -> 0.40-0.52
        # (c10-40) while prev-output covers only ~5% of new (the rho credit above cannot
        # carry it; chat shows no surplus — rho explains chat exactly, and the two
        # quantile rules NEST: both count sessions from sid 0 up, so where rho already
        # grants full-new credit the duplicate credit adds nothing).
        # ``qsim_duplicate_session_fraction`` applies the measured fraction as a
        # DETERMINISTIC session quantile over sids >= 1 (a duplicate needs an earlier
        # twin — sid 0 can never be one, so c1 cells are untouched), gated on
        # ``dup_primed`` (first successful admission of the run, the shared-APC-prefix
        # priming pattern: the twin's blocks are this-run MRU; an empty cache cannot be
        # hit). NOT capped by the OWN-session snapshot — the hit lives in the TWIN's
        # blocks (own-snapshot capping would collapse this credit back to the rho
        # credit). None (default) and 0.0 are byte-identical legacy for unpinned configs.
        fdup = getattr(state.params, "qsim_duplicate_session_fraction", None)
        dup_resident = 0.0
        if (fdup is not None and fdup > 0.0 and sid >= 1 and state.sessions
                and state.dup_primed and not cache.pressure_seen
                and (sid + 0.5) / len(state.sessions) <= float(fdup)):
            # ``not pressure_seen``: once the pool over-subscribes (tier-2 trims), the
            # twin's blocks recycle between turns and the cross-session hit DIES (S8:
            # ratio 0.40-0.52 at no-eviction c10-40, 3.94 at c80 over-pool — measured;
            # the standalone uncapped pin was gate-FALSIFIED on exactly the eviction
            # cells, x4 e2el 24.59 -> 35.10 with tb/swe/osworld c120-320 +38..+64).
            dup_resident = head.cached + head.new_prefill
        resident_credit = max(resident_prefix, shared_resident, dup_resident)
        # Split the resident credit front-to-back: cached tokens first, then new (the shared block
        # is the front of new at turn-0). reprefill = the un-resident remainder of each.
        cached_hit = min(head.cached, resident_credit)
        new_hit = min(head.new_prefill, max(0.0, resident_credit - head.cached))
        reprefill_cached = head.cached - cached_hit
        reprefill_new = head.new_prefill - new_hit
        head.is_miss = reprefill_cached > 0.0            # MISS = own prefix was evicted (unchanged semantics)
        head.remaining_prefill = reprefill_cached + reprefill_new
        head.resident_prefix = resident_credit           # HIT prefix attended each prefill step
        head.prefill_total = max(1.0, head.remaining_prefill)  # to spread the cached-attn cost across chunks
        target_blocks = cache.tokens_to_blocks(head.kv_tokens)
        # Reserve the full context: reclaim the surviving prefix, then free residents, then
        # (only under genuine over-subscription) PARTIALLY trim idle herd residents
        # (``herd_pending`` minus in-flight) LRU-oldest — the engine's free-queue recycling
        # (S7). A trimmed session re-prefills its lost tail on its turn (a partial MISS).
        # In-flight KV is never evicted. If even that can't free the delta (one over-large
        # head behind pinned in-flight KV), DEFER and retry on a completion — never a hard
        # stall.
        if not cache.grow_to(sid, target_blocks, in_flight, state.herd_pending, state.preempt_policy):
            deferred.append(head)
            continue
        state.prefilling[head.rid] = head
        in_flight.add(sid)
        # The first session to physically admit primes the shared prefix (it just paid the
        # full prefill incl. S); thereafter every peer/turn credits it as resident. Set AFTER
        # the grow_to success so a DEFERRED head never falsely primes a block nobody prefilled.
        if state.shared_prefix_tokens > 0.0:
            state.shared_primed = True
        state.dup_primed = True
        budget -= min(head.remaining_prefill, state.sched_long_prefill_threshold)
    state.waiting = deferred


def _on_arrival(state: _ServerState, session_id: int, turn_index: int) -> None:
    sess = state.sessions[session_id]
    spec = sess.turns[turn_index]
    cached = spec.cached_context_tokens
    new = spec.new_prefill_tokens
    req = _Req(
        rid=_encode_rid(session_id, turn_index),
        session_id=session_id,
        turn_index=turn_index,
        arrival_epoch=state.clock,
        cached=cached,
        new_prefill=new,
        output=spec.output_tokens,
        remaining_prefill=new,  # provisional; _schedule sets hit/miss at admission
        output_left=max(1, int(round(spec.output_tokens))),
        kv_tokens=cached + new,
        # the previous turn's generated response (it sits at the FRONT of this turn's
        # ``new_prefill`` in the benchmark's prompt-basis accounting) — the S7
        # response-resident credit basis; 0.0 at turn 0 (no previous response).
        prev_output=(sess.turns[turn_index - 1].output_tokens if turn_index >= 1 else 0.0),
    )
    state.waiting.append(req)
    state.results[(session_id, turn_index)] = {"arrival_epoch": state.clock}
    _schedule(state)
    _ensure_step(state)


def _ensure_step(state: _ServerState) -> None:
    if state.step_scheduled:
        return
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _prefill_host_fa3_rates(p: RooflineParams) -> tuple[float, float, float]:
    """(fa3_coef, host_shared_rate, host_perreq_rate) for ``_price_step``'s prefill terms.

    Per-config MEASURED prefill HOST-cached SUM + FA3 coefficient (L11 round 2: the
    like-for-like tp1/tpN stage-split pair, ``build_stage_split_rates.py``). ``None`` — every
    config that does not pin them — returns the tp1-measured module constants UNCHANGED
    (byte-identical). A pinned host SUM keeps the production measured shared/per-request
    FRACTION (0.5236, prefill_host_split_H100.json; no tpN B-sweep exists) applied to the
    per-config SUM. Explicit ``is None`` checks so a 0.0 pin is respected."""
    fa3_coef = getattr(p, "prefill_fa3_ms_per_token2", None)
    if fa3_coef is None:
        fa3_coef = PREFILL_FA3_MS_PER_TOKEN2
    host_sum = getattr(p, "prefill_host_cached_ms_per_token", None)
    if host_sum is None:
        return fa3_coef, PREFILL_HOST_SHARED_MS_PER_TOKEN, PREFILL_HOST_PERREQ_MS_PER_TOKEN
    prod_sum = PREFILL_HOST_SHARED_MS_PER_TOKEN + PREFILL_HOST_PERREQ_MS_PER_TOKEN
    return (fa3_coef,
            host_sum * (PREFILL_HOST_SHARED_MS_PER_TOKEN / prod_sum),
            host_sum * (PREFILL_HOST_PERREQ_MS_PER_TOKEN / prod_sum))


def _price_step(state: _ServerState) -> float:
    """One mixed prefill+decode step. Decode = measured kernel. Prefill = three measured terms:
      * NEW   : serving per-(re)prefilled-token rate (linear, batched ∝ total chunk tokens),
      * FA3   : pipeline attention kernel (super-linear; per request, ∝ M·(R + M/2) where
                M = tokens this req (re-)prefills this turn, R = its resident prefix — tiny for
                a HIT, the quadratic re-encode for a MISS),
      * HOST  : re-tokenize the re-sent cached context — the dominant HIT cost; split into a
                per-step SHARED part (amortized across the batch) + a per-request part. Both
                FA3 and HOST·perreq are charged ONCE per request, frac-spread across chunks.
    The vLLM v1 chunked budget (decode-first, <= long_prefill_token_threshold per req) governs
    which reqs advance each step."""
    p = state.params
    decode_batch = len(state.running)
    decode_ms = decode_step_ms(decode_batch, _running_ctx_mean(state), p) if decode_batch > 0 else 0.0

    fa3_coef, host_shared_rate, host_perreq_rate = _prefill_host_fa3_rates(p)

    budget = max(0, state.sched_max_num_batched_tokens - decode_batch)
    total_chunk = 0.0       # batched NEW-token rate scales with total tokens this step
    gpu_fa3_ms = 0.0        # pipeline FA3 attention (per request; super-linear for re-prefills)
    host_perreq_ms = 0.0    # per-request host re-tokenize (summed over concurrent prefills)
    cached_w_sum = 0.0      # frac-weighted cached, for the per-step SHARED host term
    cached_w_n = 0
    any_prefill = False
    for r in state.prefilling.values():  # dict insertion order == FIFO admission order
        chunk = min(r.remaining_prefill, float(state.sched_long_prefill_threshold), float(budget))
        if chunk <= 0:
            r._chunk = 0.0  # type: ignore[attr-defined]
            continue
        r._chunk = chunk  # type: ignore[attr-defined]
        budget -= chunk
        any_prefill = True
        total_chunk += chunk
        frac = chunk / r.prefill_total if r.prefill_total > 0 else 1.0
        M = r.prefill_total          # tokens this turn (re-)prefills (reprefill_cached + new)
        R = r.resident_prefix        # resident prefix the (re-)prefill attends
        gpu_fa3_ms += fa3_coef * M * (R + 0.5 * M) * frac
        host_perreq_ms += host_perreq_rate * r.cached * frac
        cached_w_sum += r.cached * frac
        cached_w_n += 1
    if any_prefill:
        gpu_new_ms = (_prefill_gemm_per_tok_loaded(p, total_chunk) + PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN) * total_chunk
        mean_cached = cached_w_sum / cached_w_n if cached_w_n else 0.0
        host_shared_ms = host_shared_rate * mean_cached  # amortized once/step
        # Per-STEP fixed cost is the scheduler/launch overhead (one engine tick), NOT the full
        # PREFILL_FLOOR — the floor's first-token-emit/detok/return part is a per-REQUEST cost
        # added ONCE at first-token (see _on_first_token). Charging the full floor every step
        # accumulated ~steps×(FLOOR−sched) of phantom latency across a multi-step cohort, which
        # over-served the turn-0 cold-start at high concurrency.
        prefill_ms = (
            p.scheduler_overhead_ms_per_step + gpu_new_ms + gpu_fa3_ms + host_shared_ms + host_perreq_ms
        )
    else:
        prefill_ms = 0.0
    return max(decode_ms + p.scheduler_overhead_ms_per_step, prefill_ms)


def _on_step(state: _ServerState) -> None:
    state.step_scheduled = False
    if not state.running and not state.prefilling:
        return

    step_ms = _price_step(state)
    state.clock += step_ms

    # --- prefill bookkeeping: decrement chunks (blocks were reserved at admission),
    #     emit FIRST_TOKEN on completion. ---
    finished_prefill: list[int] = []
    for rid, r in state.prefilling.items():
        r.remaining_prefill -= getattr(r, "_chunk", 0.0)
        if r.remaining_prefill <= 1e-9:
            finished_prefill.append(rid)
    for rid in finished_prefill:
        state.push(state.clock, _FIRST_TOKEN, rid)

    # --- decode bookkeeping: one token per running req; the SESSION's KV grows by one token,
    #     reserving a fresh block at block boundaries (reclaiming free residents, then
    #     preempting idle herd residents under over-subscription — in-flight KV pinned). ---
    decode_in_flight = {r.session_id for r in state.prefilling.values()}
    decode_in_flight |= {r.session_id for r in state.running.values()}
    finished_decode: list[int] = []
    for rid in list(state.running.keys()):
        r = state.running.get(rid)
        if r is None:
            continue
        r.kv_tokens += 1.0
        need = state.cache.tokens_to_blocks(r.kv_tokens)
        if need > state.cache.cached_blocks(r.session_id):
            state.cache.grow_to(
                r.session_id, need, decode_in_flight, state.herd_pending, state.preempt_policy
            )  # best-effort
        r.output_left -= 1
        if r.output_left <= 0:
            finished_decode.append(rid)
    for rid in finished_decode:
        r = state.running.get(rid)
        if r is not None:
            state.push(state.clock, _DEPART, (r.session_id, r.turn_index))

    _schedule(state)
    if state.running or state.prefilling:
        state.push(state.clock, _STEP, None)
        state.step_scheduled = True


def _on_first_token(state: _ServerState, rid: int) -> None:
    r = state.prefilling.pop(rid, None)
    if r is None:
        return
    # Every turn records its first token on prefill completion. For a MISS turn this is
    # AFTER re-prefilling the full context, so its (higher) TTFT correctly reflects the
    # recompute cost — that is the climb.
    # The per-request floor residual (FLOOR minus the per-step scheduler overhead already paid
    # each prefill step): first-token emit + detok + return, charged ONCE here, not per step.
    floor_residual = max(0.0, state.prefill_floor_ms - state.params.scheduler_overhead_ms_per_step)
    state.results[(r.session_id, r.turn_index)]["first_token_epoch"] = state.clock + floor_residual
    state.running[rid] = r
    _ensure_step(state)


def _on_depart(state: _ServerState, session_id: int, turn_index: int) -> None:
    rid = _encode_rid(session_id, turn_index)
    state.running.pop(rid, None)
    sess = state.sessions[session_id]
    state.herd_pending.discard(session_id)  # completed this round -> now evictable (rotation)
    state.cache.touch(session_id)  # just finished a turn -> MRU (evicted last); KV persists

    rec = state.results.get((session_id, turn_index))
    if rec is not None:
        rec["completion_epoch"] = state.clock
        rec.setdefault("first_token_epoch", state.clock)

    sess.next_turn_idx = max(sess.next_turn_idx, turn_index + 1)
    # Barrier round-robin (matches the harness: asyncio.gather() between turns, see
    # _release_herd). The next turn's herd is released only after EVERY request in the
    # current turn has departed. The session keeps its KV resident across the barrier.
    state.herd_remaining -= 1
    if state.herd_remaining <= 0:
        _advance_herd(state)
    _ensure_step(state)


def _release_herd(state: _ServerState, turn_idx: int) -> None:
    """Release turn ``turn_idx``'s synchronized **herd**: EVERY surviving session (one with
    ``turn_count > turn_idx``) arrives at the SAME epoch (the barrier-release time).

    This mirrors the benchmark harness exactly (``run_multi_turn_benchmark``): interleaved
    round-robin — all sessions' turn-N requests are dispatched together and ``asyncio.gather``
    waits for the whole turn before turn N+1. So per-turn TTFT is dominated by the queue wait
    of C contemporaneous arrivals, not by per-session spacing — the missing low-turn climb."""
    herd = [s for s in state.sessions if s.turn_count > turn_idx]
    state.current_turn = turn_idx
    state.herd_remaining = len(herd)
    # Freeze each herd member's resident prefix AT release: this turn's hit/miss is decided
    # against what was cache-resident when the herd was scheduled. RETAINED COMPENSATING
    # RULE (audit-v2 S8): the engine decides hits LIVE at scheduling — the trace-adjudicated
    # unfreeze was gate-rejected 2026-06-10 (see the S8 comment in ``_schedule`` and
    # defit_log_entries/L4-queue.md). Physical eviction still runs live below; cross-turn
    # eviction IS visible (the snapshot is re-taken at every barrier) — only MID-TURN
    # erosion is hidden from hit accounting.
    state.resident_at_barrier = {s.session_id: state.cache.cached_blocks(s.session_id) for s in herd}
    # Herd members still awaiting their turn are TIER-2 victims only: free residents (dead /
    # completed-this-round sessions) are reclaimed first, and only genuine over-subscription
    # partially trims an idle herd member's prefix (LRU-oldest — see ``_evict``). This tier
    # structure is the counterfactual-validated combination (within 1.3-12.6% of engine
    # re-prefill truth when paired with live lookups); the engine itself draws NO herd
    # distinction (S9: waiting herd members supplied 43.0-69.6% of evicted blocks in the
    # pressure traces, and just-finished sessions' blocks are evicted LAST, not first) — the
    # residual session-granular tier ordering is the documented honest stop-point of this
    # cache model.
    state.herd_pending = {s.session_id for s in herd}
    for s in herd:
        state.push(state.clock, _ARRIVAL, (s.session_id, turn_idx))


def _advance_herd(state: _ServerState) -> None:
    """Barrier reached (all of the current turn departed): release the next turn's herd.

    A session whose conversation has ENDED is NOT freed — vLLM does not proactively release a
    finished request's KV blocks; with prefix caching on they stay cache-resident under LRU
    and are reclaimed only when another allocation needs the space. Because a dead session is
    never touched again, its blocks are the LRU-oldest, so the cache evicts THEM first — they
    are the eviction buffer that shields still-active sessions' prefixes. (With the
    retention-correct PrefixLRUCache this finally works; the old free-on-evict pool would have
    cascaded to full re-prefills here regardless.)"""
    next_turn = state.current_turn + 1
    if any(s.turn_count > next_turn for s in state.sessions):
        _release_herd(state, next_turn)


def _run_sim(
    sessions: list[Session], params: RooflineParams, max_events: int,
    preempt_policy: str = "lru",
    shared_prefix_tokens: float = 0.0,
    prefill_floor_ms: float | None = None,
    sched: QSimSchedConfig | None = None,
) -> dict[tuple[int, int], float]:
    cache = PrefixLRUCache(params.available_kv_blocks, params.cache_block_size)
    # Per-config scheduler truth: None (or None fields) -> the module H100 constants
    # (byte-identical default). See QSimSchedConfig.
    _sched = sched or QSimSchedConfig()
    state = _ServerState(
        params=params, cache=cache, sessions=sessions, preempt_policy=preempt_policy,
        shared_prefix_tokens=max(0.0, float(shared_prefix_tokens)),
        prefill_floor_ms=PREFILL_FLOOR_MS if prefill_floor_ms is None else float(prefill_floor_ms),
        sched_max_num_batched_tokens=float(
            MAX_NUM_BATCHED_TOKENS if _sched.max_num_batched_tokens is None
            else _sched.max_num_batched_tokens),
        sched_long_prefill_threshold=float(
            LONG_PREFILL_TOKEN_THRESHOLD if _sched.long_prefill_token_threshold is None
            else _sched.long_prefill_token_threshold),
        sched_max_num_seqs=(
            MAX_NUM_SEQS if _sched.max_num_seqs is None else int(_sched.max_num_seqs)),
    )

    _release_herd(state, 0)  # turn-0 herd: all sessions arrive at epoch 0

    events = 0
    while state.heap and events < max_events:
        epoch, _seq, kind, payload = heapq.heappop(state.heap)
        state.clock = epoch
        events += 1
        if kind == _ARRIVAL:
            _on_arrival(state, payload[0], payload[1])
        elif kind == _STEP:
            _on_step(state)
        elif kind == _FIRST_TOKEN:
            _on_first_token(state, payload)
        elif kind == _DEPART:
            _on_depart(state, payload[0], payload[1])

    ttfts: dict[tuple[int, int], float] = {}
    for key, rec in state.results.items():
        if "first_token_epoch" in rec and "arrival_epoch" in rec:
            ttfts[key] = rec["first_token_epoch"] - rec["arrival_epoch"]
    return ttfts


def _aggregate(
    ttfts: dict[tuple[int, int], float],
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams,
    gpu_key: str | None = None,
) -> list[float]:
    by_idx: dict[int, list[float]] = {}
    for (_sid, ti), v in ttfts.items():
        if v > 0:
            by_idx.setdefault(ti, []).append(v)
    out: list[float] = []
    for t in turns:
        ti = int(t.get("turn_index", 0))
        vals = by_idx.get(ti)
        out.append(statistics.median(vals) if vals else _fallback_ttft(t, profile, concurrency, params, gpu_key))
    return out


def _fallback_ttft(
    turn: dict[str, Any], profile: str, concurrency: float, params: RooflineParams,
    gpu_key: str | None = None,
) -> float:
    """Forward static-formula fallback for a turn no session reached (keeps list length).
    ``gpu_key`` selects the per-(conc,gpu) cohort survival (pooled fallback when absent)."""
    from simulator.ramp_tpot import sched_hat

    ti = int(turn.get("turn_index", 0))
    cached = float(turn.get("cached_context_tokens") or 0.0)
    new = float(turn.get("new_prefill_tokens") or 0.0)
    out = float(turn.get("output_tokens") or 1.0)
    sched = sched_hat(profile, float(concurrency), ti, gpu_key) if profile in PROFILE_DIST else float(concurrency)
    tpot = predict_cell_tpot([KernelTurnInput(cached, new, out, sched)], params)[0]
    return predict_turn_ttft(cached, new, out, sched, tpot, params)


# --------------------------------------------------------- oracle (validation only)


def _build_cohort_oracle(
    turns: list[dict[str, Any]], profile: str, concurrency: float,
    bench_root: "Path | str | None" = None,
) -> list[Session] | None:
    """Validation-only: build the cohort from measured session_timelines (per-session turn
    lists) instead of the survival quantile. Off the forward path; None if unavailable.

    ``bench_root`` selects which GPU/model store to read the measured timelines from (the
    per-request JSONs carry ``session_id`` + ``turn_index``). Defaults to the H100 tp1 store
    for back-compat; pass the matching store (e.g. the A100 dir) to run oracle on that config —
    this is what makes the oracle-vs-forward drain/amplifier split available for non-H100."""
    try:
        from pathlib import Path

        from profiling.process.extract_benchmark_per_request import (
            collect_session_timelines,
        )
    except Exception:
        return None
    bench_root = Path(bench_root) if bench_root is not None else Path(
        "/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm"
    )
    if not bench_root.exists():
        return None
    try:
        timelines = collect_session_timelines(bench_root)
    except Exception:
        return None
    cell = timelines.get(f"{profile}__{int(round(float(concurrency)))}")
    if not cell or not cell.get("sessions"):
        return None
    sessions: list[Session] = []
    for sid, sess_turns in enumerate(cell["sessions"]):
        specs = [
            TurnSpec(
                turn_index=int(tt["turn_index"]),
                cached_context_tokens=float(tt.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(tt.get("new_prefill_tokens") or 0.0),
                output_tokens=max(1.0, float(tt.get("output_tokens") or 1.0)),
            )
            for tt in sess_turns
        ]
        if specs:
            sessions.append(Session(session_id=sid, turn_count=len(specs), turns=specs))
    return sessions or None


# ------------------------------------------------------------------- public API


def predict_cell_ttft_qsim(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams | None = None,
    *,
    oracle: bool = False,
    max_events: int = 4_000_000,
    preempt_policy: str = "lru",
    shared_prefix_tokens: float = 0.0,
    oracle_bench_root: "Path | str | None" = None,
    gpu_key: str | None = None,
    prefill_floor_ms: float | None = None,
    sched: QSimSchedConfig | None = None,
    _survival_override: list[float] | None = None,
    _scale_override: list[float] | None = None,
) -> list[float]:
    """Per-turn TTFT (ms) for a (profile, concurrency) cell, emergent from a forward
    closed-loop event-driven queue sim with session-persistent KV + RECOMPUTE preemption.

    Returns one TTFT per input turn (median over sessions reaching that turn_index), aligned
    to ``turns`` order; ``[]`` for empty; a turn_index reached by no session falls back to the
    forward static predictor. Forward by default (cohort from ``forward_survival``);
    ``oracle=True`` overlays measured ``session_timelines`` (validation only). ``preempt_policy``
    orders tier-2 partial trims: ``'lru'`` (oldest-first — ENGINE-FAITHFUL, the production
    default since the S7 re-derivation 2026-06-10) or ``'tail'`` (MRU-first; trace-falsified,
    kept only as the adjudication falsification seam).

    ``shared_prefix_tokens`` (S) models a profile-constant cross-session APC prefix (the
    ``prefix_aware_synthetic`` workloads): the front S tokens of every session's context are an
    identical shared block that vLLM dedups — ONE session prefills it, the rest HIT. Threaded by
    the emitter from ``request_metadata.shared_prefix_actual_tokens``; ``0.0`` (default) is a
    no-op (byte-identical). NOT a fitted constant (per-cell measured workload input).

    ``sched`` (QSimSchedConfig) is the per-deployment vLLM scheduler truth for the admission
    arithmetic (token budget / per-request chunk cap / running-set cap); ``None`` (default)
    resolves to the module-level H100 constants — byte-identical. See QSimSchedConfig."""
    if not turns:
        return []
    p = params or RooflineParams()

    sessions: list[Session] | None = None
    if oracle:
        sessions = _build_cohort_oracle(turns, profile, float(concurrency), oracle_bench_root)
    if sessions is None:
        sessions = _build_cohort(
            turns, profile, float(concurrency), gpu_key,
            survival_override=_survival_override, scale_override=_scale_override,
        )

    # Prefill floor is resolved INDEPENDENTLY of the cohort gpu_key: an explicit ``prefill_floor_ms``
    # wins (the emitter passes the config's measured floor for ALL configs), else fall back to the
    # gpu_key lookup, else the legacy 26.0. This decouples the per-config floor (always correct) from
    # the trajectory-replay cohort (gpu_key), which the emitter may gate to tp1 only.
    floor = prefill_floor_ms if prefill_floor_ms is not None else _prefill_floor_for(gpu_key)
    ttfts = _run_sim(
        sessions, p, max_events, preempt_policy=preempt_policy,
        shared_prefix_tokens=shared_prefix_tokens,
        prefill_floor_ms=floor,
        sched=sched,
    )
    return _aggregate(ttfts, turns, profile, float(concurrency), p, gpu_key)


def predict_cell_e2el_qsim(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    ttft_qsim: list[float],
    tpot_preds: list[float] | None = None,
    params: RooflineParams | None = None,
) -> list[float]:
    """E2EL composition: ``e2el_qsim[t] = ttft_qsim[t] + output_tokens[t] * tpot[t]`` (tpot
    defaults to the kernel TPOT the emitter passes — byte-identical to the ``e2el_pred`` line)."""
    if not turns:
        return []
    p = params or RooflineParams()
    if tpot_preds is None:
        inputs = [
            KernelTurnInput(
                cached_context_tokens=float(t.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(t.get("new_prefill_tokens") or 0.0),
                output_tokens=float(t.get("output_tokens") or 0.0),
                scheduled_requests=float(concurrency),
            )
            for t in turns
        ]
        tpot_preds = predict_cell_tpot(inputs, p)
    out: list[float] = []
    for t, ttft, tpot in zip(turns, ttft_qsim, tpot_preds):
        out.append(float(ttft) + float(t.get("output_tokens") or 0.0) * float(tpot))
    return out
