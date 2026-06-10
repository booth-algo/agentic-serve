"""Closed-form TPOT / TTFT predictor for vLLM-style continuous batching.

Replaces the per-step scheduler-shape replay (block pool admission, partial
prefill caps, cohort-aware multi-turn loop) with a single closed-form
expression per (profile, concurrency, turn_index) row, built on the existing
analytical roofline params at ``profile_data/kernels/roofline_params_*.json``.

Motivating bet: the trace-aligned aggregate inputs (mean cached/prefill/output
tokens per turn) carry most of the signal; per-step replay was overfitting to
vLLM internals without proportional accuracy gain.

For a single turn at concurrency ``c`` with mean output ``O``, mean new prefill
tokens ``P_new``, and mean cached prefix tokens ``P_cache``:

    # Steady-state decode step (one token per request per step)
    decode_compute_ms   = 2 · n_params · c           / (peak_flops · util_flops) · 1e3
    decode_bandwidth_ms = ( weights
                          + c · ctx_mid · kv_bpt     # KV reads (per-running ctx)
                          + c · kv_bpt               # KV writes (1 token / req)
                          ) / (peak_bw · util_bw) · 1e3
    tpot_ms             = max(decode_compute_ms, decode_bandwidth_ms)

    # Cohort prefill (all c sessions fire turn N+1 after turn N finishes;
    # vLLM admits them across enough steps to spend the prefill tokens)
    prefill_total_tokens = c · (P_new − P_cache)
    prefill_compute_ms   = 2 · n_params · prefill_total_tokens / (peak_flops · util_flops) · 1e3
    prefill_bandwidth_ms = ( weights_amortized
                           + prefill_total_tokens · kv_bpt   # read prefix + write new
                           ) / (peak_bw · util_bw) · 1e3
    ttft_ms              = max(prefill_compute_ms, prefill_bandwidth_ms)

    e2e_ms = ttft_ms + (O − 1) · tpot_ms

``ctx_mid`` is the per-request context length at decode midpoint:
``P_new + O/2``.  ``cached_context_tokens`` does not contribute to ctx_mid
because cached blocks are already resident — only newly-generated tokens and
new-prefill tokens consume KV reads during this turn's decode.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping


# ---------------------------------------------------------------------- params


@dataclass(frozen=True)
class RooflineParams:
    """Hardware + model constants + two utilization scalars + KV capacity.

    Field names match the existing ``profile_data/kernels/roofline_params_*.json``
    schema so we can reuse the calibrated H100 / Llama-3.1-8B file unchanged.
    ``available_kv_blocks`` and ``cache_block_size`` are hardware/runtime
    specs (not fit knobs) — they bound how many concurrent decode requests
    fit in HBM and drive wave-factor amplification under KV pressure.
    """

    n_params: int = 8_030_000_000
    peak_flops_per_s: float = 989e12
    peak_bw_bytes_per_s: float = 3.35e12
    kv_bytes_per_token: float = 131072.0
    util_flops: float = 0.65  # PINNED; pre-registered re-derivation over the 4 serving-wall traces gives 0.5886 (n=7, thin; submit-wall convention unreliable under AsyncScheduler) — see profile_data/kernels/roofline_utils_H100.json + defit_log_entries/L6-utils.md. R1 curve (prefill_gemm_util_H100.json) is the authoritative prefill source.
    util_bw: float = 0.93     # PINNED; does NOT reproduce from any documented computation (audit-v2 G4). Pre-registered recipe over the 4 serving-wall traces gives 0.8111 (62 steady-state decode steps, byte-quartile-stable) — profile_data/kernels/roofline_utils_H100.json. Candidate 0.8111 was wired and gated 2026-06-10 (replay-ON) and REJECTED: H100 tpot_cell +0.67pt, chat +0.77pt vs baseline (see defit_log_entries/L6-utils.md). Kept as pinned-with-measured-disagreement — 0.93 compensates host overhead the decode pricing carries no separate term for.
    bytes_per_param: float = 2.0
    available_kv_blocks: int = 27_250          # empirical H100 (max free_kv_blocks across 4 vLLM engine traces 2026-05-27; ~1.5% less than spec-derived 27651)
    cache_block_size: int = 16                 # vLLM v1 default
    max_num_batched_tokens: int = 8192         # ENGINE CONFIG, not a fit: vLLM OPENAI_API_SERVER resolved default for >=70GiB non-A100 devices (vllm/engine/arg_utils.py _set_default_args; A100-named devices resolve 2048 — set per-deployment JSON). Benchmark servers launched with the flag unset (server metadata: null), so the resolved default is what ran. Consumed by kernel_tpot._overflow_weight as the chunked-prefill per-step token budget.
    scheduler_overhead_ms_per_step: float = 5.7  # PINNED; per-step Python/dispatcher cost, originally mean over 99 lowest-work decode steps of swe_c40. Pre-registered re-derivation (median over 2666 batch-1 decode steps, all 4 traces) gives 4.5574 — roofline_utils_H100.json; 5.7 sits at the top of the per-trace range, not the center.
    tensor_parallel: int = 1                   # TP degree: shards KV (by head) + weights across GPUs. tp=1 = single-GPU (unchanged).
    kv_heads: int = 8                          # GQA KV heads (Llama-3.1-8B); caps KV sharding at min(tp, kv_heads).

    @property
    def kv_capacity_bytes(self) -> int:
        return self.available_kv_blocks * self.cache_block_size * int(
            self.kv_bytes_per_token
        )

    @classmethod
    def from_json(cls, path: Path) -> "RooflineParams":
        data = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


# ---------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class TurnInput:
    """One (profile, concurrency, turn_index) row's aggregate workload.

    Tokens are means over the cohort's `concurrency` sessions for that turn.
    For a single session (concurrency=1) these are just that session's values.

    ``shared_prefix_fraction`` ∈ [0, 1] says what fraction of the
    ``cached_context_tokens`` is shared across sessions in the cohort
    (i.e. all sessions hash to the same KV blocks for that prefix).  This
    matters under KV pressure: a shared prefix occupies the block pool only
    once for the cohort, and shows up in step bandwidth as a single KV-read
    rather than ``c`` reads.

    Workloads with a shared system prompt across sessions (e.g.
    ``osworld-multiturn-synth``) approach 1.0; workloads where each session
    is an independent agent thread (e.g. ``swebench-multiturn-synth``)
    approach 0.0.  The predictor estimates this from per-cohort variance
    in turn-0 ``new_prefill_tokens`` — low variance ⇒ shared template.
    """

    profile: str
    concurrency: int
    turn_index: int
    output_tokens: float
    new_prefill_tokens: float
    cached_context_tokens: float
    shared_prefix_fraction: float = 0.0

    @property
    def fresh_prefill_tokens(self) -> float:
        """Tokens that actually need to be prefilled (cache miss)."""
        return max(0.0, self.new_prefill_tokens - self.cached_context_tokens)

    @property
    def shared_cached_tokens(self) -> float:
        """Cached tokens that are deduped to one KV-block set across the cohort."""
        f = max(0.0, min(1.0, float(self.shared_prefix_fraction)))
        return f * float(self.cached_context_tokens)

    @property
    def per_session_cached_tokens(self) -> float:
        """Cached tokens that occupy KV blocks once per session in the cohort."""
        return max(0.0, float(self.cached_context_tokens) - self.shared_cached_tokens)


@dataclass(frozen=True)
class TurnPrediction:
    """Closed-form TPOT/TTFT/e2e for one turn row.

    `tpot_compute_ms` / `tpot_bandwidth_ms` are exposed for residual analysis.
    `effective_decode_batch` and `wave_factor` expose KV-pressure state:
    when `c × ctx_mid_blocks > available_kv_blocks`, only `effective_decode_batch`
    requests fit per step and per-request TPOT amplifies by `wave_factor`.
    """

    profile: str
    concurrency: int
    turn_index: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    tpot_compute_ms: float
    tpot_bandwidth_ms: float
    classification: str  # "compute_bound" | "bandwidth_bound" | "empty"
    effective_decode_batch: int = 0
    wave_factor: float = 1.0


# ------------------------------------------------------------------ predictor


class ClosedFormTpotPredictor:
    """Pure analytical TPOT / TTFT / e2e from aggregate per-turn inputs.

    No per-step simulation, no block pool, no scheduler-shape replay.

    ``kv_pressure_enabled`` (default ``False``) gates the KV-capacity wave
    factor.  When off, ``wave_factor`` is always 1 and the predictor behaves
    as a pure roofline.  When on, the predictor models KV-block exhaustion by
    capping the per-step decode batch at ``B_eff = available_blocks /
    blocks_per_req`` and amplifying TPOT by ``c / B_eff``.  This helps
    per-session-prefix workloads (e.g. swebench) and HURTS shared-prefix
    workloads (e.g. osworld) unless ``shared_prefix_fraction`` is set on each
    ``TurnInput``.  Default off because we don't yet have a reliable
    discriminator from aggregate sidecar features.
    """

    def __init__(
        self,
        params: RooflineParams,
        *,
        kv_pressure_enabled: bool = False,
    ) -> None:
        if params.peak_flops_per_s <= 0:
            raise ValueError("peak_flops_per_s must be > 0")
        if params.peak_bw_bytes_per_s <= 0:
            raise ValueError("peak_bw_bytes_per_s must be > 0")
        if not (0.0 < params.util_flops <= 2.0):
            raise ValueError(f"util_flops out of range: {params.util_flops}")
        if not (0.0 < params.util_bw <= 2.0):
            raise ValueError(f"util_bw out of range: {params.util_bw}")
        self._p = params
        self._kv_pressure_enabled = bool(kv_pressure_enabled)

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        kv_pressure_enabled: bool = False,
    ) -> "ClosedFormTpotPredictor":
        return cls(
            RooflineParams.from_json(path),
            kv_pressure_enabled=kv_pressure_enabled,
        )

    @property
    def kv_pressure_enabled(self) -> bool:
        return self._kv_pressure_enabled

    @property
    def params(self) -> RooflineParams:
        return self._p

    # ---------------------------------------------------------- decode (TPOT)

    def _decode_step_ms(
        self,
        c: int,
        ctx_per_req: float,
        shared_ctx: float = 0.0,
    ) -> tuple[float, float]:
        """Per-step (compute_ms, bandwidth_ms) for a decode-only step.

        c running requests, each contributing `ctx_per_req` unique KV-read
        tokens, plus `shared_ctx` tokens read ONCE (because they're shared
        across all c sessions via prefix-cache dedup).  Compute is c new
        tokens through 2·n_params FLOPs each; bandwidth is weights +
        shared_ctx KV reads + c · ctx_per_req KV reads + c KV writes.
        """
        p = self._p
        if c <= 0:
            return 0.0, 0.0
        compute_ms = (2.0 * float(p.n_params) * c) \
            / (p.peak_flops_per_s * p.util_flops) * 1e3
        kv_bpt = float(p.kv_bytes_per_token)
        bytes_total = (
            float(p.bytes_per_param) * float(p.n_params)         # weights
            + max(0.0, shared_ctx) * kv_bpt                      # shared KV reads (1×)
            + c * max(0.0, ctx_per_req) * kv_bpt                 # per-session KV reads
            + c * kv_bpt                                         # KV writes
        )
        bw_ms = bytes_total / (p.peak_bw_bytes_per_s * p.util_bw) * 1e3
        return compute_ms, bw_ms

    # --------------------------------------------------------- prefill (TTFT)

    def _prefill_ms(self, total_prefill_tokens: float) -> tuple[float, float]:
        """Cohort prefill: all `c` sessions admit ``c·P_fresh`` total tokens.

        Modeled as one big prefill burst — vLLM spreads it across enough
        steps that wall time ≈ aggregate roofline.  Weights are amortized
        across the burst (read once, used for many tokens).
        """
        p = self._p
        n = max(0.0, total_prefill_tokens)
        if n == 0.0:
            return 0.0, 0.0
        compute_ms = (2.0 * float(p.n_params) * n) \
            / (p.peak_flops_per_s * p.util_flops) * 1e3
        kv_bpt = float(p.kv_bytes_per_token)
        # Weights read once for the burst; per-token reads/writes scale linearly.
        bytes_total = (
            float(p.bytes_per_param) * float(p.n_params)         # weights once
            + n * kv_bpt                                          # prefix reads
            + n * kv_bpt                                          # new KV writes
        )
        bw_ms = bytes_total / (p.peak_bw_bytes_per_s * p.util_bw) * 1e3
        return compute_ms, bw_ms

    # -------------------------------------------------------------- per-turn

    def predict(self, inp: TurnInput) -> TurnPrediction:
        c = int(inp.concurrency)
        if c <= 0:
            raise ValueError(f"concurrency must be >= 1, got {c}")

        # Per-session context at decode start = cached prefix + new prefill;
        # midpoint adds half the output for steady-state KV-read accounting.
        # We track shared vs per-session splits even when KV pressure is off,
        # so the dataclass exposes a consistent view — but the bandwidth model
        # only consumes the split when pressure is enabled.
        full_ctx_start = float(inp.cached_context_tokens) + float(inp.new_prefill_tokens)
        full_ctx_mid = full_ctx_start + 0.5 * float(inp.output_tokens)
        shared_ctx = inp.shared_cached_tokens
        per_session_ctx_start = inp.per_session_cached_tokens + float(inp.new_prefill_tokens)
        per_session_ctx_mid = per_session_ctx_start + 0.5 * float(inp.output_tokens)

        # KV-capacity-pressure wave factor — only applied when explicitly
        # enabled.  Total KV blocks needed = shared_blocks + c × per_session
        # _blocks; if it exceeds available blocks, only B_eff requests fit
        # per decode step and per-request TPOT amplifies by c / B_eff.
        # Gated because mis-applying it to shared-prefix workloads (osworld)
        # over-predicts; we don't yet have a reliable per-cohort discriminator
        # for whether the cached prefix is shared across sessions.
        p = self._p
        if self._kv_pressure_enabled:
            block_size = max(1, p.cache_block_size)
            shared_blocks = math.ceil(shared_ctx / block_size) if shared_ctx > 0 else 0
            per_session_blocks = max(1, math.ceil(per_session_ctx_mid / block_size))
            available_for_running = max(1, p.available_kv_blocks - shared_blocks)
            capacity_batch = max(1, available_for_running // per_session_blocks)
            B_eff = max(1, min(c, capacity_batch))
            wave_factor = c / B_eff if B_eff > 0 else 1.0
            bw_ctx_per_req = per_session_ctx_mid
            bw_shared_ctx = shared_ctx
        else:
            B_eff = c
            wave_factor = 1.0
            # Without pressure modeling, ignore the shared/per-session split:
            # bandwidth treats every session as reading the full ctx.
            bw_ctx_per_req = full_ctx_mid
            bw_shared_ctx = 0.0

        # Cost the step at B_eff requests (not c) — under KV pressure, fewer
        # requests' KV reads share each step's weight-load amortization.
        # Shared KV reads are counted once regardless of B_eff.
        dec_c, dec_bw = self._decode_step_ms(
            B_eff, bw_ctx_per_req, shared_ctx=bw_shared_ctx
        )
        if dec_c == 0.0 and dec_bw == 0.0:
            step_ms = 0.0
            classification = "empty"
        elif dec_bw >= dec_c:
            step_ms = dec_bw
            classification = "bandwidth_bound"
        else:
            step_ms = dec_c
            classification = "compute_bound"
        tpot_ms = step_ms * wave_factor

        # Prefill: cohort fires together, so total fresh tokens across the
        # cohort = c · P_fresh.  Aggregate roofline ≈ wall for the burst.
        cohort_fresh = c * inp.fresh_prefill_tokens
        pf_c, pf_bw = self._prefill_ms(cohort_fresh)
        ttft_ms = max(pf_c, pf_bw)

        # e2e = TTFT + (O − 1) decode tokens.  Guard O=0/1 to avoid negative.
        decode_tokens = max(0.0, float(inp.output_tokens) - 1.0)
        e2e_ms = ttft_ms + decode_tokens * tpot_ms

        return TurnPrediction(
            profile=inp.profile,
            concurrency=c,
            turn_index=int(inp.turn_index),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            tpot_compute_ms=dec_c,
            tpot_bandwidth_ms=dec_bw,
            classification=classification,
            effective_decode_batch=B_eff,
            wave_factor=wave_factor,
        )


# -------------------------------------------------------- sidecar input helpers


def turn_input_from_sidecar(
    per_turn_bucket: Mapping[str, object],
) -> TurnInput:
    """Build :class:`TurnInput` from one ``per_turn`` sidecar entry.

    Means across the request_count samples in the bucket.  Missing or empty
    lists default to 0.
    """

    def _mean(name: str) -> float:
        vals = per_turn_bucket.get(name, [])
        if not isinstance(vals, list) or not vals:
            return 0.0
        nums = [float(v) for v in vals]
        return fmean(nums) if nums else 0.0

    return TurnInput(
        profile=str(per_turn_bucket.get("profile", "")),
        concurrency=int(per_turn_bucket.get("concurrency", 0) or 0),
        turn_index=int(per_turn_bucket.get("turn_index", 0) or 0),
        output_tokens=_mean("output_tokens"),
        new_prefill_tokens=_mean("new_prefill_tokens"),
        cached_context_tokens=_mean("cached_context_tokens"),
    )


def iter_sidecar_inputs(
    per_turn_payload: Mapping[str, Mapping[str, object]],
) -> Iterable[TurnInput]:
    """Yield one :class:`TurnInput` per ``per_turn`` bucket in the sidecar."""
    for bucket in per_turn_payload.values():
        if not isinstance(bucket, Mapping):
            continue
        yield turn_input_from_sidecar(bucket)


# ------------------------------------------------------------ llm-d sidecar fit


@dataclass(frozen=True)
class LlmDInferenceSimConfig:
    """Closed-form parameters consumed by llm-d/llm-d-inference-sim CLI flags.

    All four duration fields are in milliseconds; ``llm-d-inference-sim``
    accepts Go duration strings like ``"8.2ms"`` — formatters below convert.
    """

    prefill_overhead_ms: float
    prefill_time_per_token_ms: float
    inter_token_latency_ms: float
    time_factor_under_load: float
    max_num_seqs: int

    def to_llm_d_dict(self) -> dict:
        return {
            "prefill-overhead": f"{self.prefill_overhead_ms:.3f}ms",
            "prefill-time-per-token": f"{self.prefill_time_per_token_ms:.4f}ms",
            "inter-token-latency": f"{self.inter_token_latency_ms:.3f}ms",
            "time-factor-under-load": round(self.time_factor_under_load, 3),
            "max-num-seqs": int(self.max_num_seqs),
        }


def fit_llm_d_config(predictions: list[TurnPrediction]) -> LlmDInferenceSimConfig:
    """Fit four llm-d-inference-sim params from a swept-c TPOT/TTFT curve.

    Inputs: predictions across multiple concurrency values for a single
    profile.  We project them onto the llm-d closed form:

        TTFT  = prefill_overhead + (P_fresh) · prefill_time_per_token
        ITL   = base_itl · (1 + (tf − 1) · (c − 1) / (max_c − 1))

    Strategy:
      - prefill_time_per_token = slope of (ttft_ms, P_fresh) at c=1 (the
        single-request prediction is least confounded by batching).  We don't
        have P_fresh on the prediction directly; the caller must compute it
        from inputs and pass it via parallel ordering.  Below we use a
        simpler two-anchor formula on the predictions at min-c and max-c.
      - inter_token_latency = predicted tpot at c=1
      - time_factor_under_load = predicted tpot(c=max) / predicted tpot(c=1)
      - prefill_overhead = predicted ttft(c=1)
      - max_num_seqs = max c seen
    """
    if not predictions:
        raise ValueError("need at least one prediction to fit llm-d config")

    by_c = {p.concurrency: p for p in predictions}
    cs = sorted(by_c.keys())
    c_min = cs[0]
    c_max = cs[-1]

    base = by_c[c_min]
    peak = by_c[c_max]
    base_itl = max(base.tpot_ms, 1e-6)
    tf = peak.tpot_ms / base_itl if base_itl > 0 else 1.0

    # prefill_time_per_token from the c=1 single-request prediction: TTFT was
    # produced by a single prefill burst of `P_fresh` tokens.  We don't know
    # P_fresh here directly; caller can override by passing in a richer fitter.
    # For now use ttft(c=1) / output_tokens proxy = 0 fallback when unknown.
    prefill_overhead_ms = base.ttft_ms
    prefill_time_per_token_ms = 0.0

    return LlmDInferenceSimConfig(
        prefill_overhead_ms=prefill_overhead_ms,
        prefill_time_per_token_ms=prefill_time_per_token_ms,
        inter_token_latency_ms=base_itl,
        time_factor_under_load=tf,
        max_num_seqs=int(c_max),
    )
