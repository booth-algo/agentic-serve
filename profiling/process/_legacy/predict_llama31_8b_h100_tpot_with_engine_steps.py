"""Predict TPOT by pricing simulated vLLM scheduler engine steps.

This model keeps benchmark TPOT as validation-only.  It uses benchmark turn
shape to replay scheduler steps, prices each step as a mixed decode/prefill
forward, then aggregates predicted per-request ITLs the same way the benchmark
computes TPOT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.process._legacy.predict_llama31_8b_h100_tpot_from_interpolated_kernels import (  # noqa: E402
    DEFAULT_ATTENTION_PROFILE,
    DEFAULT_GEMM_SUMMARY,
    DEFAULT_SMALL_KERNEL_SUMMARY,
    ComponentModel,
    ComponentPrediction,
    build_component_models,
    load_attention_components,
    load_gemm_components,
    load_small_kernel_components,
    _sum_component_predictions,
)
from simulator._legacy.clean_step_predictor import CleanStepPredictor  # noqa: E402
from simulator.closed_form_tpot import (  # noqa: E402
    ClosedFormTpotPredictor,
    TurnInput as ClosedFormTurnInput,
)
from simulator._legacy.roofline_step_predictor import (  # noqa: E402
    RooflineStepPredictor,
)
from simulator._legacy.vllm_engine_step_cost import (  # noqa: E402
    TokenCostFn,
    VllmEngineStepCost,
    VllmEngineStepCostModel,
    engine_step_cost_ms,
    predict_mean_tpot_from_engine_steps,
    predict_mean_tpot_from_predictor,
    predict_mean_tpot_from_roofline,
    prefill_attention_work,
)
from simulator._legacy.vllm_fluid_scheduler import (  # noqa: E402
    VllmFluidSchedulerConfig,
    VllmFluidTurnInput,
    VllmFluidTurnSummary,
    simulate_fluid_turn,
)
from simulator._legacy.vllm_scheduler_shape import (  # noqa: E402
    MultiTurnReplayResult,
    PrefillMode,
    VllmPrefillChunk,
    VllmSchedulerConfig,
    benchmark_row_requests,
    simulate_vllm_multi_turn_replay,
    simulate_vllm_turn_shape,
    summarize_replay_for_turn,
)
from simulator._legacy.vllm_prefix_cache_residency import (  # noqa: E402
    VllmPrefixCacheResidencyConfig,
    VllmPrefixCacheResidencyInput,
    estimate_prefix_cache_residency,
)


DEFAULT_BASE_PREDICTIONS = Path(
    "profile_data/results/llama31_8b_h100_interpolated_kernel_tpot_predictions.csv"
)
DEFAULT_BENCHMARK_TURNS = Path(
    "profile_data/results/benchmark_turns_llama31_8b_h100_vllm.csv"
)
DEFAULT_BENCHMARK_PER_REQUEST = Path(
    "profile_data/results/benchmark_per_request_llama31_8b_h100_vllm.json"
)
DEFAULT_FORWARD_GEMM = Path("profile_data/kernels/forward_pass/gemm/H100.csv")
DEFAULT_FORWARD_ATTENTION = Path("profile_data/kernels/forward_pass/flash_attn/H100.csv")
DEFAULT_SERVING_ATTENTION = Path("profile_data/kernels/llama31_8b_attention.csv")
DEFAULT_FA3_PREFILL = Path("profile_data/kernels/fa3_prefill_H100.csv")
DEFAULT_VLLM_PREFILL_MEASURED = Path("profile_data/kernels/prefill_profile_H100.csv")
DEFAULT_NSYS_COMPILED = Path("profile_data/kernels/prefill_compiled_breakdown_H100.csv")
DEFAULT_FULL_PREFILL_CSV = Path("profile_data/kernels/prefill_profile_H100_dense.csv")
DEFAULT_CACHED_PREFILL_CSV = Path("profile_data/results/cached_prefill_v3_H100.csv")
DEFAULT_DECODE_PROFILE_CSV = Path(
    "profile_data/results/decode_profile_H100_large_2026-05-17.csv"
)
DEFAULT_ROOFLINE_PARAMS = Path(
    "profile_data/kernels/roofline_params_H100_llama31_8b.json"
)
DEFAULT_OUTPUT = Path(
    "profile_data/results/llama31_8b_h100_engine_step_tpot_predictions.csv"
)
DEFAULT_REPORT = Path(
    "profile_data/results/llama31_8b_h100_engine_step_tpot_report.md"
)
DEFAULT_DASHBOARD_OUTPUT = Path(
    "inference-benchmark/dashboard/public/llama31-8b-h100-tpot-fit.json"
)

NUM_LAYERS = 32
ENGINE_STEP_MODEL = "mixed_forward_step_v0"
FLUID_WAVE_MODEL = "fluid_wave_v1"
WavePolicy = str
EffectiveCachePolicy = str
SchedulerModel = str


@dataclass(frozen=True)
class EngineStepComponentModels:
    attention_model: ComponentModel
    gemm_models: dict[str, ComponentModel]
    small_kernel_models: dict[str, ComponentModel]
    forward_gemm_model: "ForwardPassGemmModel | None" = None
    forward_attention_model: "ForwardPassAttentionModel | None" = None
    nsys_compiled_model: "NsysCompiledPrefillModel | None" = None


@dataclass(frozen=True)
class EngineStepPredictionRow:
    component_model: str
    engine_step_model: str
    scheduler_model: str
    prefill_mode: str
    dense_source: str
    wave_policy: str
    profile: str
    concurrency: int
    turn_index: int
    primary_eval: bool
    batch_size: int
    scheduled_request_count: int
    successful_request_count: int
    turn_num_requests: int
    context_len: int
    output_tokens: int
    new_prefill_tokens: int
    cached_context_tokens: int
    cache_hit_rate: float
    effective_cache_policy: str
    sim_new_prefill_tokens: int
    sim_cached_context_tokens: int
    engine_effective_cached_tokens: float
    capacity_feasible_cached_tokens: int
    engine_uncached_prefill_tokens: int
    cache_residency_classification: str
    measured_tpot_ms: float
    base_pred_tpot_ms: float
    pred_tpot_ms: float
    base_pct_error: float
    pct_error: float
    base_signed_error_ms: float
    signed_error_ms: float
    base_abs_error_ms: float
    abs_error_ms: float
    sim_steps: int
    sim_decode_only_steps: int
    sim_prefill_only_steps: int
    sim_mixed_decode_prefill_steps: int
    sim_total_decode_slots: int
    sim_total_prefill_tokens: int
    sim_max_decode_batch: int
    sim_mean_decode_batch: float
    sim_mean_decode_batch_ratio: float
    sim_min_free_kv_blocks: int
    sim_max_used_kv_blocks: int
    sim_max_capacity_waiting_requests: int
    sim_total_preemptions: int
    sim_total_recompute_prefill_tokens: int
    decode_residency_wave_factor: float
    total_step_ms: float
    pooled_itl_ms: float
    decoded_request_count: int
    dense_ms: float
    decode_attention_ms: float
    prefill_attention_ms: float
    small_kernel_ms: float
    logits_sampling_ms: float
    runtime_overhead_ms: float
    recompute_or_preemption_ms: float
    prefill_attention_work_units: float
    fluid_pred_low_tpot_ms: float
    fluid_pred_mid_tpot_ms: float
    fluid_pred_high_tpot_ms: float
    fluid_active_decode_capacity: int
    fluid_decode_steps: int
    fluid_decode_waves: int
    fluid_prefill_only_steps: int
    fluid_mixed_decode_prefill_steps: int
    fluid_kv_pressure: float
    fluid_prefill_debt_tokens: int
    fluid_prefill_debt_per_decode_token: float
    fluid_cache_optimistic_tokens: int
    fluid_cache_mid_tokens: int
    fluid_cache_pessimistic_tokens: int
    fluid_scheduler_sensitivity: str
    diagnostic_reason: str


@dataclass(frozen=True)
class ErrorSummary:
    rows: int
    base_mape: float
    engine_mape: float
    base_median_ape: float
    engine_median_ape: float
    base_max_ape: float
    engine_max_ape: float
    base_mean_signed_error_ms: float
    engine_mean_signed_error_ms: float
    base_mean_abs_error_ms: float
    engine_mean_abs_error_ms: float
    engine_max_abs_error_ms: float


@dataclass(frozen=True)
class FluidTpotEstimate:
    pred_tpot_ms: float
    dense_ms: float
    decode_attention_ms: float
    prefill_attention_ms: float
    small_kernel_ms: float
    logits_sampling_ms: float
    runtime_overhead_ms: float
    recompute_or_preemption_ms: float
    total_step_ms: float
    pooled_itl_ms: float
    prefill_attention_work_units: float


class NsysCompiledPrefillModel:
    """Prefill non-attention cost from nsys-derived compiled breakdown.

    Reads ``profile_data/kernels/prefill_compiled_breakdown_H100.csv`` (one row per N
    with ``gemm_compiled_ms``, ``elementwise_ms``, ``kv_write_ms``, ``other_ms``).
    Returns the sum (= entire non-attention prefill cost).  FA3 attention is
    handled by the existing ``forward_attention_model`` path — do not add
    ``small_kernel_ms`` on top, the dense already absorbs elementwise/KV/other.
    """

    _NON_ATTENTION_COMPONENTS = (
        "gemm_compiled_ms",
        "elementwise_ms",
        "kv_write_ms",
        "other_ms",
    )

    def __init__(self, samples: Sequence[tuple[int, float]]):
        if not samples:
            raise ValueError("NsysCompiledPrefillModel requires at least one sample")
        self._samples = _dedupe_samples(samples)
        self._min_tokens = self._samples[0][0]
        self._max_tokens = self._samples[-1][0]

    @classmethod
    def from_csv(cls, path: Path) -> "NsysCompiledPrefillModel":
        samples: list[tuple[int, float]] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    n = int(row["prefill_tokens"])
                except (KeyError, ValueError):
                    continue
                total = 0.0
                for key in cls._NON_ATTENTION_COMPONENTS:
                    try:
                        total += float(row[key])
                    except (KeyError, ValueError):
                        pass
                samples.append((n, total))
        return cls(samples)

    def predict_non_attention_ms(self, tokens: int) -> float:
        return _log_linear_predict(self._samples, max(1, int(tokens)))


class Fa3FullCausalAttentionModel:
    """FA3 full-causal attention: flash_full_model_ms from fa3_prefill_H100.csv.

    This is the vLLM-native FA3 kernel cost for q=kv=N, 32 layers.
    Interpolates in log2(N) space.  Use this for full prefill where q≈kv.
    For q≪kv shapes (cached prefill, decode), use the 2D serving lattice instead.
    """

    def __init__(self, samples: Sequence[tuple[int, float]]):
        if not samples:
            raise ValueError("FA3 attention model requires at least one sample")
        self._samples = _dedupe_samples(samples)

    @classmethod
    def from_csv(cls, path: Path, *, min_tokens: int = 64) -> "Fa3FullCausalAttentionModel":
        points: list[tuple[int, float]] = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                n = int(row["prefill_tokens"])
                if n >= min_tokens:
                    points.append((n, float(row["flash_full_model_ms"])))
        return cls(points)

    def predict_ms(self, tokens: int) -> float:
        return _log_linear_predict(self._samples, max(1, int(tokens)))


class ForwardCurve(Protocol):
    def predict_ms(self, tokens: int) -> float:
        ...


class ForwardPassGemmModel:
    def __init__(self, samples: Sequence[tuple[int, float]]):
        if not samples:
            raise ValueError("forward GEMM model requires at least one sample")
        self._samples = _dedupe_samples(samples)

    @classmethod
    def from_csv(cls, path: Path, *, num_layers: int = NUM_LAYERS) -> "ForwardPassGemmModel":
        grouped: dict[int, float] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                tokens = int(float(row["M"]))
                grouped[tokens] = grouped.get(tokens, 0.0) + float(row["latency_us"])
        return cls([
            (tokens, latency_us * num_layers / 1000.0)
            for tokens, latency_us in grouped.items()
        ])

    def predict_ms(self, tokens: int) -> float:
        return _log_linear_predict(self._samples, max(1, int(tokens)))


class ForwardPassAttentionModel:
    def __init__(self, samples: Sequence[tuple[int, float]]):
        if not samples:
            raise ValueError("forward attention model requires at least one sample")
        self._samples = _dedupe_samples(samples)
        self._lattice: dict[tuple[int, int], float] = {}

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        num_layers: int = NUM_LAYERS,
    ) -> "ForwardPassAttentionModel":
        samples: list[tuple[int, float]] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                q_len = int(float(row["q_len"]))
                kv_len = int(float(row["kv_len"]))
                if q_len != kv_len:
                    continue
                samples.append((q_len, float(row["latency_us"]) * num_layers / 1000.0))
        return cls(samples)

    @classmethod
    def from_serving_csv(
        cls,
        path: Path,
        *,
        num_layers: int = NUM_LAYERS,
    ) -> "ForwardPassAttentionModel":
        """Load a 2D (q_len, kv_len) latency lattice, filtering bad rows.

        Skips: q==kv rows (FA3 model handles these), the (1,32768) outlier,
        and rows where latency drops >20% as kv increases for fixed q.
        """
        raw: dict[tuple[int, int], float] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                q_len = int(float(row["q_len"]))
                kv_len = int(float(row["kv_len"]))
                us = float(row["latency_us"])
                key = (q_len, kv_len)
                raw[key] = us * num_layers / 1000.0

        # Filter
        lattice: dict[tuple[int, int], float] = {}
        q_kv_pairs: dict[int, list[tuple[int, float]]] = {}
        for (q, kv), ms in raw.items():
            if q == kv:
                continue  # FA3 model handles full-causal
            if q == 1 and kv >= 32768:
                continue  # physically impossible outlier
            q_kv_pairs.setdefault(q, []).append((kv, ms))

        for q, pairs in q_kv_pairs.items():
            pairs.sort()
            # Filter: latency must increase with kv (within 20% tolerance)
            filtered: list[tuple[int, float]] = []
            max_seen = 0.0
            for kv, ms in pairs:
                if ms < max_seen * 0.8:
                    continue  # implausible drop
                max_seen = max(max_seen, ms)
                filtered.append((kv, ms))
            for kv, ms in filtered:
                lattice[(q, kv)] = ms

        model = cls([(1, 1.0)])  # dummy — only lattice is used
        model._lattice = lattice
        return model

    def full_causal_ms(self, tokens: int) -> float:
        return _log_linear_predict(self._samples, max(1, int(tokens)))

    def _lookup_ms(self, q_len: int, kv_len: int) -> float | None:
        """2D log-linear interpolation over the (q_len, kv_len) lattice.

        Returns None when a corner vertex is missing (signals fallback needed).
        """
        if not self._lattice:
            return None
        if (q_len, kv_len) in self._lattice:
            return self._lattice[(q_len, kv_len)]
        qs = sorted(set(q for q, _ in self._lattice))
        kvs = sorted(set(kv for _, kv in self._lattice))
        q_lo, q_hi = _bracket(qs, q_len)
        kv_lo, kv_hi = _bracket(kvs, kv_len)

        def _log_ms(q: int, kv: int) -> float | None:
            if (q, kv) not in self._lattice:
                return None
            return math.log(max(0.001, self._lattice[(q, kv)]))

        logs = [
            _log_ms(q_lo, kv_lo),
            _log_ms(q_hi, kv_lo),
            _log_ms(q_lo, kv_hi),
            _log_ms(q_hi, kv_hi),
        ]
        if any(v is None for v in logs):
            return None  # missing corner — caller should fall back

        log_q_lo = math.log(max(1, q_len))
        log_q_hi = math.log(max(1, q_hi)) if q_hi > q_lo else log_q_lo + 1
        log_kv_lo = math.log(max(1, kv_len))
        log_kv_hi = math.log(max(1, kv_hi)) if kv_hi > kv_lo else log_kv_lo + 1
        f_q = 0.0 if q_hi <= q_lo else (log_q_lo - math.log(max(1, q_lo))) / (math.log(max(1, q_hi)) - math.log(max(1, q_lo)))
        f_kv = 0.0 if kv_hi <= kv_lo else (log_kv_lo - math.log(max(1, kv_lo))) / (math.log(max(1, kv_hi)) - math.log(max(1, kv_lo)))
        f_q = max(0.0, min(1.0, f_q))
        f_kv = max(0.0, min(1.0, f_kv))
        log_val = (
            (1 - f_q) * (1 - f_kv) * logs[0]  # type: ignore[operator]
            + f_q * (1 - f_kv) * logs[1]      # type: ignore[operator]
            + (1 - f_q) * f_kv * logs[2]      # type: ignore[operator]
            + f_q * f_kv * logs[3]            # type: ignore[operator]
        )
        return max(0.001, math.exp(log_val))

    def set_fa3_model(self, model: "Fa3FullCausalAttentionModel | None") -> None:
        self._fa3 = model

    def _fa3_workratio_ms(self, q: int, prefix: int, kv_len: int) -> float:
        """FA3 full-causal attention scaled by causal work fraction."""
        fa3_full = self._fa3.predict_ms(kv_len)
        full_causal_work = kv_len * (kv_len + 1) / 2.0
        chunk_work = q * prefix + q * (q + 1) / 2.0
        return fa3_full * chunk_work / full_causal_work

    def _blend_weight(self, ratio: float) -> float:
        """0.0 = pure lattice (q<<kv), 1.0 = pure FA3 (q~kv).

        Linear ramp from ratio=0.0625 (q*16<kv) to ratio=0.25 (q*4>kv).
        Center at 0.125 (q*8=kv) matches the old hard boundary.
        """
        lo, hi = 0.0625, 0.25
        if ratio <= lo:
            return 0.0
        if ratio >= hi:
            return 1.0
        return (ratio - lo) / (hi - lo)

    def predict_chunks_ms(self, chunks: Sequence[VllmPrefillChunk]) -> float:
        total = 0.0
        for chunk in chunks:
            q = max(1, int(chunk.scheduled_tokens))
            prefix = max(0, int(chunk.prefix_tokens))
            kv_len = max(1, prefix + q)

            # Try both models
            lattice_val: float | None = None
            if self._lattice:
                lattice_val = self._lookup_ms(q, kv_len)

            fa3_val: float | None = None
            if getattr(self, "_fa3", None) is not None:
                fa3_val = self._fa3_workratio_ms(q, prefix, kv_len)

            if lattice_val is not None and fa3_val is not None:
                # Smooth blend based on q/kv ratio
                ratio = q / max(1, kv_len)
                alpha = self._blend_weight(ratio)
                total += (1.0 - alpha) * lattice_val + alpha * fa3_val
            elif fa3_val is not None:
                total += fa3_val
            elif lattice_val is not None:
                total += lattice_val
            else:
                # Fallback: NCU full-causal scaled by work ratio
                full_causal_work = kv_len * (kv_len + 1) / 2.0
                chunk_work = q * prefix + q * (q + 1) / 2.0
                total += self.full_causal_ms(kv_len) * chunk_work / full_causal_work
        return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--benchmark-turns", type=Path, default=DEFAULT_BENCHMARK_TURNS)
    parser.add_argument(
        "--benchmark-per-request-json",
        type=Path,
        default=DEFAULT_BENCHMARK_PER_REQUEST,
        help=(
            "JSON sidecar with per-(profile, c, turn) request-level "
            "output_tokens / new_prefill_tokens / cached_context_tokens "
            "lists.  When present, the simulator gets the actual per-request "
            "distribution instead of a single per-turn mean.  Missing file "
            "is fine — predictor falls back to scalar means."
        ),
    )
    parser.add_argument(
        "--per-request-fields",
        default="all",
        help=(
            "Which fields to plumb per-request from the sidecar. Comma list "
            "of {output_tokens, new_prefill_tokens, cached_context_tokens} "
            "or one of {prefill, all, none}.  Default 'all' uses every "
            "per-request field — best step-count match where same-run "
            "paired bench JSONs exist.  'prefill' keeps OSL uniform "
            "(useful when paired data is unavailable and the bench OSL "
            "distribution doesn't match the trace window).  'none' "
            "disables the sidecar entirely."
        ),
    )
    parser.add_argument("--attention-profile", type=Path, default=DEFAULT_ATTENTION_PROFILE)
    parser.add_argument("--gemm-summary", type=Path, default=DEFAULT_GEMM_SUMMARY)
    parser.add_argument("--small-kernel-summary", type=Path, default=DEFAULT_SMALL_KERNEL_SUMMARY)
    parser.add_argument("--forward-gemm", type=Path, default=DEFAULT_FORWARD_GEMM)
    parser.add_argument("--forward-attention", type=Path, default=DEFAULT_FORWARD_ATTENTION)
    parser.add_argument("--component-model", choices=("xgboost", "log_interp"), default="xgboost")
    parser.add_argument(
        "--prefill-mode",
        choices=("benchmark_cache", "synthetic_shared_prefix"),
        default="benchmark_cache",
    )
    parser.add_argument(
        "--dense-source",
        choices=("decode_components", "forward_pass", "nsys_compiled"),
        default="nsys_compiled",
        help=(
            "Prefill dense kernel source.  Default nsys_compiled reads the "
            "nsys-derived per-N breakdown (GEMM + elementwise + KV_write + "
            "other) and skips the small_kernel adder for prefill steps "
            "(already absorbed).  forward_pass uses isolated cuBLAS GEMM "
            "(overestimates at large N).  decode_components uses NCU decode GEMM."
        ),
    )
    parser.add_argument(
        "--nsys-compiled-csv",
        type=Path,
        default=DEFAULT_NSYS_COMPILED,
        help="CSV produced by profiling/process/extractors/extract_nsys_prefill_breakdown.py.",
    )
    parser.add_argument(
        "--wave-policy",
        choices=("none", "decode_residency"),
        default="none",
        help=(
            "Optional no-fit diagnostic. decode_residency scales only "
            "decode-side step components by scheduled_request_count / "
            "simulated max decode batch."
        ),
    )
    parser.add_argument(
        "--scheduler-model",
        choices=("exact_shape", "fluid_wave"),
        default="exact_shape",
        help=(
            "exact_shape replays the aggregate turn simulator. fluid_wave uses "
            "a no-fit decode-wave/prefill-debt approximation and emits cache "
            "envelope diagnostics without exact queue-order matching."
        ),
    )
    parser.add_argument("--shared-prefix-tokens", type=int, default=0)
    parser.add_argument("--bos-token-offset", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16_384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=16_384)
    parser.add_argument("--cache-block-size", type=int, default=16)
    parser.add_argument("--available-gpu-kv-blocks", type=int, default=27_651)
    # 3D roofline (bandwidth) inputs.  Zero on any of these disables the
    # bandwidth roofline and preserves the legacy 2D capacity-only shape.
    # H100 effective HBM bandwidth ~3 TB/s; defaults below assume Llama-3.1-8B
    # (32 layers, 8 KV heads, head_dim=128, fp16): 2 * 32 * 8 * 128 * 2 = 131072
    # bytes per token.
    parser.add_argument("--hbm-bw-gbps", type=float, default=0.0)
    parser.add_argument("--kv-bytes-per-token", type=float, default=131072.0)
    parser.add_argument("--step-budget-ms", type=float, default=0.0)
    parser.add_argument(
        "--bandwidth-caps-decode",
        action="store_true",
        default=True,
        help="Split decode into bandwidth-feasible waves.",
    )
    parser.add_argument(
        "--no-bandwidth-caps-decode",
        dest="bandwidth_caps_decode",
        action="store_false",
    )
    parser.add_argument(
        "--bandwidth-caps-prefill",
        action="store_true",
        default=True,
        help="Cap prefill chunk size by remaining bandwidth budget.",
    )
    parser.add_argument(
        "--no-bandwidth-caps-prefill",
        dest="bandwidth_caps_prefill",
        action="store_false",
    )
    parser.add_argument(
        "--bandwidth-decode-priority",
        action="store_true",
        default=True,
        help="Defer prefill to next step when decode saturates bandwidth.",
    )
    parser.add_argument(
        "--no-bandwidth-decode-priority",
        dest="bandwidth_decode_priority",
        action="store_false",
    )
    parser.add_argument(
        "--use-block-pool",
        action="store_true",
        default=False,
        help=(
            "Use the per-step staircase admission scheduler with a simulated "
            "BlockPool.  Produces the mixed/intrusion steps that real vLLM "
            "exhibits.  Default off until the migration completes."
        ),
    )
    parser.add_argument(
        "--max-admissions-per-step",
        type=int,
        default=0,
        help=(
            "When --use-block-pool, cap new admissions per scheduler step. "
            "0 (default) = unlimited (matches vLLM v1, which admits as many "
            "as fit in the token budget + KV pool)."
        ),
    )
    parser.add_argument(
        "--use-step-predictor",
        action="store_true",
        default=False,
        help=(
            "Price scheduler steps via the validated full_prefill/cached_prefill/"
            "decode step-level predictors instead of kernel composition.  Mixed "
            "steps use max(prefill_total, decode_total) per the GPU overlap "
            "physics.  Opt-in; defaults preserve the legacy kernel-composition "
            "pricer."
        ),
    )
    parser.add_argument(
        "--full-prefill-csv", type=Path, default=DEFAULT_FULL_PREFILL_CSV,
    )
    parser.add_argument(
        "--cached-prefill-csv", type=Path, default=DEFAULT_CACHED_PREFILL_CSV,
    )
    parser.add_argument(
        "--decode-profile-csv", type=Path, default=DEFAULT_DECODE_PROFILE_CSV,
    )
    parser.add_argument(
        "--use-roofline-predictor",
        action="store_true",
        default=False,
        help=(
            "Price scheduler steps via pure analytical 3D roofline (compute / "
            "bandwidth max).  No ML, no fitting.  Two GPU-spec utilization "
            "factors loaded from --roofline-params-json.  Overrides "
            "--use-step-predictor when both are set."
        ),
    )
    parser.add_argument(
        "--roofline-params-json", type=Path, default=DEFAULT_ROOFLINE_PARAMS,
    )
    parser.add_argument(
        "--closed-form",
        action="store_true",
        default=False,
        help=(
            "Bypass scheduler-shape simulation entirely.  Predict TPOT/TTFT/e2e "
            "directly from aggregate per-turn workload (mean output/prefill/cached "
            "tokens at concurrency c) under the same analytical roofline params. "
            "Drops the per-step admission loop, block pool, multi-turn replay. "
            "Use --closed-form-params-json to override; defaults to --roofline-params-json."
        ),
    )
    parser.add_argument(
        "--closed-form-params-json",
        type=Path,
        default=None,
        help=(
            "Roofline params JSON for the closed-form predictor.  Defaults to "
            "--roofline-params-json if unset."
        ),
    )
    parser.add_argument(
        "--kv-pressure-wave",
        action="store_true",
        default=False,
        help=(
            "Enable the KV-capacity wave-factor amplification inside the "
            "closed-form predictor.  Helps per-session-prefix workloads "
            "(swebench) where total KV exceeds GPU capacity, but mis-applied "
            "to shared-prefix workloads (osworld) it over-predicts.  Off by "
            "default until a reliable cross-cohort prefix discriminator lands."
        ),
    )
    parser.add_argument(
        "--effective-cache-policy",
        choices=("none", "vllm_residency"),
        default="none",
        help=(
            "Optional no-fit cache truth model. vllm_residency replaces "
            "benchmark cached_context/new_prefill with an estimate of the "
            "prefix tokens that vLLM's KVCacheManager can actually find under "
            "the configured GPU KV block budget."
        ),
    )
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--include-single-turn", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dashboard-output", type=Path, default=DEFAULT_DASHBOARD_OUTPUT)
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Write CSV/report only. Use while validating new predictor variants.",
    )
    return parser.parse_args()


_PER_REQUEST_FIELD_ALIASES: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "all": frozenset({"output_tokens", "new_prefill_tokens", "cached_context_tokens"}),
    "prefill": frozenset({"new_prefill_tokens", "cached_context_tokens"}),
}
_PER_REQUEST_FIELD_NAMES = frozenset(
    {"output_tokens", "new_prefill_tokens", "cached_context_tokens"}
)


def _parse_per_request_fields(spec: str) -> frozenset[str]:
    spec = (spec or "").strip().lower()
    if spec in _PER_REQUEST_FIELD_ALIASES:
        return _PER_REQUEST_FIELD_ALIASES[spec]
    out = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in _PER_REQUEST_FIELD_NAMES:
            raise ValueError(
                f"unknown per-request field {token!r}; "
                f"expected one of {sorted(_PER_REQUEST_FIELD_NAMES)} "
                f"or alias in {sorted(_PER_REQUEST_FIELD_ALIASES)}"
            )
        out.add(token)
    return frozenset(out)


def build_engine_step_prediction_rows(
    prediction_rows: Iterable[dict[str, str]],
    *,
    benchmark_turns: dict[tuple[str, int, int], dict[str, str]],
    component_models: EngineStepComponentModels,
    component_model: str = "xgboost",
    prefill_mode: PrefillMode = "benchmark_cache",
    dense_source: str = "forward_pass",
    scheduler_model: SchedulerModel = "exact_shape",
    wave_policy: WavePolicy = "none",
    effective_cache_policy: EffectiveCachePolicy = "none",
    config: VllmSchedulerConfig | None = None,
    include_diagnostic: bool = True,
    include_single_turn: bool = False,
    step_predictor: CleanStepPredictor | None = None,
    roofline_predictor: RooflineStepPredictor | None = None,
    per_request_by_key: dict[tuple[str, int, int], dict[str, list[int]]] | None = None,
    per_request_fields: frozenset[str] = frozenset(),
    session_timelines: dict[tuple[str, int], list[list[dict]]] | None = None,
) -> list[EngineStepPredictionRow]:
    cfg = config or VllmSchedulerConfig()
    per_request_by_key = per_request_by_key or {}
    per_request_used = 0
    per_request_total = 0
    session_timelines = session_timelines or {}

    # Precompute multi-turn wall-clock replay results per (profile, c).
    # The replay needs a per-step pricer; we use the roofline pricer when
    # available (it gives a clean analytical step_ms with no fitting).
    replay_by_key: dict[tuple[str, int], MultiTurnReplayResult] = {}
    replay_used = 0
    if session_timelines and roofline_predictor is not None and cfg.use_block_pool:
        def _step_pricer(step, context_lens):
            return float(roofline_predictor.step_ms(step, context_lens).total_ms)
        for (profile_key, conc_key), sessions in session_timelines.items():
            if not sessions:
                continue
            try:
                replay_by_key[(profile_key, conc_key)] = (
                    simulate_vllm_multi_turn_replay(
                        sessions, cfg, step_pricer=_step_pricer
                    )
                )
            except Exception as exc:
                print(
                    f"Multi-turn replay failed for {profile_key} c={conc_key}: {exc}; "
                    f"falling back to single-turn shape for those rows."
                )
        print(
            f"Precomputed multi-turn replays for {len(replay_by_key)} (profile, c) groups"
        )

    rows: list[EngineStepPredictionRow] = []
    for row in prediction_rows:
        if row["component_model"] != component_model:
            continue
        if not include_diagnostic and row.get("primary_eval") != "true":
            continue
        if not include_single_turn and _is_single_turn_profile(row["profile"]):
            continue

        profile = row["profile"]
        concurrency = _to_int(row.get("concurrency"), 1)
        turn_index = _to_int(row.get("turn_index"), 0)
        metadata = benchmark_turns.get((profile, concurrency, turn_index), {})
        context_len = max(1, _to_int(metadata.get("context_len") or row.get("context_len"), 1))
        batch_size = max(1, _to_int(row.get("batch_size"), 1))
        scheduled = max(
            1,
            _to_int(
                metadata.get("scheduled_request_count"),
                _to_int(metadata.get("turn_num_requests"), batch_size),
            ),
        )
        successful = max(
            1,
            _to_int(metadata.get("successful_request_count"), batch_size),
        )
        turn_num_requests = max(
            1,
            _to_int(metadata.get("turn_num_requests"), scheduled),
        )
        output_tokens = max(1, _to_int(metadata.get("output_tokens"), 1))
        new_prefill = max(0, _to_int(metadata.get("new_prefill_tokens"), context_len))
        cached_context = max(0, _to_int(metadata.get("cached_context_tokens"), 0))
        cache_hit_rate = _to_float(metadata.get("cache_hit_rate"), 0.0)
        cache_estimate = _effective_cache_estimate(
            request_count=scheduled,
            context_len=context_len,
            cached_context_tokens=cached_context,
            new_prefill_tokens=new_prefill,
            prefill_mode=prefill_mode,
            effective_cache_policy=effective_cache_policy,
            config=cfg,
        )
        sim_cached_context = cache_estimate["sim_cached_context_tokens"]
        sim_new_prefill = cache_estimate["sim_new_prefill_tokens"]
        base_pred = float(row["pred_tpot_ms"])
        measured = float(row["measured_tpot_ms"])
        base_signed_error = float(row.get("signed_error_ms") or (base_pred - measured))
        cost_model = _build_cost_model(
            component_models,
            context_len=context_len,
            dense_source=dense_source,
        )
        if scheduler_model == "fluid_wave":
            rows.append(
                _build_fluid_prediction_row(
                    source_row=row,
                    component_model=component_model,
                    prefill_mode=prefill_mode,
                    dense_source=dense_source,
                    wave_policy=wave_policy,
                    effective_cache_policy=effective_cache_policy,
                    profile=profile,
                    concurrency=concurrency,
                    turn_index=turn_index,
                    batch_size=batch_size,
                    scheduled=scheduled,
                    successful=successful,
                    turn_num_requests=turn_num_requests,
                    context_len=context_len,
                    output_tokens=output_tokens,
                    new_prefill=new_prefill,
                    cached_context=cached_context,
                    cache_hit_rate=cache_hit_rate,
                    measured=measured,
                    base_pred=base_pred,
                    base_signed_error=base_signed_error,
                    diagnostic_reason=(
                        metadata.get("diagnostic_reason")
                        or row.get("diagnostic_reason", "")
                    ),
                    cost_model=cost_model,
                    config=cfg,
                )
            )
            continue
        per_request_total += 1
        per_req = per_request_by_key.get((profile, concurrency, turn_index))
        osl_list = None
        new_prefill_list = None
        cached_list = None
        if (
            per_req is not None
            and per_req.get("output_tokens")
            and per_request_fields
        ):
            per_request_used += 1
            if "output_tokens" in per_request_fields:
                osl_list = per_req["output_tokens"]
            if "new_prefill_tokens" in per_request_fields:
                new_prefill_list = per_req.get("new_prefill_tokens") or None
            if "cached_context_tokens" in per_request_fields:
                cached_list = per_req.get("cached_context_tokens") or None
        # Multi-turn continuity: when an original request finishes its
        # turn-N decode, the single-turn sim spawns a turn-(N+1) follow-up
        # so that turn-boundary-crossing mixed steps show up.  When a
        # wall-clock multi-turn replay is precomputed for this (profile, c),
        # we skip the followup-spawn here — the replay already models
        # inter-turn arrivals from the bench's per-session timing data, so
        # spawning followups in the single-turn sim too would double-count.
        next_turn_prefill = (
            0
            if (profile, concurrency) in replay_by_key
            else max(0, int(sim_new_prefill))
        )
        requests = benchmark_row_requests(
            request_count=scheduled,
            context_len=context_len,
            output_tokens=output_tokens,
            new_prefill_tokens=sim_new_prefill,
            cached_context_tokens=sim_cached_context,
            mode=prefill_mode,
            shared_prefix_tokens=cfg.shared_prefix_tokens,
            bos_token_offset=cfg.bos_token_offset,
            output_tokens_per_request=osl_list,
            new_prefill_tokens_per_request=new_prefill_list,
            cached_context_tokens_per_request=cached_list,
            next_turn_new_prefill_tokens=next_turn_prefill,
        )
        shape = simulate_vllm_turn_shape(requests, cfg)
        # Multi-turn replay overlay: when we have a wall-clock replay for
        # this (profile, c), overwrite the step-count summary with the
        # turn-bucketed view from the replay.  TPOT pricing still uses the
        # single-turn shape (the replay's per-request emission timing isn't
        # plumbed yet — out of scope for this iteration).
        replay = replay_by_key.get((profile, concurrency))
        if replay is not None:
            replay_summary = summarize_replay_for_turn(replay, turn_index)
            if replay_summary.steps > 0:
                replay_used += 1
                shape = dataclass_replace(shape, summary=replay_summary)
        if roofline_predictor is not None:
            timing = predict_mean_tpot_from_roofline(
                shape.steps,
                context_len=context_len,
                predictor=roofline_predictor,
            )
        elif step_predictor is not None:
            timing = predict_mean_tpot_from_predictor(
                shape.steps,
                context_len=context_len,
                predictor=step_predictor,
            )
        else:
            timing = predict_mean_tpot_from_engine_steps(
                shape.steps,
                context_len=context_len,
                cost_model=cost_model,
            )
        wave_factor = _decode_residency_wave_factor(
            scheduled_request_count=scheduled,
            max_decode_batch=shape.summary.max_decode_batch,
        )
        if wave_policy == "decode_residency":
            adjusted = _predict_mean_tpot_with_decode_residency_wave(
                shape.steps,
                context_len=context_len,
                cost_model=cost_model,
                wave_factor=wave_factor,
            )
            component_sums = adjusted["components"]
            pred = float(adjusted["mean_tpot_ms"])
            pooled_itl_ms = float(adjusted["pooled_itl_ms"])
            total_step_ms = float(adjusted["total_step_ms"])
        else:
            component_sums = _sum_step_components(timing.step_costs)
            pred = timing.mean_tpot_ms
            pooled_itl_ms = timing.pooled_itl_ms
            total_step_ms = timing.total_step_ms
        signed_error = pred - measured
        rows.append(EngineStepPredictionRow(
            component_model=component_model,
            engine_step_model=ENGINE_STEP_MODEL,
            scheduler_model=scheduler_model,
            prefill_mode=prefill_mode,
            dense_source=dense_source,
            wave_policy=wave_policy,
            profile=profile,
            concurrency=concurrency,
            turn_index=turn_index,
            primary_eval=row.get("primary_eval") == "true",
            batch_size=batch_size,
            scheduled_request_count=scheduled,
            successful_request_count=successful,
            turn_num_requests=turn_num_requests,
            context_len=context_len,
            output_tokens=output_tokens,
            new_prefill_tokens=new_prefill,
            cached_context_tokens=cached_context,
            cache_hit_rate=cache_hit_rate,
            effective_cache_policy=effective_cache_policy,
            sim_new_prefill_tokens=sim_new_prefill,
            sim_cached_context_tokens=sim_cached_context,
            engine_effective_cached_tokens=cache_estimate[
                "engine_effective_cached_tokens"
            ],
            capacity_feasible_cached_tokens=cache_estimate[
                "capacity_feasible_cached_tokens"
            ],
            engine_uncached_prefill_tokens=cache_estimate[
                "engine_uncached_prefill_tokens"
            ],
            cache_residency_classification=cache_estimate[
                "cache_residency_classification"
            ],
            measured_tpot_ms=measured,
            base_pred_tpot_ms=base_pred,
            pred_tpot_ms=pred,
            base_pct_error=abs(float(row["pct_error"])),
            pct_error=percent_error(measured, pred),
            base_signed_error_ms=base_signed_error,
            signed_error_ms=signed_error,
            base_abs_error_ms=abs(base_signed_error),
            abs_error_ms=abs(signed_error),
            sim_steps=shape.summary.steps,
            sim_decode_only_steps=shape.summary.decode_only_steps,
            sim_prefill_only_steps=shape.summary.prefill_only_steps,
            sim_mixed_decode_prefill_steps=shape.summary.mixed_decode_prefill_steps,
            sim_total_decode_slots=shape.summary.total_decode_slots,
            sim_total_prefill_tokens=shape.summary.total_prefill_tokens,
            sim_max_decode_batch=shape.summary.max_decode_batch,
            sim_mean_decode_batch=shape.summary.mean_decode_batch,
            sim_mean_decode_batch_ratio=shape.summary.mean_decode_batch_ratio,
            sim_min_free_kv_blocks=shape.summary.min_free_kv_blocks,
            sim_max_used_kv_blocks=shape.summary.max_used_kv_blocks,
            sim_max_capacity_waiting_requests=shape.summary.max_capacity_waiting_requests,
            sim_total_preemptions=shape.summary.total_preemptions,
            sim_total_recompute_prefill_tokens=shape.summary.total_recompute_prefill_tokens,
            decode_residency_wave_factor=wave_factor,
            total_step_ms=total_step_ms,
            pooled_itl_ms=pooled_itl_ms,
            decoded_request_count=timing.decoded_request_count,
            dense_ms=component_sums["dense_ms"],
            decode_attention_ms=component_sums["decode_attention_ms"],
            prefill_attention_ms=component_sums["prefill_attention_ms"],
            small_kernel_ms=component_sums["small_kernel_ms"],
            logits_sampling_ms=component_sums["logits_sampling_ms"],
            runtime_overhead_ms=component_sums["runtime_overhead_ms"],
            recompute_or_preemption_ms=component_sums["recompute_or_preemption_ms"],
            prefill_attention_work_units=sum(
                prefill_attention_work(step.prefill_chunks)
                for step in shape.steps
            ),
            fluid_pred_low_tpot_ms=pred,
            fluid_pred_mid_tpot_ms=pred,
            fluid_pred_high_tpot_ms=pred,
            fluid_active_decode_capacity=shape.summary.max_decode_batch,
            fluid_decode_steps=shape.summary.decode_only_steps
            + shape.summary.mixed_decode_prefill_steps,
            fluid_decode_waves=math.ceil(
                scheduled / max(1, shape.summary.max_decode_batch)
            ),
            fluid_prefill_only_steps=shape.summary.prefill_only_steps,
            fluid_mixed_decode_prefill_steps=shape.summary.mixed_decode_prefill_steps,
            fluid_kv_pressure=(
                shape.summary.max_used_kv_blocks
                / max(1, cfg.available_gpu_kv_blocks)
            ),
            fluid_prefill_debt_tokens=shape.summary.total_prefill_tokens,
            fluid_prefill_debt_per_decode_token=(
                shape.summary.total_prefill_tokens
                / max(1, shape.summary.total_decode_slots)
            ),
            fluid_cache_optimistic_tokens=sim_cached_context,
            fluid_cache_mid_tokens=sim_cached_context,
            fluid_cache_pessimistic_tokens=sim_cached_context,
            fluid_scheduler_sensitivity="exact_shape",
            diagnostic_reason=(
                metadata.get("diagnostic_reason") or row.get("diagnostic_reason", "")
            ),
        ))
    if per_request_total > 0:
        print(
            f"Used per-request OSL distribution for {per_request_used} of "
            f"{per_request_total} rows ({per_request_used / per_request_total * 100:.1f}%)"
        )
    if replay_by_key:
        print(
            f"Used multi-turn replay for {replay_used} rows (rest fell back "
            f"to single-turn shape)"
        )
    return sorted(rows, key=lambda item: (item.profile, item.concurrency, item.turn_index))


def build_closed_form_prediction_rows(
    prediction_rows: Iterable[dict[str, str]],
    benchmark_turns: dict[tuple[str, int, int], dict[str, str]],
    closed_form_predictor: ClosedFormTpotPredictor,
    *,
    component_model: str = "xgboost",
    per_request_by_key: dict[tuple[str, int, int], dict[str, list[int]]] | None = None,
    per_request_fields: frozenset[str] = frozenset(),
    shared_prefix_by_key: dict[tuple[str, int], float] | None = None,
    include_diagnostic: bool = True,
    include_single_turn: bool = False,
) -> list[EngineStepPredictionRow]:
    """Predict TPOT directly from per-turn aggregate workload.

    Bypasses the scheduler-shape simulator (block pool, partial-prefill cap,
    multi-turn replay).  Joins ``prediction_rows`` (for measured_tpot_ms and
    base_pred_tpot_ms ground truth) with ``benchmark_turns`` metadata, then
    plugs (output_tokens, new_prefill_tokens, cached_context_tokens,
    concurrency) into the closed-form predictor.  ``per_request_*`` lists
    refine the means used for the inputs.

    All ``sim_*`` scheduler-shape fields are zeroed — they have no meaning
    under the closed-form path.  The comparator must tolerate this.
    """
    per_request_by_key = per_request_by_key or {}
    shared_prefix_by_key = shared_prefix_by_key or {}
    per_request_used = 0
    per_request_total = 0

    rows: list[EngineStepPredictionRow] = []
    for row in prediction_rows:
        if row.get("component_model") != component_model:
            continue
        if not include_diagnostic and row.get("primary_eval") != "true":
            continue
        if not include_single_turn and _is_single_turn_profile(row["profile"]):
            continue

        profile = row["profile"]
        concurrency = max(1, _to_int(row.get("concurrency"), 1))
        turn_index = _to_int(row.get("turn_index"), 0)
        metadata = benchmark_turns.get((profile, concurrency, turn_index), {})
        shared_prefix_fraction = float(
            shared_prefix_by_key.get((profile, concurrency), 0.0)
        )
        context_len = max(1, _to_int(metadata.get("context_len") or row.get("context_len"), 1))
        batch_size = max(1, _to_int(row.get("batch_size"), 1))
        scheduled = max(
            1,
            _to_int(
                metadata.get("scheduled_request_count"),
                _to_int(metadata.get("turn_num_requests"), batch_size),
            ),
        )
        successful = max(1, _to_int(metadata.get("successful_request_count"), batch_size))
        turn_num_requests = max(1, _to_int(metadata.get("turn_num_requests"), scheduled))
        output_tokens = max(1, _to_int(metadata.get("output_tokens"), 1))
        new_prefill = max(0, _to_int(metadata.get("new_prefill_tokens"), context_len))
        cached_context = max(0, _to_int(metadata.get("cached_context_tokens"), 0))
        cache_hit_rate = _to_float(metadata.get("cache_hit_rate"), 0.0)

        # Optional per-request refinement of the means.
        per_request_total += 1
        per_req = per_request_by_key.get((profile, concurrency, turn_index))
        mean_output = float(output_tokens)
        mean_new_prefill = float(new_prefill)
        mean_cached = float(cached_context)
        if per_req and per_request_fields:
            per_request_used += 1
            if "output_tokens" in per_request_fields and per_req.get("output_tokens"):
                vals = per_req["output_tokens"]
                mean_output = sum(vals) / len(vals)
            if "new_prefill_tokens" in per_request_fields and per_req.get("new_prefill_tokens"):
                vals = per_req["new_prefill_tokens"]
                mean_new_prefill = sum(vals) / len(vals)
            if "cached_context_tokens" in per_request_fields and per_req.get(
                "cached_context_tokens"
            ):
                vals = per_req["cached_context_tokens"]
                mean_cached = sum(vals) / len(vals)

        cf_input = ClosedFormTurnInput(
            profile=profile,
            concurrency=concurrency,
            turn_index=turn_index,
            output_tokens=mean_output,
            new_prefill_tokens=mean_new_prefill,
            cached_context_tokens=mean_cached,
            shared_prefix_fraction=shared_prefix_fraction,
        )
        cf_out = closed_form_predictor.predict(cf_input)

        measured = float(row["measured_tpot_ms"])
        base_pred = float(row["pred_tpot_ms"])
        base_signed_error = float(row.get("signed_error_ms") or (base_pred - measured))
        pred = float(cf_out.tpot_ms)
        signed_error = pred - measured

        rows.append(EngineStepPredictionRow(
            component_model=component_model,
            engine_step_model="closed_form_roofline",
            scheduler_model="closed_form",
            prefill_mode="benchmark_cache",
            dense_source="closed_form",
            wave_policy="none",
            profile=profile,
            concurrency=concurrency,
            turn_index=turn_index,
            primary_eval=row.get("primary_eval") == "true",
            batch_size=batch_size,
            scheduled_request_count=scheduled,
            successful_request_count=successful,
            turn_num_requests=turn_num_requests,
            context_len=context_len,
            output_tokens=int(round(mean_output)),
            new_prefill_tokens=int(round(mean_new_prefill)),
            cached_context_tokens=int(round(mean_cached)),
            cache_hit_rate=cache_hit_rate,
            effective_cache_policy="none",
            sim_new_prefill_tokens=int(round(mean_new_prefill)),
            sim_cached_context_tokens=int(round(mean_cached)),
            engine_effective_cached_tokens=mean_cached,
            capacity_feasible_cached_tokens=int(round(mean_cached)),
            engine_uncached_prefill_tokens=max(0, int(round(mean_new_prefill - mean_cached))),
            cache_residency_classification="closed_form",
            measured_tpot_ms=measured,
            base_pred_tpot_ms=base_pred,
            pred_tpot_ms=pred,
            base_pct_error=abs(float(row.get("pct_error") or 0.0)),
            pct_error=percent_error(measured, pred),
            base_signed_error_ms=base_signed_error,
            signed_error_ms=signed_error,
            base_abs_error_ms=abs(base_signed_error),
            abs_error_ms=abs(signed_error),
            # No per-step shape under closed-form — leave the sim_* fields at 0
            # except the ones the KV-pressure model produces analytically.
            sim_steps=0,
            sim_decode_only_steps=0,
            sim_prefill_only_steps=0,
            sim_mixed_decode_prefill_steps=0,
            sim_total_decode_slots=0,
            sim_total_prefill_tokens=0,
            sim_max_decode_batch=int(cf_out.effective_decode_batch),
            sim_mean_decode_batch=float(cf_out.effective_decode_batch),
            sim_mean_decode_batch_ratio=(
                float(cf_out.effective_decode_batch) / float(concurrency)
                if concurrency > 0 else 0.0
            ),
            sim_min_free_kv_blocks=0,
            sim_max_used_kv_blocks=0,
            sim_max_capacity_waiting_requests=0,
            sim_total_preemptions=0,
            sim_total_recompute_prefill_tokens=0,
            decode_residency_wave_factor=float(cf_out.wave_factor),
            total_step_ms=pred,
            pooled_itl_ms=pred,
            decoded_request_count=concurrency,
            dense_ms=0.0,
            decode_attention_ms=0.0,
            prefill_attention_ms=0.0,
            small_kernel_ms=0.0,
            logits_sampling_ms=0.0,
            runtime_overhead_ms=0.0,
            recompute_or_preemption_ms=0.0,
            prefill_attention_work_units=0.0,
            fluid_pred_low_tpot_ms=pred,
            fluid_pred_mid_tpot_ms=pred,
            fluid_pred_high_tpot_ms=pred,
            fluid_active_decode_capacity=concurrency,
            fluid_decode_steps=int(round(mean_output)),
            fluid_decode_waves=1,
            fluid_prefill_only_steps=0,
            fluid_mixed_decode_prefill_steps=0,
            fluid_kv_pressure=0.0,
            fluid_prefill_debt_tokens=0,
            fluid_prefill_debt_per_decode_token=0.0,
            fluid_cache_optimistic_tokens=int(round(mean_cached)),
            fluid_cache_mid_tokens=int(round(mean_cached)),
            fluid_cache_pessimistic_tokens=int(round(mean_cached)),
            fluid_scheduler_sensitivity="closed_form",
            diagnostic_reason=(
                metadata.get("diagnostic_reason") or row.get("diagnostic_reason", "")
            ),
        ))

    print(
        f"Closed-form predictor: {per_request_used}/{per_request_total} rows "
        f"used per-request distributions"
    )
    return sorted(rows, key=lambda item: (item.profile, item.concurrency, item.turn_index))


def build_default_component_models(
    *,
    component_model: str,
    attention_profile: Path,
    gemm_summary: Path,
    small_kernel_summary: Path,
    forward_gemm: Path,
    forward_attention: Path,
    fa3_prefill: Path = DEFAULT_FA3_PREFILL,
    nsys_compiled_csv: Path = DEFAULT_NSYS_COMPILED,
) -> EngineStepComponentModels:
    attention_models = build_component_models(
        load_attention_components(attention_profile),
        model_name=component_model,
    )
    return EngineStepComponentModels(
        attention_model=attention_models["attention"],
        gemm_models=build_component_models(
            load_gemm_components(gemm_summary),
            model_name=component_model,
        ),
        small_kernel_models=build_component_models(
            load_small_kernel_components(small_kernel_summary),
            model_name=component_model,
        ),
        forward_gemm_model=ForwardPassGemmModel.from_csv(forward_gemm)
        if forward_gemm.exists()
        else None,
        forward_attention_model=_build_attention_model(forward_attention, fa3_path=fa3_prefill),
        nsys_compiled_model=NsysCompiledPrefillModel.from_csv(nsys_compiled_csv)
        if nsys_compiled_csv.exists()
        else None,
    )


def _build_attention_model(
    forward_attention: Path,
    fa3_path: Path = DEFAULT_FA3_PREFILL,
) -> ForwardPassAttentionModel | None:
    """Build attention model preferring 2D serving lattice over full-causal.
    Attaches FA3 full-causal model for q≈kv prefill shapes."""
    serving_path = DEFAULT_SERVING_ATTENTION
    model: ForwardPassAttentionModel | None = None
    if serving_path.exists():
        model = ForwardPassAttentionModel.from_serving_csv(serving_path)
    elif forward_attention.exists():
        model = ForwardPassAttentionModel.from_csv(forward_attention)
    if model is not None and fa3_path.exists():
        model.set_fa3_model(Fa3FullCausalAttentionModel.from_csv(fa3_path))
    return model


def summarize(rows: Sequence[EngineStepPredictionRow]) -> ErrorSummary:
    if not rows:
        nan = math.nan
        return ErrorSummary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    base_errors = [row.base_pct_error for row in rows]
    engine_errors = [row.pct_error for row in rows]
    return ErrorSummary(
        rows=len(rows),
        base_mape=mean(base_errors),
        engine_mape=mean(engine_errors),
        base_median_ape=median(base_errors),
        engine_median_ape=median(engine_errors),
        base_max_ape=max(base_errors),
        engine_max_ape=max(engine_errors),
        base_mean_signed_error_ms=mean(row.base_signed_error_ms for row in rows),
        engine_mean_signed_error_ms=mean(row.signed_error_ms for row in rows),
        base_mean_abs_error_ms=mean(row.base_abs_error_ms for row in rows),
        engine_mean_abs_error_ms=mean(row.abs_error_ms for row in rows),
        engine_max_abs_error_ms=max(row.abs_error_ms for row in rows),
    )


def write_rows(path: Path, rows: Sequence[EngineStepPredictionRow]) -> None:
    fields = list(EngineStepPredictionRow.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: _fmt(getattr(row, field))
                if isinstance(getattr(row, field), float)
                else _bool_text(getattr(row, field))
                if isinstance(getattr(row, field), bool)
                else getattr(row, field)
                for field in fields
            })


def write_report(path: Path, rows: Sequence[EngineStepPredictionRow]) -> None:
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Engine-Step TPOT v0",
        "",
        "## Scope",
        "",
        "- Predictor: simulated vLLM scheduler steps priced as mixed forward passes.",
        "- Benchmark TPOT remains validation-only.",
        "- TPOT aggregation follows benchmark per-request ITL semantics.",
        "- Optional decode-residency wave mode scales only decode-side components.",
        "- Optional effective-cache mode estimates vLLM KVCacheManager-visible "
        "prefix hits before scheduling.",
        "- Logits/sampling and recompute/preemption are currently zero unless a cost model supplies them.",
        "",
        "## Overall",
        "",
        "| rows | base MAPE | engine-step MAPE | engine signed err | engine MAE | engine max abs err |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary.rows} | {summary.base_mape:.2f}% | "
            f"{summary.engine_mape:.2f}% | "
            f"{summary.engine_mean_signed_error_ms:.3f} | "
            f"{summary.engine_mean_abs_error_ms:.3f} | "
            f"{summary.engine_max_abs_error_ms:.3f} |"
        ),
        "",
        "## Worst Rows",
        "",
        "| profile | c | turn | mode | cache | B | scheduled | T | measured | base | engine | APE | signed | eff cache | sim prefill | steps | mixed | used KV | cap wait | prefill ms | dense ms |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item.pct_error, reverse=True)[:20]:
        lines.append(
            "| "
            f"{row.profile} | {row.concurrency} | {row.turn_index} | "
            f"{row.prefill_mode} | {row.cache_residency_classification} | "
            f"{row.successful_request_count} | "
            f"{row.scheduled_request_count} | {row.context_len} | "
            f"{row.measured_tpot_ms:.3f} | {row.base_pred_tpot_ms:.3f} | "
            f"{row.pred_tpot_ms:.3f} | {row.pct_error:.2f}% | "
            f"{row.signed_error_ms:.3f} | "
            f"{row.engine_effective_cached_tokens:.1f} | "
            f"{row.sim_new_prefill_tokens} | {row.sim_steps} | "
            f"{row.sim_mixed_decode_prefill_steps} | "
            f"{row.sim_max_used_kv_blocks} | "
            f"{row.sim_max_capacity_waiting_requests} | "
            f"{row.prefill_attention_ms:.3f} | {row.dense_ms:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_dashboard_json(
    path: Path,
    rows: Sequence[EngineStepPredictionRow],
    *,
    predictions_output: Path,
    report_output: Path,
    prefill_mode: str,
    dense_source: str,
) -> None:
    summary = summarize(rows)
    wave_policy = rows[0].wave_policy if rows else "none"
    backend = (
        "kernel-fluid-wave-v1"
        if rows and rows[0].scheduler_model == "fluid_wave"
        else
        "kernel-effective-cache-v0"
        if rows and rows[0].effective_cache_policy == "vllm_residency"
        else "kernel-xgb-engine-step-decode-wave-v0"
        if wave_policy == "decode_residency"
        else "kernel-xgb-engine-step-v0"
    )
    comparison = {
        "backend": backend,
        "label": f"engine-step-v0 {prefill_mode} {wave_policy}",
        "rows": summary.rows,
        "ttft_mape": None,
        "ttft_median_ape": None,
        "ttft_max_ape": None,
        "tpot_mape": summary.engine_mape,
        "tpot_median_ape": summary.engine_median_ape,
        "tpot_max_ape": summary.engine_max_ape,
        "e2el_mape": None,
        "e2el_median_ape": None,
        "e2el_max_ape": None,
    }
    payload = {
        "experiment": {
            "name": "Llama-3.1-8B H100 engine-step TPOT v0",
            "model": "Llama-3.1-8B",
            "gpu": "H100",
            "backend": backend,
            "target": "all multi-turn benchmark-turn TPOT",
            "scope_note": (
                "TPOT-only kernel composition with simulated vLLM scheduler "
                f"steps, prefill_mode={prefill_mode}, dense_source={dense_source}. "
                f"wave_policy={wave_policy}. "
                f"effective_cache_policy={rows[0].effective_cache_policy if rows else 'none'}. "
                "TTFT, E2EL, and fitted residuals are intentionally N/A."
            ),
            "dashboard_scope": "synthetic_distributional",
        },
        "sources": {
            "engine_step_predictions": str(predictions_output),
            "engine_step_report": str(report_output),
        },
        "fit_summary": {
            "rows": summary.rows,
            "kernel_composed_mape": summary.engine_mape,
            "kernel_composed_median_ape": summary.engine_median_ape,
            "kernel_composed_max_ape": summary.engine_max_ape,
            "kernel_composed_mean_signed_error_ms": summary.engine_mean_signed_error_ms,
            "kernel_composed_mean_abs_error_ms": summary.engine_mean_abs_error_ms,
            "kernel_composed_max_abs_error_ms": summary.engine_max_abs_error_ms,
            "physics_loo_mape": summary.engine_mape,
            "physics_loo_median_ape": summary.engine_median_ape,
            "physics_loo_max_ape": summary.engine_max_ape,
            "interp_loo_mape": summary.base_mape,
            "interp_loo_median_ape": summary.base_median_ape,
            "interp_loo_max_ape": summary.base_max_ape,
            "base_kernel_composed_mape": summary.base_mape,
            "base_kernel_composed_median_ape": summary.base_median_ape,
            "base_kernel_composed_max_ape": summary.base_max_ape,
            "engine_step_model": rows[0].engine_step_model if rows else ENGINE_STEP_MODEL,
            "scheduler_model": rows[0].scheduler_model if rows else "exact_shape",
            "prefill_mode": prefill_mode,
            "dense_source": dense_source,
            "wave_policy": wave_policy,
            "effective_cache_policy": rows[0].effective_cache_policy if rows else "none",
        },
        "dashboard_comparison": [comparison],
        "page_comparisons": {
            "simulator": [comparison],
            "serving": [],
        },
        "worst_rows": {
            "physics_loo": [
                {
                    "batch_size": row.scheduled_request_count,
                    "context_len": row.context_len,
                    "actual_ms": row.measured_tpot_ms,
                    "physics_loo_pred_ms": row.pred_tpot_ms,
                    "physics_loo_pct_error": row.pct_error,
                    "interp_loo_pred_ms": row.base_pred_tpot_ms,
                    "interp_loo_pct_error": row.base_pct_error,
                }
                for row in sorted(rows, key=lambda item: item.pct_error, reverse=True)[:5]
            ],
            "interpolation_loo": [
                {
                    "batch_size": row.scheduled_request_count,
                    "context_len": row.context_len,
                    "actual_ms": row.measured_tpot_ms,
                    "physics_loo_pred_ms": row.pred_tpot_ms,
                    "physics_loo_pct_error": row.pct_error,
                    "interp_loo_pred_ms": row.base_pred_tpot_ms,
                    "interp_loo_pct_error": row.base_pct_error,
                }
                for row in sorted(rows, key=lambda item: item.base_pct_error, reverse=True)[:5]
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_benchmark_turns(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    if not path.exists():
        return {}
    return {
        (row["profile"], int(row["concurrency"]), int(row["turn_index"])): row
        for row in load_csv(path)
    }


def load_per_request_distributions(
    path: Path,
) -> dict[tuple[str, int, int], dict[str, list[int]]]:
    """Load the per-(profile, c, turn) request-level distribution sidecar.

    Missing file is non-fatal — the predictor falls back to scalar means.
    Accepts both the legacy flat schema (entries keyed directly at the top
    level) and the new nested schema (entries under ``per_turn``).
    """
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    # New schema nests per-turn entries under "per_turn".
    if isinstance(payload, dict) and "per_turn" in payload:
        entries = payload["per_turn"]
    else:
        entries = payload
    out: dict[tuple[str, int, int], dict[str, list[int]]] = {}
    for entry in entries.values():
        try:
            key = (
                str(entry["profile"]),
                int(entry["concurrency"]),
                int(entry["turn_index"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = {
            "output_tokens": [int(x) for x in entry.get("output_tokens", [])],
            "new_prefill_tokens": [
                int(x) for x in entry.get("new_prefill_tokens", [])
            ],
            "cached_context_tokens": [
                int(x) for x in entry.get("cached_context_tokens", [])
            ],
        }
    return out


def load_shared_prefix_fractions(
    path: Path,
    *,
    cv_threshold: float = 0.30,
) -> dict[tuple[str, int], float]:
    """Estimate cross-cohort prefix sharing per (profile, concurrency).

    Workloads where every session in a cohort starts from the same prompt
    template (e.g. ``osworld``: shared system prompt + per-session task)
    will have **low variance** in turn-0 ``new_prefill_tokens`` across the
    cohort.  Workloads where each session is an independent agent thread
    (e.g. ``swebench``: distinct bug reports) have **high variance**.

    Returns a dict mapping (profile, concurrency) → ``shared_prefix_fraction``
    ∈ {0.0, 1.0}.  ``1.0`` means the cohort's cached prefix is shared across
    sessions and KV-block dedup applies; ``0.0`` means per-session unique
    prefix.

    Heuristic: coefficient of variation of turn-0 new_prefill_tokens.
    CV < ``cv_threshold`` ⇒ shared (1.0), otherwise per-session (0.0).
    """
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    entries = payload.get("per_turn") if isinstance(payload, dict) else payload
    if not isinstance(entries, dict):
        return {}
    out: dict[tuple[str, int], float] = {}
    for entry in entries.values():
        try:
            turn_index = int(entry.get("turn_index", -1))
        except (TypeError, ValueError):
            continue
        if turn_index != 0:
            continue
        vals = entry.get("new_prefill_tokens") or []
        if not isinstance(vals, list) or len(vals) < 2:
            continue
        nums = [float(v) for v in vals]
        m = sum(nums) / len(nums)
        if m <= 0:
            continue
        var = sum((x - m) ** 2 for x in nums) / (len(nums) - 1)
        cv = (var ** 0.5) / m
        try:
            key = (str(entry["profile"]), int(entry["concurrency"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = 1.0 if cv < cv_threshold else 0.0
    return out


def load_session_timelines(
    path: Path,
) -> dict[tuple[str, int], list[list[dict]]]:
    """Load per-(profile, c) session timelines for wall-clock multi-turn replay.

    Returns a dict mapping (profile, concurrency) → list of sessions, where
    each session is an ordered list of turn dicts:
    ``{turn_index, arrival_offset_ms, completion_offset_ms,
        new_prefill_tokens, cached_context_tokens, output_tokens}``.
    Missing file or absent ``session_timelines`` key returns ``{}``; the
    predictor then falls back to the single-turn path.
    """
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "session_timelines" not in payload:
        return {}
    out: dict[tuple[str, int], list[list[dict]]] = {}
    for entry in payload["session_timelines"].values():
        try:
            key = (str(entry["profile"]), int(entry["concurrency"]))
        except (KeyError, TypeError, ValueError):
            continue
        sessions = entry.get("sessions") or []
        # Pass through as-is — sim layer reads dict fields it needs.
        out[key] = sessions
    return out


def percent_error(measured: float, predicted: float) -> float:
    if measured <= 0.0:
        return 0.0
    return abs(predicted - measured) / measured * 100.0


def _effective_cache_estimate(
    *,
    request_count: int,
    context_len: int,
    cached_context_tokens: int,
    new_prefill_tokens: int,
    prefill_mode: PrefillMode,
    effective_cache_policy: EffectiveCachePolicy,
    config: VllmSchedulerConfig,
) -> dict[str, int | float | str]:
    if effective_cache_policy == "none":
        return {
            "sim_cached_context_tokens": max(0, int(cached_context_tokens)),
            "sim_new_prefill_tokens": max(0, int(new_prefill_tokens)),
            "engine_effective_cached_tokens": float(max(0, int(cached_context_tokens))),
            "capacity_feasible_cached_tokens": max(0, int(cached_context_tokens)),
            "engine_uncached_prefill_tokens": max(0, int(new_prefill_tokens)),
            "cache_residency_classification": "not_applied",
        }

    result = estimate_prefix_cache_residency(
        VllmPrefixCacheResidencyInput(
            request_count=request_count,
            context_len=context_len,
            cached_context_tokens=cached_context_tokens,
            new_prefill_tokens=new_prefill_tokens,
            mode=prefill_mode,
            shared_prefix_tokens=config.shared_prefix_tokens,
        ),
        VllmPrefixCacheResidencyConfig(
            cache_block_size=config.cache_block_size,
            available_gpu_kv_blocks=config.available_gpu_kv_blocks,
        ),
    )
    if prefill_mode == "synthetic_shared_prefix":
        # The synthetic path already encodes the prefix anchor/follower shape in
        # benchmark_row_requests.  Keep the scheduler inputs unchanged and emit
        # the cache truth estimate for reporting only.
        return {
            "sim_cached_context_tokens": max(0, int(cached_context_tokens)),
            "sim_new_prefill_tokens": max(0, int(new_prefill_tokens)),
            "engine_effective_cached_tokens": result.engine_effective_cached_tokens,
            "capacity_feasible_cached_tokens": result.capacity_feasible_cached_tokens,
            "engine_uncached_prefill_tokens": result.engine_uncached_prefill_tokens,
            "cache_residency_classification": result.cache_residency_classification,
        }

    effective_cached = int(round(result.engine_effective_cached_tokens))
    return {
        "sim_cached_context_tokens": effective_cached,
        "sim_new_prefill_tokens": result.engine_uncached_prefill_tokens,
        "engine_effective_cached_tokens": result.engine_effective_cached_tokens,
        "capacity_feasible_cached_tokens": result.capacity_feasible_cached_tokens,
        "engine_uncached_prefill_tokens": result.engine_uncached_prefill_tokens,
        "cache_residency_classification": result.cache_residency_classification,
    }


def _build_fluid_prediction_row(
    *,
    source_row: dict[str, str],
    component_model: str,
    prefill_mode: PrefillMode,
    dense_source: str,
    wave_policy: WavePolicy,
    effective_cache_policy: EffectiveCachePolicy,
    profile: str,
    concurrency: int,
    turn_index: int,
    batch_size: int,
    scheduled: int,
    successful: int,
    turn_num_requests: int,
    context_len: int,
    output_tokens: int,
    new_prefill: int,
    cached_context: int,
    cache_hit_rate: float,
    measured: float,
    base_pred: float,
    base_signed_error: float,
    diagnostic_reason: str,
    cost_model: VllmEngineStepCostModel,
    config: VllmSchedulerConfig,
) -> EngineStepPredictionRow:
    fluid = simulate_fluid_turn(
        VllmFluidTurnInput(
            scheduled_request_count=scheduled,
            context_len=context_len,
            output_tokens=output_tokens,
            cached_context_tokens=cached_context,
            new_prefill_tokens=new_prefill,
            cache_hit_rate=cache_hit_rate,
        ),
        VllmFluidSchedulerConfig(
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            cache_block_size=config.cache_block_size,
            available_gpu_kv_blocks=config.available_gpu_kv_blocks,
            decode_counts_against_token_budget=config.decode_counts_against_token_budget,
        ),
    )
    estimates = {
        summary.cache_policy: _predict_tpot_with_fluid_summary(
            summary,
            context_len=context_len,
            cost_model=cost_model,
        )
        for summary in fluid.envelopes
    }
    mid_summary = fluid.fluid_mid
    mid = estimates["fluid_mid"]
    low = min(estimate.pred_tpot_ms for estimate in estimates.values())
    high = max(estimate.pred_tpot_ms for estimate in estimates.values())
    signed_error = mid.pred_tpot_ms - measured
    return EngineStepPredictionRow(
        component_model=component_model,
        engine_step_model=FLUID_WAVE_MODEL,
        scheduler_model="fluid_wave",
        prefill_mode=prefill_mode,
        dense_source=dense_source,
        wave_policy=wave_policy,
        profile=profile,
        concurrency=concurrency,
        turn_index=turn_index,
        primary_eval=source_row.get("primary_eval") == "true",
        batch_size=batch_size,
        scheduled_request_count=scheduled,
        successful_request_count=successful,
        turn_num_requests=turn_num_requests,
        context_len=context_len,
        output_tokens=output_tokens,
        new_prefill_tokens=new_prefill,
        cached_context_tokens=cached_context,
        cache_hit_rate=cache_hit_rate,
        effective_cache_policy=(
            "fluid_envelope"
            if effective_cache_policy == "none"
            else f"fluid_envelope+{effective_cache_policy}"
        ),
        sim_new_prefill_tokens=mid_summary.uncached_prefill_tokens_per_request,
        sim_cached_context_tokens=mid_summary.effective_cached_tokens,
        engine_effective_cached_tokens=float(mid_summary.effective_cached_tokens),
        capacity_feasible_cached_tokens=mid_summary.capacity_feasible_cached_tokens,
        engine_uncached_prefill_tokens=mid_summary.uncached_prefill_tokens_per_request,
        cache_residency_classification=mid_summary.cache_residency_classification,
        measured_tpot_ms=measured,
        base_pred_tpot_ms=base_pred,
        pred_tpot_ms=mid.pred_tpot_ms,
        base_pct_error=abs(float(source_row["pct_error"])),
        pct_error=percent_error(measured, mid.pred_tpot_ms),
        base_signed_error_ms=base_signed_error,
        signed_error_ms=signed_error,
        base_abs_error_ms=abs(base_signed_error),
        abs_error_ms=abs(signed_error),
        sim_steps=mid_summary.estimated_steps,
        sim_decode_only_steps=mid_summary.decode_only_steps,
        sim_prefill_only_steps=mid_summary.prefill_only_steps,
        sim_mixed_decode_prefill_steps=mid_summary.mixed_decode_prefill_steps,
        sim_total_decode_slots=mid_summary.decode_slots,
        sim_total_prefill_tokens=mid_summary.fresh_prefill_debt_tokens,
        sim_max_decode_batch=mid_summary.max_decode_batch,
        sim_mean_decode_batch=mid_summary.mean_decode_batch,
        sim_mean_decode_batch_ratio=mid_summary.mean_decode_batch_ratio,
        sim_min_free_kv_blocks=mid_summary.min_free_kv_blocks,
        sim_max_used_kv_blocks=mid_summary.max_used_kv_blocks,
        sim_max_capacity_waiting_requests=max(
            0,
            scheduled - mid_summary.active_decode_capacity,
        ),
        sim_total_preemptions=0,
        sim_total_recompute_prefill_tokens=0,
        decode_residency_wave_factor=mid_summary.decode_wave_factor,
        total_step_ms=mid.total_step_ms,
        pooled_itl_ms=mid.pooled_itl_ms,
        decoded_request_count=scheduled,
        dense_ms=mid.dense_ms,
        decode_attention_ms=mid.decode_attention_ms,
        prefill_attention_ms=mid.prefill_attention_ms,
        small_kernel_ms=mid.small_kernel_ms,
        logits_sampling_ms=mid.logits_sampling_ms,
        runtime_overhead_ms=mid.runtime_overhead_ms,
        recompute_or_preemption_ms=mid.recompute_or_preemption_ms,
        prefill_attention_work_units=mid.prefill_attention_work_units,
        fluid_pred_low_tpot_ms=low,
        fluid_pred_mid_tpot_ms=mid.pred_tpot_ms,
        fluid_pred_high_tpot_ms=high,
        fluid_active_decode_capacity=mid_summary.active_decode_capacity,
        fluid_decode_steps=mid_summary.decode_steps,
        fluid_decode_waves=mid_summary.decode_waves,
        fluid_prefill_only_steps=mid_summary.prefill_only_steps,
        fluid_mixed_decode_prefill_steps=mid_summary.mixed_decode_prefill_steps,
        fluid_kv_pressure=mid_summary.kv_pressure,
        fluid_prefill_debt_tokens=mid_summary.fresh_prefill_debt_tokens,
        fluid_prefill_debt_per_decode_token=(
            mid_summary.prefill_debt_per_decode_token
        ),
        fluid_cache_optimistic_tokens=fluid.optimistic.effective_cached_tokens,
        fluid_cache_mid_tokens=fluid.fluid_mid.effective_cached_tokens,
        fluid_cache_pessimistic_tokens=fluid.pessimistic.effective_cached_tokens,
        fluid_scheduler_sensitivity=mid_summary.scheduler_sensitivity,
        diagnostic_reason=diagnostic_reason,
    )


def _predict_tpot_with_fluid_summary(
    summary: VllmFluidTurnSummary,
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel,
) -> FluidTpotEstimate:
    """Price each step type from the fluid summary via kernel composition.

    Total wall time is the sum across step types; TPOT divides by decode slots.
    """
    active_batch = max(1, int(summary.active_decode_capacity))
    decode_slots = max(1, summary.decode_slots)

    # --- per-step kernel costs (decode-only uses composition) ---
    decode_per_step = (
        float(cost_model.dense_ms(active_batch))
        + float(cost_model.decode_attention_ms(active_batch, context_len))
        + float(cost_model.small_kernel_ms(active_batch))
        + float(cost_model.logits_sampling_ms(active_batch))
    )
    decode_total = decode_per_step * summary.decode_only_steps

    # --- mixed steps ---
    mixed_steps = summary.mixed_decode_prefill_steps
    mixed_total = 0.0
    if mixed_steps > 0:
        mixed_tokens_per_step = math.ceil(
            summary.prefill_tokens_in_mixed_steps / mixed_steps
        )
        total_batch = active_batch + mixed_tokens_per_step
        chunks = (
            VllmPrefillChunk(
                request_id=-1,
                scheduled_tokens=mixed_tokens_per_step,
                prefix_tokens=max(0, context_len - mixed_tokens_per_step),
            ),
        )
        mixed_per_step = (
            float(cost_model.dense_ms(total_batch))
            + float(cost_model.small_kernel_ms(total_batch))
            + float(cost_model.logits_sampling_ms(active_batch))
            + float(cost_model.decode_attention_ms(active_batch, context_len))
            + float(cost_model.prefill_attention_ms(chunks))
        )
        mixed_total = mixed_per_step * mixed_steps

    # --- prefill-only steps ---
    prefill_only_steps = summary.prefill_only_steps
    prefill_only_total = 0.0
    if prefill_only_steps > 0:
        pf_tokens_per_step = max(
            1,
            summary.prefill_tokens_after_decode // prefill_only_steps,
        )
        pf_chunks = (
            VllmPrefillChunk(
                request_id=-1,
                scheduled_tokens=pf_tokens_per_step,
                prefix_tokens=max(0, context_len - pf_tokens_per_step),
            ),
        )
        pf_per_step = (
            float(cost_model.dense_ms(pf_tokens_per_step))
            + float(cost_model.small_kernel_ms(pf_tokens_per_step))
            + float(cost_model.prefill_attention_ms(pf_chunks))
            )
        prefill_only_total = pf_per_step * prefill_only_steps

    # --- TPOT = total wall time / total decode slots ---
    total_step_ms = decode_total + mixed_total + prefill_only_total
    pred = total_step_ms / decode_slots

    return FluidTpotEstimate(
        pred_tpot_ms=pred,
        dense_ms=float(cost_model.dense_ms(active_batch))
        * summary.decode_only_steps / decode_slots,
        decode_attention_ms=float(cost_model.decode_attention_ms(active_batch, context_len))
        * summary.decode_only_steps / decode_slots,
        prefill_attention_ms=0.0,
        small_kernel_ms=float(cost_model.small_kernel_ms(active_batch))
        * summary.decode_only_steps / decode_slots,
        logits_sampling_ms=float(cost_model.logits_sampling_ms(active_batch))
        * summary.decode_only_steps / decode_slots,
        runtime_overhead_ms=0.0,
        recompute_or_preemption_ms=0.0,
        total_step_ms=total_step_ms,
        pooled_itl_ms=pred,
        prefill_attention_work_units=0.0,
    )


def _build_cost_model(
    models: EngineStepComponentModels,
    *,
    context_len: int,
    dense_source: str,
) -> VllmEngineStepCostModel:
    dense_cache: dict[int, float] = {}
    decode_attention_cache: dict[tuple[int, int], float] = {}
    small_cache: dict[int, float] = {}

    prefill_small_kernel_override: TokenCostFn | None = None
    if dense_source == "nsys_compiled" and models.nsys_compiled_model is not None:
        raw_dense_ms = models.nsys_compiled_model.predict_non_attention_ms
        # Dense already includes elementwise + KV_write + other for prefill;
        # zero the small-kernel adder on prefill-bearing steps.  Decode steps
        # keep the normal small_kernel_ms adder via the dataclass default.
        prefill_small_kernel_override = lambda _tokens: 0.0  # noqa: E731
    elif dense_source == "forward_pass" and models.forward_gemm_model is not None:
        raw_dense_ms = models.forward_gemm_model.predict_ms
    else:
        raw_dense_ms = lambda tokens: _sum_component_predictions(  # noqa: E731
            models.gemm_models,
            max(1, int(tokens)),
            None,
        ).value_ms

    def dense_ms(tokens: int) -> float:
        key = max(1, int(tokens))
        if key not in dense_cache:
            dense_cache[key] = raw_dense_ms(key)
        return dense_cache[key]

    # Per-B training T max for attention model.  XGBoost degrades
    # when T exceeds the maximum seen during training for that B.
    _t_max_by_b: dict[int, int] = {}
    if hasattr(models.attention_model, "_samples"):
        for s in models.attention_model._samples:  # type: ignore[union-attr]
            b = int(s.batch_size)
            t = int(s.context_len)
            _t_max_by_b[b] = max(_t_max_by_b.get(b, 0), t)

    # Bandwidth model for decode attention at large B×T.
    # KV bytes = 2 × B × T × n_kv × d × bytes × L
    # Effective BW ~600 GB/s from FA3 q=1 measurements at T>=4096.
    _KV_BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 32   # = 131072 bytes per token per layer... no, total
    # Per-request KV: 2 × T × n_kv(8) × d(128) × bytes(2) × L(32) = 131072 × T bytes
    # For B requests: 131072 × B × T bytes
    _BW_EFFECTIVE_GB_S = 600.0  # GB/s, calibrated from FA3 q=1 data

    def decode_attention_ms(decode_batch: int, query_context_len: int) -> float:
        if decode_batch <= 0:
            return 0.0
        db = max(1, int(decode_batch))
        qcl = max(1, int(query_context_len))
        key = (db, qcl)
        if key not in decode_attention_cache:
            pred = models.attention_model.predict(db, qcl)
            ms = pred.value_ms
            # If T exceeds training max for this B, scale by bandwidth ratio
            t_max = _t_max_by_b.get(db)
            if t_max is not None and qcl > t_max:
                base_pred = models.attention_model.predict(db, t_max)
                if base_pred.value_ms > 0:
                    ms = base_pred.value_ms * (qcl / t_max)
            # Bandwidth floor: at large B×T the KV scan is bandwidth-bound.
            # Only apply for T>2000 where KV read dominates.
            # Effective BW scales with B (better SM occupancy at larger batches).
            # Calibrated from decode profile: B=1→400GB/s, B=8→1300, B=32→1900.
            if qcl > 2000:
                import math
                eff_bw = 400.0 + 300.0 * math.log2(max(1.0, float(db)))
                kv_gb = 131072.0 * db * qcl / 1e9
                bw_ms = kv_gb / eff_bw * 1000.0
                ms = max(ms, bw_ms)
            decode_attention_cache[key] = ms
        return decode_attention_cache[key]

    def prefill_attention_ms(chunks: Sequence[VllmPrefillChunk]) -> float:
        if not chunks:
            return 0.0
        if models.forward_attention_model is not None:
            return models.forward_attention_model.predict_chunks_ms(chunks)
        decode_reference = models.attention_model.predict(1, max(1, context_len)).value_ms
        return decode_reference * prefill_attention_work(chunks) / max(1, context_len)

    def small_kernel_ms(tokens: int) -> float:
        if tokens <= 0 or not models.small_kernel_models:
            return 0.0
        key = max(1, int(tokens))
        if key not in small_cache:
            small_cache[key] = _sum_component_predictions(
                models.small_kernel_models,
                key,
                context_len,
            ).value_ms
        return small_cache[key]

    # decode_dense_ms always uses NCU GEMM (validated 7.4% MAPE for decode)
    def decode_dense_ms(tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        t = max(1, int(tokens))
        return _sum_component_predictions(
            models.gemm_models, t, None,
        ).value_ms

    return VllmEngineStepCostModel(
        dense_ms=dense_ms,
        decode_dense_ms=decode_dense_ms,
        decode_attention_ms=decode_attention_ms,
        prefill_attention_ms=prefill_attention_ms,
        small_kernel_ms=small_kernel_ms,
        prefill_small_kernel_ms=prefill_small_kernel_override,
    )


def _decode_residency_wave_factor(
    *,
    scheduled_request_count: int,
    max_decode_batch: int,
) -> float:
    if scheduled_request_count <= 0 or max_decode_batch <= 0:
        return 1.0
    return max(1.0, float(scheduled_request_count) / float(max_decode_batch))


def _predict_mean_tpot_with_decode_residency_wave(
    steps: Sequence[object],
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel,
    wave_factor: float,
) -> dict[str, object]:
    request_itls: dict[int, list[float]] = {}
    components = {
        "dense_ms": 0.0,
        "decode_attention_ms": 0.0,
        "prefill_attention_ms": 0.0,
        "small_kernel_ms": 0.0,
        "logits_sampling_ms": 0.0,
        "runtime_overhead_ms": 0.0,
        "recompute_or_preemption_ms": 0.0,
    }
    pooled_numerator = 0.0
    pooled_denominator = 0
    total_step_ms = 0.0

    for step in steps:
        split = _split_step_cost_for_decode_wave(
            step,
            context_len=context_len,
            cost_model=cost_model,
            wave_factor=wave_factor,
        )
        for field in components:
            components[field] += float(getattr(split, field))
        step_ms = split.total_ms
        total_step_ms += step_ms
        decoded_ids = tuple(getattr(step, "decoded_request_ids"))
        for request_id in decoded_ids:
            request_itls.setdefault(int(request_id), []).append(step_ms)
        pooled_numerator += step_ms * len(decoded_ids)
        pooled_denominator += len(decoded_ids)

    if not request_itls:
        raise ValueError("cannot compute TPOT without decoded scheduler steps")

    request_means = [mean(values) for values in request_itls.values() if values]
    return {
        "mean_tpot_ms": mean(request_means),
        "pooled_itl_ms": pooled_numerator / pooled_denominator,
        "total_step_ms": total_step_ms,
        "components": components,
    }


def _split_step_cost_for_decode_wave(
    step: object,
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel,
    wave_factor: float,
) -> VllmEngineStepCost:
    full = engine_step_cost_ms(
        step,  # type: ignore[arg-type]
        context_len=context_len,
        cost_model=cost_model,
    )
    decode_batch = max(0, int(getattr(step, "decode_batch")))
    prefill_tokens = max(0, int(getattr(step, "prefill_tokens")))
    total_tokens = decode_batch + prefill_tokens

    decode_dense = float(cost_model.dense_ms(decode_batch)) if decode_batch else 0.0
    decode_small = (
        float(cost_model.small_kernel_ms(decode_batch)) if decode_batch else 0.0
    )
    decode_samples = decode_batch
    decode_logits = (
        float(cost_model.logits_sampling_ms(decode_samples))
        if decode_samples
        else 0.0
    )

    prefill_dense = max(0.0, full.dense_ms - decode_dense)
    prefill_small = max(0.0, full.small_kernel_ms - decode_small)
    prefill_logits = max(0.0, full.logits_sampling_ms - decode_logits)

    return VllmEngineStepCost(
        dense_ms=decode_dense * wave_factor + prefill_dense,
        decode_attention_ms=full.decode_attention_ms * wave_factor,
        prefill_attention_ms=full.prefill_attention_ms,
        small_kernel_ms=decode_small * wave_factor + prefill_small,
        logits_sampling_ms=decode_logits * wave_factor + prefill_logits,
        runtime_overhead_ms=full.runtime_overhead_ms,
        recompute_or_preemption_ms=full.recompute_or_preemption_ms,
    )


def _sum_step_components(step_costs: Sequence[object]) -> dict[str, float]:
    fields = [
        "dense_ms",
        "decode_attention_ms",
        "prefill_attention_ms",
        "small_kernel_ms",
        "logits_sampling_ms",
        "runtime_overhead_ms",
        "recompute_or_preemption_ms",
    ]
    return {
        field: sum(float(getattr(cost, field)) for cost in step_costs)
        for field in fields
    }


def _bracket(values: Sequence[int], target: int) -> tuple[int, int]:
    """Return (lo, hi) bracketing values for interpolation."""
    if not values:
        return target, target
    sorted_vals = sorted(values)
    lo = sorted_vals[0]
    hi = sorted_vals[-1]
    for v in sorted_vals:
        if v <= target:
            lo = v
        if v >= target and hi == sorted_vals[-1]:
            hi = v
            break
    if lo == hi:
        hi = max(hi + 1, target + 1)
    return lo, hi


def _dedupe_samples(samples: Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    grouped: dict[int, list[float]] = {}
    for tokens, value in samples:
        grouped.setdefault(max(1, int(tokens)), []).append(float(value))
    return tuple(sorted((tokens, mean(values)) for tokens, values in grouped.items()))


def _log_linear_predict(samples: Sequence[tuple[int, float]], tokens: int) -> float:
    points = [(_log2(tokens), value) for tokens, value in sorted(samples)]
    x = _log2(tokens)
    if len(points) == 1:
        return points[0][1]
    if x <= points[0][0]:
        return _line(points[0], points[1], x)
    if x >= points[-1][0]:
        return _line(points[-2], points[-1], x)
    for low, high in zip(points, points[1:]):
        if low[0] <= x <= high[0]:
            return _line(low, high, x)
    return min(points, key=lambda point: abs(point[0] - x))[1]


def _line(low: tuple[float, float], high: tuple[float, float], x: float) -> float:
    if high[0] == low[0]:
        return low[1]
    weight = (x - low[0]) / (high[0] - low[0])
    return low[1] + weight * (high[1] - low[1])


def _log2(value: int) -> float:
    return math.log2(max(1, int(value)))


def _is_single_turn_profile(profile: str) -> bool:
    normalized = profile.replace("_", "-").lower()
    return "singleturn" in normalized or "single-turn" in normalized


def _to_int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    return int(round(float(str(value))))


def _to_float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    return float(str(value))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    args = parse_args()
    config = VllmSchedulerConfig(
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        cache_block_size=args.cache_block_size,
        available_gpu_kv_blocks=args.available_gpu_kv_blocks,
        shared_prefix_tokens=args.shared_prefix_tokens,
        bos_token_offset=args.bos_token_offset,
        hbm_bw_gbps=args.hbm_bw_gbps,
        kv_bytes_per_token=args.kv_bytes_per_token,
        step_budget_ms=args.step_budget_ms,
        bandwidth_caps_decode=args.bandwidth_caps_decode,
        bandwidth_caps_prefill=args.bandwidth_caps_prefill,
        bandwidth_decode_priority=args.bandwidth_decode_priority,
        use_block_pool=args.use_block_pool,
        max_admissions_per_step=args.max_admissions_per_step,
    )
    component_models = build_default_component_models(
        component_model=args.component_model,
        attention_profile=args.attention_profile,
        gemm_summary=args.gemm_summary,
        small_kernel_summary=args.small_kernel_summary,
        forward_gemm=args.forward_gemm,
        forward_attention=args.forward_attention,
        nsys_compiled_csv=args.nsys_compiled_csv,
    )
    step_predictor: CleanStepPredictor | None = None
    if args.use_step_predictor:
        step_predictor = CleanStepPredictor.from_csvs(
            full_prefill_path=args.full_prefill_csv,
            cached_prefill_path=args.cached_prefill_csv,
            decode_path=args.decode_profile_csv,
        )
    roofline_predictor: RooflineStepPredictor | None = None
    if args.use_roofline_predictor:
        roofline_predictor = RooflineStepPredictor.from_json(
            args.roofline_params_json
        )
    if args.closed_form:
        closed_form_params_path = (
            args.closed_form_params_json or args.roofline_params_json
        )
        closed_form_predictor = ClosedFormTpotPredictor.from_json(
            closed_form_params_path,
            kv_pressure_enabled=args.kv_pressure_wave,
        )
        # Note: load_shared_prefix_fractions() exists but isn't wired in by
        # default — its CV-of-turn-0-new-prefill-tokens proxy mis-classifies
        # chat/terminal as "shared prefix" (their prompts have similar
        # length but different content).  Until a token-content
        # discriminator lands (requires bench JSON metadata we don't yet
        # extract), shared_prefix_fraction stays at 0 for every row.
        rows = build_closed_form_prediction_rows(
            load_csv(args.base_predictions),
            benchmark_turns=load_benchmark_turns(args.benchmark_turns),
            closed_form_predictor=closed_form_predictor,
            component_model=args.component_model,
            per_request_by_key=load_per_request_distributions(
                args.benchmark_per_request_json
            ),
            per_request_fields=_parse_per_request_fields(args.per_request_fields),
            shared_prefix_by_key=None,
            include_diagnostic=not args.primary_only,
            include_single_turn=args.include_single_turn,
        )
        write_rows(args.output, rows)
        write_report(args.report_output, rows)
        if not args.skip_dashboard:
            write_dashboard_json(args.dashboard_output, rows)
        return
    rows = build_engine_step_prediction_rows(
        load_csv(args.base_predictions),
        benchmark_turns=load_benchmark_turns(args.benchmark_turns),
        component_models=component_models,
        component_model=args.component_model,
        prefill_mode=args.prefill_mode,
        dense_source=args.dense_source,
        scheduler_model=args.scheduler_model,
        wave_policy=args.wave_policy,
        effective_cache_policy=args.effective_cache_policy,
        config=config,
        include_diagnostic=not args.primary_only,
        include_single_turn=args.include_single_turn,
        step_predictor=step_predictor,
        roofline_predictor=roofline_predictor,
        per_request_by_key=load_per_request_distributions(args.benchmark_per_request_json),
        per_request_fields=_parse_per_request_fields(args.per_request_fields),
        session_timelines=load_session_timelines(args.benchmark_per_request_json),
    )
    write_rows(args.output, rows)
    write_report(args.report_output, rows)
    if not args.skip_dashboard:
        write_dashboard_json(
            args.dashboard_output,
            rows,
            predictions_output=args.output,
            report_output=args.report_output,
            prefill_mode=args.prefill_mode,
            dense_source=args.dense_source,
        )
    summary = summarize(rows)
    print(
        f"engine-step rows={summary.rows} MAPE={summary.engine_mape:.2f}% "
        f"base={summary.base_mape:.2f}% signed={summary.engine_mean_signed_error_ms:.3f}ms"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")
    if args.skip_dashboard:
        print("Skipped dashboard output")
    else:
        print(f"Wrote {args.dashboard_output}")


if __name__ == "__main__":
    main()
