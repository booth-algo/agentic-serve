"""Timing primitives for vLLM scheduler-step simulation.

The functions here price one model-runner forward for a scheduler step.  They
do not fit residuals or read benchmark targets; callers provide the component
cost curves they want to test.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .vllm_scheduler_shape import VllmPrefillChunk, VllmStepShape


TokenCostFn = Callable[[int], float]
DecodeAttentionCostFn = Callable[[int, int], float]
PrefillAttentionCostFn = Callable[[Sequence[VllmPrefillChunk]], float]
SampleCostFn = Callable[[int], float]
StepCostFn = Callable[[VllmStepShape], float]


def _zero_token_cost(tokens: int) -> float:
    del tokens
    return 0.0


def _zero_decode_attention_cost(decode_batch: int, context_len: int) -> float:
    del decode_batch, context_len
    return 0.0


def _zero_prefill_attention_cost(chunks: Sequence[VllmPrefillChunk]) -> float:
    del chunks
    return 0.0


def _zero_sample_cost(sampled_positions: int) -> float:
    del sampled_positions
    return 0.0


def _zero_step_cost(step: VllmStepShape) -> float:
    del step
    return 0.0


@dataclass(frozen=True)
class VllmEngineStepCostModel:
    dense_ms: TokenCostFn = _zero_token_cost
    decode_dense_ms: TokenCostFn = _zero_token_cost  # NCU GEMM for decode tokens
    decode_attention_ms: DecodeAttentionCostFn = _zero_decode_attention_cost
    prefill_attention_ms: PrefillAttentionCostFn = _zero_prefill_attention_cost
    small_kernel_ms: TokenCostFn = _zero_token_cost
    # Override for prefill-bearing steps.  When dense_ms already absorbs
    # elementwise + KV_write + other (e.g. nsys-derived compiled breakdown),
    # set this to ``_zero_token_cost`` to prevent double-counting.  Default
    # ``None`` falls back to ``small_kernel_ms`` for all step types, preserving
    # legacy behavior.
    prefill_small_kernel_ms: TokenCostFn | None = None
    logits_sampling_ms: SampleCostFn = _zero_sample_cost
    runtime_overhead_ms: StepCostFn = _zero_step_cost
    recompute_or_preemption_ms: StepCostFn = _zero_step_cost


@dataclass(frozen=True)
class VllmEngineStepCost:
    dense_ms: float
    decode_attention_ms: float
    prefill_attention_ms: float
    small_kernel_ms: float
    logits_sampling_ms: float
    runtime_overhead_ms: float
    recompute_or_preemption_ms: float

    @property
    def total_ms(self) -> float:
        return (
            self.dense_ms
            + self.decode_attention_ms
            + self.prefill_attention_ms
            + self.small_kernel_ms
            + self.logits_sampling_ms
            + self.runtime_overhead_ms
            + self.recompute_or_preemption_ms
        )


@dataclass(frozen=True)
class VllmTurnTimingResult:
    mean_tpot_ms: float
    pooled_itl_ms: float
    total_step_ms: float
    total_decode_slots: int
    decoded_request_count: int
    step_costs: tuple[VllmEngineStepCost, ...]


def prefill_attention_work(chunks: Sequence[VllmPrefillChunk]) -> float:
    """Return q*prefix + q*(q+1)/2 causal-attention work units."""
    work = 0.0
    for chunk in chunks:
        q = max(0, int(chunk.scheduled_tokens))
        prefix = max(0, int(chunk.prefix_tokens))
        work += q * prefix + q * (q + 1) / 2.0
    return work


def engine_step_cost_ms(
    step: VllmStepShape,
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel | None = None,
    sampled_positions: int | None = None,
) -> VllmEngineStepCost:
    """Price one mixed decode/prefill scheduler step.

    ``context_len`` is the fallback prefix length for aggregate step rows that
    do not carry per-request prefill chunks.
    """
    model = cost_model or VllmEngineStepCostModel()
    decode_batch = int(step.decode_batch)
    prefill_tokens = int(step.prefill_tokens)
    total_scheduled_tokens = decode_batch + prefill_tokens
    chunks = _prefill_chunks_or_aggregate_fallback(step, context_len=context_len)
    samples = (
        max(0, int(sampled_positions))
        if sampled_positions is not None
        else decode_batch + len(step.completed_prefill_request_ids)
    )
    # Dense: prefill tokens in the batch → compiled dense benefits all tokens.
    # Decode-only → use NCU GEMM (validated 7.4% MAPE).
    if prefill_tokens > 0:
        total_dense = float(model.dense_ms(total_scheduled_tokens))
    else:
        total_dense = float(model.decode_dense_ms(decode_batch)) if decode_batch else 0.0

    # Small kernels: by default use ``small_kernel_ms`` for both prefill and
    # decode steps.  If the cost model overrides ``prefill_small_kernel_ms``
    # (e.g. when ``dense_ms`` already absorbs elementwise/KV_write/other for
    # prefill), use the override on prefill-bearing steps to avoid double-count.
    if prefill_tokens > 0 and model.prefill_small_kernel_ms is not None:
        small_kernel_total = float(model.prefill_small_kernel_ms(total_scheduled_tokens))
    else:
        small_kernel_total = float(model.small_kernel_ms(total_scheduled_tokens))

    # KV pressure penalty: when KV blocks are near capacity, waiting requests
    # cause queue delays not captured by kernel costs alone.
    # Disabled for now — bandwidth model is the primary fix.
    kv_penalty_ms = 0.0

    return VllmEngineStepCost(
        dense_ms=total_dense,
        decode_attention_ms=float(
            model.decode_attention_ms(int(step.decode_batch), int(context_len))
        ),
        prefill_attention_ms=float(model.prefill_attention_ms(chunks)),
        small_kernel_ms=small_kernel_total,
        logits_sampling_ms=float(model.logits_sampling_ms(samples)),
        runtime_overhead_ms=float(model.runtime_overhead_ms(step)) + kv_penalty_ms,
        recompute_or_preemption_ms=float(model.recompute_or_preemption_ms(step)),
    )


def mixed_forward_cost_ms(
    step: VllmStepShape,
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel | None = None,
    sampled_positions: int | None = None,
) -> VllmEngineStepCost:
    """Alias using the terminology from the experiment notes."""
    return engine_step_cost_ms(
        step,
        context_len=context_len,
        cost_model=cost_model,
        sampled_positions=sampled_positions,
    )


def predict_mean_tpot_from_engine_steps(
    steps: Sequence[VllmStepShape],
    *,
    context_len: int,
    cost_model: VllmEngineStepCostModel | None = None,
) -> VllmTurnTimingResult:
    """Aggregate step costs using benchmark-style per-request mean ITL."""
    request_itls: dict[int, list[float]] = defaultdict(list)
    step_costs: list[VllmEngineStepCost] = []
    pooled_numerator = 0.0
    pooled_denominator = 0
    total_step_ms = 0.0

    for step in steps:
        cost = engine_step_cost_ms(
            step,
            context_len=context_len,
            cost_model=cost_model,
        )
        step_costs.append(cost)
        step_ms = cost.total_ms
        total_step_ms += step_ms
        for request_id in step.decoded_request_ids:
            request_itls[request_id].append(step_ms)
        pooled_numerator += step_ms * len(step.decoded_request_ids)
        pooled_denominator += len(step.decoded_request_ids)

    if not request_itls:
        raise ValueError("cannot compute TPOT without decoded scheduler steps")

    request_means = [
        statistics.fmean(itls) for itls in request_itls.values() if itls
    ]
    return VllmTurnTimingResult(
        mean_tpot_ms=statistics.fmean(request_means),
        pooled_itl_ms=pooled_numerator / pooled_denominator,
        total_step_ms=total_step_ms,
        total_decode_slots=pooled_denominator,
        decoded_request_count=len(request_itls),
        step_costs=tuple(step_costs),
    )


def engine_step_cost_ms_from_predictor(
    step: "VllmStepShape",
    *,
    predictor: "CleanStepPredictor",  # type: ignore[name-defined]
    context_lens: "Mapping[int, int]",  # type: ignore[name-defined]
) -> VllmEngineStepCost:
    """Step pricing via the validated clean step predictors.

    Replaces kernel composition with type dispatch:
      * decode-only  → predictor.decode_ms(B, T_avg)
      * prefill-only → sum predictor.cached_prefill_ms(U_i, P_i) per chunk
      * mixed        → max(prefill_total, decode_total)

    The resulting ``total_ms`` is laid into ``prefill_attention_ms`` for
    prefill-bearing steps and ``decode_attention_ms`` for decode-only steps,
    so downstream code that introspects component fields keeps reasonable
    annotations.  Only ``total_ms`` matters for TPOT aggregation.
    """
    cost = predictor.step_ms(step, context_lens)
    decode_field = cost.decode_ms if cost.classification != "prefill_only" else 0.0
    prefill_field = cost.prefill_ms if cost.classification != "decode_only" else 0.0
    if cost.classification == "mixed":
        # Map total_ms onto a single bucket so callers summing component
        # fields land on total_ms.  The "winning" side gets the ms; the
        # other side reports zero.
        if cost.prefill_ms >= cost.decode_ms:
            prefill_field = cost.total_ms
            decode_field = 0.0
        else:
            decode_field = cost.total_ms
            prefill_field = 0.0
    return VllmEngineStepCost(
        dense_ms=0.0,
        decode_attention_ms=decode_field,
        prefill_attention_ms=prefill_field,
        small_kernel_ms=0.0,
        logits_sampling_ms=0.0,
        runtime_overhead_ms=0.0,
        recompute_or_preemption_ms=0.0,
    )


def predict_mean_tpot_from_predictor(
    steps: "Sequence[VllmStepShape]",
    *,
    context_len: int,
    predictor: "CleanStepPredictor",  # type: ignore[name-defined]
) -> VllmTurnTimingResult:
    """Aggregate per-step costs via the clean step predictor.

    Computes per-request ITLs as **wall-clock deltas between consecutive
    decode emissions**, which is what real-benchmark TPOT measures.  When a
    request decodes in steps [k, m, n], its ITLs are
    [sum(step_k+1..step_m), sum(step_m+1..step_n)] — i.e. how much wall time
    elapsed between consecutive token emissions for that request.

    This is the correct semantic to use when steps differ in cost (e.g. the
    mixed/intrusion steps the staircase generates), versus the legacy
    "append step_ms when decoded" approximation that overcounts mixed-step
    cost.
    """
    step_costs: list[VllmEngineStepCost] = []
    step_wall_ms: list[float] = []
    total_step_ms = 0.0

    for step in steps:
        ctx_map = {rid: context_len for rid in step.decoded_request_ids}
        cost = engine_step_cost_ms_from_predictor(
            step, predictor=predictor, context_lens=ctx_map
        )
        step_costs.append(cost)
        step_wall_ms.append(cost.total_ms)
        total_step_ms += cost.total_ms

    # Cumulative wall at end of each step (index = step index).
    cumulative: list[float] = []
    running = 0.0
    for ms in step_wall_ms:
        running += ms
        cumulative.append(running)

    # Per-request: ordered list of step indices in which it was decoded.
    decode_step_indices: dict[int, list[int]] = defaultdict(list)
    for idx, step in enumerate(steps):
        for request_id in step.decoded_request_ids:
            decode_step_indices[request_id].append(idx)

    if not decode_step_indices:
        raise ValueError("cannot compute TPOT without decoded scheduler steps")

    request_itl_means: list[float] = []
    pooled_numerator = 0.0
    pooled_denominator = 0
    for indices in decode_step_indices.values():
        if len(indices) < 2:
            # Single emission has no ITL to measure.  Fall back to the cost
            # of the step it was decoded in (matches the legacy semantic
            # for output_tokens=1 turns).
            single_ms = step_wall_ms[indices[0]]
            request_itl_means.append(single_ms)
            pooled_numerator += single_ms
            pooled_denominator += 1
            continue
        itls = [
            cumulative[indices[i + 1]] - cumulative[indices[i]]
            for i in range(len(indices) - 1)
        ]
        request_itl_means.append(statistics.fmean(itls))
        # Count one slot per emission (matches pooled_itl semantics of the
        # legacy function, which used len(decoded_request_ids) per step).
        pooled_numerator += sum(itls)
        pooled_denominator += len(itls)

    pooled_itl = pooled_numerator / pooled_denominator if pooled_denominator else 0.0
    return VllmTurnTimingResult(
        mean_tpot_ms=statistics.fmean(request_itl_means),
        pooled_itl_ms=pooled_itl,
        total_step_ms=total_step_ms,
        total_decode_slots=sum(len(v) for v in decode_step_indices.values()),
        decoded_request_count=len(decode_step_indices),
        step_costs=tuple(step_costs),
    )


def engine_step_cost_ms_from_roofline(
    step: "VllmStepShape",
    *,
    predictor: "RooflineStepPredictor",  # type: ignore[name-defined]
    context_lens: "Mapping[int, int]",  # type: ignore[name-defined]
) -> VllmEngineStepCost:
    """Step pricing via the analytical 3D roofline floor.

    Routes through :meth:`RooflineStepPredictor.step_ms` and lays the result
    onto the existing ``VllmEngineStepCost`` schema so downstream aggregation
    works unchanged.  The total_ms is placed in either ``prefill_attention_ms``
    or ``decode_attention_ms`` depending on which side of the roofline
    dominated, preserving introspection.
    """
    cost = predictor.step_ms(step, context_lens)
    # When compute-bound, the work was dominated by matmuls on the scheduled
    # tokens — surface that as the dense_ms component.  Bandwidth-bound work
    # is dominated by KV/weight reads — surface as attention_ms.
    if cost.classification == "compute_bound":
        return VllmEngineStepCost(
            dense_ms=cost.total_ms,
            decode_attention_ms=0.0,
            prefill_attention_ms=0.0,
            small_kernel_ms=0.0,
            logits_sampling_ms=0.0,
            runtime_overhead_ms=0.0,
            recompute_or_preemption_ms=0.0,
        )
    if cost.classification == "bandwidth_bound":
        # Split between decode and prefill attention based on which dominated
        # the step's token mix.
        prefill_field = cost.total_ms if step.prefill_tokens > 0 and step.decode_batch == 0 else 0.0
        decode_field = cost.total_ms if step.decode_batch > 0 and step.prefill_tokens == 0 else 0.0
        # Mixed bandwidth-bound: route to whichever side had more tokens.
        if step.prefill_tokens > 0 and step.decode_batch > 0:
            if step.prefill_tokens >= step.decode_batch:
                prefill_field = cost.total_ms
            else:
                decode_field = cost.total_ms
        return VllmEngineStepCost(
            dense_ms=0.0,
            decode_attention_ms=decode_field,
            prefill_attention_ms=prefill_field,
            small_kernel_ms=0.0,
            logits_sampling_ms=0.0,
            runtime_overhead_ms=0.0,
            recompute_or_preemption_ms=0.0,
        )
    return VllmEngineStepCost(
        dense_ms=0.0,
        decode_attention_ms=0.0,
        prefill_attention_ms=0.0,
        small_kernel_ms=0.0,
        logits_sampling_ms=0.0,
        runtime_overhead_ms=0.0,
        recompute_or_preemption_ms=0.0,
    )


def predict_mean_tpot_from_roofline(
    steps: "Sequence[VllmStepShape]",
    *,
    context_len: int,
    predictor: "RooflineStepPredictor",  # type: ignore[name-defined]
) -> VllmTurnTimingResult:
    """TPOT via the analytical roofline, using wall-clock ITL semantics.

    Mirrors ``predict_mean_tpot_from_predictor``: ITL between consecutive
    decode emissions per request, NOT step_ms per emission.  This is the
    correct semantic when per-step costs vary (which they do with the
    staircase admission).
    """
    step_costs: list[VllmEngineStepCost] = []
    step_wall_ms: list[float] = []
    total_step_ms = 0.0

    for step in steps:
        ctx_map = {rid: context_len for rid in step.decoded_request_ids}
        cost = engine_step_cost_ms_from_roofline(
            step, predictor=predictor, context_lens=ctx_map
        )
        step_costs.append(cost)
        step_wall_ms.append(cost.total_ms)
        total_step_ms += cost.total_ms

    cumulative: list[float] = []
    running = 0.0
    for ms in step_wall_ms:
        running += ms
        cumulative.append(running)

    decode_step_indices: dict[int, list[int]] = defaultdict(list)
    for idx, step in enumerate(steps):
        for request_id in step.decoded_request_ids:
            decode_step_indices[request_id].append(idx)

    if not decode_step_indices:
        raise ValueError("cannot compute TPOT without decoded scheduler steps")

    request_itl_means: list[float] = []
    pooled_numerator = 0.0
    pooled_denominator = 0
    for indices in decode_step_indices.values():
        if len(indices) < 2:
            single_ms = step_wall_ms[indices[0]]
            request_itl_means.append(single_ms)
            pooled_numerator += single_ms
            pooled_denominator += 1
            continue
        itls = [
            cumulative[indices[i + 1]] - cumulative[indices[i]]
            for i in range(len(indices) - 1)
        ]
        request_itl_means.append(statistics.fmean(itls))
        pooled_numerator += sum(itls)
        pooled_denominator += len(itls)

    pooled_itl = pooled_numerator / pooled_denominator if pooled_denominator else 0.0
    return VllmTurnTimingResult(
        mean_tpot_ms=statistics.fmean(request_itl_means),
        pooled_itl_ms=pooled_itl,
        total_step_ms=total_step_ms,
        total_decode_slots=sum(len(v) for v in decode_step_indices.values()),
        decoded_request_count=len(decode_step_indices),
        step_costs=tuple(step_costs),
    )


def _prefill_chunks_or_aggregate_fallback(
    step: VllmStepShape,
    *,
    context_len: int,
) -> tuple[VllmPrefillChunk, ...]:
    if step.prefill_chunks:
        return step.prefill_chunks
    if step.prefill_tokens <= 0:
        return ()
    return (
        VllmPrefillChunk(
            request_id=-1,
            scheduled_tokens=int(step.prefill_tokens),
            prefix_tokens=max(0, int(context_len)),
        ),
    )
