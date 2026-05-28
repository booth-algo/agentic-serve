from __future__ import annotations

import pytest

from simulator._legacy.vllm_engine_step_cost import (
    VllmEngineStepCostModel,
    engine_step_cost_ms,
    predict_mean_tpot_from_engine_steps,
    prefill_attention_work,
)
from simulator._legacy.vllm_scheduler_shape import (
    VllmRequestShape,
    VllmSchedulerConfig,
    VllmStepShape,
    simulate_vllm_turn_shape,
)


def test_engine_step_cost_prices_all_mixed_forward_components() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(0, 1, 0, 1, 2, prefill_work_tokens=1),
            VllmRequestShape(1, 4, 0, 4, 1, prefill_work_tokens=4),
        ],
        VllmSchedulerConfig(max_num_batched_tokens=4),
    )
    mixed_step = next(
        step
        for step in result.steps
        if step.decode_batch > 0 and step.prefill_tokens > 0
    )
    model = VllmEngineStepCostModel(
        dense_ms=lambda scheduled_tokens: scheduled_tokens * 10.0,
        decode_attention_ms=lambda decode_batch, context_len: decode_batch + context_len,
        prefill_attention_ms=prefill_attention_work,
        small_kernel_ms=lambda scheduled_tokens: scheduled_tokens,
        logits_sampling_ms=lambda sampled_positions: sampled_positions * 0.5,
        runtime_overhead_ms=lambda step: 7.0 if step.prefill_tokens else 0.0,
    )

    cost = engine_step_cost_ms(mixed_step, context_len=4, cost_model=model)

    assert mixed_step.decode_batch == 1
    assert mixed_step.decoded_request_ids == (0,)
    assert mixed_step.prefill_tokens == 1
    assert mixed_step.completed_prefill_request_ids == (1,)
    assert [(c.request_id, c.scheduled_tokens, c.prefix_tokens) for c in mixed_step.prefill_chunks] == [
        (1, 1, 3),
    ]
    assert cost.dense_ms == 20.0
    assert cost.decode_attention_ms == 5.0
    assert cost.prefill_attention_ms == 4.0
    assert cost.small_kernel_ms == 2.0
    assert cost.logits_sampling_ms == 1.0
    assert cost.runtime_overhead_ms == 7.0
    assert cost.total_ms == 39.0


def test_prefill_attention_work_uses_prefix_plus_intra_chunk_causal_work() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(0, 8, 4, 4, 1, prefill_work_tokens=4),
        ],
        VllmSchedulerConfig(max_num_batched_tokens=4),
    )

    work = prefill_attention_work(result.steps[0].prefill_chunks)

    assert result.steps[0].prefill_chunks[0].prefix_tokens == 4
    assert work == 4 * 4 + 4 * 5 / 2


def test_predict_mean_tpot_assigns_full_step_cost_to_decoded_requests() -> None:
    steps = (
        _step(step_id=0, decode_ids=(0, 1)),
        _step(step_id=1, decode_ids=(0,)),
    )
    model = VllmEngineStepCostModel(
        runtime_overhead_ms=lambda step: 10.0 if step.step_id == 0 else 30.0,
    )

    timing = predict_mean_tpot_from_engine_steps(
        steps,
        context_len=1,
        cost_model=model,
    )

    assert timing.mean_tpot_ms == 15.0
    assert timing.pooled_itl_ms == pytest.approx((10.0 * 2 + 30.0) / 3)
    assert timing.total_step_ms == 40.0
    assert timing.total_decode_slots == 3
    assert timing.decoded_request_count == 2
    assert timing.mean_tpot_ms != timing.total_step_ms / timing.total_decode_slots


def test_predict_mean_tpot_requires_decoded_steps() -> None:
    with pytest.raises(ValueError, match="without decoded scheduler steps"):
        predict_mean_tpot_from_engine_steps(
            (_step(step_id=0, decode_ids=()),),
            context_len=1,
        )


def test_prefill_small_kernel_override_zeros_only_prefill_bearing_steps() -> None:
    model = VllmEngineStepCostModel(
        dense_ms=lambda scheduled: scheduled * 10.0,
        decode_dense_ms=lambda batch: batch * 5.0,
        small_kernel_ms=lambda scheduled: scheduled * 0.5,
        prefill_small_kernel_ms=lambda _scheduled: 0.0,
    )
    decode_only = _step(step_id=0, decode_ids=(0, 1, 2))
    prefill_step = VllmStepShape(
        step_id=1,
        decode_batch=0,
        decoded_request_ids=(),
        prefill_seqs=1,
        prefill_tokens=16,
        prefill_chunks=(),
        waiting_queue=0,
        running_queue=1,
        free_kv_blocks=1,
        completed_prefill_request_ids=(),
        completed_request_ids=(),
    )

    decode_cost = engine_step_cost_ms(decode_only, context_len=4, cost_model=model)
    prefill_cost = engine_step_cost_ms(prefill_step, context_len=4, cost_model=model)

    # Decode-only step keeps the default small_kernel_ms adder.
    assert decode_cost.small_kernel_ms == 1.5  # 3 * 0.5
    # Prefill-bearing step uses the override, so adder is zero.
    assert prefill_cost.small_kernel_ms == 0.0


def test_engine_step_cost_from_predictor_decode_only() -> None:
    from simulator._legacy.clean_step_predictor import CleanStepPredictor
    from simulator._legacy.vllm_engine_step_cost import engine_step_cost_ms_from_predictor

    predictor = CleanStepPredictor(
        full_prefill_samples=[(64, 5.0), (1024, 20.0)],
        cached_prefill_lattice={(64, 512): 10.0, (128, 512): 11.0},
        decode_lattice={(40, 1024): 8.0, (40, 8192): 30.0},
    )
    step = _step(step_id=0, decode_ids=tuple(range(40)))
    cost = engine_step_cost_ms_from_predictor(
        step, predictor=predictor, context_lens={i: 1024 for i in range(40)}
    )
    assert cost.decode_attention_ms == 8.0
    assert cost.prefill_attention_ms == 0.0
    assert cost.dense_ms == 0.0
    assert cost.total_ms == 8.0


def test_engine_step_cost_from_predictor_mixed_takes_max() -> None:
    from simulator._legacy.clean_step_predictor import CleanStepPredictor
    from simulator._legacy.vllm_engine_step_cost import engine_step_cost_ms_from_predictor
    from simulator._legacy.vllm_scheduler_shape import VllmPrefillChunk

    predictor = CleanStepPredictor(
        full_prefill_samples=[(64, 5.0)],
        cached_prefill_lattice={(64, 512): 10.0},
        decode_lattice={(1, 1024): 6.5, (40, 8192): 30.0},
    )
    # 40 decoders × 8192 ctx = 30.0; 1 chunk @ (64, 512) = 10.0; max = 30
    step = VllmStepShape(
        step_id=0,
        decode_batch=40,
        decoded_request_ids=tuple(range(40)),
        prefill_seqs=1,
        prefill_tokens=64,
        prefill_chunks=(VllmPrefillChunk(99, 64, 512),),
        waiting_queue=0,
        running_queue=41,
        free_kv_blocks=1,
        completed_prefill_request_ids=(),
        completed_request_ids=(),
    )
    cost = engine_step_cost_ms_from_predictor(
        step, predictor=predictor, context_lens={i: 8192 for i in range(40)}
    )
    # Decode wins → 30 lands in decode_attention; prefill side is zero.
    assert cost.decode_attention_ms == 30.0
    assert cost.prefill_attention_ms == 0.0
    assert cost.total_ms == 30.0


def test_predict_mean_tpot_from_predictor_aggregates_per_request() -> None:
    from simulator._legacy.clean_step_predictor import CleanStepPredictor
    from simulator._legacy.vllm_engine_step_cost import predict_mean_tpot_from_predictor

    predictor = CleanStepPredictor(
        full_prefill_samples=[(64, 5.0)],
        cached_prefill_lattice={(64, 512): 10.0},
        decode_lattice={(2, 1024): 4.0},
    )
    # 2 decode steps, each request decodes both times.
    steps = (
        _step(step_id=0, decode_ids=(0, 1)),
        _step(step_id=1, decode_ids=(0, 1)),
    )
    result = predict_mean_tpot_from_predictor(
        steps, context_len=1024, predictor=predictor
    )
    # Each step costs 4.0 ms → each request's mean ITL is 4.0 → mean TPOT 4.0.
    assert result.mean_tpot_ms == 4.0
    assert result.total_step_ms == 8.0
    assert result.total_decode_slots == 4


def test_engine_step_cost_from_roofline_bandwidth_bound() -> None:
    from simulator._legacy.roofline_step_predictor import (
        RooflineParams,
        RooflineStepPredictor,
    )
    from simulator._legacy.vllm_engine_step_cost import engine_step_cost_ms_from_roofline

    predictor = RooflineStepPredictor(
        RooflineParams(
            n_params=1_000_000,
            peak_flops_per_s=1e12,
            peak_bw_bytes_per_s=1e9,  # slow BW so bandwidth dominates
            kv_bytes_per_token=1024.0,
            util_flops=1.0,
            util_bw=1.0,
            bytes_per_param=2.0,
        )
    )
    step = _step(step_id=0, decode_ids=(0, 1))
    cost = engine_step_cost_ms_from_roofline(
        step, predictor=predictor, context_lens={0: 1024, 1: 1024}
    )
    assert cost.dense_ms == 0.0
    assert cost.total_ms > 0
    # Decode-only bandwidth-bound → decode_attention_ms gets the cost.
    assert cost.decode_attention_ms == cost.total_ms


def test_predict_mean_tpot_from_roofline_uses_wall_clock_itl() -> None:
    from simulator._legacy.roofline_step_predictor import (
        RooflineParams,
        RooflineStepPredictor,
    )
    from simulator._legacy.vllm_engine_step_cost import predict_mean_tpot_from_roofline

    predictor = RooflineStepPredictor(
        RooflineParams(
            n_params=1_000_000,
            peak_flops_per_s=1e12,
            peak_bw_bytes_per_s=1e10,
            kv_bytes_per_token=1024.0,
            util_flops=1.0,
            util_bw=1.0,
            bytes_per_param=2.0,
        )
    )
    # 2 decode steps; each request emits both times.  Per-request ITL is the
    # wall delta between consecutive emissions.
    steps = (
        _step(step_id=0, decode_ids=(0, 1)),
        _step(step_id=1, decode_ids=(0, 1)),
    )
    result = predict_mean_tpot_from_roofline(
        steps, context_len=1024, predictor=predictor
    )
    # Each step has the same cost; the wall delta between emission 1 and 2
    # is one step's wall.  So mean_tpot == that step's wall.
    assert result.mean_tpot_ms > 0
    # 2 requests × 2 emissions each = 4 total decode slots.
    assert result.total_decode_slots == 4
    assert result.decoded_request_count == 2


def test_prefill_small_kernel_default_falls_back_to_small_kernel_ms() -> None:
    model = VllmEngineStepCostModel(
        dense_ms=lambda scheduled: 0.0,
        small_kernel_ms=lambda scheduled: scheduled * 0.25,
    )
    prefill_step = VllmStepShape(
        step_id=0,
        decode_batch=0,
        decoded_request_ids=(),
        prefill_seqs=1,
        prefill_tokens=8,
        prefill_chunks=(),
        waiting_queue=0,
        running_queue=1,
        free_kv_blocks=1,
        completed_prefill_request_ids=(),
        completed_request_ids=(),
    )

    cost = engine_step_cost_ms(prefill_step, context_len=4, cost_model=model)

    # No override → both step types still call small_kernel_ms.
    assert cost.small_kernel_ms == 2.0  # 8 * 0.25


def _step(*, step_id: int, decode_ids: tuple[int, ...]) -> VllmStepShape:
    return VllmStepShape(
        step_id=step_id,
        decode_batch=len(decode_ids),
        decoded_request_ids=decode_ids,
        prefill_seqs=0,
        prefill_tokens=0,
        prefill_chunks=(),
        waiting_queue=0,
        running_queue=len(decode_ids),
        free_kv_blocks=1,
        completed_prefill_request_ids=(),
        completed_request_ids=(),
    )
