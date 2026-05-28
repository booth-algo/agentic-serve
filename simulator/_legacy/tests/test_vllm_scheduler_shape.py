from __future__ import annotations

import pytest

from simulator._legacy.vllm_scheduler_shape import (
    VllmRequestShape,
    VllmSchedulerConfig,
    benchmark_row_requests,
    simulate_vllm_turn_shape,
)


def test_single_request_prefills_then_decodes_without_latency() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(
                request_id=0,
                total_context_tokens=4,
                cached_context_tokens=0,
                new_prefill_tokens=4,
                output_tokens=3,
            )
        ],
        VllmSchedulerConfig(max_num_batched_tokens=4),
    )

    assert result.summary.steps == 4
    assert result.summary.prefill_only_steps == 1
    assert result.summary.decode_only_steps == 3
    assert result.summary.mixed_decode_prefill_steps == 0
    assert result.summary.total_decode_slots == 3
    assert result.summary.total_prefill_tokens == 4


def test_mixed_step_uses_leftover_budget_after_decode() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(0, 1, 0, 1, 2, prefill_work_tokens=1),
            VllmRequestShape(1, 4, 0, 4, 1, prefill_work_tokens=4),
        ],
        VllmSchedulerConfig(max_num_batched_tokens=4),
    )

    mixed_steps = [
        step
        for step in result.steps
        if step.decode_batch > 0 and step.prefill_tokens > 0
    ]
    assert mixed_steps
    assert mixed_steps[0].decode_batch == 1
    assert mixed_steps[0].prefill_tokens == 1
    assert result.summary.prefill_intrusion_candidate


def test_decode_tokens_can_consume_prefill_budget() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(0, 1, 0, 1, 2, prefill_work_tokens=1),
            VllmRequestShape(1, 1, 0, 1, 2, prefill_work_tokens=1),
            VllmRequestShape(2, 4, 0, 4, 1, prefill_work_tokens=4),
        ],
        VllmSchedulerConfig(max_num_batched_tokens=3),
    )

    assert result.steps[1].decode_batch == 2
    assert result.steps[1].prefill_tokens == 1


def test_max_num_seqs_queues_waiting_requests() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(i, 1, 0, 1, 1, prefill_work_tokens=1)
            for i in range(3)
        ],
        VllmSchedulerConfig(max_num_batched_tokens=8, max_num_seqs=2),
    )

    assert result.summary.max_running_queue == 2
    assert result.summary.max_waiting_queue == 1
    assert result.summary.max_decode_batch == 2


def test_cached_context_counts_as_resident_kv_blocks() -> None:
    result = simulate_vllm_turn_shape(
        [
            VllmRequestShape(
                request_id=0,
                total_context_tokens=64,
                cached_context_tokens=64,
                new_prefill_tokens=0,
                output_tokens=1,
                prefill_work_tokens=1,
            )
        ],
        VllmSchedulerConfig(
            max_num_batched_tokens=4,
            cache_block_size=16,
            available_gpu_kv_blocks=10,
        ),
    )

    assert result.steps[0].used_kv_blocks == 5
    assert result.steps[0].free_kv_blocks == 5
    assert result.summary.max_used_kv_blocks == 5


def test_kv_capacity_limits_admission_when_cached_context_is_resident() -> None:
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=64,
            cached_context_tokens=64,
            new_prefill_tokens=0,
            output_tokens=1,
            prefill_work_tokens=1,
        )
        for i in range(2)
    ]

    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=8,
            max_num_seqs=2,
            cache_block_size=16,
            available_gpu_kv_blocks=8,
        ),
    )

    assert result.summary.max_running_queue == 1
    assert result.summary.max_capacity_waiting_requests == 1
    assert result.summary.kv_pressure_candidate
    assert result.steps[0].admitted_request_ids == (0,)


def test_decode_allocation_failure_preempts_tail_running_request() -> None:
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=32,
            cached_context_tokens=16,
            new_prefill_tokens=16,
            output_tokens=1,
            prefill_work_tokens=16,
        )
        for i in range(2)
    ]

    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=2,
            cache_block_size=16,
            available_gpu_kv_blocks=4,
        ),
    )

    preempting_steps = [step for step in result.steps if step.preemptions > 0]
    assert len(preempting_steps) == 1
    assert preempting_steps[0].decoded_request_ids == (0,)
    assert preempting_steps[0].preempted_request_ids == (1,)
    assert preempting_steps[0].recompute_prefill_tokens == 16
    assert result.summary.total_preemptions == 1
    assert result.summary.total_recompute_prefill_tokens == 16
    assert result.summary.total_prefill_tokens == 48
    assert result.steps[-1].completed_request_ids == (1,)


def test_waiting_capacity_failure_does_not_preempt_decoders() -> None:
    requests = [
        VllmRequestShape(
            request_id=0,
            total_context_tokens=15,
            cached_context_tokens=14,
            new_prefill_tokens=1,
            output_tokens=1,
            prefill_work_tokens=1,
        ),
        VllmRequestShape(
            request_id=1,
            total_context_tokens=15,
            cached_context_tokens=14,
            new_prefill_tokens=1,
            output_tokens=1,
            prefill_work_tokens=1,
        ),
        VllmRequestShape(
            request_id=2,
            total_context_tokens=32,
            cached_context_tokens=32,
            new_prefill_tokens=1,
            output_tokens=1,
            prefill_work_tokens=1,
        ),
    ]

    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=16,
            max_num_seqs=3,
            cache_block_size=16,
            available_gpu_kv_blocks=3,
        ),
    )

    assert result.steps[0].admitted_request_ids == (0, 1)
    assert result.steps[0].prefill_tokens == 2
    assert result.steps[1].capacity_waiting_requests == 1
    assert result.steps[1].decoded_request_ids == (0, 1)
    assert result.steps[0].preemptions == 0
    assert result.summary.total_preemptions == 0
    assert result.summary.max_capacity_waiting_requests == 1


def test_synthetic_shared_prefix_matches_offline_trace_prefill_accounting() -> None:
    requests = benchmark_row_requests(
        request_count=80,
        context_len=6188,
        output_tokens=28,
        new_prefill_tokens=138,
        cached_context_tokens=5904,
        mode="synthetic_shared_prefix",
        shared_prefix_tokens=1024,
        bos_token_offset=1,
    )

    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=16_384,
            max_num_seqs=512,
            shared_prefix_tokens=1024,
            bos_token_offset=1,
            available_gpu_kv_blocks=27_651,
        ),
    )

    assert sum(request.prefill_work_tokens or 0 for request in requests) == 414_224
    assert result.summary.total_prefill_tokens == 414_224
    assert result.summary.max_decode_batch == 80
    assert result.summary.min_free_kv_blocks > 0


def test_raises_when_kv_capacity_prevents_any_progress() -> None:
    with pytest.raises(RuntimeError, match="made no progress"):
        simulate_vllm_turn_shape(
            [
                VllmRequestShape(
                    request_id=0,
                    total_context_tokens=1024,
                    cached_context_tokens=0,
                    new_prefill_tokens=1024,
                    output_tokens=1,
                )
            ],
            VllmSchedulerConfig(
                available_gpu_kv_blocks=1,
                admit_by_kv_capacity=True,
            ),
        )


def test_bandwidth_disabled_by_default_preserves_legacy_shape() -> None:
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=1024,
            cached_context_tokens=1024,
            new_prefill_tokens=0,
            output_tokens=4,
            prefill_work_tokens=1,
        )
        for i in range(64)
    ]
    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_seqs=64,
            available_gpu_kv_blocks=8192,
        ),
    )
    assert result.summary.bandwidth_capped_steps == 0
    assert result.summary.max_decode_batch == 64
    assert all(not step.bandwidth_capped for step in result.steps)


def test_bandwidth_caps_decode_batch_into_waves() -> None:
    # 64 requests, each with ~1024 tokens resident.  KV bytes per token =
    # 128 KB → each request = 128 MB.  Bandwidth budget = 100 GB/s * 10 ms
    # = 1 GB → exactly 8 requests fit per step under bandwidth.
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=1024,
            cached_context_tokens=1024,
            new_prefill_tokens=0,
            output_tokens=1,
            prefill_work_tokens=1,
        )
        for i in range(64)
    ]
    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=4096,
            max_num_seqs=64,
            available_gpu_kv_blocks=8192,
            hbm_bw_gbps=100.0,
            kv_bytes_per_token=128 * 1024,
            step_budget_ms=10.0,
        ),
    )
    decode_only_steps = [step for step in result.steps if step.decode_batch > 0]
    assert decode_only_steps, "expected at least one decode step"
    # Each decode step should hold <=8 requests under the bandwidth budget.
    assert max(step.decode_batch for step in decode_only_steps) <= 8
    assert result.summary.bandwidth_capped_steps >= 1


def test_bandwidth_metrics_track_decode_kv_bytes() -> None:
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=512,
            cached_context_tokens=512,
            new_prefill_tokens=0,
            output_tokens=1,
            prefill_work_tokens=1,
        )
        for i in range(4)
    ]
    result = simulate_vllm_turn_shape(
        requests,
        VllmSchedulerConfig(
            max_num_batched_tokens=256,
            max_num_seqs=4,
            available_gpu_kv_blocks=1024,
            hbm_bw_gbps=3000.0,
            kv_bytes_per_token=128 * 1024,
            step_budget_ms=20.0,
        ),
    )
    decode_step = next(step for step in result.steps if step.decode_batch == 4)
    # 4 requests * 512 ctx + 1 emitted decode token each = (513) * 4 * 128KB
    expected_lower_bound = 4 * 512 * 128 * 1024
    assert decode_step.decode_kv_read_bytes >= expected_lower_bound
    assert result.summary.total_decode_kv_read_bytes >= expected_lower_bound


def test_block_pool_staircase_produces_mixed_steps_when_legacy_does_not() -> None:
    # Reproduce the c=40 t=12 swebench gap: 40 requests, ctx=7188, cached
    # cross-turn = 6568, only 144 new tokens per request from the benchmark's
    # cache-hit view.  The legacy path admits all 40 in step 0 (zero mixed
    # steps).  The block-pool path admits in stages, producing the mixed
    # steps that real vLLM exhibits.
    #
    # NB: this test pins ``max_admissions_per_step=4`` to exercise the
    # bounded staircase deliberately.  The production default is 0
    # (unlimited) — which together with intra-turn prefix sharing collapses
    # mixed steps further (~1 here) because cached siblings admit cheaply.
    # Real vLLM at this workload has 15 mixed and 39 total steps.
    legacy_requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=7188,
            cached_context_tokens=6568,
            new_prefill_tokens=144,
            output_tokens=22,
            prefill_work_tokens=144,  # cached-mode shortcut the legacy path uses
        )
        for i in range(40)
    ]
    legacy_cfg = VllmSchedulerConfig(
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        available_gpu_kv_blocks=27_651,
        cache_block_size=16,
        use_block_pool=False,
    )
    legacy = simulate_vllm_turn_shape(legacy_requests, legacy_cfg).summary

    block_pool_cfg = VllmSchedulerConfig(
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        available_gpu_kv_blocks=27_651,
        cache_block_size=16,
        use_block_pool=True,
        max_admissions_per_step=4,
    )
    staircase = simulate_vllm_turn_shape(legacy_requests, block_pool_cfg).summary

    # Legacy still produces 0 mixed steps for this workload (the regression we
    # are explicitly working around).
    assert legacy.mixed_decode_prefill_steps == 0
    # Qualitative check: the bounded staircase produces a non-trivial
    # number of mixed steps where legacy produces zero.  Exact count
    # depends on (cap, partial_prefill_cap, sharing) which evolve as the
    # simulator matches real vLLM more closely; the band is intentionally
    # wide.  Real has 15 mixed at c=40 t=12.
    assert 5 <= staircase.mixed_decode_prefill_steps <= 60
    # Total step count: real is 39; sim lands within ~50% either way.
    assert 20 <= staircase.steps <= 70


def test_block_pool_caches_blocks_after_anchor_completes() -> None:
    # 2 requests sharing a 64-token (4-block) prefix.  Anchor finishes
    # prefill first; sibling should hit those 4 cached blocks on admission.
    shared = 64
    ctx = 128
    output = 1
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=ctx,
            cached_context_tokens=0,
            new_prefill_tokens=ctx,
            output_tokens=output,
            shared_prefix_tokens=shared,
        )
        for i in range(2)
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=2,
        available_gpu_kv_blocks=32,
        cache_block_size=16,
        use_block_pool=True,
        max_admissions_per_step=2,
    )
    result = simulate_vllm_turn_shape(requests, cfg)
    # Per-request prefill scheduled tokens, summed.
    tokens_by_request: dict[int, int] = {}
    for step in result.steps:
        for chunk in step.prefill_chunks:
            tokens_by_request[chunk.request_id] = (
                tokens_by_request.get(chunk.request_id, 0) + chunk.scheduled_tokens
            )
    # Anchor does full ctx; sibling skips the 4 cached blocks (64 tokens).
    assert tokens_by_request[0] == ctx
    assert tokens_by_request[1] == ctx - shared


def test_block_pool_admission_fails_when_pool_too_small() -> None:
    # 4 requests each needing 64 tokens (4 blocks).  Pool holds only 6 blocks.
    # First admission consumes 4; second can't fit.
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=64,
            cached_context_tokens=0,
            new_prefill_tokens=64,
            output_tokens=1,
        )
        for i in range(4)
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        available_gpu_kv_blocks=6,
        cache_block_size=16,
        use_block_pool=True,
        max_admissions_per_step=4,
    )
    result = simulate_vllm_turn_shape(requests, cfg)
    # At least one early step had at most 1 admitted; the pool gates the
    # rest until earlier requests free their blocks.
    step_zero_admits = result.steps[0].admitted_request_ids
    assert len(step_zero_admits) == 1
    # And all 4 requests eventually completed.
    assert result.summary.total_decode_slots == 4


def test_block_pool_path_is_off_by_default() -> None:
    # Sanity: a workload that the staircase would resolve differently from
    # the legacy path defaults to the legacy answer.
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=128,
            cached_context_tokens=0,
            new_prefill_tokens=128,
            output_tokens=1,
        )
        for i in range(8)
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        available_gpu_kv_blocks=64,
        cache_block_size=16,
    )
    # Legacy default: use_block_pool=False.
    assert cfg.use_block_pool is False
    result = simulate_vllm_turn_shape(requests, cfg)
    # The legacy path admits all 8 in step 0; staircase would admit in stages.
    assert result.steps[0].running_queue == 8


def test_benchmark_row_requests_uses_per_request_output_tokens_list() -> None:
    osls = [10, 100, 1000]
    requests = benchmark_row_requests(
        request_count=3,
        context_len=128,
        output_tokens=42,  # ignored because the per-request list is provided
        new_prefill_tokens=64,
        cached_context_tokens=0,
        output_tokens_per_request=osls,
    )
    assert [r.output_tokens for r in requests] == [10, 100, 1000]


def test_benchmark_row_requests_pads_short_per_request_list_by_cycling() -> None:
    # Three values, eight requests → the distribution cycles to preserve shape.
    requests = benchmark_row_requests(
        request_count=8,
        context_len=64,
        output_tokens=1,
        new_prefill_tokens=8,
        cached_context_tokens=0,
        output_tokens_per_request=[5, 50, 500],
    )
    assert [r.output_tokens for r in requests] == [5, 50, 500, 5, 50, 500, 5, 50]


def test_benchmark_row_requests_truncates_long_per_request_list() -> None:
    requests = benchmark_row_requests(
        request_count=2,
        context_len=64,
        output_tokens=1,
        new_prefill_tokens=8,
        cached_context_tokens=0,
        output_tokens_per_request=[1, 2, 3, 4, 5],
    )
    assert [r.output_tokens for r in requests] == [1, 2]


def test_benchmark_row_requests_falls_back_to_scalar_when_list_omitted() -> None:
    requests = benchmark_row_requests(
        request_count=3,
        context_len=64,
        output_tokens=77,
        new_prefill_tokens=8,
        cached_context_tokens=0,
    )
    assert [r.output_tokens for r in requests] == [77, 77, 77]


def test_benchmark_row_requests_threads_per_request_prefill_and_cache() -> None:
    requests = benchmark_row_requests(
        request_count=3,
        context_len=128,
        output_tokens=10,
        new_prefill_tokens=8,
        cached_context_tokens=0,
        new_prefill_tokens_per_request=[16, 32, 64],
        cached_context_tokens_per_request=[0, 16, 32],
    )
    assert [r.new_prefill_tokens for r in requests] == [16, 32, 64]
    assert [r.cached_context_tokens for r in requests] == [0, 16, 32]
    # prefill_work_tokens should track per-request new_prefill in benchmark_cache mode.
    assert [r.prefill_work_tokens for r in requests] == [16, 32, 64]


def test_multi_turn_replay_cohort_turn_index_is_monotonic_and_partitions_steps() -> None:
    from simulator._legacy.vllm_scheduler_shape import (
        simulate_vllm_multi_turn_replay,
        summarize_replay_for_turn,
    )
    # 2 sessions × 3 turns each, staggered arrivals.  Verify
    # cohort_turn_index never decreases AND the per-turn summary step
    # counts sum to total step count (no double-counting, no gaps).
    sessions = [
        [
            {"turn_index": 0, "arrival_offset_ms": 0.0,   "completion_offset_ms": 50,  "new_prefill_tokens": 64, "cached_context_tokens": 0,  "output_tokens": 4},
            {"turn_index": 1, "arrival_offset_ms": 100.0, "completion_offset_ms": 200, "new_prefill_tokens": 32, "cached_context_tokens": 64, "output_tokens": 3},
            {"turn_index": 2, "arrival_offset_ms": 300.0, "completion_offset_ms": 400, "new_prefill_tokens": 32, "cached_context_tokens": 96, "output_tokens": 2},
        ],
        [
            {"turn_index": 0, "arrival_offset_ms": 5.0,   "completion_offset_ms": 55,  "new_prefill_tokens": 64, "cached_context_tokens": 0,  "output_tokens": 6},
            {"turn_index": 1, "arrival_offset_ms": 120.0, "completion_offset_ms": 220, "new_prefill_tokens": 32, "cached_context_tokens": 64, "output_tokens": 4},
            {"turn_index": 2, "arrival_offset_ms": 320.0, "completion_offset_ms": 420, "new_prefill_tokens": 32, "cached_context_tokens": 96, "output_tokens": 3},
        ],
    ]
    cfg = VllmSchedulerConfig(
        use_block_pool=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        available_gpu_kv_blocks=128,
    )
    result = simulate_vllm_multi_turn_replay(
        sessions, cfg, step_pricer=lambda step, ctx: 1.0
    )
    # cohort_turn_index is monotonically non-decreasing.
    tags = [s.cohort_turn_index for s in result.steps]
    assert tags == sorted(tags), f"cohort_turn_index not monotonic: {tags}"
    # First step where cohort flips to N coincides with the first
    # admission of any turn-N request.
    seen_admit_turn: set[int] = set()
    for i, s in enumerate(result.steps):
        for chunk_turn in s.prefill_turn_indices:
            if chunk_turn not in seen_admit_turn:
                seen_admit_turn.add(chunk_turn)
                # cohort_turn_index for this step should be >= chunk_turn.
                assert s.cohort_turn_index >= chunk_turn
    # Per-turn summaries partition the steps (no double-counting).
    total = len(result.steps)
    summed = sum(
        summarize_replay_for_turn(result, t).steps for t in (0, 1, 2)
    )
    # Steps before any admission have cohort_turn_index == -1 and won't
    # be claimed by any positive target turn — allow that gap.
    leading_uncounted = sum(1 for s in result.steps if s.cohort_turn_index < 0)
    assert summed + leading_uncounted == total


def test_block_pool_spawns_next_turn_followup_when_original_finishes() -> None:
    # 4 siblings with varied OSLs (some short, some long).  next_turn
    # prefill of 64 tokens means each completing original triggers a new
    # follow-up admission while others are still decoding → real-vLLM-
    # style turn-boundary mixed steps.
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=64,
            cached_context_tokens=0,
            new_prefill_tokens=64,
            output_tokens=osl,
            next_turn_new_prefill_tokens=64,
        )
        for i, osl in enumerate([2, 4, 6, 8])  # spread completion across steps
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=256,
        max_num_seqs=8,
        available_gpu_kv_blocks=64,
        cache_block_size=16,
        use_block_pool=True,
    )
    result = simulate_vllm_turn_shape(requests, cfg)
    # At least one mixed step: a follow-up admits while at least one
    # original is still decoding.
    assert result.summary.mixed_decode_prefill_steps >= 1
    # All four originals completed their full decode.
    total_originals_decoded = sum(
        len(step.decoded_request_ids) for step in result.steps
        if any(rid >= 0 for rid in step.decoded_request_ids)
    )
    # Originals collectively emit 2+4+6+8 = 20 tokens (decode_request_ids
    # includes follow-up emissions counted separately by sign).
    original_emissions = sum(
        1 for step in result.steps for rid in step.decoded_request_ids if rid >= 0
    )
    assert original_emissions == 2 + 4 + 6 + 8


def test_block_pool_followups_do_not_gate_completion() -> None:
    # A single request with a giant next_turn prefill must NOT keep the
    # simulator running indefinitely after the original finishes.
    requests = [
        VllmRequestShape(
            request_id=0,
            total_context_tokens=32,
            cached_context_tokens=0,
            new_prefill_tokens=32,
            output_tokens=2,
            next_turn_new_prefill_tokens=8192,  # huge — would take many steps
        )
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=128,  # small budget → followup would chunk
        max_num_seqs=4,
        available_gpu_kv_blocks=128,
        cache_block_size=16,
        use_block_pool=True,
    )
    result = simulate_vllm_turn_shape(requests, cfg)
    # Original needs ~1 prefill step + 2 decode steps = ~3 steps total.
    # Followup chunking would take 8192/128 = 64 steps, so a properly-
    # gated sim must stop well before that.
    assert result.summary.steps < 10


def test_block_pool_shares_intra_turn_cached_prefix_when_siblings_arrive_together() -> None:
    # 8 sibling requests with identical leading 96 tokens (6 blocks).  In
    # benchmark_cache mode (shared_prefix_tokens == 0) the new
    # _default_block_hashes should give the first 6 blocks a shared anchor
    # key so siblings hit each other's cache once the first one prefills.
    # The pool pre-population also pre-caches those blocks, so EVERY
    # request — including request 0 — sees the cached prefix.
    shared_tokens = 96  # 6 blocks of 16
    total = 160  # 10 blocks
    requests = [
        VllmRequestShape(
            request_id=i,
            total_context_tokens=total,
            cached_context_tokens=shared_tokens,
            new_prefill_tokens=total - shared_tokens,
            output_tokens=1,
            # Critical: shared_prefix_tokens left at default 0 — this is
            # benchmark_cache mode, the path we just fixed.
        )
        for i in range(8)
    ]
    cfg = VllmSchedulerConfig(
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        available_gpu_kv_blocks=256,
        cache_block_size=16,
        use_block_pool=True,
        # This test is about prefix-cache sharing — opt out of the
        # per-step admission cap so all 8 siblings can admit together
        # and we can verify they all skip the cached prefix.
        max_num_partial_prefills=0,
    )
    result = simulate_vllm_turn_shape(requests, cfg)
    tokens_by_request: dict[int, int] = {}
    for step in result.steps:
        for chunk in step.prefill_chunks:
            tokens_by_request[chunk.request_id] = (
                tokens_by_request.get(chunk.request_id, 0) + chunk.scheduled_tokens
            )
    # Every sibling should skip the 96 cached-prefix tokens and only prefill
    # the 64-token tail.
    for i in range(8):
        assert tokens_by_request[i] == total - shared_tokens, (
            f"request {i} prefilled {tokens_by_request[i]}, expected "
            f"{total - shared_tokens}"
        )
    # And the small tail (64 tokens each) easily fits in the 1024-token
    # budget, so the 8 admissions collapse into one mixed step.
    assert result.summary.mixed_decode_prefill_steps <= 2
