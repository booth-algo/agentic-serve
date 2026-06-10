#!/usr/bin/env python3
"""Collect best-effort vLLM scheduler-step evidence for TPOT cliff analysis.

This script is intentionally not a latency profiler. It runs benchmark-shaped
offline vLLM generations and monkeypatches scheduler `schedule()` methods so we
can inspect what the engine actually scheduled at each step: decode batch,
prefill tokens, queue sizes, KV-block pressure, and preemption/swap/recompute
counters when this vLLM version exposes them.

The trace is diagnostic evidence for separating:

* effective microbatching / decode waves,
* fresh-prefill work sharing engine iterations with decode,
* graph-bucket changes,
* KV-block pressure and preemption behavior.

Because vLLM scheduler internals vary by version, extraction is best-effort and
preserves a compact `raw_summary` column for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TURNS = Path("profiling/results/benchmark_turns_llama31_8b_h100_vllm.csv")
DEFAULT_OUTPUT = Path("profiling/results/vllm_engine_step_trace.csv")
BENCHMARK_REQUEST_ID_PREFIX = "agenticbench"

TRACE_FIELDS = [
    "run_id",
    "profile",
    "concurrency",
    "turn_index",
    "trace_scope",
    "trace_phase",
    "target_turn_index",
    "primary_eval",
    "diagnostic_reason",
    "scheduled_request_count",
    "successful_request_count",
    "batch_size",
    "context_len",
    "output_tokens",
    "new_prefill_tokens",
    "cached_context_tokens",
    "cache_hit_rate",
    "prompt_shape_mode",
    "trace_prompt_tokens",
    "trace_shared_prefix_tokens",
    "trace_warmup_cached_tokens",
    "trace_unique_tail_tokens",
    "scheduler_label",
    "step_id",
    "engine_step_wall_ms",
    "model_submit_wall_ms",
    "model_wait_wall_ms",
    "sample_wall_ms",
    "scheduler_update_wall_ms",
    "model_executed",
    "worker_execute_wall_ms",
    "worker_execute_cuda_sync_ms",
    "scheduler_wall_ms",
    "decode_batch",
    "decode_request_ids",
    "prefill_seqs",
    "prefill_tokens",
    "prefill_request_ids",
    "scheduled_request_ids",
    "total_scheduled_tokens",
    "waiting_queue",
    "running_queue",
    "waiting_request_ids",
    "running_request_ids",
    "skipped_waiting_request_ids",
    "free_kv_blocks",
    "graph_bucket",
    "preemptions",
    "preempted_request_ids",
    "swaps",
    "recomputes",
    "engine_computed_tokens_sum",
    "engine_uncached_prefill_tokens_sum",
    "engine_cache_hit_rate_mean",
    "engine_cache_lookup_request_count",
    "engine_new_block_count",
    "engine_allocate_none_count",
    "engine_cache_truth",
    "raw_summary",
]

WALL_TRACE_FIELDS = [
    "engine_step_wall_ms",
    "model_submit_wall_ms",
    "model_wait_wall_ms",
    "sample_wall_ms",
    "scheduler_update_wall_ms",
    "model_executed",
    "worker_execute_wall_ms",
    "worker_execute_cuda_sync_ms",
]

_ENGINE_TRACE_STATE: dict[str, Any] = {
    "engine_wall_enabled": False,
    "pending_rows": [],
    "worker_timings": [],
    "kv_events": [],
}


@dataclass(frozen=True)
class TraceCase:
    profile: str
    concurrency: int
    turn_index: int
    batch_size: int
    context_len: int
    output_tokens: int
    scheduled_request_count: int
    successful_request_count: int
    new_prefill_tokens: int
    cached_context_tokens: int = 0
    cache_hit_rate: float = 0.0
    primary_eval: bool = True
    diagnostic_reason: str = ""
    trace_scope: str = "single-turn"
    trace_phase: str = "target_turn"
    target_turn_index: int = -1
    prompt_shape_mode: str = "synthetic-shared-prefix"
    trace_prompt_tokens: int = 0
    trace_shared_prefix_tokens: int = 0
    trace_warmup_cached_tokens: int = 0
    trace_unique_tail_tokens: int = 0


@dataclass(frozen=True)
class PromptBatch:
    target_prompts: list[Any]
    warmup_prompts: list[Any]
    trace_prompt_tokens: int
    trace_shared_prefix_tokens: int
    trace_warmup_cached_tokens: int
    trace_unique_tail_tokens: int


class FullCellPromptState:
    """Synthetic per-session prompt history for full-cell trace replay.

    Single-turn trace mode warms a target prefix explicitly. Full-cell mode uses
    previous turns as the warmup instead: each synthetic session reuses its
    actual prompt + generated token history when constructing later turns.
    """

    def __init__(self) -> None:
        self.histories: dict[int, list[int]] = {}
        self.records: list[dict[str, Any]] = []

    def make_prompt_batch(self, tokenizer: Any, case: TraceCase) -> PromptBatch:
        target_prompts: list[Any] = []
        for session_index in range(case.batch_size):
            prompt_ids = self._prompt_ids_for_session(
                tokenizer,
                case=case,
                session_index=session_index,
            )
            target_prompts.append(make_token_prompt(prompt_ids))
        return PromptBatch(
            target_prompts=target_prompts,
            warmup_prompts=[],
            trace_prompt_tokens=case.context_len,
            trace_shared_prefix_tokens=0,
            trace_warmup_cached_tokens=0,
            trace_unique_tail_tokens=case.new_prefill_tokens,
        )

    def update_from_outputs(
        self,
        prompt_batch: PromptBatch,
        outputs: Sequence[Any],
        case: TraceCase | None = None,
    ) -> None:
        for session_index, prompt in enumerate(prompt_batch.target_prompts):
            prompt_ids = list(prompt.get("prompt_token_ids", []))
            generated_ids = _generated_token_ids(_list_get(outputs, session_index))
            self.histories[session_index] = prompt_ids + generated_ids
            if case is not None:
                request_index = (
                    case.turn_index * case.scheduled_request_count + session_index
                )
                self.records.append({
                    "request_id": make_trace_request_id(
                        profile=case.profile,
                        concurrency=case.concurrency,
                        session_id=session_index,
                        turn_index=case.turn_index,
                        request_index=request_index,
                    ),
                    "profile": case.profile,
                    "concurrency": case.concurrency,
                    "turn_index": case.turn_index,
                    "session_id": session_index,
                    "request_index": request_index,
                    "prompt_tokens": len(prompt_ids),
                    "output_tokens": len(generated_ids),
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": generated_ids,
                })

    def _prompt_ids_for_session(
        self,
        tokenizer: Any,
        *,
        case: TraceCase,
        session_index: int,
    ) -> list[int]:
        history = self.histories.get(session_index, [])
        if not history:
            return make_prompt_token_ids(
                tokenizer,
                case.context_len,
                salt=(
                    f"{case.profile}_{case.concurrency}_session_{session_index}_"
                    "initial_context"
                ),
            )

        desired_cached = max(0, min(case.cached_context_tokens, case.context_len))
        fallback_cached = max(0, case.context_len - max(0, case.new_prefill_tokens))
        prefix_len = min(len(history), desired_cached or fallback_cached, case.context_len)
        tail_len = max(0, case.context_len - prefix_len)
        tail_ids = make_prompt_token_ids(
            tokenizer,
            tail_len,
            salt=(
                f"{case.profile}_{case.concurrency}_session_{session_index}_"
                f"turn_{case.turn_index}_tail"
            ),
        )
        return (history[:prefix_len] + tail_ids)[: case.context_len]


@dataclass(frozen=True)
class SchedulerCandidate:
    label: str
    scheduler: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--benchmark-turns", type=Path, default=DEFAULT_TURNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--token-history-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON artifact with prompt/output token IDs observed during "
            "full-cell direct vLLM generation. This is for simulator token "
            "identity validation, not a TPOT predictor feature."
        ),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["terminalbench-multiturn-synth", "swebench-multiturn-synth"],
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=[40, 80, 160, 320],
    )
    parser.add_argument(
        "--max-turns-per-cell",
        type=int,
        default=4,
        help="Limit benchmark-derived cases per profile/concurrency cell.",
    )
    parser.add_argument(
        "--trace-scope",
        choices=("single-turn", "full-cell"),
        default="single-turn",
        help=(
            "single-turn traces each selected row independently. full-cell "
            "replays all turns up to the requested target turn in one vLLM "
            "process so prefix cache and KV state evolve across turns."
        ),
    )
    parser.add_argument(
        "--turn-indices",
        nargs="+",
        type=int,
        default=[],
        help="Optional explicit benchmark turn indices to keep for each cell.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional global cap after filters. Zero means no cap.",
    )
    parser.add_argument(
        "--include-diagnostic",
        action="store_true",
        help="Include benchmark rows previously marked diagnostic.",
    )
    parser.add_argument(
        "--synthetic-case",
        action="append",
        default=[],
        metavar="PROFILE:CONCURRENCY:TURN:BATCH:CONTEXT:OUTPUT[:NEW_PREFILL]",
        help=(
            "Add an explicit synthetic trace case. Useful for smoke runs without "
            "benchmark CSV dependence."
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=1024,
        help="Synthetic shared-prefix length used before per-request unique tails.",
    )
    parser.add_argument(
        "--prompt-shape",
        choices=("synthetic-shared-prefix", "benchmark-cache-faithful"),
        default="synthetic-shared-prefix",
        help=(
            "Prompt construction mode. synthetic-shared-prefix keeps the old "
            "fixed shared-prefix trace. benchmark-cache-faithful warms each "
            "request's estimated cached prefix, then traces the target turn."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-scheduler-search-depth",
        type=int,
        default=7,
        help="Depth for best-effort scheduler discovery in the LLM object graph.",
    )
    parser.add_argument(
        "--hook-mode",
        choices=("auto", "vllm-v1-class", "object"),
        default="auto",
        help=(
            "Scheduler hook strategy. vLLM V1 keeps the scheduler inside a "
            "forked EngineCore process, so the class hook is the H100 path."
        ),
    )
    parser.add_argument(
        "--enable-engine-wall-trace",
        action="store_true",
        help=(
            "In vLLM V1 class-hook mode, wrap EngineCore.step() and attach "
            "per-engine-step wall timing fields to the scheduler trace row."
        ),
    )
    parser.add_argument(
        "--enable-worker-wall-trace",
        action="store_true",
        help=(
            "Diagnostic-only: also wrap the vLLM GPU worker execute_model() "
            "path and attach worker wall timing when it runs in the same "
            "process as the EngineCore hook."
        ),
    )
    parser.add_argument(
        "--enable-worker-cuda-sync",
        action="store_true",
        help=(
            "Diagnostic-only: synchronize CUDA around the worker execute_model "
            "hook. This can perturb async execution, so keep it off for the "
            "primary wall-time trace."
        ),
    )
    return parser.parse_args(argv)


def parse_synthetic_case(spec: str) -> TraceCase:
    parts = spec.split(":")
    if len(parts) not in (6, 7):
        raise ValueError(
            "--synthetic-case must be "
            "PROFILE:CONCURRENCY:TURN:BATCH:CONTEXT:OUTPUT[:NEW_PREFILL]"
        )
    profile, concurrency, turn, batch, context, output = parts[:6]
    new_prefill = parts[6] if len(parts) == 7 else context
    return TraceCase(
        profile=profile,
        concurrency=int(concurrency),
        turn_index=int(turn),
        batch_size=int(batch),
        context_len=int(context),
        output_tokens=int(output),
        scheduled_request_count=int(batch),
        successful_request_count=int(batch),
        new_prefill_tokens=int(new_prefill),
        primary_eval=True,
    )


def load_trace_cases(
    path: Path,
    *,
    profiles: set[str],
    concurrencies: set[int],
    max_turns_per_cell: int,
    include_diagnostic: bool,
    turn_indices: set[int] | None = None,
    max_cases: int = 0,
) -> list[TraceCase]:
    cases_by_cell: dict[tuple[str, int], list[TraceCase]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            profile = row.get("profile", "")
            concurrency = _to_int(row.get("concurrency"), 0)
            if profile not in profiles or concurrency not in concurrencies:
                continue
            turn_index = _to_int(row.get("turn_index"), 0)
            if turn_indices and turn_index not in turn_indices:
                continue
            primary_eval = _to_bool(row.get("primary_eval", "true"))
            if not include_diagnostic and not primary_eval:
                continue
            case = TraceCase(
                profile=profile,
                concurrency=concurrency,
                turn_index=turn_index,
                batch_size=_to_int(row.get("batch_size"), concurrency),
                context_len=max(1, _to_int(row.get("context_len"), 1)),
                output_tokens=max(1, _to_int(row.get("output_tokens"), 1)),
                scheduled_request_count=_to_int(
                    row.get("scheduled_request_count"),
                    _to_int(row.get("batch_size"), concurrency),
                ),
                successful_request_count=_to_int(
                    row.get("successful_request_count"),
                    _to_int(row.get("batch_size"), concurrency),
                ),
                new_prefill_tokens=max(0, _to_int(row.get("new_prefill_tokens"), 0)),
                cached_context_tokens=max(0, _to_int(row.get("cached_context_tokens"), 0)),
                cache_hit_rate=_to_float(row.get("cache_hit_rate"), 0.0),
                primary_eval=primary_eval,
                diagnostic_reason=row.get("diagnostic_reason", ""),
            )
            cases_by_cell.setdefault((profile, concurrency), []).append(case)

    cases: list[TraceCase] = []
    for key in sorted(cases_by_cell):
        cell_cases = sorted(cases_by_cell[key], key=lambda item: item.turn_index)
        cases.extend(cell_cases[:max_turns_per_cell])

    if max_cases > 0:
        return cases[:max_cases]
    return cases


def load_full_cell_trace_cases(
    path: Path,
    *,
    profiles: set[str],
    concurrencies: set[int],
    target_turn_indices: set[int],
    max_turns_per_cell: int,
    include_diagnostic: bool,
) -> list[TraceCase]:
    """Load chronological full-cell turn trajectories.

    When target turn indices are supplied, each matching profile/concurrency
    cell includes every earlier turn so cache/KV state can be built by actual
    generations. Without explicit targets, this falls back to the first
    `max_turns_per_cell` turns and marks them as targets.
    """

    cases_by_cell: dict[tuple[str, int], list[TraceCase]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            profile = row.get("profile", "")
            concurrency = _to_int(row.get("concurrency"), 0)
            if profile not in profiles or concurrency not in concurrencies:
                continue
            primary_eval = _to_bool(row.get("primary_eval", "true"))
            if not include_diagnostic and not primary_eval:
                continue
            case = TraceCase(
                profile=profile,
                concurrency=concurrency,
                turn_index=_to_int(row.get("turn_index"), 0),
                batch_size=_to_int(row.get("batch_size"), concurrency),
                context_len=max(1, _to_int(row.get("context_len"), 1)),
                output_tokens=max(1, _to_int(row.get("output_tokens"), 1)),
                scheduled_request_count=_to_int(
                    row.get("scheduled_request_count"),
                    _to_int(row.get("batch_size"), concurrency),
                ),
                successful_request_count=_to_int(
                    row.get("successful_request_count"),
                    _to_int(row.get("batch_size"), concurrency),
                ),
                new_prefill_tokens=max(0, _to_int(row.get("new_prefill_tokens"), 0)),
                cached_context_tokens=max(0, _to_int(row.get("cached_context_tokens"), 0)),
                cache_hit_rate=_to_float(row.get("cache_hit_rate"), 0.0),
                primary_eval=primary_eval,
                diagnostic_reason=row.get("diagnostic_reason", ""),
                trace_scope="full-cell",
            )
            cases_by_cell.setdefault((profile, concurrency), []).append(case)

    cases: list[TraceCase] = []
    for key in sorted(cases_by_cell):
        cell_cases = sorted(cases_by_cell[key], key=lambda item: item.turn_index)
        existing_targets = (
            {case.turn_index for case in cell_cases if case.turn_index in target_turn_indices}
            if target_turn_indices
            else {case.turn_index for case in cell_cases[:max_turns_per_cell]}
        )
        if not existing_targets:
            continue
        max_target = max(existing_targets)
        for case in cell_cases:
            if case.turn_index > max_target:
                continue
            phase = "target_turn" if case.turn_index in existing_targets else "warmup_turn"
            cases.append(
                replace(
                    case,
                    trace_phase=phase,
                    target_turn_index=max_target,
                )
            )
    return cases


def find_scheduler_candidates(root: Any, *, max_depth: int = 7) -> list[SchedulerCandidate]:
    """Find scheduler-like objects with a callable `schedule` method.

    vLLM has changed where schedulers live across releases. This object walk is
    bounded and conservative: it follows instance attributes, containers, and
    mappings, skips modules/classes/primitives, and records only vLLM-looking
    objects that expose `schedule`.
    """

    candidates: list[SchedulerCandidate] = []
    seen: set[int] = set()
    stack: list[tuple[str, Any, int]] = [("llm", root, 0)]

    while stack:
        label, obj, depth = stack.pop()
        if depth > max_depth or not _should_visit(obj):
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        schedule = getattr(obj, "schedule", None)
        if callable(schedule) and _looks_like_vllm_object(obj):
            candidates.append(SchedulerCandidate(label=label, scheduler=obj))

        for child_label, child in _iter_children(label, obj):
            if _should_visit(child):
                stack.append((child_label, child, depth + 1))

    unique: dict[int, SchedulerCandidate] = {}
    for candidate in candidates:
        unique.setdefault(id(candidate.scheduler), candidate)
    return list(unique.values())


class SchedulerTraceRecorder:
    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []
        self.current_case: TraceCase | None = None
        self.step_id = 0

    @contextmanager
    def case_context(self, case: TraceCase) -> Iterator[None]:
        previous_case = self.current_case
        self.current_case = case
        try:
            yield
        finally:
            self.current_case = previous_case

    def wrap(self, candidate: SchedulerCandidate) -> None:
        scheduler = candidate.scheduler
        original = scheduler.schedule
        recorder = self

        def wrapped_schedule(*args: Any, **kwargs: Any) -> Any:
            before = scheduler_state_summary(scheduler)
            start = time.perf_counter()
            output = original(*args, **kwargs)
            scheduler_wall_ms = (time.perf_counter() - start) * 1000.0
            after = scheduler_state_summary(scheduler)
            recorder.record(
                scheduler_label=candidate.label,
                scheduler_output=output,
                before=before,
                after=after,
                scheduler_wall_ms=scheduler_wall_ms,
            )
            return output

        setattr(scheduler, "schedule", wrapped_schedule)

    def record(
        self,
        *,
        scheduler_label: str,
        scheduler_output: Any,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        scheduler_wall_ms: float | None = None,
    ) -> None:
        case = self.current_case
        if case is None:
            return
        self.step_id += 1
        output = scheduler_output_summary(scheduler_output)
        row = {
            "run_id": self.run_id,
            "profile": case.profile,
            "concurrency": case.concurrency,
            "turn_index": case.turn_index,
            "trace_scope": case.trace_scope,
            "trace_phase": case.trace_phase,
            "target_turn_index": case.target_turn_index,
            "primary_eval": "true" if case.primary_eval else "false",
            "diagnostic_reason": case.diagnostic_reason,
            "scheduled_request_count": case.scheduled_request_count,
            "successful_request_count": case.successful_request_count,
            "batch_size": case.batch_size,
            "context_len": case.context_len,
            "output_tokens": case.output_tokens,
            "new_prefill_tokens": case.new_prefill_tokens,
            "cached_context_tokens": case.cached_context_tokens,
            "cache_hit_rate": _fmt_float(case.cache_hit_rate),
            "prompt_shape_mode": case.prompt_shape_mode,
            "trace_prompt_tokens": case.trace_prompt_tokens,
            "trace_shared_prefix_tokens": case.trace_shared_prefix_tokens,
            "trace_warmup_cached_tokens": case.trace_warmup_cached_tokens,
            "trace_unique_tail_tokens": case.trace_unique_tail_tokens,
            "scheduler_label": scheduler_label,
            "step_id": self.step_id,
            "scheduler_wall_ms": _fmt_optional_float(scheduler_wall_ms),
            "decode_batch": output["decode_batch"],
            "decode_request_ids": output["decode_request_ids"],
            "prefill_seqs": output["prefill_seqs"],
            "prefill_tokens": output["prefill_tokens"],
            "prefill_request_ids": output["prefill_request_ids"],
            "scheduled_request_ids": output["scheduled_request_ids"],
            "total_scheduled_tokens": output["total_scheduled_tokens"],
            "waiting_queue": _first_numeric(after, before, "waiting_queue"),
            "running_queue": _first_numeric(after, before, "running_queue"),
            "waiting_request_ids": _first_text(after, before, "waiting_request_ids"),
            "running_request_ids": _first_text(after, before, "running_request_ids"),
            "skipped_waiting_request_ids": _first_text(
                after,
                before,
                "skipped_waiting_request_ids",
            ),
            "free_kv_blocks": _first_numeric(after, before, "free_kv_blocks"),
            "graph_bucket": output["graph_bucket"],
            "preemptions": output["preemptions"]
            or _first_numeric(after, before, "preemptions"),
            "preempted_request_ids": output["preempted_request_ids"],
            "swaps": _first_numeric(after, before, "swaps"),
            "recomputes": _first_numeric(after, before, "recomputes"),
            "engine_computed_tokens_sum": output["engine_computed_tokens_sum"],
            "engine_uncached_prefill_tokens_sum": output[
                "engine_uncached_prefill_tokens_sum"
            ],
            "engine_cache_hit_rate_mean": output["engine_cache_hit_rate_mean"],
            "engine_cache_lookup_request_count": output[
                "engine_cache_lookup_request_count"
            ],
            "engine_new_block_count": output["engine_new_block_count"],
            "engine_allocate_none_count": output["engine_allocate_none_count"],
            "engine_cache_truth": output["engine_cache_truth"],
            "raw_summary": output["raw_summary"],
        }
        row.update(_empty_wall_trace_fields())
        self.rows.append(row)


def install_vllm_v1_class_hook(
    *,
    trace_jsonl_path: Path,
    case_json_path: Path | None,
    run_id: str,
    enable_engine_wall_trace: bool = False,
    enable_worker_wall_trace: bool = False,
    enable_worker_cuda_sync: bool = False,
    server_trace_from_request_ids: bool = False,
    include_unmatched_server_steps: bool = False,
) -> bool:
    """Patch vLLM V1 Scheduler.schedule before EngineCore is forked."""

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return False

    engine_wall_ready = False
    if enable_engine_wall_trace:
        engine_wall_ready = install_engine_core_step_hook(
            trace_jsonl_path=trace_jsonl_path,
        )
    _ENGINE_TRACE_STATE["engine_wall_enabled"] = engine_wall_ready
    if enable_worker_wall_trace:
        install_worker_wall_trace_hook(cuda_sync=enable_worker_cuda_sync)
    install_kv_cache_truth_hook()

    if not getattr(Scheduler.schedule, "_agentic_trace_wrapped", False):
        original = Scheduler.schedule
        step_counter = {"value": 0}

        def traced_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
            before = scheduler_state_summary(self)
            kv_events: list[dict[str, Any]] = _ENGINE_TRACE_STATE["kv_events"]
            kv_start = len(kv_events)
            start = time.perf_counter()
            try:
                output = original(self, *args, **kwargs)
                return output
            finally:
                scheduler_wall_ms = (time.perf_counter() - start) * 1000.0
                step_kv_events = kv_events[kv_start:]
                del kv_events[kv_start:]
                after = scheduler_state_summary(self)
                case = _read_case_json(case_json_path) if case_json_path else None
                output_obj = locals().get("output")
                if (
                    case is None
                    and output_obj is not None
                    and server_trace_from_request_ids
                ):
                    output_summary = scheduler_output_summary(
                        output_obj,
                        kv_events=step_kv_events,
                    )
                    case = benchmark_trace_case_from_output_summary(
                        output_summary,
                        include_unmatched=include_unmatched_server_steps,
                    )
                if case and output_obj is not None:
                    step_counter["value"] += 1
                    row = build_trace_row(
                        run_id=run_id,
                        case=case,
                        scheduler_label=type(self).__module__
                        + "."
                        + type(self).__name__,
                        scheduler_output=output_obj,
                        before=before,
                        after=after,
                        scheduler_wall_ms=scheduler_wall_ms,
                        kv_events=step_kv_events,
                    )
                    row["step_id"] = step_counter["value"]
                    if _ENGINE_TRACE_STATE.get("engine_wall_enabled"):
                        _ENGINE_TRACE_STATE["pending_rows"].append(row)
                    else:
                        _append_jsonl(trace_jsonl_path, row)

        setattr(traced_schedule, "_agentic_trace_wrapped", True)
        Scheduler.schedule = traced_schedule
    return True


def benchmark_trace_case_from_output_summary(
    output: Mapping[str, Any],
    *,
    include_unmatched: bool = False,
) -> dict[str, Any] | None:
    metas = [
        meta
        for request_id in _split_joined_ids(
            str(output.get("scheduled_request_ids") or "")
        )
        for meta in [parse_benchmark_request_id(request_id)]
        if meta is not None
    ]
    if not metas:
        if not include_unmatched:
            return None
        return {
            "profile": "unmatched",
            "concurrency": "",
            "turn_index": "",
            "trace_scope": "benchmark-serving",
            "trace_phase": "unmatched_request",
            "target_turn_index": "",
            "primary_eval": "false",
            "diagnostic_reason": "no_agenticbench_request_id",
            "scheduled_request_count": "",
            "successful_request_count": "",
            "batch_size": "",
            "context_len": "",
            "output_tokens": "",
            "new_prefill_tokens": "",
            "cached_context_tokens": "",
            "cache_hit_rate": "",
            "prompt_shape_mode": "actual-benchmark-request-stream",
        }

    profile = _single_or_join(meta["profile"] for meta in metas)
    concurrency = _single_or_join(meta["concurrency"] for meta in metas)
    turn_index = _single_or_join(meta["turn_index"] for meta in metas)
    return {
        "profile": profile,
        "concurrency": concurrency,
        "turn_index": turn_index,
        "trace_scope": "benchmark-serving",
        "trace_phase": "live_server",
        "target_turn_index": turn_index,
        "primary_eval": "true",
        "diagnostic_reason": "",
        "scheduled_request_count": concurrency,
        "successful_request_count": concurrency,
        "batch_size": len({meta["request_id"] for meta in metas}),
        "context_len": "",
        "output_tokens": "",
        "new_prefill_tokens": "",
        "cached_context_tokens": "",
        "cache_hit_rate": "",
        "prompt_shape_mode": "actual-benchmark-request-stream",
        "trace_prompt_tokens": "",
        "trace_shared_prefix_tokens": "",
        "trace_warmup_cached_tokens": "",
        "trace_unique_tail_tokens": "",
    }


def parse_benchmark_request_id(request_id: str) -> dict[str, str] | None:
    marker = f"{BENCHMARK_REQUEST_ID_PREFIX}__p="
    start = request_id.find(marker)
    if start < 0:
        return None
    payload = request_id[start:]
    parts = payload.split("__")
    values: dict[str, str] = {"request_id": request_id}
    for part in parts:
        if part == BENCHMARK_REQUEST_ID_PREFIX:
            continue
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "p":
            values["profile"] = value
        elif key == "c":
            values["concurrency"] = value
        elif key == "t":
            values["turn_index"] = value
        elif key == "s":
            values["session_id"] = _leading_int_string(value)
        elif key == "i":
            values["request_index"] = _leading_int_string(value)
    required = {"profile", "concurrency", "turn_index", "session_id", "request_index"}
    if not required.issubset(values):
        return None
    return values


def make_trace_request_id(
    *,
    profile: str,
    concurrency: int,
    session_id: int,
    turn_index: int,
    request_index: int,
) -> str:
    return (
        f"{BENCHMARK_REQUEST_ID_PREFIX}__p={profile}"
        f"__c={concurrency}"
        f"__t={turn_index}"
        f"__s={session_id}"
        f"__i={request_index}"
    )


def _leading_int_string(value: str) -> str:
    out = []
    for char in value:
        if char.isdigit() or (char == "-" and not out):
            out.append(char)
            continue
        break
    return "".join(out)


def _single_or_join(values: Iterable[str]) -> str:
    unique = sorted({str(value) for value in values if str(value) != ""})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return ";".join(unique)


def _split_joined_ids(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def install_engine_core_step_hook(*, trace_jsonl_path: Path) -> bool:
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:
        return False
    return wrap_engine_core_step_class(
        EngineCore,
        trace_jsonl_path=trace_jsonl_path,
    )


def wrap_engine_core_step_class(
    engine_core_cls: type,
    *,
    trace_jsonl_path: Path,
) -> bool:
    wrapped_any = False
    for method_name in ("step", "step_with_batch_queue"):
        original = getattr(engine_core_cls, method_name, None)
        if original is None:
            continue
        if getattr(original, "_agentic_engine_wall_wrapped", False):
            wrapped_any = True
            continue
        setattr(
            engine_core_cls,
            method_name,
            _make_traced_engine_core_method(
                original,
                trace_jsonl_path=trace_jsonl_path,
            ),
        )
        wrapped_any = True
    return wrapped_any


def _make_traced_engine_core_method(
    original: Callable[..., Any],
    *,
    trace_jsonl_path: Path,
) -> Callable[..., Any]:
    def traced_engine_core_method(self: Any, *args: Any, **kwargs: Any) -> Any:
        timings: dict[str, Any] = {
            "model_submit_wall_ms": 0.0,
            "model_wait_wall_ms": 0.0,
            "sample_wall_ms": 0.0,
            "scheduler_update_wall_ms": 0.0,
            "model_executed": False,
        }
        pending_rows: list[dict[str, Any]] = _ENGINE_TRACE_STATE["pending_rows"]
        worker_timings: list[dict[str, float | None]] = _ENGINE_TRACE_STATE[
            "worker_timings"
        ]
        pending_start = len(pending_rows)
        worker_start = len(worker_timings)
        patches: list[tuple[Any, str, Any]] = []
        _patch_engine_step_collaborators(self, timings, patches)
        start = time.perf_counter()
        result: Any = None
        try:
            result = original(self, *args, **kwargs)
            if isinstance(result, tuple) and len(result) > 1:
                timings["model_executed"] = bool(result[1])
            return result
        finally:
            engine_step_wall_ms = (time.perf_counter() - start) * 1000.0
            for obj, name, value in reversed(patches):
                try:
                    setattr(obj, name, value)
                except Exception:
                    pass
            new_worker_timings = worker_timings[worker_start:]
            del worker_timings[worker_start:]
            worker_wall_ms = _sum_optional_timing(
                item.get("worker_execute_wall_ms") for item in new_worker_timings
            )
            worker_cuda_sync_ms = _sum_optional_timing(
                item.get("worker_execute_cuda_sync_ms")
                for item in new_worker_timings
            )
            timing_fields = _engine_wall_timing_fields(
                timings,
                engine_step_wall_ms=engine_step_wall_ms,
                worker_execute_wall_ms=worker_wall_ms,
                worker_execute_cuda_sync_ms=worker_cuda_sync_ms,
            )
            new_rows = pending_rows[pending_start:]
            del pending_rows[pending_start:]
            for row in new_rows:
                row.update(timing_fields)
                _append_jsonl(trace_jsonl_path, row)

    setattr(traced_engine_core_method, "_agentic_engine_wall_wrapped", True)
    return traced_engine_core_method


def _patch_engine_step_collaborators(
    engine_core: Any,
    timings: dict[str, Any],
    patches: list[tuple[Any, str, Any]],
) -> None:
    model_executor = getattr(engine_core, "model_executor", None)
    scheduler = getattr(engine_core, "scheduler", None)
    if model_executor is not None and hasattr(model_executor, "execute_model"):
        original_execute = model_executor.execute_model

        def traced_execute_model(*args: Any, **kwargs: Any) -> Any:
            scheduler_output = args[0] if args else kwargs.get("scheduler_output")
            total_tokens = _get_any(scheduler_output, ["total_num_scheduled_tokens"])
            if _to_int(total_tokens, 0) > 0:
                timings["model_executed"] = True
            start = time.perf_counter()
            future = original_execute(*args, **kwargs)
            timings["model_submit_wall_ms"] += (time.perf_counter() - start) * 1000.0
            if hasattr(future, "result") and callable(getattr(future, "result")):
                return _TimedFuture(future, timings)
            return future

        _patch_attr(model_executor, "execute_model", traced_execute_model, patches)

    if model_executor is not None and hasattr(model_executor, "sample_tokens"):
        original_sample = model_executor.sample_tokens

        def traced_sample_tokens(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_sample(*args, **kwargs)
            finally:
                timings["sample_wall_ms"] += (time.perf_counter() - start) * 1000.0

        _patch_attr(model_executor, "sample_tokens", traced_sample_tokens, patches)

    if scheduler is not None and hasattr(scheduler, "update_from_output"):
        original_update = scheduler.update_from_output

        def traced_update_from_output(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_update(*args, **kwargs)
            finally:
                timings["scheduler_update_wall_ms"] += (
                    time.perf_counter() - start
                ) * 1000.0

        _patch_attr(scheduler, "update_from_output", traced_update_from_output, patches)


class _TimedFuture:
    def __init__(self, future: Any, timings: dict[str, Any]) -> None:
        self._future = future
        self._timings = timings

    def result(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._future.result(*args, **kwargs)
        finally:
            self._timings["model_wait_wall_ms"] += (
                time.perf_counter() - start
            ) * 1000.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._future, name)


def install_worker_wall_trace_hook(*, cuda_sync: bool = False) -> bool:
    for module_name, class_name in [
        ("vllm.v1.worker.gpu_worker", "GPUWorker"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ]:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception:
            continue
        if wrap_worker_execute_model_class(cls, cuda_sync=cuda_sync):
            return True
    return False


def wrap_worker_execute_model_class(cls: type, *, cuda_sync: bool = False) -> bool:
    original = getattr(cls, "execute_model", None)
    if original is None:
        return False
    if getattr(original, "_agentic_worker_wall_wrapped", False):
        return True

    def traced_execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        torch_module = _torch_module() if cuda_sync else None
        start = time.perf_counter()
        sync_start: float | None = None
        if torch_module is not None:
            _cuda_synchronize(torch_module)
            sync_start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            sync_ms: float | None = None
            if torch_module is not None and sync_start is not None:
                _cuda_synchronize(torch_module)
                sync_ms = (time.perf_counter() - sync_start) * 1000.0
            wall_ms = (time.perf_counter() - start) * 1000.0
            _ENGINE_TRACE_STATE["worker_timings"].append({
                "worker_execute_wall_ms": wall_ms,
                "worker_execute_cuda_sync_ms": sync_ms,
            })

    setattr(traced_execute_model, "_agentic_worker_wall_wrapped", True)
    setattr(cls, "execute_model", traced_execute_model)
    return True


def install_kv_cache_truth_hook() -> bool:
    """Patch vLLM KV-cache methods to record per-step cache/admission truth.

    The scheduler output already exposes enough fields for a fallback summary,
    but these hooks catch the precise prefix-cache lookup and allocation calls
    made inside the forked EngineCore process.
    """

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:
        return False

    wrapped_any = False
    get_computed_blocks = getattr(KVCacheManager, "get_computed_blocks", None)
    if get_computed_blocks is not None and not getattr(
        get_computed_blocks,
        "_agentic_cache_truth_wrapped",
        False,
    ):

        def traced_get_computed_blocks(self: Any, request: Any) -> Any:
            result = get_computed_blocks(self, request)
            computed_blocks, computed_tokens = result
            _ENGINE_TRACE_STATE["kv_events"].append({
                "event": "get_computed_blocks",
                "request_id": str(_request_id(request) or ""),
                "request_num_tokens": _to_int(_get_any(request, ["num_tokens"]), 0),
                "request_num_prompt_tokens": _to_int(
                    _get_any(request, ["num_prompt_tokens"]),
                    0,
                ),
                "computed_tokens": _to_int(computed_tokens, 0),
                "computed_block_count": _block_count(computed_blocks),
                "max_cache_hit_length": max(
                    0,
                    _to_int(_get_any(request, ["num_tokens"]), 0) - 1,
                ),
            })
            return result

        setattr(
            traced_get_computed_blocks,
            "_agentic_cache_truth_wrapped",
            True,
        )
        setattr(KVCacheManager, "get_computed_blocks", traced_get_computed_blocks)
        wrapped_any = True

    allocate_slots = getattr(KVCacheManager, "allocate_slots", None)
    if allocate_slots is not None and not getattr(
        allocate_slots,
        "_agentic_cache_truth_wrapped",
        False,
    ):

        def traced_allocate_slots(self: Any, request: Any, num_new_tokens: int, *args: Any, **kwargs: Any) -> Any:
            result = allocate_slots(self, request, num_new_tokens, *args, **kwargs)
            _ENGINE_TRACE_STATE["kv_events"].append({
                "event": "allocate_slots",
                "request_id": str(_request_id(request) or ""),
                "request_num_tokens": _to_int(_get_any(request, ["num_tokens"]), 0),
                "request_num_prompt_tokens": _to_int(
                    _get_any(request, ["num_prompt_tokens"]),
                    0,
                ),
                "request_num_computed_tokens": _to_int(
                    _get_any(request, ["num_computed_tokens"]),
                    0,
                ),
                "num_new_tokens": _to_int(num_new_tokens, 0),
                "num_new_computed_tokens": _to_int(
                    kwargs.get("num_new_computed_tokens"),
                    0,
                ),
                "num_external_computed_tokens": _to_int(
                    kwargs.get("num_external_computed_tokens"),
                    0,
                ),
                "new_block_count": _block_count(result),
                "allocation_failed": result is None,
            })
            return result

        setattr(traced_allocate_slots, "_agentic_cache_truth_wrapped", True)
        setattr(KVCacheManager, "allocate_slots", traced_allocate_slots)
        wrapped_any = True

    return wrapped_any


def build_trace_row(
    *,
    run_id: str,
    case: Mapping[str, Any],
    scheduler_label: str,
    scheduler_output: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    scheduler_wall_ms: float | None = None,
    kv_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output = scheduler_output_summary(scheduler_output, kv_events=kv_events)
    row = {
        "run_id": run_id,
        "profile": case.get("profile", ""),
        "concurrency": case.get("concurrency", ""),
        "turn_index": case.get("turn_index", ""),
        "trace_scope": case.get("trace_scope", ""),
        "trace_phase": case.get("trace_phase", ""),
        "target_turn_index": case.get("target_turn_index", ""),
        "primary_eval": case.get("primary_eval", ""),
        "diagnostic_reason": case.get("diagnostic_reason", ""),
        "scheduled_request_count": case.get("scheduled_request_count", ""),
        "successful_request_count": case.get("successful_request_count", ""),
        "batch_size": case.get("batch_size", ""),
        "context_len": case.get("context_len", ""),
        "output_tokens": case.get("output_tokens", ""),
        "new_prefill_tokens": case.get("new_prefill_tokens", ""),
        "cached_context_tokens": case.get("cached_context_tokens", ""),
        "cache_hit_rate": case.get("cache_hit_rate", ""),
        "prompt_shape_mode": case.get("prompt_shape_mode", ""),
        "trace_prompt_tokens": case.get("trace_prompt_tokens", ""),
        "trace_shared_prefix_tokens": case.get("trace_shared_prefix_tokens", ""),
        "trace_warmup_cached_tokens": case.get("trace_warmup_cached_tokens", ""),
        "trace_unique_tail_tokens": case.get("trace_unique_tail_tokens", ""),
        "scheduler_label": scheduler_label,
        "step_id": _next_step_id(after, before),
        "scheduler_wall_ms": _fmt_optional_float(scheduler_wall_ms),
        "decode_batch": output["decode_batch"],
        "decode_request_ids": output["decode_request_ids"],
        "prefill_seqs": output["prefill_seqs"],
        "prefill_tokens": output["prefill_tokens"],
        "prefill_request_ids": output["prefill_request_ids"],
        "scheduled_request_ids": output["scheduled_request_ids"],
        "total_scheduled_tokens": output["total_scheduled_tokens"],
        "waiting_queue": _first_numeric(after, before, "waiting_queue"),
        "running_queue": _first_numeric(after, before, "running_queue"),
        "waiting_request_ids": _first_text(after, before, "waiting_request_ids"),
        "running_request_ids": _first_text(after, before, "running_request_ids"),
        "skipped_waiting_request_ids": _first_text(
            after,
            before,
            "skipped_waiting_request_ids",
        ),
        "free_kv_blocks": _first_numeric(after, before, "free_kv_blocks"),
        "graph_bucket": output["graph_bucket"],
        "preemptions": output["preemptions"]
        or _first_numeric(after, before, "preemptions"),
        "preempted_request_ids": output["preempted_request_ids"],
        "swaps": _first_numeric(after, before, "swaps"),
        "recomputes": _first_numeric(after, before, "recomputes"),
        "engine_computed_tokens_sum": output["engine_computed_tokens_sum"],
        "engine_uncached_prefill_tokens_sum": output[
            "engine_uncached_prefill_tokens_sum"
        ],
        "engine_cache_hit_rate_mean": output["engine_cache_hit_rate_mean"],
        "engine_cache_lookup_request_count": output[
            "engine_cache_lookup_request_count"
        ],
        "engine_new_block_count": output["engine_new_block_count"],
        "engine_allocate_none_count": output["engine_allocate_none_count"],
        "engine_cache_truth": output["engine_cache_truth"],
        "raw_summary": output["raw_summary"],
    }
    row.update(_empty_wall_trace_fields())
    return row


def scheduler_output_summary(
    output: Any,
    *,
    kv_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    token_map = _get_any(output, [
        "num_scheduled_tokens",
        "scheduled_num_tokens",
        "scheduled_tokens",
        "num_batched_tokens",
    ])
    total_scheduled_tokens = _sum_tokens(token_map)

    scheduled_new = _get_any(output, ["scheduled_new_reqs", "scheduled_new_seq_groups"])
    scheduled_cached = _get_any(output, [
        "scheduled_cached_reqs",
        "scheduled_prefill_reqs",
        "scheduled_prefills",
    ])
    scheduled_running = _get_any(output, [
        "scheduled_running_reqs",
        "scheduled_decode_reqs",
        "scheduled_decodes",
    ])
    preempted = _get_any(output, ["preempted_req_ids", "preempted_reqs"])

    decode_request_ids = _unique_ids(
        _request_ids(scheduled_running),
        _cached_decode_req_ids(scheduled_cached, token_map),
    )
    prefill_request_ids = _unique_ids(
        _request_ids(scheduled_new),
        _cached_context_req_ids(scheduled_cached, token_map),
    )
    scheduled_request_ids = _unique_ids(decode_request_ids, prefill_request_ids)
    if not scheduled_request_ids:
        scheduled_request_ids = _positive_token_req_ids(token_map)
    preempted_request_ids = _request_ids(preempted)

    prefill_seqs = _first_positive_int([
        _get_any(output, ["num_prefill_groups", "num_prefills", "prefill_seqs"]),
        _len_or_zero(scheduled_new)
        + _count_cached_context_reqs(scheduled_cached, token_map),
    ])
    decode_batch = _first_positive_int([
        _get_any(output, ["num_decode_groups", "num_decodes", "decode_batch"]),
        _len_or_zero(scheduled_running),
        _count_cached_decode_reqs(scheduled_cached, token_map),
    ])

    prefill_tokens = _first_positive_int([
        _get_any(output, [
            "num_prefill_tokens",
            "prefill_tokens",
            "scheduled_prefill_tokens",
        ]),
        _tokens_for_selected_reqs(token_map, scheduled_new)
        + _cached_context_tokens(scheduled_cached, token_map),
    ])
    preemptions = len(preempted_request_ids)

    graph_bucket = _get_any(output, [
        "graph_bucket",
        "cuda_graph_batch_size",
        "cudagraph_batch_size",
        "selected_graph_batch_size",
    ])
    if graph_bucket is None and total_scheduled_tokens:
        graph_bucket = _next_power_of_two_or_capture_bucket(total_scheduled_tokens)

    cache_truth = engine_cache_truth_summary(
        output,
        token_map=token_map,
        kv_events=kv_events or [],
    )
    raw_summary = compact_raw_summary(output)
    return {
        "decode_batch": decode_batch,
        "decode_request_ids": _join_ids(decode_request_ids),
        "prefill_seqs": prefill_seqs,
        "prefill_tokens": prefill_tokens,
        "prefill_request_ids": _join_ids(prefill_request_ids),
        "scheduled_request_ids": _join_ids(scheduled_request_ids),
        "total_scheduled_tokens": total_scheduled_tokens,
        "graph_bucket": "" if graph_bucket is None else str(graph_bucket),
        "preemptions": preemptions,
        "preempted_request_ids": _join_ids(preempted_request_ids),
        "engine_computed_tokens_sum": cache_truth["engine_computed_tokens_sum"],
        "engine_uncached_prefill_tokens_sum": cache_truth[
            "engine_uncached_prefill_tokens_sum"
        ],
        "engine_cache_hit_rate_mean": cache_truth["engine_cache_hit_rate_mean"],
        "engine_cache_lookup_request_count": cache_truth[
            "engine_cache_lookup_request_count"
        ],
        "engine_new_block_count": cache_truth["engine_new_block_count"],
        "engine_allocate_none_count": cache_truth["engine_allocate_none_count"],
        "engine_cache_truth": cache_truth["engine_cache_truth"],
        "raw_summary": json.dumps(raw_summary, sort_keys=True),
    }


def engine_cache_truth_summary(
    output: Any,
    *,
    token_map: Any,
    kv_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request_truth: list[dict[str, Any]] = []

    scheduled_new = _get_any(output, ["scheduled_new_reqs", "scheduled_new_seq_groups"])
    for item in _iter_request_items(scheduled_new):
        request_id = str(
            _get_any(item, ["req_id", "request_id", "id"]) or _request_id(item) or ""
        )
        prompt_tokens = _prompt_token_count(item)
        computed_tokens = _to_int(_get_any(item, ["num_computed_tokens"]), 0)
        scheduled_tokens = _sum_tokens_for_req(token_map, request_id)
        request_truth.append({
            "source": "scheduled_new",
            "phase": "prefill",
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "engine_computed_tokens": computed_tokens,
            "scheduled_tokens": scheduled_tokens,
            "engine_uncached_remaining_tokens": (
                max(0, prompt_tokens - computed_tokens)
                if prompt_tokens
                else ""
            ),
            "new_block_count": _block_count(_get_any(item, ["block_ids"])),
        })

    cached = _get_any(output, ["scheduled_cached_reqs"])
    req_ids = _as_list(_get_any(cached, ["req_ids"]))
    computed_values = _as_list(_get_any(cached, ["num_computed_tokens"]))
    output_values = _as_list(_get_any(cached, ["num_output_tokens"]))
    new_block_values = _as_list(_get_any(cached, ["new_block_ids"]))
    all_token_ids = _get_any(cached, ["all_token_ids"])
    for index, req_id_obj in enumerate(req_ids):
        request_id = str(req_id_obj)
        num_output_tokens = _to_int(_list_get(output_values, index), 0)
        phase = "prefill" if num_output_tokens == 0 else "decode"
        prompt_tokens = 0
        if isinstance(all_token_ids, Mapping):
            prompt_tokens = _len_or_zero(all_token_ids.get(request_id))
        computed_tokens = _to_int(_list_get(computed_values, index), 0)
        scheduled_tokens = _sum_tokens_for_req(token_map, request_id)
        request_truth.append({
            "source": "scheduled_cached",
            "phase": phase,
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "engine_computed_tokens": computed_tokens,
            "scheduled_tokens": scheduled_tokens,
            "engine_uncached_remaining_tokens": (
                max(0, prompt_tokens - computed_tokens)
                if prompt_tokens and phase == "prefill"
                else ""
            ),
            "new_block_count": _block_count(_list_get(new_block_values, index)),
        })

    prefill_truth = [
        item
        for item in request_truth
        if item.get("phase") == "prefill" and _to_int(item.get("scheduled_tokens"), 0) > 0
    ]
    computed_sum = sum(
        _to_int(item.get("engine_computed_tokens"), 0) for item in prefill_truth
    )
    uncached_sum = sum(_to_int(item.get("scheduled_tokens"), 0) for item in prefill_truth)
    rates = [
        _to_int(item.get("engine_computed_tokens"), 0)
        / _to_int(item.get("prompt_tokens"), 0)
        for item in prefill_truth
        if _to_int(item.get("prompt_tokens"), 0) > 0
    ]

    hook_new_block_count = sum(
        _to_int(item.get("new_block_count"), 0)
        for item in kv_events
        if item.get("event") == "allocate_slots"
    )
    fallback_new_block_count = sum(
        _to_int(item.get("new_block_count"), 0) for item in request_truth
    )
    allocate_none_count = sum(
        1
        for item in kv_events
        if item.get("event") == "allocate_slots" and item.get("allocation_failed")
    )
    lookup_count = sum(
        1 for item in kv_events if item.get("event") == "get_computed_blocks"
    )
    if not lookup_count:
        lookup_count = len([
            item
            for item in prefill_truth
            if _to_int(item.get("prompt_tokens"), 0) > 0
        ])

    payload = {
        "requests": request_truth,
        "kv_events": [dict(item) for item in kv_events],
    }
    return {
        "engine_computed_tokens_sum": computed_sum,
        "engine_uncached_prefill_tokens_sum": uncached_sum,
        "engine_cache_hit_rate_mean": (
            _fmt_optional_float(sum(rates) / len(rates)) if rates else ""
        ),
        "engine_cache_lookup_request_count": lookup_count,
        "engine_new_block_count": hook_new_block_count or fallback_new_block_count,
        "engine_allocate_none_count": allocate_none_count,
        "engine_cache_truth": json.dumps(payload, sort_keys=True),
    }


def scheduler_state_summary(scheduler: Any) -> dict[str, Any]:
    return {
        "waiting_queue": _queue_length_from_names(scheduler, [
            "waiting",
            "waiting_queue",
            "waiting_reqs",
            "waiting_seq_groups",
        ]),
        "running_queue": _queue_length_from_names(scheduler, [
            "running",
            "running_queue",
            "running_reqs",
            "running_seq_groups",
        ]),
        "waiting_request_ids": _join_ids(_queue_request_ids_from_names(scheduler, [
            "waiting",
            "waiting_queue",
            "waiting_reqs",
            "waiting_seq_groups",
        ])),
        "running_request_ids": _join_ids(_queue_request_ids_from_names(scheduler, [
            "running",
            "running_queue",
            "running_reqs",
            "running_seq_groups",
        ])),
        "skipped_waiting_request_ids": _join_ids(
            _queue_request_ids_from_names(scheduler, [
                "skipped_waiting",
                "skipped_waiting_queue",
                "skipped_waiting_reqs",
            ])
        ),
        "free_kv_blocks": _free_kv_blocks(scheduler),
        "preemptions": _counter_from_names(scheduler, [
            "num_cumulative_preemption",
            "num_preemptions",
            "preemptions",
            "preempted_reqs",
        ]),
        "swaps": _counter_from_names(scheduler, [
            "num_swapped",
            "num_swaps",
            "swaps",
            "swapped_reqs",
        ]),
        "recomputes": _counter_from_names(scheduler, [
            "num_recomputed",
            "num_recomputes",
            "recomputes",
        ]),
    }


def compact_raw_summary(obj: Any) -> dict[str, Any]:
    """Return scalar/list-length metadata without dumping huge vLLM objects."""

    summary: dict[str, Any] = {}
    for key, value in _iter_public_attrs(obj):
        if key.startswith("_"):
            continue
        if _is_scalar(value):
            summary[key] = value
        elif isinstance(value, Mapping):
            summary[key] = {
                "type": type(value).__name__,
                "len": len(value),
                "token_sum": _sum_tokens(value),
            }
        elif isinstance(value, (list, tuple, set, frozenset)):
            summary[key] = {
                "type": type(value).__name__,
                "len": len(value),
                "token_sum": _sum_tokens(value),
            }
    return summary


def make_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Build synthetic text that tokenizes to at least `target_tokens`.

    The exact words are irrelevant for TPOT decode-shape tracing. We only need a
    stable prompt whose token count is close to the target.
    """

    unit = " analysis"
    text = unit * max(1, target_tokens)
    encode = getattr(tokenizer, "encode")
    tokens = encode(text, add_special_tokens=False)
    if len(tokens) < target_tokens:
        repeat = max(1, int(target_tokens / max(1, len(tokens))) + 2)
        text = text * repeat
        tokens = encode(text, add_special_tokens=False)
    if len(tokens) <= target_tokens:
        return text
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return decode(tokens[:target_tokens])
    return text


def make_prompt_token_ids(
    tokenizer: Any,
    target_tokens: int,
    *,
    salt: str = "",
) -> list[int]:
    if target_tokens <= 0:
        return []
    marker = f" {salt} " if salt else ""
    text = marker + (" analysis" * max(1, target_tokens + 8))
    encode = getattr(tokenizer, "encode")
    tokens = encode(text, add_special_tokens=False)
    if len(tokens) < target_tokens:
        repeat = max(1, int(target_tokens / max(1, len(tokens))) + 2)
        tokens = encode(text * repeat, add_special_tokens=False)
    return list(tokens[:target_tokens])


def make_token_prompt(token_ids: Sequence[int]) -> dict[str, list[int]]:
    return {"prompt_token_ids": list(token_ids)}


def make_case_prompts(
    tokenizer: Any,
    *,
    target_tokens: int,
    batch_size: int,
    shared_prefix_tokens: int,
) -> list[str]:
    shared_tokens = max(0, min(shared_prefix_tokens, target_tokens))
    tail_tokens = max(0, target_tokens - shared_tokens)
    shared_text = make_prompt(tokenizer, shared_tokens) if shared_tokens else ""
    prompts = []
    for request_index in range(batch_size):
        unique_marker = f" request_{request_index}_unique "
        tail_text = make_prompt(tokenizer, tail_tokens + 16) if tail_tokens else ""
        text = shared_text + unique_marker + tail_text
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < target_tokens:
            filler = make_prompt(tokenizer, target_tokens - len(token_ids) + 16)
            token_ids = tokenizer.encode(text + filler, add_special_tokens=False)
        if len(token_ids) > target_tokens:
            text = tokenizer.decode(token_ids[:target_tokens])
        prompts.append(text)
    return prompts


def make_case_prompt_batch(
    tokenizer: Any,
    case: TraceCase,
    *,
    prompt_shape_mode: str,
    shared_prefix_tokens: int,
) -> PromptBatch:
    if prompt_shape_mode == "synthetic-shared-prefix":
        return make_synthetic_shared_prefix_prompt_batch(
            tokenizer,
            target_tokens=case.context_len,
            batch_size=case.batch_size,
            shared_prefix_tokens=shared_prefix_tokens,
        )
    if prompt_shape_mode == "benchmark-cache-faithful":
        return make_benchmark_cache_faithful_prompt_batch(tokenizer, case)
    raise ValueError(f"Unsupported prompt shape mode: {prompt_shape_mode}")


def make_synthetic_shared_prefix_prompt_batch(
    tokenizer: Any,
    *,
    target_tokens: int,
    batch_size: int,
    shared_prefix_tokens: int,
) -> PromptBatch:
    shared_tokens = max(0, min(shared_prefix_tokens, target_tokens))
    tail_tokens = max(0, target_tokens - shared_tokens)
    shared_ids = make_prompt_token_ids(
        tokenizer,
        shared_tokens,
        salt="synthetic_shared_prefix",
    )
    target_prompts = []
    for request_index in range(batch_size):
        tail_ids = make_prompt_token_ids(
            tokenizer,
            tail_tokens,
            salt=f"synthetic_tail_{request_index}",
        )
        target_prompts.append(make_token_prompt(shared_ids + tail_ids))
    return PromptBatch(
        target_prompts=target_prompts,
        warmup_prompts=[],
        trace_prompt_tokens=target_tokens,
        trace_shared_prefix_tokens=shared_tokens,
        trace_warmup_cached_tokens=0,
        trace_unique_tail_tokens=tail_tokens,
    )


def make_benchmark_cache_faithful_prompt_batch(
    tokenizer: Any,
    case: TraceCase,
) -> PromptBatch:
    cached_tokens = max(0, min(case.cached_context_tokens, case.context_len))
    tail_tokens = max(0, case.context_len - cached_tokens)
    target_prompts = []
    warmup_prompts = []
    for request_index in range(case.batch_size):
        prefix_ids = make_prompt_token_ids(
            tokenizer,
            cached_tokens,
            salt=(
                f"{case.profile}_{case.concurrency}_{case.turn_index}_"
                f"session_{request_index}_cached_prefix"
            ),
        )
        tail_ids = make_prompt_token_ids(
            tokenizer,
            tail_tokens,
            salt=(
                f"{case.profile}_{case.concurrency}_{case.turn_index}_"
                f"session_{request_index}_target_tail"
            ),
        )
        if prefix_ids:
            warmup_prompts.append(make_token_prompt(prefix_ids))
        target_prompts.append(make_token_prompt((prefix_ids + tail_ids)[: case.context_len]))
    return PromptBatch(
        target_prompts=target_prompts,
        warmup_prompts=warmup_prompts,
        trace_prompt_tokens=case.context_len,
        trace_shared_prefix_tokens=0,
        trace_warmup_cached_tokens=cached_tokens,
        trace_unique_tail_tokens=tail_tokens,
    )


def write_trace_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TRACE_FIELDS})


def write_token_history_json(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "vllm-full-cell-token-history-v1",
        "per_request": list(records),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cases(args: argparse.Namespace, cases: Sequence[TraceCase]) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    trace_jsonl_path = args.output.with_suffix(".jsonl")
    case_json_path = args.output.with_suffix(".current_case.json")
    if trace_jsonl_path.exists():
        trace_jsonl_path.unlink()
    if case_json_path.exists():
        case_json_path.unlink()

    using_class_hook = False
    if args.hook_mode in {"auto", "vllm-v1-class"}:
        using_class_hook = install_vllm_v1_class_hook(
            trace_jsonl_path=trace_jsonl_path,
            case_json_path=case_json_path,
            run_id=args.run_id,
            enable_engine_wall_trace=args.enable_engine_wall_trace,
            enable_worker_wall_trace=args.enable_worker_wall_trace,
            enable_worker_cuda_sync=args.enable_worker_cuda_sync,
        )
        if args.hook_mode == "vllm-v1-class" and not using_class_hook:
            raise RuntimeError("Requested vLLM V1 class hook, but import failed.")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True

    llm = LLM(**llm_kwargs)
    recorder = SchedulerTraceRecorder(run_id=args.run_id)
    if not using_class_hook:
        candidates = find_scheduler_candidates(
            llm,
            max_depth=args.max_scheduler_search_depth,
        )
        if not candidates:
            raise RuntimeError(
                "Could not find a vLLM scheduler object with a schedule() method. "
                "This vLLM version may require --hook-mode vllm-v1-class."
            )
        for candidate in candidates:
            recorder.wrap(candidate)

    tokenizer = llm.get_tokenizer()
    full_cell_state = FullCellPromptState()
    for case in cases:
        if args.trace_scope == "full-cell":
            prompt_batch = full_cell_state.make_prompt_batch(tokenizer, case)
        else:
            prompt_batch = make_case_prompt_batch(
                tokenizer,
                case,
                prompt_shape_mode=args.prompt_shape,
                shared_prefix_tokens=args.shared_prefix_tokens,
            )
        case = replace(
            case,
            trace_scope=args.trace_scope,
            prompt_shape_mode=args.prompt_shape,
            trace_prompt_tokens=prompt_batch.trace_prompt_tokens,
            trace_shared_prefix_tokens=prompt_batch.trace_shared_prefix_tokens,
            trace_warmup_cached_tokens=prompt_batch.trace_warmup_cached_tokens,
            trace_unique_tail_tokens=prompt_batch.trace_unique_tail_tokens,
        )
        sampling_params = SamplingParams(
            max_tokens=case.output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            ignore_eos=True,
        )
        if prompt_batch.warmup_prompts:
            _unlink_if_exists(case_json_path)
            warmup_sampling_params = SamplingParams(
                max_tokens=1,
                temperature=args.temperature,
                top_p=args.top_p,
                ignore_eos=True,
            )
            llm.generate(
                prompt_batch.warmup_prompts,
                warmup_sampling_params,
                use_tqdm=False,
            )
        _write_case_json(case_json_path, case)
        with recorder.case_context(case):
            outputs = llm.generate(
                prompt_batch.target_prompts,
                sampling_params,
                use_tqdm=False,
            )
        if args.trace_scope == "full-cell":
            full_cell_state.update_from_outputs(prompt_batch, outputs, case=case)
    if args.token_history_output and full_cell_state.records:
        write_token_history_json(args.token_history_output, full_cell_state.records)
    if using_class_hook:
        return _read_jsonl(trace_jsonl_path)
    return recorder.rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prompt_shape == "benchmark-cache-faithful" and not args.enable_prefix_caching:
        raise SystemExit(
            "--prompt-shape benchmark-cache-faithful requires "
            "--enable-prefix-caching so the warmup prefix can be reused."
        )
    if args.trace_scope == "full-cell" and args.prompt_shape != "benchmark-cache-faithful":
        raise SystemExit(
            "--trace-scope full-cell currently requires "
            "--prompt-shape benchmark-cache-faithful so prior turns build "
            "benchmark-shaped prefix-cache state."
        )
    run_id = args.run_id or os.environ.get("RUN_ID") or "vllm_engine_step_trace"
    args.run_id = run_id

    cases = [parse_synthetic_case(spec) for spec in args.synthetic_case]
    if args.benchmark_turns.exists():
        if args.trace_scope == "full-cell":
            cases.extend(
                load_full_cell_trace_cases(
                    args.benchmark_turns,
                    profiles=set(args.profiles),
                    concurrencies=set(args.concurrencies),
                    target_turn_indices=set(args.turn_indices),
                    max_turns_per_cell=args.max_turns_per_cell,
                    include_diagnostic=args.include_diagnostic,
                )
            )
        else:
            cases.extend(
                load_trace_cases(
                    args.benchmark_turns,
                    profiles=set(args.profiles),
                    concurrencies=set(args.concurrencies),
                    turn_indices=set(args.turn_indices) if args.turn_indices else None,
                    max_turns_per_cell=args.max_turns_per_cell,
                    include_diagnostic=args.include_diagnostic,
                    max_cases=args.max_cases,
                )
            )
    if args.max_cases > 0 and args.trace_scope != "full-cell":
        cases = cases[:args.max_cases]
    if not cases:
        raise SystemExit("No trace cases selected.")

    rows = run_cases(args, cases)
    write_trace_csv(args.output, rows)
    print(f"Wrote {len(rows)} engine-step rows to {args.output}")
    return 0


def _to_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt_float(value: float) -> str:
    return f"{value:.6g}"


def _fmt_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _fmt_float(float(value))


def _empty_wall_trace_fields() -> dict[str, str]:
    return {field: "" for field in WALL_TRACE_FIELDS}


def _engine_wall_timing_fields(
    timings: Mapping[str, Any],
    *,
    engine_step_wall_ms: float,
    worker_execute_wall_ms: float | None,
    worker_execute_cuda_sync_ms: float | None,
) -> dict[str, str]:
    return {
        "engine_step_wall_ms": _fmt_optional_float(engine_step_wall_ms),
        "model_submit_wall_ms": _fmt_optional_float(
            _to_float(timings.get("model_submit_wall_ms"), 0.0)
        ),
        "model_wait_wall_ms": _fmt_optional_float(
            _to_float(timings.get("model_wait_wall_ms"), 0.0)
        ),
        "sample_wall_ms": _fmt_optional_float(
            _to_float(timings.get("sample_wall_ms"), 0.0)
        ),
        "scheduler_update_wall_ms": _fmt_optional_float(
            _to_float(timings.get("scheduler_update_wall_ms"), 0.0)
        ),
        "model_executed": "true" if timings.get("model_executed") else "false",
        "worker_execute_wall_ms": _fmt_optional_float(worker_execute_wall_ms),
        "worker_execute_cuda_sync_ms": _fmt_optional_float(
            worker_execute_cuda_sync_ms
        ),
    }


def _sum_optional_timing(values: Iterable[float | None]) -> float | None:
    seen = False
    total = 0.0
    for value in values:
        if value is None:
            continue
        seen = True
        total += float(value)
    return total if seen else None


def _patch_attr(
    obj: Any,
    name: str,
    value: Any,
    patches: list[tuple[Any, str, Any]],
) -> bool:
    try:
        original = getattr(obj, name)
        setattr(obj, name, value)
    except Exception:
        return False
    patches.append((obj, name, original))
    return True


def _torch_module() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _cuda_synchronize(torch_module: Any) -> None:
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    except Exception:
        return


def _case_to_json(case: TraceCase) -> dict[str, Any]:
    return {
        "profile": case.profile,
        "concurrency": case.concurrency,
        "turn_index": case.turn_index,
        "trace_scope": case.trace_scope,
        "trace_phase": case.trace_phase,
        "target_turn_index": case.target_turn_index,
        "batch_size": case.batch_size,
        "context_len": case.context_len,
        "output_tokens": case.output_tokens,
        "scheduled_request_count": case.scheduled_request_count,
        "successful_request_count": case.successful_request_count,
        "new_prefill_tokens": case.new_prefill_tokens,
        "cached_context_tokens": case.cached_context_tokens,
        "cache_hit_rate": _fmt_float(case.cache_hit_rate),
        "prompt_shape_mode": case.prompt_shape_mode,
        "trace_prompt_tokens": case.trace_prompt_tokens,
        "trace_shared_prefix_tokens": case.trace_shared_prefix_tokens,
        "trace_warmup_cached_tokens": case.trace_warmup_cached_tokens,
        "trace_unique_tail_tokens": case.trace_unique_tail_tokens,
        "primary_eval": "true" if case.primary_eval else "false",
        "diagnostic_reason": case.diagnostic_reason,
    }


def _write_case_json(path: Path, case: TraceCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_case_to_json(case), sort_keys=True))
    tmp_path.replace(path)


def _read_case_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _should_visit(obj: Any) -> bool:
    if obj is None or _is_scalar(obj):
        return False
    if isinstance(obj, ModuleType):
        return False
    if isinstance(obj, type):
        return False
    module = type(obj).__module__
    if module.startswith(("builtins", "typing", "pathlib")):
        return False
    return True


def _looks_like_vllm_object(obj: Any) -> bool:
    module = type(obj).__module__.lower()
    name = type(obj).__name__.lower()
    return "vllm" in module or "scheduler" in name


def _iter_children(label: str, obj: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield f"{label}.{key}", value
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield f"{label}[{index}]", value
        return
    attrs = getattr(obj, "__dict__", {})
    if isinstance(attrs, Mapping):
        for key, value in attrs.items():
            if key.startswith("__"):
                continue
            yield f"{label}.{key}", value


def _iter_public_attrs(obj: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(obj, Mapping):
        yield from obj.items()
        return
    attrs = getattr(obj, "__dict__", {})
    if isinstance(attrs, Mapping):
        yield from attrs.items()


def _get_any(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _len_or_zero(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, frozenset):
        return list(value)
    return [value]


def _list_get(values: Sequence[Any], index: int) -> Any:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _generated_token_ids(output: Any) -> list[int]:
    if output is None:
        return []
    outputs = _get_any(output, ["outputs"])
    first_output = _list_get(_as_list(outputs), 0)
    token_ids = _get_any(first_output, ["token_ids"])
    if token_ids is None:
        token_ids = _get_any(output, ["token_ids", "output_token_ids"])
    result: list[int] = []
    for token_id in _as_list(token_ids):
        try:
            result.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return result


def _prompt_token_count(value: Any) -> int:
    token_ids = _get_any(value, ["prompt_token_ids", "prefill_token_ids"])
    return _len_or_zero(token_ids)


def _block_count(value: Any) -> int:
    if value is None:
        return 0
    block_ids = _get_any(value, ["block_ids"])
    if block_ids is not None and block_ids is not value:
        return _block_count(block_ids)
    get_block_ids = getattr(value, "get_block_ids", None)
    if callable(get_block_ids):
        try:
            return _block_count(get_block_ids(allow_none=True))
        except TypeError:
            try:
                return _block_count(get_block_ids())
            except Exception:
                return 0
        except Exception:
            return 0
    if isinstance(value, Mapping):
        return sum(_block_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        total = 0
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set, frozenset, Mapping)):
                total += _block_count(item)
            else:
                total += 1
        return total
    return 1


def _sum_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, Mapping):
        return sum(_sum_tokens(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_sum_tokens(item) for item in value)
    token_count = _get_any(value, ["num_tokens", "token_count", "num_scheduled_tokens"])
    if token_count is not None:
        return _sum_tokens(token_count)
    token_ids = _get_any(value, ["token_ids", "input_ids", "output_token_ids"])
    if token_ids is not None:
        return _len_or_zero(token_ids)
    return 0


def _tokens_for_selected_reqs(token_map: Any, selected: Any) -> int:
    if not isinstance(token_map, Mapping) or selected is None:
        return 0
    selected_ids = set()
    for item in _iter_request_items(selected):
        request_id = _request_id(item)
        if request_id is not None:
            selected_ids.add(str(request_id))
    total = 0
    for key, value in token_map.items():
        if str(key) in selected_ids:
            total += _sum_tokens(value)
    return total


def _request_ids(value: Any) -> list[str]:
    ids: list[str] = []
    for item in _iter_request_items(value):
        request_id = _request_id(item)
        if request_id is not None:
            ids.append(str(request_id))
    return _dedupe_ids(ids)


def _iter_request_items(value: Any) -> Iterator[Any]:
    if value is None:
        return
    if isinstance(value, Mapping):
        yield from value.values()
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        yield from value
        return
    yield value


def _request_id(value: Any) -> Any:
    if isinstance(value, (str, int)):
        return value
    return _get_any(value, ["request_id", "req_id", "seq_group_id", "id"])


def _dedupe_ids(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _unique_ids(*groups: Sequence[str]) -> list[str]:
    return _dedupe_ids(item for group in groups for item in group)


def _join_ids(values: Sequence[str]) -> str:
    return " ".join(values)


def _positive_token_req_ids(token_map: Any) -> list[str]:
    if not isinstance(token_map, Mapping):
        return []
    return _dedupe_ids(
        str(key)
        for key, value in token_map.items()
        if _sum_tokens(value) > 0
    )


def _cached_req_ids(cached_request_data: Any) -> list[str]:
    req_ids = _get_any(cached_request_data, ["req_ids"])
    if req_ids is None:
        return []
    if isinstance(req_ids, Mapping):
        return [str(item) for item in req_ids.keys()]
    if isinstance(req_ids, (list, tuple, set, frozenset)):
        return [str(item) for item in req_ids]
    return [str(req_ids)]


def _cached_is_context_phase(cached_request_data: Any, req_id: str) -> bool:
    method = getattr(cached_request_data, "is_context_phase", None)
    if callable(method):
        try:
            return bool(method(req_id))
        except Exception:
            return False
    return False


def _count_cached_context_reqs(cached_request_data: Any, token_map: Any) -> int:
    count = 0
    for req_id in _cached_req_ids(cached_request_data):
        if _sum_tokens_for_req(token_map, req_id) <= 0:
            continue
        if _cached_is_context_phase(cached_request_data, req_id):
            count += 1
    return count


def _count_cached_decode_reqs(cached_request_data: Any, token_map: Any) -> int:
    count = 0
    for req_id in _cached_req_ids(cached_request_data):
        if _sum_tokens_for_req(token_map, req_id) <= 0:
            continue
        if not _cached_is_context_phase(cached_request_data, req_id):
            count += 1
    return count


def _cached_context_req_ids(cached_request_data: Any, token_map: Any) -> list[str]:
    return [
        req_id
        for req_id in _cached_req_ids(cached_request_data)
        if _sum_tokens_for_req(token_map, req_id) > 0
        and _cached_is_context_phase(cached_request_data, req_id)
    ]


def _cached_decode_req_ids(cached_request_data: Any, token_map: Any) -> list[str]:
    return [
        req_id
        for req_id in _cached_req_ids(cached_request_data)
        if _sum_tokens_for_req(token_map, req_id) > 0
        and not _cached_is_context_phase(cached_request_data, req_id)
    ]


def _cached_context_tokens(cached_request_data: Any, token_map: Any) -> int:
    total = 0
    for req_id in _cached_req_ids(cached_request_data):
        if _cached_is_context_phase(cached_request_data, req_id):
            total += _sum_tokens_for_req(token_map, req_id)
    return total


def _sum_tokens_for_req(token_map: Any, req_id: str) -> int:
    if not isinstance(token_map, Mapping):
        return 0
    if req_id in token_map:
        return _sum_tokens(token_map[req_id])
    for key, value in token_map.items():
        if str(key) == str(req_id):
            return _sum_tokens(value)
    return 0


def _first_positive_int(values: Sequence[Any]) -> int:
    for value in values:
        number = _to_int(value, 0)
        if number > 0:
            return number
    return 0


def _next_power_of_two_or_capture_bucket(value: int) -> int:
    capture_sizes = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
        272,
        288,
        304,
        320,
        336,
        352,
        368,
        384,
        400,
        416,
        432,
        448,
        464,
        480,
        496,
        512,
    ]
    for size in capture_sizes:
        if value <= size:
            return size
    return value


def _next_step_id(after: Mapping[str, Any], before: Mapping[str, Any]) -> str:
    for state in (after, before):
        value = state.get("scheduler_step")
        if value != "":
            return str(value)
    return ""


def _queue_length_from_names(obj: Any, names: Sequence[str]) -> int | str:
    value = _get_any(obj, names)
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return int(value)
    length = _len_or_zero(value)
    if length:
        return length
    return ""


def _queue_request_ids_from_names(obj: Any, names: Sequence[str]) -> list[str]:
    value = _get_any(obj, names)
    if value is None:
        return []
    return _request_ids(_queue_items(value))


def _queue_items(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.values()
    for attr in ("requests", "queue", "_queue", "_items"):
        items = _get_any(value, [attr])
        if items is not None:
            return items
    if not isinstance(value, (str, bytes)):
        try:
            return list(value)
        except TypeError:
            pass
    return value


def _counter_from_names(obj: Any, names: Sequence[str]) -> int | str:
    value = _get_any(obj, names)
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return int(value)
    length = _len_or_zero(value)
    return length if length else ""


def _free_kv_blocks(obj: Any) -> int | str:
    kv_cache_manager = _get_any(obj, ["kv_cache_manager"])
    block_pool = _get_any(kv_cache_manager, ["block_pool"]) if kv_cache_manager else None
    candidates = [
        obj,
        _get_any(obj, ["block_manager", "cache_manager", "kv_cache_manager"]),
        _get_any(obj, ["scheduler"]),
        block_pool,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for method_name in [
            "get_num_free_gpu_blocks",
            "get_num_free_blocks",
            "get_num_free_blocks_gpu",
        ]:
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    return int(method())
                except Exception:
                    continue
        value = _get_any(candidate, [
            "num_free_gpu_blocks",
            "free_gpu_blocks",
            "free_kv_blocks",
            "free_blocks",
        ])
        if isinstance(value, (int, float)):
            return int(value)
        length = _len_or_zero(value)
        if length:
            return length
    return ""


def _first_numeric(primary: Mapping[str, Any], fallback: Mapping[str, Any], key: str) -> Any:
    value = primary.get(key, "")
    if value != "":
        return value
    return fallback.get(key, "")


def _first_text(primary: Mapping[str, Any], fallback: Mapping[str, Any], key: str) -> str:
    value = primary.get(key, "")
    if value not in (None, ""):
        return str(value)
    value = fallback.get(key, "")
    return "" if value in (None, "") else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
