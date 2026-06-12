#!/usr/bin/env python3
"""Launch vLLM's OpenAI API server with EngineCore trace hooks installed."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from profiling.profile.vllm.engine_trace.serving_engine_steps import (
    install_vllm_v1_class_hook,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--trace-run-id", default="benchmark-serving")
    parser.add_argument("--enable-engine-wall-trace", action="store_true")
    parser.add_argument("--enable-worker-wall-trace", action="store_true")
    parser.add_argument("--enable-worker-cuda-sync", action="store_true")
    parser.add_argument("--include-unmatched-server-steps", action="store_true")
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed after '--' to vllm.entrypoints.openai.api_server.",
    )
    args = parser.parse_args(argv)
    if args.vllm_args and args.vllm_args[0] == "--":
        args.vllm_args = args.vllm_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace_jsonl = args.trace_output.with_suffix(".jsonl")
    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if trace_jsonl.exists():
        trace_jsonl.unlink()

    _export_child_hook_env(args=args, trace_jsonl=trace_jsonl)

    installed = install_vllm_v1_class_hook(
        trace_jsonl_path=trace_jsonl,
        case_json_path=None,
        run_id=args.trace_run_id,
        enable_engine_wall_trace=args.enable_engine_wall_trace,
        enable_worker_wall_trace=args.enable_worker_wall_trace,
        enable_worker_cuda_sync=args.enable_worker_cuda_sync,
        server_trace_from_request_ids=True,
        include_unmatched_server_steps=args.include_unmatched_server_steps,
    )
    if not installed:
        raise RuntimeError("Could not install vLLM V1 scheduler trace hook")

    sys.argv = ["vllm.entrypoints.openai.api_server", *args.vllm_args]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
    return 0


def _export_child_hook_env(*, args: argparse.Namespace, trace_jsonl: Path) -> None:
    """Make spawned vLLM EngineCore processes install the same trace hooks."""

    sitecustomize_dir = Path(__file__).resolve().parent
    pythonpath_parts = [
        part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part
    ]
    if str(sitecustomize_dir) not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(sitecustomize_dir), *pythonpath_parts]
        )

    os.environ["AGENTIC_VLLM_ENGINE_TRACE_JSONL"] = str(trace_jsonl)
    os.environ["AGENTIC_VLLM_ENGINE_TRACE_RUN_ID"] = args.trace_run_id
    os.environ["AGENTIC_VLLM_ENGINE_WALL_TRACE"] = (
        "1" if args.enable_engine_wall_trace else "0"
    )
    os.environ["AGENTIC_VLLM_WORKER_WALL_TRACE"] = (
        "1" if args.enable_worker_wall_trace else "0"
    )
    os.environ["AGENTIC_VLLM_WORKER_CUDA_SYNC"] = (
        "1" if args.enable_worker_cuda_sync else "0"
    )
    os.environ["AGENTIC_VLLM_SERVER_TRACE_FROM_REQUEST_IDS"] = "1"
    os.environ["AGENTIC_VLLM_INCLUDE_UNMATCHED_SERVER_STEPS"] = (
        "1" if args.include_unmatched_server_steps else "0"
    )


if __name__ == "__main__":
    raise SystemExit(main())
