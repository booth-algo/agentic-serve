"""Auto-install agentic vLLM EngineCore trace hooks in spawned workers.

This file is intentionally inert unless AGENTIC_VLLM_ENGINE_TRACE_JSONL is set.
`run_instrumented_api_server.py` prepends this directory to PYTHONPATH before
starting vLLM, so Python subprocesses spawned by vLLM import this module at
interpreter startup and receive the same trace hooks as the API-server process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _install_from_env() -> None:
    trace_jsonl = os.environ.get("AGENTIC_VLLM_ENGINE_TRACE_JSONL")
    if not trace_jsonl:
        return

    try:
        from profiling.profile.vllm.engine_trace.serving_engine_steps import (
            install_vllm_v1_class_hook,
        )

        install_vllm_v1_class_hook(
            trace_jsonl_path=Path(trace_jsonl),
            case_json_path=None,
            run_id=os.environ.get("AGENTIC_VLLM_ENGINE_TRACE_RUN_ID", ""),
            enable_engine_wall_trace=_truthy("AGENTIC_VLLM_ENGINE_WALL_TRACE"),
            enable_worker_wall_trace=_truthy("AGENTIC_VLLM_WORKER_WALL_TRACE"),
            enable_worker_cuda_sync=_truthy("AGENTIC_VLLM_WORKER_CUDA_SYNC"),
            server_trace_from_request_ids=_truthy(
                "AGENTIC_VLLM_SERVER_TRACE_FROM_REQUEST_IDS"
            ),
            include_unmatched_server_steps=_truthy(
                "AGENTIC_VLLM_INCLUDE_UNMATCHED_SERVER_STEPS"
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive child-process hook.
        print(
            f"[agentic-trace] failed to install vLLM trace hook: {exc}",
            file=sys.stderr,
        )


_install_from_env()
