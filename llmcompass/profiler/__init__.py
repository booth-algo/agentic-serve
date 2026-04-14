"""
LLMCompass GPU Kernel Profiler
==============================

Profiles real GPU kernel latencies (GEMM, attention, elementwise) and
saves CSV lookup tables used by the analytical cost model.

Usage
-----
    python -m llmcompass.profiler.kernel_profiler --device cuda:0 --output-dir profiler/profiles/H100
    python -m llmcompass.profiler.kernel_profiler --quick   # reduced grid for testing
"""

from .kernel_profiler import (
    profile_gemm,
    profile_attention,
    profile_elementwise,
)

__all__ = [
    "profile_gemm",
    "profile_attention",
    "profile_elementwise",
]
