"""splits.py — LOMO + held-out split iterators for per-op training.

Thin wrapper over `llm_predict.training.per_kernel.splits`: the split
logic is identical (same `held_out` flag convention, same LOMO
invariants, same `MIN_TRAIN_ROWS` gate). Re-exporting from here keeps
the per_op trainer self-contained (no cross-package imports at call
site) and leaves room for per-op-specific split overrides later
without touching the per_kernel module.

See `llm_predict/training/per_kernel/splits.py` for the canonical
docstring, invariants, and CLI.
"""
from __future__ import annotations

from llm_predict.training.per_kernel.splits import (  # noqa: F401
    MIN_TRAIN_ROWS,
    heldout_split,
    held_out_short_names,
    lomo_models,
    lomo_splits,
)


__all__ = [
    "MIN_TRAIN_ROWS",
    "heldout_split",
    "held_out_short_names",
    "lomo_models",
    "lomo_splits",
]
