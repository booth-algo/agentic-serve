"""Feature specification for per-kernel XGBoost predictors.

Single source of truth for:
- per-family feature column lists (must match `predictors/per_kernel/predictor.py`)
- the derived-column builders that convert raw shape inputs into training rows
- dtype byte sizes + leak-guard substrings

Extracted from `trainer.py` so that the data-pipeline layer
(`split_by_family.py`) and the training layer both import the same
definitions and can't drift. The runtime predictor at
`llm_predict/predictors/per_kernel/predictor.py` must also use these
feature lists when constructing its inference-time feature vector.

Design note — per-family CSVs under `data/per_family/{gpu}/` store only
RAW shape columns (M/N/K/dtype for gemm; bs/seq/n_heads/head_dim/kv_heads
for flash_attn; numel/op_type for elementwise; family+shape for misc).
Derived features (log, analytical_flops, one-hots, etc.) are materialised
at train/predict time by the `build_*_features` functions below. This
keeps the CSVs small and ensures feature-engineering changes only
require re-importing, not re-splitting.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Per `.claude/paper/predictor_notes.md`: RTX2080Ti is pinned to fp16;
# A100 / RTX3090 are bf16. fp8 is placeholder for future FP8 coverage.
DTYPE_BYTES: dict[str, int] = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}

# Raw ncu metric columns that must never appear as training features —
# they leak target time information or hardware-counter signal the
# runtime predictor cannot reproduce from shape alone.
FORBIDDEN_SUBSTRINGS: list[str] = [
    "cycles", "throughput", "launch_", "dram_bytes_sum", "dram_bytes_read",
    "dram_bytes_write", "register", "occupancy", "sm_active",
    "compute_throughput", "memory_throughput",
]


# ───────── Raw CSV columns per family (what split_by_family.py writes) ─────────

GEMM_RAW_COLS: list[str] = [
    "gpu", "model", "held_out", "kernel_name",
    "M", "N", "K", "dtype", "op_type",
    "gpu_time_duration_ms", "target_us",
]

FLASH_ATTN_RAW_COLS: list[str] = [
    "gpu", "model", "held_out", "kernel_name",
    "bs", "seq", "n_heads", "head_dim", "kv_heads", "dtype",
    "gpu_time_duration_ms", "target_us",
]

ELEMENTWISE_RAW_COLS: list[str] = [
    "gpu", "model", "held_out", "kernel_name",
    "numel", "op_type", "dtype",
    "gpu_time_duration_ms", "target_us",
]

MISC_RAW_COLS: list[str] = [
    "gpu", "model", "held_out", "kernel_name",
    "kernel_family",           # reduce | splitk_reduce | cast | copy
    "M", "N", "K", "numel", "dtype",
    "gpu_time_duration_ms", "target_us",
]


# ───────── Feature lists (what the trained XGB models see) ─────────

GEMM_FEATURES: list[str] = [
    "M", "N", "K", "log_M", "log_N", "log_K",
    "analytical_flops", "analytical_bytes", "analytical_ai",
    "log_flops", "log_bytes", "dtype_onehot_bf16",
]

FLASH_ATTN_FEATURES: list[str] = [
    "bs", "seq", "n_heads", "head_dim", "kv_heads",
    "total_flops", "log_seq", "log_heads",
]

ELEM_OPS: list[str] = [
    "rmsnorm", "silu", "mul", "residual", "rope", "neg", "fill",
    "compare", "other",
]

ELEMENTWISE_FEATURES: list[str] = (
    ["numel", "log_numel"]
    + [f"op_type_onehot_{op}" for op in ELEM_OPS]
)

MISC_FAMILIES: list[str] = ["reduce", "splitk_reduce", "cast", "copy"]

MISC_FEATURES: list[str] = (
    ["numel_or_shape_total", "log_numel_or_shape",
     "size_m", "size_n", "size_k",
     "log_size_m", "log_size_n", "log_size_k"]
    + [f"kernel_family_onehot_{fam}" for fam in MISC_FAMILIES]
)


# ───────── Builders: raw → training matrix ─────────

def build_gemm_features(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["M"].notna() & df["N"].notna() & df["K"].notna()].copy()
    sub["dtype_bytes"] = sub["dtype"].map(DTYPE_BYTES).fillna(2.0)
    sub["log_M"] = np.log1p(sub["M"])
    sub["log_N"] = np.log1p(sub["N"])
    sub["log_K"] = np.log1p(sub["K"])
    sub["analytical_flops"] = 2.0 * sub["M"] * sub["N"] * sub["K"]
    sub["analytical_bytes"] = (
        sub["M"] * sub["K"] + sub["N"] * sub["K"] + sub["M"] * sub["N"]
    ) * sub["dtype_bytes"]
    sub["analytical_ai"] = sub["analytical_flops"] / np.maximum(sub["analytical_bytes"], 1.0)
    sub["log_flops"] = np.log1p(sub["analytical_flops"])
    sub["log_bytes"] = np.log1p(sub["analytical_bytes"])
    sub["dtype_onehot_bf16"] = (sub["dtype"] == "bf16").astype(int)
    return sub


def build_flash_attn_features(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub["total_flops"] = 4.0 * sub["bs"] * sub["n_heads"] * (sub["seq"] ** 2) * sub["head_dim"]
    sub["log_seq"] = np.log1p(sub["seq"])
    sub["log_heads"] = np.log1p(sub["n_heads"])
    return sub


def build_elementwise_features(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub["log_numel"] = np.log1p(sub["numel"].fillna(0))
    for op in ELEM_OPS:
        sub[f"op_type_onehot_{op}"] = (sub["op_type"] == op).astype(int)
    known = set(ELEM_OPS)
    sub.loc[~sub["op_type"].isin(known), "op_type_onehot_other"] = 1
    return sub


def build_misc_features(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    m = sub["M"].fillna(0.0)
    n = sub["N"].fillna(0.0)
    k = sub["K"].fillna(0.0)
    nn = sub["numel"].fillna(0.0)
    size_proxy = np.where(nn > 0, nn, m * n)
    sub["numel_or_shape_total"] = size_proxy
    sub["log_numel_or_shape"] = np.log1p(size_proxy)
    sub["size_m"] = m
    sub["size_n"] = n
    sub["size_k"] = k
    sub["log_size_m"] = np.log1p(m)
    sub["log_size_n"] = np.log1p(n)
    sub["log_size_k"] = np.log1p(k)
    for fam in MISC_FAMILIES:
        sub[f"kernel_family_onehot_{fam}"] = (sub["kernel_family"] == fam).astype(int)
    return sub


# ───────── FAMILY_CONFIG: the one table trainer + splitter share ─────────

FAMILY_CONFIG: dict[str, dict[str, Any]] = {
    "gemm": {
        "kernel_families": ["gemm"],
        "raw_cols": GEMM_RAW_COLS,
        "features": GEMM_FEATURES,
        "builder": build_gemm_features,
    },
    "flash_attn": {
        "kernel_families": ["flash_attn"],
        "raw_cols": FLASH_ATTN_RAW_COLS,
        "features": FLASH_ATTN_FEATURES,
        "builder": build_flash_attn_features,
    },
    "elementwise": {
        "kernel_families": ["elementwise"],
        "raw_cols": ELEMENTWISE_RAW_COLS,
        "features": ELEMENTWISE_FEATURES,
        "builder": build_elementwise_features,
    },
    "misc": {
        "kernel_families": MISC_FAMILIES,
        "raw_cols": MISC_RAW_COLS,
        "features": MISC_FEATURES,
        "builder": build_misc_features,
    },
}


# ───────── Qwen3.5 exclusion (per .claude/paper/predictor_notes.md) ─────────

# Models that are excluded from specific family training pools.
# Qwen3.5 uses hybrid attention (3/4 of layers are linear_attention via
# triton chunk_gated_delta_rule); only 1/4 emit flash_fwd_* kernels.
# Admitting those flash_attn rows would misrepresent the attention
# distribution — the per-op composer would over-predict flash_attn
# time 4×. Other families (gemm/elementwise/misc) are shape-only and
# safe to admit.
FAMILY_EXCLUDED_MODELS: dict[str, set[str]] = {
    "flash_attn": {"Qwen3.5-9B", "Qwen3.5-27B"},
}


def audit_features(feature_cols: list[str]) -> list[tuple[str, str]]:
    """Returns [(feat, matched_bad_substring), ...] if any feature name
    contains a forbidden substring. Empty list ⇒ no leaks."""
    leaks: list[tuple[str, str]] = []
    for feat in feature_cols:
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in feat.lower():
                leaks.append((feat, bad))
    return leaks


__all__ = [
    "DTYPE_BYTES", "FORBIDDEN_SUBSTRINGS",
    "GEMM_RAW_COLS", "FLASH_ATTN_RAW_COLS", "ELEMENTWISE_RAW_COLS", "MISC_RAW_COLS",
    "GEMM_FEATURES", "FLASH_ATTN_FEATURES", "ELEMENTWISE_FEATURES", "MISC_FEATURES",
    "ELEM_OPS", "MISC_FAMILIES",
    "build_gemm_features", "build_flash_attn_features",
    "build_elementwise_features", "build_misc_features",
    "FAMILY_CONFIG", "FAMILY_EXCLUDED_MODELS",
    "audit_features",
]
