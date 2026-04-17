"""Kernel name classification + per-arch tile extraction.

Consolidates the regex tables that lived inline at
`per-kernel-rebuild/label_all_v3.py:65-92,126`. Two responsibilities:

1. `classify_kernel(name)` → (subtype, family). Deterministic first-match on
   an ordered pattern table. Families are the four training targets
   {gemm, flash_attn, elementwise, misc}; subtypes are finer-grained for
   per-kernel op_type labeling and for the training `misc` one-hot.

2. `parse_gemm_tile(name, gpu)` → (tile_m, tile_n) or None. Ampere kernels
   use `s16816gemm_*_MxN_…stage`; Turing (sm_75) kernels use
   `s1688gemm_fp16_MxM_…` (no stages suffix). Hopper adds `s64128gemm_*`
   variants that are structurally similar to ampere.

The legacy `(ampere|hopper|sm\\d+).*gemm` classifier pattern at
label_all_v3.py:68 silently dropped every Turing GEMM kernel into `other`,
because `turing_fp16_s1688gemm_*` does not match `ampere|hopper|sm\\d+`. This
module fixes that bug by adding `turing` to the gemm family pattern and
splitting tile parsing by GPU arch.
"""
from __future__ import annotations

import re
from typing import Optional


# ───────────────────────── Classifier ────────────────────────────────────

# First-match wins. Order matters — splitk_reduce must come before gemm
# because its kernel name can contain the substring `gemm`.
KERNEL_PATTERNS: list[tuple[str, str]] = [
    ("splitk_reduce",   r"splitKreduce|splitK_reduce"),
    ("gemv",            r"gemmk1_kernel|gemvx_kernel"),
    ("gemm",            r"(ampere|hopper|turing|volta|maxwell|pascal|kepler|sm\d+).*gemm|cublas.*gemm|cutlass.*gemm|magma_sgemm"),
    ("flash_attn",      r"flash_fwd|flash_bwd|fmha|pytorch_flash"),
    ("silu",            r"silu_kernel|silu.*kernel"),
    ("rmsnorm_reduce",  r"MeanOps.*reduce_kernel|reduce_kernel.*MeanOps"),
    ("rmsnorm_pow",     r"pow_tensor_scalar_kernel"),
    ("rmsnorm_rsqrt",   r"rsqrt_kernel"),
    ("elementwise_mul", r"MulFunctor"),
    ("elementwise_add", r"AddFunctor|CUDAFunctor_add|CUDAFunctorOnSelf_add"),
    ("elementwise_neg", r"neg_kernel"),
    ("cast",            r"bfloat16_copy_kernel|LoadWithCast|StoreWithCast|copy_kernel_cuda"),
    ("copy",            r"direct_copy_kernel|Memcpy|memcpy"),
    ("cat",             r"CatArrayBatched"),
    ("rope_cos",        r"cos_kernel"),
    ("rope_sin",        r"sin_kernel"),
    ("gather",          r"vectorized_gather|gather_kernel"),
    ("index",           r"index_elementwise_kernel|indexFuncLargeIndex"),
    ("arange",          r"arange|elementwise_kernel_with_index"),
    ("random",          r"distribution_elementwise|philox|random"),
    ("compact",         r"DeviceCompact|DeviceSelectSweep|write_indices|DeviceReduceSingleTileKernel"),
    ("reduce",          r"reduce_kernel|welford|softmax"),
    ("fill",            r"FillFunctor"),
    ("compare",         r"CompareFunctor|where_kernel"),
    ("elementwise",     r"elementwise_kernel|vectorized_elementwise"),
    ("other",           r".*"),
]

FAMILY_MAP: dict[str, str] = {
    "splitk_reduce": "splitk_reduce",
    "gemv": "gemm",
    "gemm": "gemm",
    "flash_attn": "flash_attn",
    "silu": "elementwise",
    "rmsnorm_reduce": "reduce",
    "rmsnorm_pow": "elementwise",
    "rmsnorm_rsqrt": "elementwise",
    "elementwise_mul": "elementwise",
    "elementwise_add": "elementwise",
    "elementwise_neg": "elementwise",
    "cast": "cast",
    "copy": "copy",
    "cat": "copy",
    "rope_cos": "elementwise",
    "rope_sin": "elementwise",
    "gather": "copy",
    "index": "copy",
    "arange": "other",
    "random": "other",
    "compact": "other",
    "reduce": "reduce",
    "fill": "elementwise",
    "compare": "elementwise",
    "elementwise": "elementwise",
    "other": "other",
}

OP_TYPE_MAP: dict[str, str] = {
    "silu": "silu",
    "rmsnorm_reduce": "rmsnorm",
    "rmsnorm_pow": "rmsnorm",
    "rmsnorm_rsqrt": "rmsnorm",
    "elementwise_mul": "mul",
    "elementwise_add": "residual",
    "elementwise_neg": "neg",
    "rope_cos": "rope",
    "rope_sin": "rope",
    "cast": "cast",
    "fill": "fill",
    "compare": "compare",
}

_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in KERNEL_PATTERNS]


def classify_kernel(name: str) -> tuple[str, str]:
    """Return (subtype, family). Family is one of {gemm, flash_attn,
    elementwise, reduce, cast, copy, splitk_reduce, other}."""
    if not isinstance(name, str):
        return "unknown", "other"
    for label, rx in _COMPILED:
        if rx.search(name):
            return label, FAMILY_MAP.get(label, "other")
    return "other", "other"


def op_type_for(subtype: str) -> str:
    """Map a fine-grained subtype to the training `op_type` label used for
    elementwise/reduce/cast rows."""
    return OP_TYPE_MAP.get(subtype, "other")


# ───────────────────────── Tile extraction ───────────────────────────────

# Ampere (sm_80): ampere_bf16_s16816gemm_bf16_128x128_32x3_nn_align8
# Hopper  (sm_90): hopper_bf16_s64128gemm_bf16_128x128_32x5_nn_align8
_AMPERE_HOPPER_TILE_RE = re.compile(
    r"s(?:16816|64128)gemm_\w+?_(\d+)x(\d+)_", re.IGNORECASE
)

# Turing (sm_75): turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn
_TURING_TILE_RE = re.compile(
    r"s1688gemm_\w+?_(\d+)x(\d+)_", re.IGNORECASE
)


def parse_gemm_tile(name: str, gpu: Optional[str] = None) -> Optional[tuple[int, int]]:
    """Parse (tile_m, tile_n) from a tensor-core GEMM kernel name.

    If `gpu` is omitted, both ampere/hopper and turing patterns are tried in
    that order. Returns None if no pattern matches (e.g. CUTLASS generic
    kernels, gemvx, splitk_reduce).
    """
    if not isinstance(name, str):
        return None
    if gpu is None or gpu.upper().startswith(("A100", "H100", "RTX3090", "RTX40", "RTX50")):
        m = _AMPERE_HOPPER_TILE_RE.search(name)
        if m:
            return int(m.group(1)), int(m.group(2))
    if gpu is None or gpu.upper().startswith(("RTX2080", "RTX20", "T4", "TURING")):
        m = _TURING_TILE_RE.search(name)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


__all__ = [
    "classify_kernel",
    "op_type_for",
    "parse_gemm_tile",
    "KERNEL_PATTERNS",
    "FAMILY_MAP",
    "OP_TYPE_MAP",
]
