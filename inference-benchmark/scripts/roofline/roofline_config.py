"""
Roofline profiling configuration.

H100 SXM hardware specs, model registry, and kernel classification.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── H100 SXM5 80GB specs ──────────────────────────────────────────────

@dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_bf16_tflops: float      # tensor core, no sparsity
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    hbm_bandwidth_tb_s: float    # TB/s
    l2_cache_mb: float
    sm_count: int

    @property
    def ridge_point_bf16(self) -> float:
        """Arithmetic intensity where compute ceiling meets memory slope."""
        return (self.peak_bf16_tflops * 1e12) / (self.hbm_bandwidth_tb_s * 1e12)


H100_SXM = GpuSpec(
    name="H100 SXM5 80GB",
    peak_bf16_tflops=989.4,
    peak_fp16_tflops=989.4,
    peak_fp32_tflops=67.0,
    hbm_bandwidth_tb_s=3.35,
    l2_cache_mb=50,
    sm_count=132,
)


# ── Model registry ────────────────────────────────────────────────────

MODELS_DIR = Path("/workspace/models")

@dataclass
class ModelSpec:
    path: str
    tp_size: int = 1
    is_moe: bool = False
    trust_remote_code: bool = False
    extra_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Llama-3.1-8B": ModelSpec(
        path=str(MODELS_DIR / "Llama-3.1-8B-Instruct"),
        tp_size=1,
    ),
    "Qwen3.5-9B": ModelSpec(
        path=str(MODELS_DIR / "Qwen3.5-9B"),
        tp_size=1,
        trust_remote_code=True,
    ),
    "gpt-oss-20b": ModelSpec(
        path=str(MODELS_DIR / "gpt-oss-20b"),
        tp_size=1,
        is_moe=True,
        trust_remote_code=True,
    ),
    "Qwen3.5-27B": ModelSpec(
        path=str(MODELS_DIR / "Qwen3.5-27B"),
        tp_size=2,
        trust_remote_code=True,
    ),
    "Qwen3-32B": ModelSpec(
        path=str(MODELS_DIR / "Qwen3-32B"),
        tp_size=2,
        trust_remote_code=True,
    ),
    "Qwen2.5-72B": ModelSpec(
        path=str(MODELS_DIR / "Qwen2.5-72B-Instruct"),
        tp_size=2,
        trust_remote_code=True,
    ),
    "Llama-3.1-70B": ModelSpec(
        path=str(MODELS_DIR / "Llama-3.1-70B-Instruct"),
        tp_size=2,
    ),
    "gpt-oss-120b": ModelSpec(
        path=str(MODELS_DIR / "gpt-oss-120b"),
        tp_size=2,
        is_moe=True,
        trust_remote_code=True,
    ),
}


# ── Kernel classification ─────────────────────────────────────────────
# Ordered list: first match wins.

KERNEL_CATEGORIES: list[tuple[str, str]] = [
    # Attention kernels (before GEMM — flash attention uses GEMM internally)
    (r"flash_?attn|flash_attention|fmha|cutlass.*attention|sdpa|flash_fwd|flash_bwd", "attention"),
    # GEMM / linear layers (aten::mm, cuBLAS, nvjet = NVIDIA's new GEMM library)
    (r"aten::mm\b|aten::addmm|aten::bmm|aten::linear|gemm|cublas|cutlass.*gemm|wgmma|sm90_gemm|cublasLt|nvjet_", "gemm"),
    # Layer/RMS normalization
    (r"layer_?norm|rms_?norm|LayerNorm|RMSNorm", "layernorm"),
    # Softmax
    (r"softmax|Softmax", "softmax"),
    # MoE-specific
    (r"topk|moe.*gate|expert.*route|gating", "moe-routing"),
    (r"permute.*expert|scatter.*expert|expert.*permute", "moe-dispatch"),
    # Activations
    (r"aten::silu|aten::gelu|silu|gelu|swiglu|SiLU|GeLU", "activation"),
    # Elementwise ops (mul, add, pow — often part of RMSNorm or residual connections)
    (r"aten::mul\b|aten::add\b|aten::pow\b|aten::neg\b|aten::rsqrt|aten::mean\b|elementwise_kernel|vectorized_elementwise", "elementwise"),
    # Communication (TP > 1) — allreduce, not plain reduce
    (r"allreduce|nccl|ncclKernel|all_reduce", "communication"),
    # Sampling / token selection
    (r"sample|argmax|top_p|top_k.*sample", "sampling"),
    # Memory operations (copy, concat, reshape, slice)
    (r"memcpy|memset|Memcpy|Memset|aten::copy_|aten::cat\b|aten::slice|aten::reshape|CatArrayBatchedCopy|unrolled_elementwise.*copy", "memory"),
    # Rotary embeddings (cos, sin, gather for position encoding)
    (r"rotary|rope|RoPE|aten::cos\b|aten::sin\b", "rope"),
    # Reduce operations (not allreduce)
    (r"reduce_kernel|aten::sum\b", "reduce"),
    # Indexing / embedding lookup / position generation
    (r"aten::gather|aten::arange|aten::index|vectorized_gather|indexSelect|aten::embedding", "indexing"),
    # Profiler/runtime overhead (skip)
    (r"Activity Buffer|Lazy Function|Runtime Triggered|aten::random_|distribution_elementwise|distribution_nullary", "profiler-overhead"),
]

_compiled_categories = [(re.compile(pat, re.IGNORECASE), cat) for pat, cat in KERNEL_CATEGORIES]


def classify_kernel(kernel_name: str) -> str:
    """Classify a CUDA kernel name into a category."""
    for pattern, category in _compiled_categories:
        if pattern.search(kernel_name):
            return category
    return "other"


# ── Profiling defaults ────────────────────────────────────────────────

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64]
DEFAULT_SEQ_LEN = 512        # prefill input length
DEFAULT_NUM_LAYERS = 2       # layers to profile (keeps ncu fast)
NCU_PATH = "/usr/local/cuda/bin/ncu"
