"""Analytical FlashAttention2 model.

FlashAttention fuses Q×K^T + softmax + A×V into a single kernel that
tiles over sequence length, keeping intermediate attention scores in SRAM.
This avoids materializing the O(s^2) attention matrix to HBM.

For prefill (all tokens attend to all): compute-bound for long sequences.
For decode (1 new token attends to full KV cache): memory-bound.
"""

from llmcompass.software_model.utils import DataType


def flash_attention_latency(
    batch_size: int,
    n_q_heads: int,
    n_kv_heads: int,
    seq_len: int,       # query sequence length (prefill) or 1 (decode)
    kv_len: int,        # KV cache length (= seq_len for prefill, > seq_len for decode)
    head_dim: int,
    data_type: DataType,
    peak_flops: float,  # e.g., 989.4e12 for H100 BF16
    hbm_bandwidth: float,  # e.g., 3.35e12 bytes/s for H100
) -> float:
    """Estimate FlashAttention2 latency using roofline model.

    Returns latency in seconds.
    """
    word_size = data_type.word_size

    # FLOPs: 2 matmuls (QK^T and AV) + softmax (negligible vs matmul)
    # Q×K^T: [b, h_q, s, d] × [b, h_kv, d, kv_len] → need to account for GQA grouping
    # Each Q head group shares a KV head, but compute is still per Q head
    # Total FLOPs = 2 * batch * n_q_heads * seq_len * kv_len * head_dim * 2
    # (factor 2 for Q×K^T and A×V, factor 2 for multiply-accumulate)
    total_flops = 4.0 * batch_size * n_q_heads * seq_len * kv_len * head_dim

    # HBM bytes: FlashAttention2 reads Q, K, V once and writes O once
    # Q: [b, n_q_heads, seq_len, head_dim]
    # K: [b, n_kv_heads, kv_len, head_dim]
    # V: [b, n_kv_heads, kv_len, head_dim]
    # O: [b, n_q_heads, seq_len, head_dim]
    q_bytes = batch_size * n_q_heads * seq_len * head_dim * word_size
    k_bytes = batch_size * n_kv_heads * kv_len * head_dim * word_size
    v_bytes = batch_size * n_kv_heads * kv_len * head_dim * word_size
    o_bytes = batch_size * n_q_heads * seq_len * head_dim * word_size
    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes

    # Roofline: max(compute-bound, memory-bound)
    compute_time = total_flops / peak_flops if peak_flops > 0 else 0
    memory_time = total_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0

    return max(compute_time, memory_time)


def elementwise_latency(
    numel: int,
    flops_per_elem: float,
    reads: int,           # number of input tensors read
    writes: int,          # number of output tensors written
    word_size: int,       # bytes per element
    peak_flops: float,
    hbm_bandwidth: float,
) -> float:
    """Estimate latency for an elementwise/memory-bound operation.

    Used for: RoPE, residual add, RMSNorm, memory copies.
    Returns latency in seconds.
    """
    total_flops = numel * flops_per_elem
    total_bytes = numel * word_size * (reads + writes)

    compute_time = total_flops / peak_flops if peak_flops > 0 else 0
    memory_time = total_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0

    return max(compute_time, memory_time)
