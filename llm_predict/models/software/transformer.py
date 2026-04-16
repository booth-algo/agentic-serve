from llm_predict.models.software.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from llm_predict.models.software.matmul import Matmul, BatchedMatmul
from llm_predict.models.software.softmax import Softmax
from llm_predict.models.software.layernorm import LayerNorm
from llm_predict.models.software.gelu import GeLU
from llm_predict.models.software.silu import SiLU


from llm_predict.models.software.utils import Tensor, DataType
from llm_predict.models.software.communication_primitives import AllReduceMultiPCB, AllReduceHierarchical
from llm_predict.predictors.calibration.moe import post_prediction_correction
from llm_predict.predictors.calibration.allreduce import calibrated_allreduce_latency


def _make_allreduce(data_type, use_hierarchical: bool = False):
    """Create appropriate AllReduce primitive. Hierarchical is used for multi-node."""
    if use_hierarchical:
        return AllReduceHierarchical(data_type)
    return AllReduceMultiPCB(data_type)
from typing import List
from llm_predict.models.hardware.system import System


# Default GEMM calibration factors per size category.
# Real cuBLAS/nvjet kernels are auto-tuned per shape, making them faster than
# the conservative heuristic-GPU tiling schedule in matmul.py.
DEFAULT_GEMM_CALIBRATION_FACTORS = {
    'large': 0.6,   # M*N*K > 1e9: cuBLAS well-optimized
    'medium': 0.5,  # 1e6 < M*N*K < 1e9
    'small': 0.4,   # M*N*K < 1e6: kernel launch overhead dominates
}


def _apply_gemm_calibration(latency: float, M: int, N: int, K: int, factors: dict) -> float:
    """Apply calibration factor to a GEMM latency based on problem size."""
    mnk = M * N * K
    if mnk > 1e9:
        return latency * factors.get('large', 0.6)
    elif mnk > 1e6:
        return latency * factors.get('medium', 0.5)
    else:
        return latency * factors.get('small', 0.4)


def _calibrate_matmul(matmul_op, raw_latency: float, factors: dict) -> float:
    """Apply GEMM calibration to a Matmul operator's latency."""
    if hasattr(matmul_op, 'computational_graph') and matmul_op.computational_graph is not None:
        cg = matmul_op.computational_graph
        return _apply_gemm_calibration(raw_latency, cg.M, cg.N, cg.K, factors)
    return raw_latency


def _calibrate_batched_matmul(bmm_op, raw_latency: float, factors: dict) -> float:
    """Apply GEMM calibration to a BatchedMatmul operator's latency."""
    if hasattr(bmm_op, 'M') and hasattr(bmm_op, 'N') and hasattr(bmm_op, 'K'):
        return _apply_gemm_calibration(raw_latency, bmm_op.M, bmm_op.N, bmm_op.K, factors)
    return raw_latency


def _flash_attention_latency_prefill(
    batch: int, n_heads: int, seq_len: int, head_dim: int,
    word_size: int, hbm_bandwidth: float, peak_flops: float,
    n_kv_heads: int = None,
) -> float:
    """
    FlashAttention2 analytical model for prefill (full sequence attention).

    FlashAttention fuses Q*K^T, softmax, and A*V into a single pass staying in SRAM,
    avoiding full HBM round-trips for the intermediate attention matrix.

    Uses roofline: max(compute_time, memory_time).
    For GQA, n_kv_heads < n_heads reduces K/V memory traffic.
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads  # MHA default
    # Compute: 2 matmuls (QK^T and AV), each 2*b*n_q*s^2*d FLOPs
    # All Q heads compute attention (K is broadcast for GQA)
    compute_flops = 2 * 2 * batch * n_heads * seq_len * seq_len * head_dim
    compute_time = compute_flops / peak_flops if peak_flops > 0 else 0

    # Memory: read Q(n_q) + K(n_kv) + V(n_kv) + write O(n_q)
    memory_bytes = (2 * n_heads + 2 * n_kv_heads) * batch * seq_len * head_dim * word_size
    memory_time = memory_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0

    return max(compute_time, memory_time)


def _flash_attention_latency_decode(
    batch: int, n_kv_heads: int, seq_len: int, head_dim: int,
    word_size: int, hbm_bandwidth: float, peak_flops: float,
    n_q_heads: int = None,
) -> float:
    """
    FlashAttention analytical model for decode (autoregressive, single query token).

    Decode attention is memory-bound: reads full KV cache once per query token.
    For GQA, K/V cache uses n_kv_heads but compute uses n_q_heads.
    """
    if n_q_heads is None:
        n_q_heads = n_kv_heads  # MHA default
    # Memory: read K cache + V cache = 2 * b * n_kv_heads * seq * d_h * word_size
    # Q query + O output are single-token: negligible vs KV cache
    memory_bytes = 2 * batch * n_kv_heads * seq_len * head_dim * word_size
    memory_time = memory_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0

    # Compute: each Q head attends to KV cache (K broadcast for GQA)
    compute_flops = 2 * 2 * batch * n_q_heads * seq_len * head_dim
    compute_time = compute_flops / peak_flops if peak_flops > 0 else 0

    return max(compute_time, memory_time)


def _elementwise_latency(
    total_bytes: float, total_flops: float,
    hbm_bandwidth: float, peak_flops: float,
) -> float:
    """Roofline latency for an elementwise op."""
    mem_time = total_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0
    comp_time = total_flops / peak_flops if peak_flops > 0 else 0
    return max(mem_time, comp_time)


def _rope_latency(
    batch: int, seq_len: int, n_heads: int, n_kv_heads: int,
    head_dim: int, word_size: int, hbm_bandwidth: float, peak_flops: float,
) -> float:
    """RoPE: reads Q and K, applies cos/sin rotation, writes back."""
    total_heads = n_heads + n_kv_heads
    # Read Q,K + write Q,K = 4 tensors
    total_bytes = 4 * batch * seq_len * total_heads * head_dim * word_size
    # 6 FLOPs per element (mul cos, mul sin, add/sub for each pair)
    total_flops = 6 * batch * seq_len * total_heads * head_dim
    return _elementwise_latency(total_bytes, total_flops, hbm_bandwidth, peak_flops)


def _residual_add_latency(
    batch: int, seq_len: int, d_model: int, word_size: int,
    hbm_bandwidth: float,
) -> float:
    """Residual add: read x + read residual + write output = 3 tensors."""
    total_bytes = 3 * batch * seq_len * d_model * word_size
    return total_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0


def _memory_copy_latency(
    batch: int, seq_len: int, d_model: int, word_size: int,
    hbm_bandwidth: float,
) -> float:
    """Memory copy/cast: read + write = 2 tensors."""
    total_bytes = 2 * batch * seq_len * d_model * word_size
    return total_bytes / hbm_bandwidth if hbm_bandwidth > 0 else 0


_perop_predictor_cache = None
def get_perop_predictor():
    global _perop_predictor_cache
    if _perop_predictor_cache is not None:
        return _perop_predictor_cache
    import pickle, os
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkl_path = os.path.join(base, 'profiling', 'data', 'A100', 'trained', 'perop_analytical_v4.pkl')
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    _perop_predictor_cache = data['model']
    return _perop_predictor_cache


def compute_perop_features(n_tokens, op, d_model, n_heads, n_kv_heads, intermediate_size, num_experts, top_k, batch_size=1, seq_len=None, kv_cache_len=0):
    """Compute analytical features for per-op XGBoost prediction.
    
    For decode (seq=1): kv_cache_len determines attention scan length.
    For prefill (seq>1): kv_cache_len=0, attention is self-attention over seq.
    """
    import numpy as np
    tok = float(n_tokens); d = float(d_model); h = float(n_heads)
    kv = float(n_kv_heads); ffn = float(intermediate_size)
    E = float(num_experts); k = float(top_k); d_h = d / h

    is_a = float(op == "attn"); is_f = float(op == "ffn"); is_n = float(op in ("norm_pre", "norm_post"))

    tpe = max(1, tok * k / max(E, 1)) if E > 0 else 0
    aexp = min(tok * k, E) if E > 0 else 0

    import math
    if seq_len is None:
        seq_len = n_tokens
    bs_f = float(batch_size); seq_f = float(seq_len)
    
    # For decode, attention scans KV cache; for prefill, self-attention over seq
    effective_kv = max(float(kv_cache_len), seq_f) if kv_cache_len > 0 else seq_f

    # Attention FLOPs: QKV proj + attention scores (tok * effective_kv) + O proj
    attn_fl = is_a * (2*tok*d*(d + 2*kv*d_h) + 4*bs_f*h*seq_f*effective_kv*d_h + 2*tok*d*d)
    dffn_fl = is_f * (1 - min(E, 1)) * 2*tok*d*ffn*3
    mffn_fl = is_f * (2*tpe*d*ffn*3*aexp if E > 0 else 0)
    norm_fl = is_n * tok * d * 5
    total_fl = attn_fl + dffn_fl + mffn_fl + norm_fl

    wt = (is_a * d * (d + 2*kv*d_h + d) * 2 +
          is_f * (E*d*ffn*3*2 if E > 0 else d*ffn*3*2) +
          is_n * d * 2)
    ai = total_fl / wt if wt > 0 else 0

    attn_quad = is_a * bs_f * seq_f * effective_kv * h * d_h
    return [tok, math.log2(tok+1), total_fl, math.log2(total_fl+1),
            wt, math.log2(wt+1), ai, math.log2(ai+1),
            is_a, is_f, is_n, d, ffn, E, k,
            d*d, d*ffn, tpe if E > 0 else tok, aexp if E > 0 else 1, wt,
            bs_f, seq_f, math.log2(bs_f+1), math.log2(seq_f+1),
            attn_quad, math.log2(attn_quad+1) if attn_quad > 0 else 0]


class TransformerBlockInitComputationTP(Operator):
    """
    Prefill/initial computation — processes the full prompt sequence in parallel.

    This transformer block is used during the initial forward pass where the entire
    input sequence (prompt) is processed simultaneously. It computes attention over
    the full sequence without KV caching, making it efficient for parallel processing
    of the complete input.

    Supports Grouped Query Attention (GQA) via n_kv_heads parameter:
    - n_kv_heads == n_heads: standard Multi-Head Attention (MHA)
    - n_kv_heads < n_heads: GQA where multiple Q heads share the same KV head
    - n_kv_heads == 1: Multi-Query Attention (MQA)
    """
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, intermediate_size: int = None, n_kv_heads: int = None,
                 use_flash_attention: bool = True, activation_type: str = 'silu', gemm_calibration_factors: dict = None,
                 use_ml_predictor: bool = False, use_hierarchical_allreduce: bool = False,
                 use_cuda_graph: bool = False):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.device_count = device_count
        self.intermediate_size = intermediate_size or 4 * d_model
        self.use_flash_attention = use_flash_attention
        self.activation_type = activation_type
        self.gemm_calibration_factors = gemm_calibration_factors if gemm_calibration_factors is not None else dict(DEFAULT_GEMM_CALIBRATION_FACTORS)
        self.use_ml_predictor = use_ml_predictor
        self.use_hierarchical_allreduce = use_hierarchical_allreduce
        self.use_cuda_graph = use_cuda_graph
        # parameters per device
        d = d_model
        d_h = d // n_heads  # head dimension
        ffn = self.intermediate_size
        self.Wq = Tensor([d, d // device_count], data_type)
        # GQA: K and V projection sizes based on n_kv_heads, not n_heads
        kv_dim_per_device = (self.n_kv_heads * d_h) // device_count
        self.Wk = Tensor([d, kv_dim_per_device], data_type)
        self.Wv = Tensor([d, kv_dim_per_device], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, ffn // device_count], data_type)
        self.W2 = Tensor([ffn // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_activation = SiLU(data_type) if activation_type == 'silu' else GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # Backward-compatible alias
        self.H_gelu = self.H_activation

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension (sharded dimension in TP)
        # d_h: dimension per head
        b, s, d = X.shape
        # Store for compile_and_simulate to use actual dimensions
        self._last_batch_size = b
        self._last_seq_len = s
        # Transformer block expects full d_model dimension; it handles TP sharding internally
        assert d == self.d_model, \
            f"Input dimension {d} must equal d_model ({self.d_model}). TP sharding is handled internally by the block."
        h = self.n_heads
        n_kv = self.n_kv_heads
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        d_h = d // h

        # Q heads per device
        q_heads_per_dev = h // dev_cnt
        # KV heads per device (GQA: fewer KV heads)
        kv_heads_per_dev = n_kv // dev_cnt
        # GQA group size: how many Q heads share one KV head
        gqa_group_size = h // n_kv

        # Project to get Q, K, V
        Q = self.Q_proj(X, self.Wq)  # [b, s, q_heads_per_dev * d_h]
        K = self.K_proj(X, self.Wk)  # [b, s, kv_heads_per_dev * d_h]
        V = self.V_proj(X, self.Wv)  # [b, s, kv_heads_per_dev * d_h]

        # Reshape Q to [b, s, q_heads_per_dev, d_h]
        Q = self.Q_reshape(Q, [b, s, q_heads_per_dev, d_h])
        # Reshape K, V to [b, s, kv_heads_per_dev, d_h]
        K = self.K_reshape(K, [b, s, kv_heads_per_dev, d_h])
        V = self.V_reshape(V, [b, s, kv_heads_per_dev, d_h])

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, q_heads_per_dev, s, d_h]
        assert Q_T.shape == [b, q_heads_per_dev, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, kv_heads_per_dev, d_h, s]
        assert K_T.shape == [b, kv_heads_per_dev, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, kv_heads_per_dev, s, d_h]
        assert V_T.shape == [b, kv_heads_per_dev, s, d_h]

        # GQA attention: Q has q_heads_per_dev heads, K has kv_heads_per_dev heads.
        # In actual compute, K is broadcast/repeated for each Q group.
        # For performance modeling, we model the full Q×K^T compute:
        #   [b, q_heads_per_dev, s, d_h] × [b, q_heads_per_dev, d_h, s]
        # The K tensor is conceptually repeated gqa_group_size times.
        # We model this as a batched matmul with q_heads_per_dev batches.
        K_T_expanded = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        A = self.Q_mul_K(Q_T, K_T_expanded)  # [b, q_heads_per_dev, s, s]
        assert A.shape == [b, q_heads_per_dev, s, s]
        A_prob = self.A_softmax(A)

        V_T_expanded = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        H = self.A_mul_V(A_prob, V_T_expanded)  # [b, q_heads_per_dev, s, d_h]
        assert H.shape == [b, q_heads_per_dev, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  # [b, s, q_heads_per_dev, d_h]
        assert H.shape == [b, s, q_heads_per_dev, d_h]
        H = self.H_reshape(H, [b, s, q_heads_per_dev * d_h])  # [b, s, d // dev_cnt]
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  # [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, ffn / dev_cnt]
        assert H1.shape == [b, s, ffn // dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  # [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        # Set up BatchedMatmul operators with dummy tensors for roofline modeling
        # Q_mul_K: [b, h // dev_cnt, s, d_h] * [b, h // dev_cnt, d_h, s] = [b, h // dev_cnt, s, s]
        # Use dummy dimensions: batch=1, seq_len=512, d_h = d_model // n_heads
        b, s, d_h = 1, 512, self.d_model // self.n_heads
        h_per_dev = self.n_heads // self.device_count
        q_t_dummy = Tensor([b, h_per_dev, s, d_h], self.data_type)
        k_t_dummy = Tensor([b, h_per_dev, d_h, s], self.data_type)
        _ = self.Q_mul_K(q_t_dummy, k_t_dummy)
        
        # A_mul_V: [b, h // dev_cnt, s, s] * [b, h // dev_cnt, s, d_h] = [b, h // dev_cnt, s, d_h]
        a_prob_dummy = Tensor([b, h_per_dev, s, s], self.data_type)
        v_t_dummy = Tensor([b, h_per_dev, s, d_h], self.data_type)
        _ = self.A_mul_V(a_prob_dummy, v_t_dummy)
        
        # Set up Softmax operator with dummy tensor
        _ = self.A_softmax(a_prob_dummy)
        
        # Set up LayerNorm operators with dummy tensors
        # LayerNorm operates on hidden states: [b, s, d_model]
        h0_dummy = Tensor([b, s, self.d_model], self.data_type)
        _ = self.layer_norm0(h0_dummy)
        _ = self.layer_norm1(h0_dummy)
        
        # Set up GeLU operator with dummy tensor
        # GeLU operates on FFN hidden states
        h1_dummy = Tensor([b, h0_dummy.shape[1], 4 * self.d_model // self.device_count], self.data_type)
        _ = self.H_gelu(h1_dummy)

        # GQA: Q, K, V projections may differ in size; simulate each
        q_proj_latency = self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        k_proj_latency = self.K_proj.roofline_model(device) + device.compute_module.overhead.matmul
        v_proj_latency = self.V_proj.roofline_model(device) + device.compute_module.overhead.matmul
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            # Set up AllReduce with dummy tensor for roofline model
            dummy_allreduce_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        # Use actual dimensions from __call__ if available, otherwise default to 1
        b = getattr(self, '_last_batch_size', 1)
        s = getattr(self, '_last_seq_len', 1)
        d = self.d_model
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        h = self.n_heads
        n_kv = self.n_kv_heads
        d_h = d // h
        q_heads_per_dev = h // dev_cnt
        kv_heads_per_dev = n_kv // dev_cnt
        word_sz = self.data_type.word_size
        hbm_bw = device.io_module.bandwidth
        peak_flops = device.compute_module.total_systolic_array_flops

        # Rebuild computational graph with actual dimensions
        X = Tensor([b, s, d], self.data_type)
        _ = self.Q_proj(X, self.Wq)
        _ = self.K_proj(X, self.Wk)
        _ = self.V_proj(X, self.Wv)

        # GQA: Q×K^T uses q_heads_per_dev (not kv_heads_per_dev) because K is broadcast
        Q_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        K_T = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        _ = self.Q_mul_K(Q_T, K_T)

        A_prob = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        V_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        _ = self.A_mul_V(A_prob, V_T)

        H = Tensor([b, s, d // dev_cnt], self.data_type)
        _ = self.H_matmul0(H, self.W0)

        H0 = Tensor([b, s, d], self.data_type)
        _ = self.H_matmul1(H0, self.W1)

        H1 = Tensor([b, s, ffn // dev_cnt], self.data_type)
        _ = self.H_matmul2(H1, self.W2)

        A = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        _ = self.A_softmax(A)

        H0_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm0(H0_norm)

        H1_act = Tensor([b, s, ffn // dev_cnt], self.data_type)
        _ = self.H_activation(H1_act)

        # --- Per-op XGBoost prediction (preferred over per-kernel RF) ---
        perop_model = get_perop_predictor()
        if perop_model is not None and self.use_ml_predictor:
            import numpy as np
            n_tokens = b * s
            # TP-aware: pass sharded dimensions when device_count > 1
            tp_d = d  # attention: each device handles n_heads/dev_cnt heads (but d_model stays same for projection input)
            tp_ffn = self.intermediate_size // dev_cnt if dev_cnt > 1 else self.intermediate_size
            tp_heads = self.n_heads // dev_cnt if dev_cnt > 1 else self.n_heads
            tp_kv = max(1, self.n_kv_heads // dev_cnt) if dev_cnt > 1 else self.n_kv_heads
            lat_sum = 0.0
            for op in ["attn", "ffn", "norm_pre", "norm_post"]:
                features = compute_perop_features(
                    n_tokens, op, tp_d, tp_heads, tp_kv,
                    tp_ffn, 0, 0,
                    batch_size=b, seq_len=s)  # dense: num_experts=0
                pred_us = perop_model.predict(np.array([features]))[0]
                lat_sum += max(0, pred_us)
            # Add allreduce cost for TP > 1 (2 allreduces per layer: after attn + after FFN)
            if dev_cnt > 1:
                ar_bytes = b * s * d * self.data_type.word_size
                ar_lat = calibrated_allreduce_latency(ar_bytes)
                if ar_lat is not None:
                    exposed_ar = ar_lat * 2 * (1.0 - 0.95)  # 2 allreduces, 95% overlap
                    lat_sum += exposed_ar * 1e6  # seconds -> us
            lat_sum = post_prediction_correction(lat_sum, n_tokens, d, getattr(self, 'num_experts', 0), b, s)
            self.latency = lat_sum / 1e6  # us -> seconds
            return self.latency

        # --- Analytical fallback (original path) ---
        # --- GEMM latencies with calibration (Fix 2) ---
        cal = self.gemm_calibration_factors

        q_proj_latency = _calibrate_matmul(self.Q_proj,
            self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        k_proj_latency = _calibrate_matmul(self.K_proj,
            self.K_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        v_proj_latency = _calibrate_matmul(self.V_proj,
            self.V_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency

        # --- Attention (Fix 1): FlashAttention vs legacy 3-op model ---
        if self.use_flash_attention:
            flash_attn_latency = _flash_attention_latency_prefill(
                b, q_heads_per_dev, s, d_h, word_sz, hbm_bw, peak_flops,
                n_kv_heads=kv_heads_per_dev)
            attention_matmul_latency = flash_attn_latency
        else:
            q_mul_k_latency = _calibrate_batched_matmul(self.Q_mul_K,
                self.Q_mul_K.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            a_mul_v_latency = _calibrate_batched_matmul(self.A_mul_V,
                self.A_mul_V.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            attention_matmul_latency = q_mul_k_latency + a_mul_v_latency

        h_matmul0_latency = _calibrate_matmul(self.H_matmul0,
            self.H_matmul0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        h1_matmul1_latency = _calibrate_matmul(self.H_matmul1,
            self.H_matmul1.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        h2_matmul2_latency = _calibrate_matmul(self.H_matmul2,
            self.H_matmul2.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)

        matmul_total_latency = (
            qkv_latency
            + attention_matmul_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # --- Normalization ---
        if self.use_flash_attention:
            # FlashAttention fuses softmax; no separate softmax cost
            softmax_latency = 0.0
        else:
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # --- Activation (Fix 3): SiLU or GeLU ---
        activation_latency = (
            self.H_activation.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # --- Elementwise ops (Fix 4): RoPE, residual adds, memory copies ---
        rope_lat = _rope_latency(b, s, h // dev_cnt, kv_heads_per_dev, d_h, word_sz, hbm_bw, peak_flops)
        residual_lat = 2 * _residual_add_latency(b, s, d, word_sz, hbm_bw)
        copy_lat = _memory_copy_latency(b, s, d, word_sz, hbm_bw)
        elementwise_total = rope_lat + residual_lat + copy_lat

        # allreduce
        if self.device_count > 1:
            dummy_allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + activation_latency
            + elementwise_total
            + allreduce_total_latency
        )
        self.simluate_log = (
            f"qkv={qkv_latency:.6f}, attn={attention_matmul_latency:.6f}, "
            f"o_proj={h_matmul0_latency:.6f}, ffn1={h1_matmul1_latency:.6f}, "
            f"ffn2={h2_matmul2_latency:.6f}, norm={normlization_total_latency:.6f}, "
            f"act={activation_latency:.6f}, rope={rope_lat:.6f}, "
            f"residual={residual_lat:.6f}, copy={copy_lat:.6f}, "
            f"allreduce={allreduce_total_latency:.6f}"
        )
        return self.latency

    def run_on_gpu(self):
        # matmul
        # GQA: Q, K, V projections may differ in size
        qkv_latency = (
            self.Q_proj.run_on_gpu()
            + self.K_proj.run_on_gpu()
            + self.V_proj.run_on_gpu()
        )
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class TransformerBlockAutoRegressionTP(Operator):
    """
    Autoregressive generation — processes one token at a time using cached K/V from previous tokens.

    This transformer block is used during token-by-token generation/inference. It processes
    a single token at each step and maintains a KV cache of all previous tokens to avoid
    recomputing attention over the entire sequence. This is optimized for sequential generation
    where each new token depends on all previous tokens.

    Supports Grouped Query Attention (GQA) via n_kv_heads parameter:
    - n_kv_heads == n_heads: standard Multi-Head Attention (MHA)
    - n_kv_heads < n_heads: GQA where multiple Q heads share the same KV head
    - n_kv_heads == 1: Multi-Query Attention (MQA)
    """
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, intermediate_size: int = None, n_kv_heads: int = None,
                 use_flash_attention: bool = True, activation_type: str = 'silu', gemm_calibration_factors: dict = None,
                 use_ml_predictor: bool = False, use_hierarchical_allreduce: bool = False,
                 use_cuda_graph: bool = False):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.device_count = device_count
        self.intermediate_size = intermediate_size or 4 * d_model
        self.use_flash_attention = use_flash_attention
        self.activation_type = activation_type
        self.gemm_calibration_factors = gemm_calibration_factors if gemm_calibration_factors is not None else dict(DEFAULT_GEMM_CALIBRATION_FACTORS)
        self.use_ml_predictor = use_ml_predictor
        self.use_hierarchical_allreduce = use_hierarchical_allreduce
        self.use_cuda_graph = use_cuda_graph
        # parameters per device
        d = d_model
        d_h = d // n_heads
        ffn = self.intermediate_size
        self.Wq = Tensor([d, d // device_count], data_type)
        # GQA: K and V projection sizes based on n_kv_heads
        kv_dim_per_device = (self.n_kv_heads * d_h) // device_count
        self.Wk = Tensor([d, kv_dim_per_device], data_type)
        self.Wv = Tensor([d, kv_dim_per_device], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, ffn // device_count], data_type)
        self.W2 = Tensor([ffn // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_activation = SiLU(data_type) if activation_type == 'silu' else GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # Backward-compatible alias
        self.H_gelu = self.H_activation

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length (KV cache length)
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        # Store for compile_and_simulate to use actual dimensions
        self._last_batch_size = b
        self._last_seq_len = s
        h = self.n_heads
        n_kv = self.n_kv_heads
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        d_h = d // h

        q_heads_per_dev = h // dev_cnt
        kv_heads_per_dev = n_kv // dev_cnt

        # KV cache uses kv_heads_per_dev (GQA: fewer KV heads)
        K_cache = Tensor([b, kv_heads_per_dev, d_h, s], self.data_type)
        V_cache = Tensor([b, kv_heads_per_dev, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, q_heads_per_dev * d_h]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, kv_heads_per_dev * d_h]
        v = self.V_proj(x, self.Wv)  # [b, 1, kv_heads_per_dev * d_h]
        q = self.Q_reshape(q, [b, 1, q_heads_per_dev, d_h])
        k = self.K_reshape(k, [b, 1, kv_heads_per_dev, d_h])
        v = self.V_reshape(v, [b, 1, kv_heads_per_dev, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, q_heads_per_dev, 1, d_h]
        assert q_T.shape == [b, q_heads_per_dev, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, kv_heads_per_dev, d_h, 1]
        assert k_T.shape == [b, kv_heads_per_dev, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, kv_heads_per_dev, 1, d_h]
        assert v_T.shape == [b, kv_heads_per_dev, 1, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, kv_heads_per_dev, d_h, s+1]
        assert K_T.shape == [b, kv_heads_per_dev, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, kv_heads_per_dev, s+1, d_h]
        assert V_T.shape == [b, kv_heads_per_dev, s + 1, d_h]

        # GQA: Q has q_heads_per_dev heads, K has kv_heads_per_dev heads
        # K is broadcast/repeated for each Q group for the attention computation
        K_T_expanded = Tensor([b, q_heads_per_dev, d_h, s + 1], self.data_type)
        a = self.Q_mul_K(q_T, K_T_expanded)  # [b, q_heads_per_dev, 1, s+1]
        assert a.shape == [b, q_heads_per_dev, 1, s + 1]
        a_prob = self.A_softmax(a)
        V_T_expanded = Tensor([b, q_heads_per_dev, s + 1, d_h], self.data_type)
        h0 = self.A_mul_V(a_prob, V_T_expanded)  # [b, q_heads_per_dev, 1, d_h]
        assert h0.shape == [b, q_heads_per_dev, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  # [b, 1, q_heads_per_dev, d_h]
        assert h0.shape == [b, 1, q_heads_per_dev, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  # [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, ffn / dev_cnt]
        assert h1.shape == [b, 1, ffn // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  # [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        # Set up BatchedMatmul operators with dummy tensors for roofline modeling
        # GQA: Q uses q_heads_per_dev, attention uses q_heads_per_dev (K broadcast)
        b, s, d_h = 1, 512, self.d_model // self.n_heads
        q_heads_per_dev = self.n_heads // self.device_count
        q_t_dummy = Tensor([b, q_heads_per_dev, 1, d_h], self.data_type)
        k_t_dummy = Tensor([b, q_heads_per_dev, d_h, s + 1], self.data_type)
        _ = self.Q_mul_K(q_t_dummy, k_t_dummy)

        a_prob_dummy = Tensor([b, q_heads_per_dev, 1, s + 1], self.data_type)
        v_t_dummy = Tensor([b, q_heads_per_dev, s + 1, d_h], self.data_type)
        _ = self.A_mul_V(a_prob_dummy, v_t_dummy)

        _ = self.A_softmax(a_prob_dummy)

        h0_dummy = Tensor([b, 1, self.d_model], self.data_type)
        _ = self.layer_norm0(h0_dummy)
        _ = self.layer_norm1(h0_dummy)

        h1_dummy = Tensor([b, 1, 4 * self.d_model // self.device_count], self.data_type)
        _ = self.H_gelu(h1_dummy)

        # GQA: Q, K, V projections may differ in size; simulate each
        q_proj_latency = self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        k_proj_latency = self.K_proj.roofline_model(device) + device.compute_module.overhead.matmul
        v_proj_latency = self.V_proj.roofline_model(device) + device.compute_module.overhead.matmul
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            # Set up AllReduce with dummy tensor for roofline model
            dummy_allreduce_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        pcb = system.device
        interconnect = system.interconnect

        b = getattr(self, '_last_batch_size', 1)
        s = getattr(self, '_last_seq_len', 1)
        d = self.d_model
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        h = self.n_heads
        n_kv = self.n_kv_heads
        d_h = d // h
        q_heads_per_dev = h // dev_cnt
        kv_heads_per_dev = n_kv // dev_cnt
        word_sz = self.data_type.word_size
        hbm_bw = pcb.io_module.bandwidth
        peak_flops = pcb.compute_module.total_systolic_array_flops
        cal = self.gemm_calibration_factors

        # --- Per-op XGBoost prediction (preferred over per-kernel RF) ---
        perop_model = get_perop_predictor()
        if perop_model is not None and self.use_ml_predictor:
            import numpy as np
            n_tokens = b  # decode: single token per request
            # TP-aware: shard heads and FFN across devices
            tp_heads = self.n_heads // dev_cnt if dev_cnt > 1 else self.n_heads
            tp_kv = max(1, self.n_kv_heads // dev_cnt) if dev_cnt > 1 else self.n_kv_heads
            tp_ffn = self.intermediate_size // dev_cnt if dev_cnt > 1 else self.intermediate_size
            lat_sum = 0.0
            for op in ["attn", "ffn", "norm_pre", "norm_post"]:
                features = compute_perop_features(
                    n_tokens, op, d, tp_heads, tp_kv,
                    tp_ffn, 0, 0,
                    batch_size=b, seq_len=1, kv_cache_len=s)  # decode: seq=1, s=kv_cache
                pred_us = perop_model.predict(np.array([features]))[0]
                lat_sum += max(0, pred_us)
            # Add allreduce for TP > 1
            if dev_cnt > 1:
                ar_bytes = b * 1 * d * self.data_type.word_size  # decode: seq=1
                ar_lat = calibrated_allreduce_latency(ar_bytes)
                if ar_lat is not None:
                    exposed_ar = ar_lat * 2 * (1.0 - 0.95)
                    lat_sum += exposed_ar * 1e6
            lat_sum = post_prediction_correction(lat_sum, n_tokens, d, getattr(self, 'num_experts', 0), b, s)
            self.latency = lat_sum / 1e6  # us -> seconds
            return self.latency

        # --- Analytical fallback (original path) ---
        # --- GEMM latencies with calibration (Fix 2) ---
        q_proj_latency = _calibrate_matmul(self.Q_proj,
            self.Q_proj.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
        k_proj_latency = _calibrate_matmul(self.K_proj,
            self.K_proj.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
        v_proj_latency = _calibrate_matmul(self.V_proj,
            self.V_proj.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency

        # --- Attention (Fix 1): FlashAttention decode model ---
        if self.use_flash_attention:
            # Decode: single query token attending to full KV cache (s tokens)
            flash_attn_latency = _flash_attention_latency_decode(
                b, kv_heads_per_dev, s + 1, d_h, word_sz, hbm_bw, peak_flops,
                n_q_heads=q_heads_per_dev)
            attention_matmul_latency = flash_attn_latency
        else:
            q_mul_k_latency = _calibrate_batched_matmul(self.Q_mul_K,
                self.Q_mul_K.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
            a_mul_v_latency = _calibrate_batched_matmul(self.A_mul_V,
                self.A_mul_V.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
            attention_matmul_latency = q_mul_k_latency + a_mul_v_latency

        h_matmul0_latency = _calibrate_matmul(self.H_matmul0,
            self.H_matmul0.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
        h1_matmul1_latency = _calibrate_matmul(self.H_matmul1,
            self.H_matmul1.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)
        h2_matmul2_latency = _calibrate_matmul(self.H_matmul2,
            self.H_matmul2.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul, cal)

        matmul_total_latency = (
            qkv_latency
            + attention_matmul_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # --- Normalization ---
        if self.use_flash_attention:
            softmax_latency = 0.0
        else:
            softmax_latency = (
                self.A_softmax.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.layernorm
        )
        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # --- Activation (Fix 3) ---
        activation_latency = (
            self.H_activation.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.gelu
        )

        # --- Elementwise ops (Fix 4): RoPE, residual adds, memory copies ---
        # For decode, seq_len=1 (single token)
        rope_lat = _rope_latency(b, 1, h // dev_cnt, kv_heads_per_dev, d_h, word_sz, hbm_bw, peak_flops)
        residual_lat = 2 * _residual_add_latency(b, 1, d, word_sz, hbm_bw)
        copy_lat = _memory_copy_latency(b, 1, d, word_sz, hbm_bw)
        elementwise_total = rope_lat + residual_lat + copy_lat

        # allreduce
        if self.device_count > 1:
            dummy_allreduce_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + activation_latency
            + elementwise_total
            + allreduce_total_latency
        )
        self.simluate_log = (
            f"qkv={qkv_latency:.6f}, attn={attention_matmul_latency:.6f}, "
            f"o_proj={h_matmul0_latency:.6f}, ffn1={h1_matmul1_latency:.6f}, "
            f"ffn2={h2_matmul2_latency:.6f}, norm={normlization_total_latency:.6f}, "
            f"act={activation_latency:.6f}, rope={rope_lat:.6f}, "
            f"residual={residual_lat:.6f}, copy={copy_lat:.6f}, "
            f"allreduce={allreduce_total_latency:.6f}"
        )
        return self.latency

    def run_on_gpu(self):
        # matmul
        # GQA: Q, K, V projections may differ in size
        qkv_latency = (
            self.Q_proj.run_on_gpu()
            + self.K_proj.run_on_gpu()
            + self.V_proj.run_on_gpu()
        )
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )
        # gelu_latency = max(gelu_latency, 1e-7)

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class TransformerBlockInitComputationTPDP(Operator):
    """
    Transformer block with Tensor Parallelism (TP) and Data Parallelism (DP) support.
    
    This block supports:
    - TP: Model sharding across TP devices (requires AllReduce within TP group)
    - DP: Different DP groups process different batches (no sync needed for inference)
    - Per-device-pair bandwidth: Uses bandwidth matrix for accurate communication modeling
    
    Architecture:
    - TP groups: Devices that share model shards (need AllReduce synchronization)
    - DP groups: Devices that process different data batches (independent for inference)
    - Combined: Each TP group can have multiple DP replicas
    
    Example:
        TP=4, DP=2 means:
        - 2 DP groups: [0-3], [4-7]  (each processes different batch)
        - 2 TP groups within each DP: [0,1,2,3] and [4,5,6,7] (within group, sync via AllReduce)
        - device_group=[0,1,2,3] for first TP group
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device_group: List[int],  # Device IDs in the TP group (for this transformer block)
        data_type: DataType,
        intermediate_size: int = None,
        n_kv_heads: int = None,
        use_flash_attention: bool = True,
        activation_type: str = 'silu',
        gemm_calibration_factors: dict = None,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.device_group = device_group  # List of device IDs in this TP group
        self.device_count = len(device_group)  # TP size
        self.data_type = data_type
        self.intermediate_size = intermediate_size or 4 * d_model
        self.use_flash_attention = use_flash_attention
        self.activation_type = activation_type
        self.gemm_calibration_factors = gemm_calibration_factors if gemm_calibration_factors is not None else dict(DEFAULT_GEMM_CALIBRATION_FACTORS)

        # Validate device count
        if self.device_count < 1:
            raise ValueError("device_group must contain at least one device")
        if self.n_heads % self.device_count != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by device_count ({self.device_count})")

        # parameters per device
        d = d_model
        d_h = d // n_heads
        ffn = self.intermediate_size
        self.Wq = Tensor([d, d // self.device_count], data_type)
        kv_dim_per_device = (self.n_kv_heads * d_h) // self.device_count
        self.Wk = Tensor([d, kv_dim_per_device], data_type)
        self.Wv = Tensor([d, kv_dim_per_device], data_type)
        self.W0 = Tensor([d // self.device_count, d], data_type)
        self.W1 = Tensor([d, ffn // self.device_count], data_type)
        self.W2 = Tensor([ffn // self.device_count, d], data_type)

        # operators per device (same as TP version)
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))

        # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_activation = SiLU(data_type) if activation_type == 'silu' else GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # Backward-compatible alias
        self.H_gelu = self.H_activation
    
    def __call__(self, X: Tensor) -> Tensor:
        """Forward pass - same as TP version but uses device_group for communication.
        Supports GQA via n_kv_heads parameter."""
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        n_kv = self.n_kv_heads
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        d_h = d // h

        q_heads_per_dev = h // dev_cnt
        kv_heads_per_dev = n_kv // dev_cnt

        # multi-head attention
        Q = self.Q_proj(X, self.Wq)  # [b, s, q_heads_per_dev * d_h]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, kv_heads_per_dev * d_h]
        V = self.V_proj(X, self.Wv)  # [b, s, kv_heads_per_dev * d_h]
        Q = self.Q_reshape(Q, [b, s, q_heads_per_dev, d_h])
        K = self.K_reshape(K, [b, s, kv_heads_per_dev, d_h])
        V = self.V_reshape(V, [b, s, kv_heads_per_dev, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, q_heads_per_dev, s, d_h]
        assert Q_T.shape == [b, q_heads_per_dev, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, kv_heads_per_dev, d_h, s]
        assert K_T.shape == [b, kv_heads_per_dev, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, kv_heads_per_dev, s, d_h]
        assert V_T.shape == [b, kv_heads_per_dev, s, d_h]

        # GQA: K is broadcast for each Q group
        K_T_expanded = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        A = self.Q_mul_K(Q_T, K_T_expanded)  # [b, q_heads_per_dev, s, s]
        assert A.shape == [b, q_heads_per_dev, s, s]
        A_prob = self.A_softmax(A)
        V_T_expanded = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        H = self.A_mul_V(A_prob, V_T_expanded)  # [b, q_heads_per_dev, s, d_h]
        assert H.shape == [b, q_heads_per_dev, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  # [b, s, q_heads_per_dev, d_h]
        assert H.shape == [b, s, q_heads_per_dev, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, ffn / dev_cnt]
        assert H1.shape == [b, s, ffn // dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2
    
    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU", batch_size: int = None, seq_len: int = None) -> float:
        """
        Simulate the transformer block with TP+DP support.
        
        Uses device_group to determine which devices participate in AllReduce,
        allowing per-device-pair bandwidth to be used.
        
        Args:
            system: System configuration with device and interconnect
            compile_mode: Compilation mode for operators
            batch_size: Optional batch size (uses computational graph if already built, otherwise uses batch_size or 1)
            seq_len: Optional sequence length (uses computational graph if already built, otherwise uses seq_len or 1)
        """
        device = system.device
        interconnect = system.interconnect

        # Determine batch size and sequence length
        # Priority: 1) Use provided parameters, 2) Use dimensions from existing computational graph, 3) Default to 1
        b = batch_size
        s = seq_len
        
        # If not provided, try to get from existing computational graph
        if b is None or s is None:
            if hasattr(self.Q_proj, 'computational_graph') and self.Q_proj.computational_graph is not None:
                if hasattr(self.Q_proj.computational_graph, 'input1_shape'):
                    shape = self.Q_proj.computational_graph.input1_shape
                    if len(shape) >= 2:
                        b = b or shape[0]
                        s = s or shape[1]
        
        # Final defaults
        b = b or 1
        s = s or 1
        
        d = self.d_model
        dev_cnt = self.device_count
        ffn = self.intermediate_size
        
        # Build computational graph with determined dimensions
        # Note: This will rebuild if called again, but ensures correct dimensions are used
        dummy_X = Tensor([b, s, d], self.data_type)
        
        # Call matmul operators with tensors to build computational graph
        _ = self.Q_proj(dummy_X, self.Wq)
        _ = self.K_proj(dummy_X, self.Wk)
        _ = self.V_proj(dummy_X, self.Wv)
        
        # For batched matmuls, need to set up properly shaped tensors
        h = self.n_heads
        d_h = d // h
        q_heads_per_dev = h // dev_cnt
        ffn = self.intermediate_size
        # GQA: attention uses q_heads_per_dev (K broadcast)
        dummy_Q_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        dummy_K_T = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        _ = self.Q_mul_K(dummy_Q_T, dummy_K_T)

        dummy_A_prob = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        dummy_V_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        _ = self.A_mul_V(dummy_A_prob, dummy_V_T)

        dummy_H = Tensor([b, s, d // dev_cnt], self.data_type)
        _ = self.H_matmul0(dummy_H, self.W0)

        dummy_H0 = Tensor([b, s, d], self.data_type)
        _ = self.H_matmul1(dummy_H0, self.W1)

        dummy_H1 = Tensor([b, s, ffn // dev_cnt], self.data_type)
        _ = self.H_matmul2(dummy_H1, self.W2)

        # Set up dummy tensors for Softmax, LayerNorm, and GeLU
        dummy_A = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        _ = self.A_softmax(dummy_A)
        
        dummy_H0_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm0(dummy_H0_norm)
        
        dummy_H1_gelu = Tensor([b, s, ffn // dev_cnt], self.data_type)
        _ = self.H_gelu(dummy_H1_gelu)
        
        # --- GEMM latencies with calibration (Fix 2) ---
        n_kv = self.n_kv_heads
        kv_heads_per_dev = n_kv // dev_cnt
        word_sz = self.data_type.word_size
        hbm_bw = device.io_module.bandwidth
        peak_flops = device.compute_module.total_systolic_array_flops
        cal = self.gemm_calibration_factors

        q_proj_latency = _calibrate_matmul(self.Q_proj,
            self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        k_proj_latency = _calibrate_matmul(self.K_proj,
            self.K_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        v_proj_latency = _calibrate_matmul(self.V_proj,
            self.V_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency

        # --- Attention (Fix 1): FlashAttention vs legacy 3-op model ---
        if self.use_flash_attention:
            flash_attn_latency = _flash_attention_latency_prefill(
                b, q_heads_per_dev, s, d_h, word_sz, hbm_bw, peak_flops,
                n_kv_heads=kv_heads_per_dev)
            attention_matmul_latency = flash_attn_latency
        else:
            q_mul_k_latency = _calibrate_batched_matmul(self.Q_mul_K,
                self.Q_mul_K.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            a_mul_v_latency = _calibrate_batched_matmul(self.A_mul_V,
                self.A_mul_V.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            attention_matmul_latency = q_mul_k_latency + a_mul_v_latency

        h_matmul0_latency = _calibrate_matmul(self.H_matmul0,
            self.H_matmul0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        h1_matmul1_latency = _calibrate_matmul(self.H_matmul1,
            self.H_matmul1.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        h2_matmul2_latency = _calibrate_matmul(self.H_matmul2,
            self.H_matmul2.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)

        matmul_total_latency = (
            qkv_latency
            + attention_matmul_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # --- Normalization ---
        if self.use_flash_attention:
            softmax_latency = 0.0
        else:
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # --- Activation (Fix 3) ---
        activation_latency = (
            self.H_activation.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # --- Elementwise ops (Fix 4) ---
        rope_lat = _rope_latency(b, s, h // dev_cnt, kv_heads_per_dev, d_h, word_sz, hbm_bw, peak_flops)
        residual_lat = 2 * _residual_add_latency(b, s, d, word_sz, hbm_bw)
        copy_lat = _memory_copy_latency(b, s, d, word_sz, hbm_bw)
        elementwise_total = rope_lat + residual_lat + copy_lat

        # allreduce with device_group support
        allreduce_total_latency = 0.0
        if self.device_count > 1:
            allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_mha(allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect, device_group=self.device_group)
            allreduce_total_latency = allreduce_latency * 2

        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + activation_latency
            + elementwise_total
            + allreduce_total_latency
        )
        return self.latency
    
    def roofline_model(self, system: System):
        """Roofline model - same as TP version but uses device_group"""
        device = system.device
        interconnect = system.interconnect

        # Set up BatchedMatmul operators with dummy tensors for roofline modeling
        # GQA: attention uses q_heads_per_dev (K broadcast)
        b, s, d_h = 1, 512, self.d_model // self.n_heads
        q_heads_per_dev = self.n_heads // self.device_count
        q_t_dummy = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        k_t_dummy = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        _ = self.Q_mul_K(q_t_dummy, k_t_dummy)

        a_prob_dummy = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        v_t_dummy = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        _ = self.A_mul_V(a_prob_dummy, v_t_dummy)

        _ = self.A_softmax(a_prob_dummy)

        h0_dummy = Tensor([b, s, self.d_model], self.data_type)
        _ = self.layer_norm0(h0_dummy)
        _ = self.layer_norm1(h0_dummy)
        
        # Set up GeLU operator with dummy tensor
        # GeLU operates on FFN hidden states
        h1_dummy = Tensor([b, h0_dummy.shape[1], 4 * self.d_model // self.device_count], self.data_type)
        _ = self.H_gelu(h1_dummy)

        # GQA: Q, K, V projections may differ in size; simulate each
        q_proj_latency = self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        k_proj_latency = self.K_proj.roofline_model(device) + device.compute_module.overhead.matmul
        v_proj_latency = self.V_proj.roofline_model(device) + device.compute_module.overhead.matmul
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce (conservative estimate for roofline)
        if self.device_count > 1:
            dummy_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_mha(dummy_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect, device_group=self.device_group)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_total_latency = 0

        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.roofline_latency


class TransformerBlockMoETP(Operator):
    """
    Transformer block with Mixture of Experts (MoE) FFN, using Tensor Parallelism.

    Instead of a single dense FFN, this block uses a gating network to route each token
    to a subset (top_k) of num_experts smaller expert FFNs. This models architectures
    like Mixtral, DeepSeek-V2, and Qwen-MoE.

    The attention portion is identical to TransformerBlockInitComputationTP (with GQA support).
    The FFN portion replaces the dense FFN with:
    1. Gate/router: [b*s, d_model] x [d_model, num_experts] -> top_k expert selection
    2. Expert dispatch: permute tokens to selected experts (memory-bound)
    3. Per-expert FFN: top_k smaller expert FFNs, each processing b*s*top_k/num_experts tokens
    4. Expert combine: reduce outputs back (memory-bound)

    Args:
        d_model: Hidden dimension
        n_heads: Number of Q attention heads
        device_count: Number of TP devices
        data_type: Data type for tensors
        intermediate_size: FFN intermediate size (per-expert)
        n_kv_heads: Number of KV attention heads (default: n_heads for MHA)
        num_experts: Total number of experts
        top_k: Number of experts activated per token
        expert_intermediate_size: Per-expert FFN intermediate size (default: intermediate_size)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device_count: int,
        data_type: DataType,
        intermediate_size: int = None,
        n_kv_heads: int = None,
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: int = None,
        use_flash_attention: bool = True,
        activation_type: str = 'silu',
        gemm_calibration_factors: dict = None,
        use_ml_predictor: bool = False,
        use_cuda_graph: bool = False,
        use_hierarchical_allreduce: bool = False,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.device_count = device_count
        self.intermediate_size = intermediate_size or 4 * d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_intermediate_size = expert_intermediate_size or self.intermediate_size
        self.use_flash_attention = use_flash_attention
        self.activation_type = activation_type
        self.gemm_calibration_factors = gemm_calibration_factors if gemm_calibration_factors is not None else dict(DEFAULT_GEMM_CALIBRATION_FACTORS)
        self.use_ml_predictor = use_ml_predictor
        self.use_cuda_graph = use_cuda_graph
        self.use_hierarchical_allreduce = use_hierarchical_allreduce

        d = d_model
        d_h = d // n_heads
        ffn_expert = self.expert_intermediate_size

        # Attention weights (same as dense transformer with GQA)
        self.Wq = Tensor([d, d // device_count], data_type)
        kv_dim_per_device = (self.n_kv_heads * d_h) // device_count
        self.Wk = Tensor([d, kv_dim_per_device], data_type)
        self.Wv = Tensor([d, kv_dim_per_device], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)

        # Gate/router weight
        self.W_gate = Tensor([d, num_experts], data_type)

        # Per-expert FFN weights (sharded across TP devices)
        # Each expert has gate_proj, up_proj, down_proj
        self.W_expert_gate = Tensor([d, ffn_expert // device_count], data_type)
        self.W_expert_up = Tensor([d, ffn_expert // device_count], data_type)
        self.W_expert_down = Tensor([ffn_expert // device_count, d], data_type)

        # Attention operators
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))

        # MoE operators
        self.gate_proj = Matmul(data_type)  # Router
        self.expert_gate_proj = Matmul(data_type)  # Expert gate_proj (SwiGLU)
        self.expert_up_proj = Matmul(data_type)  # Expert up_proj (SwiGLU)
        self.expert_activation = SiLU(data_type) if activation_type == 'silu' else GeLU(data_type)
        self.expert_down_proj = Matmul(data_type)  # Expert down_proj
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = _make_allreduce(data_type, getattr(self, 'use_hierarchical_allreduce', False))
        # Backward-compatible aliases
        self.expert_gelu = self.expert_activation

    def __call__(self, X: Tensor) -> Tensor:
        b, s, d = X.shape
        self._last_batch_size = b
        self._last_seq_len = s
        assert d == self.d_model

        h = self.n_heads
        n_kv = self.n_kv_heads
        dev_cnt = self.device_count
        d_h = d // h
        q_heads_per_dev = h // dev_cnt
        kv_heads_per_dev = n_kv // dev_cnt

        # === Attention (same as dense with GQA) ===
        Q = self.Q_proj(X, self.Wq)
        K = self.K_proj(X, self.Wk)
        V = self.V_proj(X, self.Wv)

        Q = self.Q_reshape(Q, [b, s, q_heads_per_dev, d_h])
        K = self.K_reshape(K, [b, s, kv_heads_per_dev, d_h])
        V = self.V_reshape(V, [b, s, kv_heads_per_dev, d_h])

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        K_T = self.K_transpose(K, [0, 2, 3, 1])
        V_T = self.V_transpose(V, [0, 2, 1, 3])

        K_T_expanded = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        A = self.Q_mul_K(Q_T, K_T_expanded)
        A_prob = self.A_softmax(A)
        V_T_expanded = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        H = self.A_mul_V(A_prob, V_T_expanded)
        H = self.H_transpose(H, [0, 2, 1, 3])
        H = self.H_reshape(H, [b, s, q_heads_per_dev * d_h])
        H0 = self.H_matmul0(H, self.W0)
        H0 = self.layer_norm0(H0)
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # === MoE FFN ===
        # Gate/router: [b*s, d_model] x [d_model, num_experts]
        X_flat = Tensor([b * s, d], self.data_type)
        gate_out = self.gate_proj(X_flat, self.W_gate)  # [b*s, num_experts]
        assert gate_out.shape == [b * s, self.num_experts]

        # Per-expert FFN: each expert processes b*s*top_k/num_experts tokens
        # Total compute = top_k / num_experts * dense FFN compute
        tokens_per_expert = max(1, (b * s * self.top_k) // self.num_experts)
        expert_input = Tensor([tokens_per_expert, d], self.data_type)

        # Expert gate_proj and up_proj (SwiGLU pattern)
        expert_gate_out = self.expert_gate_proj(expert_input, self.W_expert_gate)
        expert_up_out = self.expert_up_proj(expert_input, self.W_expert_up)
        expert_act = self.expert_gelu(expert_gate_out)

        # Expert down_proj
        expert_hidden = Tensor([tokens_per_expert, self.expert_intermediate_size // dev_cnt], self.data_type)
        expert_out = self.expert_down_proj(expert_hidden, self.W_expert_down)
        assert expert_out.shape == [tokens_per_expert, d]

        H2 = Tensor([b, s, d], self.data_type)
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        return H2

    def compile_and_simulate(self, system: System, compile_mode: str):
        """Simulate the MoE transformer block latency."""
        device = system.device
        interconnect = system.interconnect

        b = getattr(self, '_last_batch_size', 1)
        s = getattr(self, '_last_seq_len', 1)
        d = self.d_model
        dev_cnt = self.device_count
        h = self.n_heads
        d_h = d // h
        q_heads_per_dev = h // dev_cnt

        # Rebuild computational graph
        X = Tensor([b, s, d], self.data_type)
        _ = self.Q_proj(X, self.Wq)
        _ = self.K_proj(X, self.Wk)
        _ = self.V_proj(X, self.Wv)

        Q_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        K_T = Tensor([b, q_heads_per_dev, d_h, s], self.data_type)
        _ = self.Q_mul_K(Q_T, K_T)

        A_prob = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        V_T = Tensor([b, q_heads_per_dev, s, d_h], self.data_type)
        _ = self.A_mul_V(A_prob, V_T)

        H = Tensor([b, s, d // dev_cnt], self.data_type)
        _ = self.H_matmul0(H, self.W0)

        A = Tensor([b, q_heads_per_dev, s, s], self.data_type)
        _ = self.A_softmax(A)

        H0_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm0(H0_norm)

        # Gate/router
        X_flat = Tensor([b * s, d], self.data_type)
        _ = self.gate_proj(X_flat, self.W_gate)

        # Expert FFN
        tokens_per_expert = max(1, (b * s * self.top_k) // self.num_experts)
        expert_input = Tensor([tokens_per_expert, d], self.data_type)
        _ = self.expert_gate_proj(expert_input, self.W_expert_gate)
        _ = self.expert_up_proj(expert_input, self.W_expert_up)

        expert_hidden = Tensor([tokens_per_expert, self.expert_intermediate_size // dev_cnt], self.data_type)
        _ = self.expert_down_proj(expert_hidden, self.W_expert_down)

        expert_gelu_input = Tensor([tokens_per_expert, self.expert_intermediate_size // dev_cnt], self.data_type)
        _ = self.expert_gelu(expert_gelu_input)

        H1_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm1(H1_norm)

        # === Simulate latencies ===
        n_kv = self.n_kv_heads
        kv_heads_per_dev = n_kv // dev_cnt
        word_sz = self.data_type.word_size
        hbm_bw = device.io_module.bandwidth
        peak_flops = device.compute_module.total_systolic_array_flops
        cal = self.gemm_calibration_factors
        expert_ffn = self.expert_intermediate_size

        # --- Per-op XGBoost prediction (preferred over per-kernel RF) ---
        perop_model = get_perop_predictor()
        if perop_model is not None and self.use_ml_predictor:
            import numpy as np
            n_tokens = b * s
            # TP-aware: shard heads and FFN across devices
            tp_heads = self.n_heads // dev_cnt if dev_cnt > 1 else self.n_heads
            tp_kv = max(1, self.n_kv_heads // dev_cnt) if dev_cnt > 1 else self.n_kv_heads
            tp_ffn = self.expert_intermediate_size // dev_cnt if dev_cnt > 1 else self.expert_intermediate_size
            lat_sum = 0.0
            for op in ["attn", "ffn", "norm_pre", "norm_post"]:
                features = compute_perop_features(
                    n_tokens, op, d, tp_heads, tp_kv,
                    tp_ffn, self.num_experts, self.top_k,
                    batch_size=b, seq_len=s)
                pred_us = perop_model.predict(np.array([features]))[0]
                lat_sum += max(0, pred_us)
            # Add allreduce for TP > 1
            if dev_cnt > 1:
                ar_bytes = b * s * d * self.data_type.word_size
                ar_lat = calibrated_allreduce_latency(ar_bytes)
                if ar_lat is not None:
                    exposed_ar = ar_lat * 2 * (1.0 - 0.95)
                    lat_sum += exposed_ar * 1e6
            lat_sum = post_prediction_correction(lat_sum, n_tokens, d, getattr(self, 'num_experts', 0), b, s)
            self.latency = lat_sum / 1e6  # us -> seconds
            return self.latency

        # --- Analytical fallback ---

        # --- GEMM with calibration (Fix 2) ---
        q_proj_latency = _calibrate_matmul(self.Q_proj,
            self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        k_proj_latency = _calibrate_matmul(self.K_proj,
            self.K_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        v_proj_latency = _calibrate_matmul(self.V_proj,
            self.V_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        qkv_latency = q_proj_latency + k_proj_latency + v_proj_latency

        # --- Attention (Fix 1) ---
        if self.use_flash_attention:
            flash_attn_latency = _flash_attention_latency_prefill(
                b, q_heads_per_dev, s, d_h, word_sz, hbm_bw, peak_flops,
                n_kv_heads=kv_heads_per_dev)
            attention_matmul_latency = flash_attn_latency
        else:
            q_mul_k_latency = _calibrate_batched_matmul(self.Q_mul_K,
                self.Q_mul_K.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            a_mul_v_latency = _calibrate_batched_matmul(self.A_mul_V,
                self.A_mul_V.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
            attention_matmul_latency = q_mul_k_latency + a_mul_v_latency

        h_matmul0_latency = _calibrate_matmul(self.H_matmul0,
            self.H_matmul0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)

        attention_latency = qkv_latency + attention_matmul_latency + h_matmul0_latency

        # --- Normalization ---
        if self.use_flash_attention:
            softmax_latency = 0.0
        else:
            softmax_latency = self.A_softmax.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.softmax
        layernorm0_latency = self.layer_norm0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.layernorm

        # MoE FFN
        gate_latency = _calibrate_matmul(self.gate_proj,
            self.gate_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)

        # Expert dispatch: memory-bound data movement
        dispatch_data_bytes = tokens_per_expert * d * self.data_type.word_size
        dispatch_latency = dispatch_data_bytes / device.io_module.bandwidth

        # Expert FFN with calibration (Fix 2) and activation fix (Fix 3)
        expert_gate_latency = _calibrate_matmul(self.expert_gate_proj,
            self.expert_gate_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        expert_up_latency = _calibrate_matmul(self.expert_up_proj,
            self.expert_up_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        expert_act_latency = self.expert_activation.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.gelu
        expert_down_latency = _calibrate_matmul(self.expert_down_proj,
            self.expert_down_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul, cal)
        # Expert count: decode (few tokens) = top_k, prefill (many tokens) = all
        is_decode_a = (s == 1)
        active_experts_a = self.top_k if is_decode_a else min(b * s * self.top_k, self.num_experts)
        single_expert_ffn = expert_gate_latency + expert_up_latency + expert_act_latency + expert_down_latency
        expert_ffn_latency = single_expert_ffn * active_experts_a
        tpe_a = max(1, (b * s * self.top_k) // self.num_experts)
        bmm_overhead_a = max(1.5, min(0.017 * (self.num_experts / 32.0) ** 2 * tpe_a + 1.3, 12.0))
        expert_ffn_latency *= bmm_overhead_a

        # MoE routing overhead (analytical)
        n_tok_a = b * s
        E_a = self.num_experts
        k_a = self.top_k
        ws = self.data_type.word_size
        bw = device.io_module.bandwidth
        router_lat_a = max(2*n_tok_a*E_a*ws*3 / bw, n_tok_a*E_a*10 / peak_flops)
        scatter_lat_a = 2*n_tok_a*k_a*d*ws / bw
        gather_lat_a = max(2*n_tok_a*k_a*d*ws / bw, n_tok_a*k_a*d*2 / peak_flops)
        cat_lat_a = 2*n_tok_a*k_a*d*ws / bw
        combine_latency = router_lat_a + scatter_lat_a + gather_lat_a + cat_lat_a

        layernorm1_latency = self.layer_norm1.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.layernorm

        moe_ffn_latency = gate_latency + dispatch_latency + expert_ffn_latency + combine_latency

        # --- Elementwise ops (Fix 4) ---
        rope_lat = _rope_latency(b, s, h // dev_cnt, kv_heads_per_dev, d_h, word_sz, hbm_bw, peak_flops)
        residual_lat = 2 * _residual_add_latency(b, s, d, word_sz, hbm_bw)
        copy_lat = _memory_copy_latency(b, s, d, word_sz, hbm_bw)
        elementwise_total = rope_lat + residual_lat + copy_lat

        # AllReduce
        allreduce_total_latency = 0
        if self.device_count > 1:
            dummy_allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2

        self.latency = (
            attention_latency
            + softmax_latency
            + layernorm0_latency + layernorm1_latency
            + moe_ffn_latency
            + elementwise_total
            + allreduce_total_latency
        )
        return self.latency


class TransformerBlockMLATP(Operator):
    """
    Transformer block with Multi-head Latent Attention (MLA).
    Used by DeepSeek-V2/V3 and GLM-5.

    MLA compresses KV into a low-rank latent vector before caching,
    reducing KV cache by 5-10x vs GQA while maintaining MHA quality.

    Forward pass per layer:
        C = X @ W_down_kv          # [b, s, d_latent]  <- stored in KV cache
        K = C @ W_up_k             # [b, s, n_heads * d_h]
        V = C @ W_up_v             # [b, s, n_heads * d_h]
        Q = X @ W_q  (or two-stage if q_lora_rank > 0)
        O = FlashAttn(Q, K, V) @ W_o
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        device_count: int,
        data_type: DataType,
        head_dim: int = 128,
        q_lora_rank: int = 0,
        qk_rope_head_dim: int = 64,
        intermediate_size: int = None,
        n_experts: int = 1,
        n_experts_active: int = 1,
        use_ml_predictor: bool = False,
        use_cuda_graph: bool = False,
        use_hierarchical_allreduce: bool = False,
        gemm_calibration_factors: dict = None,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.device_count = device_count
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.intermediate_size = intermediate_size or 4 * d_model
        self.n_experts = n_experts
        self.n_experts_active = n_experts_active
        self.use_ml_predictor = use_ml_predictor
        self.use_cuda_graph = use_cuda_graph
        self.use_hierarchical_allreduce = use_hierarchical_allreduce
        self.gemm_calibration_factors = gemm_calibration_factors if gemm_calibration_factors is not None else dict(DEFAULT_GEMM_CALIBRATION_FACTORS)

        d = d_model
        d_lat = d_latent
        d_h = head_dim
        heads_per_dev = n_heads // device_count
        kv_up_dim = heads_per_dev * d_h
        ffn = self.intermediate_size

        # KV down-projection: [d_model, d_latent] — small, shared across devices
        self.W_down_kv = Tensor([d, d_lat], data_type)
        self.down_kv_proj = Matmul(data_type)

        # KV up-projections (per device): [d_latent, heads_per_dev * d_h] x 2
        self.W_up_k = Tensor([d_lat, kv_up_dim], data_type)
        self.W_up_v = Tensor([d_lat, kv_up_dim], data_type)
        self.up_k_proj = Matmul(data_type)
        self.up_v_proj = Matmul(data_type)

        # Q projection
        if q_lora_rank > 0:
            self.W_down_q = Tensor([d, q_lora_rank], data_type)
            self.W_up_q = Tensor([q_lora_rank, heads_per_dev * d_h], data_type)
            self.down_q_proj = Matmul(data_type)
            self.up_q_proj = Matmul(data_type)
        else:
            self.W_q = Tensor([d, heads_per_dev * d_h], data_type)
            self.Q_proj = Matmul(data_type)

        # Output projection: [heads_per_dev * d_h, d_model]
        self.W_o = Tensor([heads_per_dev * d_h, d], data_type)
        self.O_proj = Matmul(data_type)

        # FFN
        if n_experts > 1:
            # MoE: gate router + per-expert SwiGLU FFN
            self.W_gate = Tensor([d, n_experts], data_type)
            self.gate_proj_op = Matmul(data_type)
            self.W1 = Tensor([d, ffn // device_count], data_type)
            self.W1_up = Tensor([d, ffn // device_count], data_type)
            self.W2 = Tensor([ffn // device_count, d], data_type)
        else:
            self.W1 = Tensor([d, ffn // device_count], data_type)
            self.W2 = Tensor([ffn // device_count, d], data_type)
        self.H_matmul1 = Matmul(data_type)
        self.H_matmul2 = Matmul(data_type)

        # Norm + activation + allreduce
        self.layer_norm0 = LayerNorm(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.H_activation = SiLU(data_type)
        self.allreduce_mha = _make_allreduce(data_type, use_hierarchical_allreduce)
        self.allreduce_ffn = _make_allreduce(data_type, use_hierarchical_allreduce)

    def __call__(self, X: Tensor) -> Tensor:
        b, s, d = X.shape
        assert d == self.d_model, \
            f"Input dimension {d} must equal d_model ({self.d_model})."
        self._last_batch_size = b
        self._last_seq_len = s
        return X

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        b = getattr(self, '_last_batch_size', 1)
        s = getattr(self, '_last_seq_len', 1)
        d = self.d_model
        dev_cnt = self.device_count
        d_lat = self.d_latent
        n_h = self.n_heads
        d_h = self.head_dim
        heads_per_dev = n_h // dev_cnt
        word_sz = self.data_type.word_size
        hbm_bw = device.io_module.bandwidth
        peak_flops = device.compute_module.total_systolic_array_flops
        ffn = self.intermediate_size
        cal = self.gemm_calibration_factors

        # Determine prefill vs decode
        is_decode = (s == 1)

        # Rebuild computational graphs so compile_and_simulate works correctly.

        # KV down-projection: [b*s, d] x [d, d_latent]
        X_flat = Tensor([b * s, d], self.data_type)
        _ = self.down_kv_proj(X_flat, self.W_down_kv)

        # KV up-projections: [b*s, d_latent] x [d_latent, heads_per_dev * d_h]
        C_flat = Tensor([b * s, d_lat], self.data_type)
        _ = self.up_k_proj(C_flat, self.W_up_k)
        _ = self.up_v_proj(C_flat, self.W_up_v)

        # Q projection
        if self.q_lora_rank > 0:
            _ = self.down_q_proj(X_flat, self.W_down_q)
            Q_lora = Tensor([b * s, self.q_lora_rank], self.data_type)
            _ = self.up_q_proj(Q_lora, self.W_up_q)
        else:
            _ = self.Q_proj(X_flat, self.W_q)

        # Output projection: [b*s, heads_per_dev * d_h] x [heads_per_dev * d_h, d]
        H_flat = Tensor([b * s, heads_per_dev * d_h], self.data_type)
        _ = self.O_proj(H_flat, self.W_o)

        # FFN
        if self.n_experts > 1:
            tokens_per_expert = max(1, b * s * self.n_experts_active // self.n_experts)
            _ = self.gate_proj_op(X_flat, self.W_gate)
            expert_input = Tensor([tokens_per_expert, d], self.data_type)
            _ = self.H_matmul1(expert_input, self.W1)
            expert_hidden = Tensor([tokens_per_expert, ffn // dev_cnt], self.data_type)
            _ = self.H_matmul2(expert_hidden, self.W2)
        else:
            H0_flat = Tensor([b * s, d], self.data_type)
            _ = self.H_matmul1(H0_flat, self.W1)
            H1_flat = Tensor([b * s, ffn // dev_cnt], self.data_type)
            _ = self.H_matmul2(H1_flat, self.W2)

        # Norms and activation
        H0_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm0(H0_norm)
        _ = self.layer_norm1(H0_norm)
        H1_act = Tensor([b * s, ffn // dev_cnt], self.data_type)
        _ = self.H_activation(H1_act)

        # --- GEMM latencies ---

        # 1. KV down-projection: [b*s, d] x [d, d_latent]
        down_kv_lat = _apply_gemm_calibration(
            self.down_kv_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul,
            b * s, d_lat, d, cal)

        # 2. KV up-projections: [b*s, d_latent] x [d_latent, heads_per_dev * d_h]
        # For decode the up-projections act on the full KV cache: [b*cached_seq, d_latent]
        # In practice this is fused into the attention kernel, so we use the
        # decode flash-attention latency which already accounts for reading latent KV.
        up_k_lat = _apply_gemm_calibration(
            self.up_k_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul,
            b * s, heads_per_dev * d_h, d_lat, cal)
        up_v_lat = _apply_gemm_calibration(
            self.up_v_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul,
            b * s, heads_per_dev * d_h, d_lat, cal)

        # 3. Q projection
        if self.q_lora_rank > 0:
            q_down_lat = _apply_gemm_calibration(
                self.down_q_proj.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, self.q_lora_rank, d, cal)
            q_up_lat = _apply_gemm_calibration(
                self.up_q_proj.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, heads_per_dev * d_h, self.q_lora_rank, cal)
            q_total = q_down_lat + q_up_lat
        else:
            q_total = _apply_gemm_calibration(
                self.Q_proj.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, heads_per_dev * d_h, d, cal)

        projection_total = down_kv_lat + up_k_lat + up_v_lat + q_total

        # 4. Attention (FlashAttention)
        if is_decode:
            # Decode: read compressed latent KV cache then attend.
            # The up-projection latency is dominated by reading d_latent × s from HBM.
            # Model as decode attention with MLA: memory cost is latent cache read
            # + decompressed K/V compute. We approximate using flash decode latency
            # with effective n_kv_heads = heads_per_dev (full MHA after decompression).
            attn_lat = _flash_attention_latency_decode(
                b, heads_per_dev, s, d_h, word_sz, hbm_bw, peak_flops)
        else:
            attn_lat = _flash_attention_latency_prefill(
                b, heads_per_dev, s, d_h, word_sz, hbm_bw, peak_flops)

        # 5. Output projection: [b*s, heads_per_dev * d_h] x [heads_per_dev * d_h, d]
        o_proj_lat = _apply_gemm_calibration(
            self.O_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul,
            b * s, d, heads_per_dev * d_h, cal)

        # 6. FFN
        if self.n_experts > 1:
            tokens_per_expert = max(1, b * s * self.n_experts_active // self.n_experts)
            gate_lat = _apply_gemm_calibration(
                self.gate_proj_op.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, self.n_experts, d, cal)
            ffn1_lat = _apply_gemm_calibration(
                self.H_matmul1.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                tokens_per_expert, ffn // dev_cnt, d, cal)
            ffn2_lat = _apply_gemm_calibration(
                self.H_matmul2.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                tokens_per_expert, d, ffn // dev_cnt, cal)
            # Expert dispatch/combine: memory-bound token permutation
            dispatch_bytes = tokens_per_expert * d * word_sz
            dispatch_lat = dispatch_bytes / hbm_bw if hbm_bw > 0 else 0
            ffn_total = gate_lat + dispatch_lat + ffn1_lat + ffn2_lat + dispatch_lat
        else:
            ffn1_lat = _apply_gemm_calibration(
                self.H_matmul1.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, ffn // dev_cnt, d, cal)
            ffn2_lat = _apply_gemm_calibration(
                self.H_matmul2.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul,
                b * s, d, ffn // dev_cnt, cal)
            ffn_total = ffn1_lat + ffn2_lat

        # 7. Activation
        activation_lat = (
            self.H_activation.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # 8. Normalization
        layernorm_lat = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        norm_total = layernorm_lat * 2

        # 9. Elementwise: partial RoPE (only qk_rope_head_dim), residual adds, copy
        # MLA uses partial RoPE on decoupled rope head dims, not full head_dim
        rope_lat = _rope_latency(b, s, n_h // dev_cnt, n_h // dev_cnt,
                                 self.qk_rope_head_dim, word_sz, hbm_bw, peak_flops)
        residual_lat = 2 * _residual_add_latency(b, s, d, word_sz, hbm_bw)
        copy_lat = _memory_copy_latency(b, s, d, word_sz, hbm_bw)
        elementwise_total = rope_lat + residual_lat + copy_lat

        # 10. Allreduce
        if dev_cnt > 1:
            dummy_allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_mha(dummy_allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            projection_total
            + attn_lat
            + o_proj_lat
            + ffn_total
            + activation_lat
            + norm_total
            + elementwise_total
            + allreduce_total_latency
        )
        self.simluate_log = (
            f"down_kv={down_kv_lat:.6f}, up_kv={up_k_lat + up_v_lat:.6f}, "
            f"q_proj={q_total:.6f}, attn={attn_lat:.6f}, o_proj={o_proj_lat:.6f}, "
            f"ffn={ffn_total:.6f}, act={activation_lat:.6f}, norm={norm_total:.6f}, "
            f"elementwise={elementwise_total:.6f}, allreduce={allreduce_total_latency:.6f}"
        )
        return self.latency

    def kv_cache_memory_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """
        MLA KV cache: stores compressed latent, not full K/V.

        MLA caches the low-rank latent C = X @ W_down_kv of shape [seq, d_latent]
        instead of K, V of shape [seq, n_heads * head_dim] each.
        This gives 5-10x smaller KV cache vs standard MHA.
        """
        return batch_size * seq_len * self.d_latent * self.data_type.word_size




# NOTE: post_prediction_correction, get_allreduce_calibration, and
# calibrated_allreduce_latency moved to llm_predict.predictors.calibration.
# See imports at top of file.


def pp_bubble_fraction(pp_stages: int, num_microbatches: int) -> float:
    """
    Compute the pipeline parallelism bubble fraction.

    In pipeline parallelism, the bubble is the idle time at the start and end of
    each training/inference step where not all pipeline stages are active.

    Args:
        pp_stages: Number of pipeline parallel stages
        num_microbatches: Number of microbatches in the pipeline

    Returns:
        Bubble fraction in [0, 1). Lower is better.
        Returns 0.0 if pp_stages <= 1 (no pipeline parallelism).
    """
    if pp_stages <= 1:
        return 0.0
    return (pp_stages - 1) / (num_microbatches + pp_stages - 1)


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass
