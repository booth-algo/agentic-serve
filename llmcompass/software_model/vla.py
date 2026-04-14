"""
OpenVLA (Vision-Language-Action) Model Performance Modeling.

This module provides performance modeling for OpenVLA models, which combine:
1. Vision Encoder (ViT-like) - processes image patches
2. LLM Backbone (Transformer blocks) - processes vision + text tokens
3. Action Prediction Head - outputs action predictions

Based on the transformer.py pattern for consistency.

Pipeline Parallelism (PP) Support:
The OpenVLAForActionPredictionPP class supports PP + TP + DP:
- PP stages split LLM layers at transformer block granularity (minimum = 1 block per stage)
- Each PP stage can have different TP and DP configurations
- P2P communication forwards activations between consecutive stages

Example usage:
    # Define PP stages (4 stages, each handling 8 LLM layers)
    pp_stages = [
        {'layer_range': (0, 8), 'tp_size': 2, 'dp_size': 1},   # Stage 0: layers 0-7, TP=2
        {'layer_range': (8, 16), 'tp_size': 4, 'dp_size': 1},  # Stage 1: layers 8-15, TP=4
        {'layer_range': (16, 24), 'tp_size': 2, 'dp_size': 2}, # Stage 2: layers 16-23, TP=2, DP=2
        {'layer_range': (24, 32), 'tp_size': 1, 'dp_size': 1}, # Stage 3: layers 24-31, TP=1
    ]
    
    model = OpenVLAForActionPredictionPP(
        pp_stages=pp_stages,
        llm_n_layers=32,
        vision_stage_idx=0,  # Vision encoder on stage 0
    )
"""

from llmcompass.software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from llmcompass.software_model.matmul import Matmul, BatchedMatmul
from llmcompass.software_model.softmax import Softmax
from llmcompass.software_model.layernorm import LayerNorm
from llmcompass.software_model.gelu import GeLU
from llmcompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
)

from llmcompass.software_model.utils import Tensor, DataType
from llmcompass.software_model.communication_primitives import AllReduceMultiPCB, P2P as P2PPrimitive
from llmcompass.hardware_model.interconnect import InterConnectModule
from math import ceil
from typing import List, Dict, Tuple, Optional
from llmcompass.hardware_model.system import System


class VisionEncoderTP(Operator):
    """
    Vision Transformer (ViT) encoder with tensor parallelism.
    
    Processes image patches through:
    1. Patch embedding (linear projection)
    2. Positional encoding
    3. Multiple transformer blocks (can reuse TransformerBlockInitComputationTP)
    """
    def __init__(
        self,
        image_size: int,  # e.g., 224
        patch_size: int,  # e.g., 14
        d_model: int,  # Hidden dimension (e.g., 1024 for ViT-L)
        n_layers: int,  # Number of transformer layers
        n_heads: int,  # Number of attention heads
        device_count: int,  # TP size
        data_type: DataType,
        device_group: Optional[List[int]] = None,  # Optional device group for TP
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device_count = device_count
        self.data_type = data_type
        # If device_group not provided, create one from 0 to device_count-1
        if device_group is None:
            self.device_group = list(range(device_count))
        else:
            self.device_group = device_group
        
        # Calculate number of patches
        self.n_patches = (image_size // patch_size) ** 2
        # Add 1 for CLS token
        self.seq_len = self.n_patches + 1
        
        # Patch embedding: [3, patch_size, patch_size] -> [d_model]
        # Input: [batch, 3, image_size, image_size]
        # Output: [batch, n_patches + 1, d_model]
        patch_dim = 3 * patch_size * patch_size
        self.patch_embed = Matmul(data_type)
        self.patch_embed_weight = Tensor([patch_dim, d_model // device_count], data_type)
        
        # CLS token embedding
        self.cls_token = Tensor([1, 1, d_model // device_count], data_type)
        
        # Positional encoding (learned, but we model as addition)
        self.pos_embed = Tensor([1, self.seq_len, d_model // device_count], data_type)
        
        # Transformer blocks (reuse existing transformer block)
        self.transformer_blocks = [
            TransformerBlockInitComputationTP(
                d_model=d_model,
                n_heads=n_heads,
                device_count=device_count,
                data_type=data_type,
            )
            for _ in range(n_layers)
        ]
        
        # Layer norm at the end
        self.layer_norm = LayerNorm(data_type)
        
        # AllReduce for TP synchronization
        self.allreduce = AllReduceMultiPCB(data_type)
    
    def __call__(self, images: Tensor) -> Tensor:
        """
        Process images through vision encoder.
        
        Args:
            images: [batch, 3, image_size, image_size]
        
        Returns:
            vision_tokens: [batch, seq_len, d_model]
        """
        batch_size = images.shape[0]
        
        # Reshape images to patches: [batch, 3, image_size, image_size] 
        # -> [batch, n_patches, 3 * patch_size * patch_size]
        # For simplicity, we model this as a reshape + linear projection
        patch_embed_input = Tensor([batch_size, self.n_patches, 3 * self.patch_size * self.patch_size], self.data_type)
        
        # Patch embedding projection (per device)
        patch_embeds = self.patch_embed(patch_embed_input, self.patch_embed_weight)  # [batch, n_patches, d_model // device_count]
        
        # Add CLS token (modeled as concat)
        cls_tokens = Tensor([batch_size, 1, self.d_model // self.device_count], self.data_type)
        vision_tokens = Concat(self.data_type)(cls_tokens, patch_embeds, 1)  # [batch, seq_len, d_model // device_count]
        
        # Positional encoding (modeled as addition, but we just track the shape)
        # In practice, this is element-wise addition, but for simulation we just track tensor shapes
        # Note: For TP, after AllReduce (if any), or for the input to transformer blocks,
        # we need the full d_model dimension. The transformer blocks internally handle TP sharding.
        # Before entering transformer blocks, the input should have the full d_model dimension
        # (the blocks will internally shard it for TP processing)
        vision_tokens = Tensor([batch_size, self.seq_len, self.d_model], self.data_type)
        
        # Process through transformer blocks
        # Transformer blocks expect full d_model dimension; they handle TP internally
        for transformer_block in self.transformer_blocks:
            vision_tokens = transformer_block(vision_tokens)  # [batch, seq_len, d_model] (after internal AllReduce)
        
        # Final layer norm
        vision_tokens = self.layer_norm(vision_tokens)
        
        # AllReduce to gather full tokens across TP devices
        if self.device_count > 1:
            vision_tokens = self.allreduce(vision_tokens)
        
        # Output: [batch, seq_len, d_model]
        return vision_tokens
    
    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU"):
        """
        Simulate the vision encoder forward pass.
        """
        device = system.device
        interconnect = system.interconnect
        
        total_latency = 0.0
        
        # Patch embedding
        # Set up the matmul with dummy tensors
        patch_embed_input_dummy = Tensor([1, self.n_patches, 3 * self.patch_size * self.patch_size], self.data_type)
        _ = self.patch_embed(patch_embed_input_dummy, self.patch_embed_weight)
        patch_embed_latency = (
            self.patch_embed.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        total_latency += patch_embed_latency
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            block_latency = transformer_block.compile_and_simulate(system, compile_mode)
            total_latency += block_latency
        
        # Layer norm - set up dummy tensor first
        layer_norm_input_dummy = Tensor([1, self.seq_len, self.d_model], self.data_type)
        _ = self.layer_norm(layer_norm_input_dummy)
        layernorm_latency = (
            self.layer_norm.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        total_latency += layernorm_latency
        
        # AllReduce (if TP > 1)
        if self.device_count > 1:
            vision_tokens_tensor = Tensor([1, self.seq_len, self.d_model], self.data_type)
            self.allreduce(vision_tokens_tensor)
            allreduce_latency = self.allreduce.simulate(interconnect, device_group=self.device_group)
            total_latency += allreduce_latency
        
        self.latency = total_latency
        return self.latency
    
    def roofline_model(self, system: System):
        """
        Roofline model for vision encoder.
        """
        device = system.device
        
        # Set up patch embedding operator with dummy tensors
        # Patch embedding: [batch, n_patches, patch_dim] * [patch_dim, d_model // device_count]
        patch_dim = 3 * self.patch_size * self.patch_size
        patch_embed_input_dummy = Tensor([1, self.n_patches, patch_dim], self.data_type)
        _ = self.patch_embed(patch_embed_input_dummy, self.patch_embed_weight)
        
        # Patch embedding roofline
        patch_embed_roofline = self.patch_embed.roofline_model(device)
        
        # Transformer blocks roofline
        transformer_roofline = sum(
            block.roofline_model(system) for block in self.transformer_blocks
        )
        
        # Set up LayerNorm operator with dummy tensor
        # LayerNorm operates on vision tokens: [batch, seq_len, d_model]
        layer_norm_dummy = Tensor([1, self.seq_len, self.d_model], self.data_type)
        _ = self.layer_norm(layer_norm_dummy)
        
        # Layer norm roofline
        layernorm_roofline = self.layer_norm.roofline_model(device)
        
        self.roofline_latency = patch_embed_roofline + transformer_roofline + layernorm_roofline
        return self.roofline_latency


class MHALayerTP(Operator):
    """
    Multi-Head Attention layer with Tensor Parallelism support.
    
    This is a single MHA layer extracted from TransformerBlockInitComputationTP.
    Supports TP for pipeline parallelism at layer granularity.
    """
    def __init__(self, d_model: int, n_heads: int, device_group: List[int], data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_group = device_group
        self.device_count = len(device_group)
        self.data_type = data_type
        
        if self.device_count < 1:
            raise ValueError("device_group must contain at least one device")
        # Support uneven head assignment: remove strict divisibility requirement
        # We'll handle uneven division in the forward pass and model performance based on bottleneck
        # if self.n_heads % self.device_count != 0:
        #     raise ValueError(f"n_heads ({self.n_heads}) must be divisible by device_count ({self.device_count})")
        
        # Parameters per device
        d = d_model
        self.Wq = Tensor([d, d // self.device_count], data_type)
        self.Wk = Tensor([d, d // self.device_count], data_type)
        self.Wv = Tensor([d, d // self.device_count], data_type)
        self.W0 = Tensor([d // self.device_count, d], data_type)
        
        # Operators per device
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
        self.allreduce_mha = AllReduceMultiPCB(data_type)
    
    def __call__(self, X: Tensor) -> Tensor:
        """Forward pass through MHA layer"""
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        
        # Support uneven head assignment:
        # - Most devices get h // dev_cnt heads each
        # - Last device gets the remainder: h - (h // dev_cnt) * (dev_cnt - 1)
        # For performance modeling, we model the bottleneck (device with most heads)
        h_per_dev_base = h // dev_cnt  # Base heads per device (for most devices)
        h_remainder = h - h_per_dev_base * (dev_cnt - 1)  # Heads for last device
        h_per_dev_max = max(h_per_dev_base, h_remainder)  # Maximum heads (bottleneck for modeling)
        # Store for performance modeling
        self.h_per_dev_model = h_per_dev_max
        
        # Project to get Q, K, V
        Q = self.Q_proj(X, self.Wq)  # [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d // dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d // dev_cnt]
        
        # Get actual dimension per device from projection output
        actual_d_per_dev = Q.shape[2]
        
        # Calculate how many complete heads fit in the actual dimension
        # This handles the case where d_per_dev is not exactly divisible by heads
        # For uneven head assignment: model the bottleneck (device with most heads)
        # But for computation, use the actual heads that fit
        h_per_dev_actual = actual_d_per_dev // d_h  # Actual complete heads that fit
        d_used = h_per_dev_actual * d_h  # Dimension used for complete heads
        
        # Reshape to heads: use the actual heads that fit
        # If there's remainder dimension (actual_d_per_dev > d_used), Reshape now allows this
        # For performance modeling, we track h_per_dev_max (bottleneck) separately
        if h_per_dev_actual > 0:
            Q = self.Q_reshape(Q, [b, s, h_per_dev_actual, d_h])
            K = self.K_reshape(K, [b, s, h_per_dev_actual, d_h])
            V = self.V_reshape(V, [b, s, h_per_dev_actual, d_h])
            h_per_dev = h_per_dev_actual
        else:
            # Edge case: no complete head fits (shouldn't happen with reasonable TP)
            # Treat as single head with available dimension
            h_per_dev = 1
            Q = self.Q_reshape(Q, [b, s, 1, actual_d_per_dev])
            K = self.K_reshape(K, [b, s, 1, actual_d_per_dev])
            V = self.V_reshape(V, [b, s, 1, actual_d_per_dev])
        
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h_per_dev, s, d_h or actual_d_per_dev]
        assert Q_T.shape[0] == b and Q_T.shape[1] == h_per_dev and Q_T.shape[2] == s
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h_per_dev, d_h or actual_d_per_dev, s]
        assert K_T.shape[0] == b and K_T.shape[1] == h_per_dev and K_T.shape[3] == s
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h_per_dev, s, d_h or actual_d_per_dev]
        assert V_T.shape[0] == b and V_T.shape[1] == h_per_dev and V_T.shape[2] == s
        A = self.Q_mul_K(Q_T, K_T)  # [b, h_per_dev, s, s]
        assert A.shape[0] == b and A.shape[1] == h_per_dev
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)  # [b, h_per_dev, s, d_h or actual_d_per_dev]
        assert H.shape[0] == b and H.shape[1] == h_per_dev and H.shape[2] == s
        H = self.H_transpose(H, [0, 2, 1, 3])  # [b, s, h_per_dev, d_h or actual_d_per_dev]
        assert H.shape[0] == b and H.shape[2] == h_per_dev and H.shape[1] == s
        # Reshape back to actual_d_per_dev for compatibility with W0
        # If we have remainder dimension (d_used < actual_d_per_dev), Reshape now supports
        # padding conceptually for performance modeling
        H = self.H_reshape(H, [b, s, actual_d_per_dev])  # Back to [b, s, actual_d_per_dev]
        assert H.shape[0] == b and H.shape[1] == s and H.shape[2] == actual_d_per_dev
        H0 = self.H_matmul0(H, self.W0)  # [b, s, d]
        H0 = self.layer_norm0(H0)
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)
        
        assert H0.shape == [b, s, d]
        return H0
    
    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU", batch_size: int = None, seq_len: int = None) -> float:
        """Simulate MHA layer"""
        device = system.device
        interconnect = system.interconnect
        
        # Determine batch size and sequence length
        b = batch_size or 1
        s = seq_len or 1
        d = self.d_model
        dev_cnt = self.device_count
        h = self.n_heads
        d_h = d // h
        
        # Calculate heads per device for uneven assignment (use bottleneck for modeling)
        h_per_dev_base = h // dev_cnt
        h_remainder = h - h_per_dev_base * (dev_cnt - 1)
        h_per_dev_max = max(h_per_dev_base, h_remainder)  # Use bottleneck for performance modeling
        
        # Get actual dimension per device
        actual_d_per_dev = d // dev_cnt
        h_per_dev_actual = actual_d_per_dev // d_h  # Actual heads that fit
        
        # Use the maximum for performance modeling (bottleneck case)
        h_per_dev_model = max(h_per_dev_actual, 1) if h_per_dev_actual > 0 else 1
        
        # Build computational graph
        dummy_X = Tensor([b, s, d], self.data_type)
        _ = self.Q_proj(dummy_X, self.Wq)
        _ = self.K_proj(dummy_X, self.Wk)
        _ = self.V_proj(dummy_X, self.Wv)
        
        # Use modeling heads (bottleneck) for performance estimation
        dummy_Q_T = Tensor([b, h_per_dev_model, s, d_h], self.data_type)
        dummy_K_T = Tensor([b, h_per_dev_model, d_h, s], self.data_type)
        _ = self.Q_mul_K(dummy_Q_T, dummy_K_T)
        
        dummy_A_prob = Tensor([b, h_per_dev_model, s, s], self.data_type)
        dummy_V_T = Tensor([b, h_per_dev_model, s, d_h], self.data_type)
        _ = self.A_mul_V(dummy_A_prob, dummy_V_T)
        
        dummy_H = Tensor([b, s, actual_d_per_dev], self.data_type)
        _ = self.H_matmul0(dummy_H, self.W0)
        
        dummy_A = Tensor([b, h_per_dev_model, s, s], self.data_type)
        _ = self.A_softmax(dummy_A)
        
        dummy_H0_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm0(dummy_H0_norm)
        
        # Compute latencies
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        
        matmul_total_latency = qkv_latency + q_mul_k_latency + a_mul_v_latency + h_matmul0_latency
        
        # AllReduce
        allreduce_latency = 0.0
        if self.device_count > 1:
            allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_mha(allreduce_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect, device_group=self.device_group)
        
        self.latency = (
            matmul_total_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        return self.latency
    
    def roofline_model(self, system: System):
        """Roofline model for MHA layer"""
        device = system.device
        interconnect = system.interconnect
        
        # Set up BatchedMatmul operators with dummy tensors for roofline modeling
        # Q_mul_K: [b, h_per_dev, s, d_h] * [b, h_per_dev, d_h, s] = [b, h_per_dev, s, s]
        # Use dummy dimensions: batch=1, seq_len=512, d_h = d_model // n_heads
        # For roofline modeling, use the bottleneck (maximum heads per device)
        b, s, d_h = 1, 512, self.d_model // self.n_heads
        h = self.n_heads
        dev_cnt = self.device_count
        
        # Calculate heads per device for uneven assignment (use bottleneck for roofline)
        h_per_dev_base = h // dev_cnt
        h_remainder = h - h_per_dev_base * (dev_cnt - 1)
        h_per_dev_max = max(h_per_dev_base, h_remainder)  # Use bottleneck for roofline modeling
        
        # Get actual dimension per device and calculate actual heads
        actual_d_per_dev = self.d_model // dev_cnt
        h_per_dev_actual = actual_d_per_dev // d_h
        # Use maximum for roofline (bottleneck case)
        h_per_dev_model = max(h_per_dev_actual, 1) if h_per_dev_actual > 0 else 1
        
        q_t_dummy = Tensor([b, h_per_dev_model, s, d_h], self.data_type)
        k_t_dummy = Tensor([b, h_per_dev_model, d_h, s], self.data_type)
        _ = self.Q_mul_K(q_t_dummy, k_t_dummy)
        
        # A_mul_V: [b, h_per_dev, s, s] * [b, h_per_dev, s, d_h] = [b, h_per_dev, s, d_h]
        a_prob_dummy = Tensor([b, h_per_dev_model, s, s], self.data_type)
        v_t_dummy = Tensor([b, h_per_dev_model, s, d_h], self.data_type)
        _ = self.A_mul_V(a_prob_dummy, v_t_dummy)
        
        # Set up Softmax operator with dummy tensor
        _ = self.A_softmax(a_prob_dummy)
        
        # Set up LayerNorm operator with dummy tensor
        # LayerNorm operates on hidden states: [b, s, d_model]
        h0_dummy = Tensor([b, s, self.d_model], self.data_type)
        _ = self.layer_norm0(h0_dummy)
        
        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
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
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        
        matmul_total_latency = qkv_latency + q_mul_k_latency + a_mul_v_latency + h_matmul0_latency
        
        allreduce_latency = 0.0
        if self.device_count > 1:
            dummy_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_mha(dummy_tensor)
            allreduce_latency = self.allreduce_mha.simulate(interconnect, device_group=self.device_group)
        
        self.roofline_latency = (
            matmul_total_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        return self.roofline_latency


class FFNLayerTP(Operator):
    """
    Feed-Forward Network layer with Tensor Parallelism support.
    
    This is a single FFN layer extracted from TransformerBlockInitComputationTP.
    Supports TP for pipeline parallelism at layer granularity.
    """
    def __init__(self, d_model: int, device_group: List[int], data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.device_group = device_group
        self.device_count = len(device_group)
        self.data_type = data_type
        
        if self.device_count < 1:
            raise ValueError("device_group must contain at least one device")
        
        # Parameters per device
        d = d_model
        self.W1 = Tensor([d, 4 * d // self.device_count], data_type)
        self.W2 = Tensor([4 * d // self.device_count, d], data_type)
        
        # Operators per device
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)
    
    def __call__(self, X: Tensor) -> Tensor:
        """Forward pass through FFN layer"""
        b, s, d = X.shape
        assert d == self.d_model
        dev_cnt = self.device_count
        
        # Feed-forward network
        H1 = self.H_matmul1(X, self.W1)  # [b, s, 4 * d / dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  # [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)
        
        assert H2.shape == [b, s, d]
        return H2
    
    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU", batch_size: int = None, seq_len: int = None) -> float:
        """Simulate FFN layer"""
        device = system.device
        interconnect = system.interconnect
        
        # Determine batch size and sequence length
        b = batch_size or 1
        s = seq_len or 1
        d = self.d_model
        dev_cnt = self.device_count
        
        # Build computational graph
        dummy_H0 = Tensor([b, s, d], self.data_type)
        _ = self.H_matmul1(dummy_H0, self.W1)
        
        dummy_H1 = Tensor([b, s, 4 * d // dev_cnt], self.data_type)
        _ = self.H_gelu(dummy_H1)
        _ = self.H_matmul2(dummy_H1, self.W2)
        
        dummy_H2_norm = Tensor([b, s, d], self.data_type)
        _ = self.layer_norm1(dummy_H2_norm)
        
        # Compute latencies
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        layernorm_latency = (
            self.layer_norm1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        
        matmul_total_latency = h1_matmul1_latency + h2_matmul2_latency
        
        # AllReduce
        allreduce_latency = 0.0
        if self.device_count > 1:
            allreduce_tensor = Tensor([b, s, d], self.data_type)
            self.allreduce_ffn(allreduce_tensor)
            allreduce_latency = self.allreduce_ffn.simulate(interconnect, device_group=self.device_group)
        
        self.latency = (
            matmul_total_latency
            + gelu_latency
            + layernorm_latency
            + allreduce_latency
        )
        return self.latency
    
    def roofline_model(self, system: System):
        """Roofline model for FFN layer"""
        device = system.device
        interconnect = system.interconnect
        
        # Set up GeLU operator with dummy tensor
        # GeLU operates on FFN hidden states: [b, s, 4 * d_model // device_count]
        b, s = 1, 512
        h1_dummy = Tensor([b, s, 4 * self.d_model // self.device_count], self.data_type)
        _ = self.H_gelu(h1_dummy)
        
        # Set up LayerNorm operator with dummy tensor
        # LayerNorm operates on hidden states: [b, s, d_model]
        h2_dummy = Tensor([b, s, self.d_model], self.data_type)
        _ = self.layer_norm1(h2_dummy)
        
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        layernorm_latency = (
            self.layer_norm1.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        
        matmul_total_latency = h1_matmul1_latency + h2_matmul2_latency
        
        allreduce_latency = 0.0
        if self.device_count > 1:
            dummy_tensor = Tensor([1, 1, self.d_model], self.data_type)
            self.allreduce_ffn(dummy_tensor)
            allreduce_latency = self.allreduce_ffn.simulate(interconnect, device_group=self.device_group)
        
        self.roofline_latency = (
            matmul_total_latency
            + gelu_latency
            + layernorm_latency
            + allreduce_latency
        )
        return self.roofline_latency


class OpenVLAForActionPredictionPP(Operator):
    """
    OpenVLA model with Pipeline Parallelism (PP) + Tensor Parallelism (TP).
    
    Architecture:
    1. Vision Encoder: Processes images to vision tokens (can be on first PP stage or separate)
    2. Vision-Language Fusion: Projects vision tokens to LLM hidden dimension
    3. LLM Backbone: Processes vision + text tokens through individual layers (MHA and FFN)
       - Partitioned across PP stages at layer granularity (each layer can be a PP stage)
       - Each PP stage can have different TP configurations
    4. Action Head: Predicts action from final hidden states (on last PP stage)
    
    PP Configuration:
    - PP stages split LLM layers at individual layer level (MHA or FFN)
    - Each PP stage can have different TP arrangements
    - P2P communication forwards activations between consecutive stages
    - Each transformer block consists of: MHA layer -> FFN layer
    
    Based on OpenVLA-7b which uses:
    - Llama-2-7b backbone (d_model=4096, n_heads=32, n_layers=32)
    - ViT vision encoder (d_model=1024 for ViT-L, projected to 4096)
    - Action prediction head (n_action_bins=256)
    """
    def __init__(
        self,
        # Pipeline parallelism config (required parameter)
        pp_stages: List[Dict],  # List of PP stage configs
        # Each stage config: {
        #   'layer_range': (start_layer_idx, end_layer_idx),  # Which layers this stage handles (layer indices, not block indices)
        #   'tp_size': int,  # Tensor parallelism size for this stage
        #   'device_group': Optional[List[int]],  # Optional explicit device IDs for TP group
        # }
        # Note: layer_idx refers to individual layers (MHA or FFN), not transformer blocks
        # For n_layers transformer blocks, there are 2*n_layers layers (MHA + FFN for each block)
        # Vision encoder config
        image_size: int = 224,
        patch_size: int = 14,
        vision_d_model: int = 1024,  # ViT-L hidden dim
        vision_n_layers: int = 24,  # ViT-L layers
        vision_n_heads: int = 16,  # ViT-L heads
        # LLM backbone config (Llama-2-7b)
        llm_d_model: int = 4096,
        llm_n_heads: int = 32,
        llm_n_layers: int = 32,
        # Action head config
        n_action_bins: int = 256,
        action_dim: int = 7,  # 6D pose + gripper
        # Other config
        data_type: DataType = None,
        # Text sequence length (for modeling)
        text_seq_len: int = 512,
        # Vision encoder placement: which PP stage handles vision encoder
        vision_stage_idx: int = 0,  # Vision encoder on first PP stage by default
    ):
        if data_type is None:
            from llmcompass.software_model.utils import data_type_dict
            data_type = data_type_dict["fp16"]
        
        super().__init__(0, 0, 0, 0, data_type)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_d_model = vision_d_model
        self.vision_n_layers = vision_n_layers
        self.vision_n_heads = vision_n_heads
        self.llm_d_model = llm_d_model
        self.llm_n_heads = llm_n_heads
        self.llm_n_layers = llm_n_layers
        self.n_action_bins = n_action_bins
        self.action_dim = action_dim
        self.pp_stages = pp_stages
        self.pp_size = len(pp_stages)
        self.text_seq_len = text_seq_len
        self.vision_stage_idx = vision_stage_idx
        
        # Total number of layers = 2 * n_layers (MHA + FFN for each transformer block)
        self.total_layers = 2 * llm_n_layers
        
        # Validate PP stage configurations
        self._validate_pp_stages()
        
        # Calculate sequence lengths
        self.vision_seq_len = ((image_size // patch_size) ** 2) + 1  # patches + CLS token
        
        # Vision encoder (only on the designated stage)
        self.vision_encoder = None
        if vision_stage_idx < self.pp_size:
            stage_tp = pp_stages[vision_stage_idx]['tp_size']
            stage_device_group = pp_stages[vision_stage_idx].get('device_group', list(range(stage_tp)))
            self.vision_encoder = VisionEncoderTP(
                image_size=image_size,
                patch_size=patch_size,
                d_model=vision_d_model,
                n_layers=vision_n_layers,
                n_heads=vision_n_heads,
                device_count=stage_tp,
                data_type=data_type,
                device_group=stage_device_group,
            )
        
        # Vision-Language fusion projection (on vision stage)
        self.vision_proj = None
        if vision_stage_idx < self.pp_size:
            stage_tp = pp_stages[vision_stage_idx]['tp_size']
            self.vision_proj = Matmul(data_type)
            self.vision_proj_weight = Tensor([vision_d_model, llm_d_model // stage_tp], data_type)
        
        # LLM backbone partitioned across PP stages at layer level
        # Store device groups for each stage to support per-device-pair bandwidth
        self.pp_stage_device_groups: List[List[int]] = []
        self.pp_stage_layers: List[List[Operator]] = []  # List of layers (MHA or FFN) per stage
        device_id_offset = 0  # Track device ID allocation across stages
        for stage_idx, stage_cfg in enumerate(pp_stages):
            start_layer_idx, end_layer_idx = stage_cfg['layer_range']
            stage_tp = stage_cfg['tp_size']
            
            # Get device group for this PP stage's TP group
            if 'device_group' in stage_cfg:
                stage_device_group = stage_cfg['device_group']
                if len(stage_device_group) != stage_tp:
                    raise ValueError(f"PP stage {stage_idx} device_group length ({len(stage_device_group)}) must match tp_size ({stage_tp})")
            else:
                # Auto-generate consecutive device IDs
                stage_device_group = list(range(device_id_offset, device_id_offset + stage_tp))
                device_id_offset += stage_tp
            
            self.pp_stage_device_groups.append(stage_device_group)
            
            # Create individual layers (MHA and FFN) for this stage
            stage_layers = []
            for layer_idx in range(start_layer_idx, end_layer_idx):
                # Determine if this is MHA (even indices) or FFN (odd indices)
                # Layer 0, 2, 4, ... are MHA layers
                # Layer 1, 3, 5, ... are FFN layers
                if layer_idx % 2 == 0:
                    # MHA layer
                    layer = MHALayerTP(
                        d_model=llm_d_model,
                        n_heads=llm_n_heads,
                        device_group=stage_device_group,
                        data_type=data_type,
                    )
                else:
                    # FFN layer
                    layer = FFNLayerTP(
                        d_model=llm_d_model,
                        device_group=stage_device_group,
                        data_type=data_type,
                    )
                stage_layers.append(layer)
            self.pp_stage_layers.append(stage_layers)
        
        # Action prediction head (on last PP stage)
        last_stage_tp = pp_stages[-1]['tp_size']
        self.action_head_proj1 = Matmul(data_type)
        self.action_head_proj1_weight = Tensor([llm_d_model // last_stage_tp, llm_d_model // last_stage_tp], data_type)
        self.action_head_gelu = GeLU(data_type)
        self.action_head_proj2 = Matmul(data_type)
        self.action_head_proj2_weight = Tensor([llm_d_model // last_stage_tp, action_dim * n_action_bins // last_stage_tp], data_type)
        self.allreduce_action = AllReduceMultiPCB(data_type)
        
        # P2P communication primitives between stages
        # For each stage boundary, we need P2P communication
        # If TP sizes differ, we may need special handling, but for now model as simple P2P
        self.p2p_comm: List[P2PPrimitive] = []
        for stage_idx in range(self.pp_size - 1):
            # Get source and destination device IDs for P2P
            # Use first device of each stage's device group
            src_device = self.pp_stage_device_groups[stage_idx][0]
            dst_device = self.pp_stage_device_groups[stage_idx + 1][0]
            p2p = P2PPrimitive(data_type, src_device=src_device, dst_device=dst_device)
            self.p2p_comm.append(p2p)
    
    def _validate_pp_stages(self):
        """Validate that PP stage configurations are valid"""
        if not self.pp_stages:
            raise ValueError("At least one PP stage is required")
        
        # Check that all layers are covered exactly once
        covered_layers = set()
        for stage_idx, stage_cfg in enumerate(self.pp_stages):
            if 'layer_range' not in stage_cfg:
                raise ValueError(f"PP stage {stage_idx} must have 'layer_range'")
            if 'tp_size' not in stage_cfg:
                raise ValueError(f"PP stage {stage_idx} must have 'tp_size'")
            
            start_layer_idx, end_layer_idx = stage_cfg['layer_range']
            if start_layer_idx < 0 or end_layer_idx > self.total_layers:
                raise ValueError(f"PP stage {stage_idx} layer_range [{start_layer_idx}, {end_layer_idx}) out of bounds [0, {self.total_layers})")
            if start_layer_idx >= end_layer_idx:
                raise ValueError(f"PP stage {stage_idx} must have start_layer_idx < end_layer_idx")
            
            if stage_cfg['tp_size'] < 1:
                raise ValueError(f"PP stage {stage_idx} tp_size must be >= 1")
            
            # Check for overlaps
            stage_layers = set(range(start_layer_idx, end_layer_idx))
            overlap = covered_layers & stage_layers
            if overlap:
                raise ValueError(f"PP stage {stage_idx} overlaps with previous stages at layers {overlap}")
            covered_layers |= stage_layers
        
        # Check that all layers are covered
        all_layers = set(range(self.total_layers))
        if covered_layers != all_layers:
            missing = all_layers - covered_layers
            raise ValueError(f"Missing layers in PP stages: {missing}")
    
    def __call__(self, images: Tensor, text_tokens: Tensor = None) -> Tensor:
        """
        Forward pass through OpenVLA model with PP.
        
        Args:
            images: [batch, 3, image_size, image_size]
            text_tokens: [batch, text_seq_len, llm_d_model] (optional text tokens)
        
        Returns:
            actions: [batch, action_dim, n_action_bins]
        """
        batch_size = images.shape[0]
        
        # Stage 0: Vision encoder (if on first stage) or initial LLM blocks
        hidden_states = None
        
        if self.vision_stage_idx == 0 and self.vision_encoder is not None:
            # Process vision encoder
            vision_tokens = self.vision_encoder(images)  # [batch, vision_seq_len, vision_d_model]
            
            # Project vision tokens to LLM hidden dimension
            vision_tokens_llm = self.vision_proj(vision_tokens, self.vision_proj_weight)
            
            # Combine vision and text tokens
            if text_tokens is not None:
                combined_tokens = Concat(self.data_type)(vision_tokens_llm, text_tokens, 1)
                seq_len = self.vision_seq_len + text_tokens.shape[1]
            else:
                combined_tokens = vision_tokens_llm
                seq_len = self.vision_seq_len
            
            # AllReduce if TP > 1
            stage_tp = self.pp_stages[0]['tp_size']
            if stage_tp > 1:
                hidden_states = Tensor([batch_size, seq_len, self.llm_d_model], self.data_type)
            else:
                hidden_states = combined_tokens
        
        # Process LLM layers in each PP stage
        for stage_idx, stage_layers in enumerate(self.pp_stage_layers):
            stage_cfg = self.pp_stages[stage_idx]
            stage_tp = stage_cfg['tp_size']
            
            # If this is the vision stage and we haven't processed it yet, skip to LLM layers
            if stage_idx == self.vision_stage_idx and hidden_states is None:
                # Vision processing already happened above, just use hidden_states
                pass
            
            # Process individual layers (MHA or FFN) in this stage
            for layer in stage_layers:
                if hidden_states is None:
                    # First stage without vision: create initial hidden states
                    seq_len = self.vision_seq_len + (self.text_seq_len if text_tokens is not None else 0)
                    hidden_states = Tensor([batch_size, seq_len, self.llm_d_model], self.data_type)
                
                hidden_states = layer(hidden_states)
            
            # P2P communication to next stage (if not last stage)
            if stage_idx < self.pp_size - 1:
                # Model P2P send: forward hidden states to next stage
                # In TP configurations, this might need special handling
                next_stage_tp = self.pp_stages[stage_idx + 1]['tp_size']
                if stage_tp != next_stage_tp:
                    # TP size changes: need to reshape/redistribute
                    # For simplicity, model as tensor shape change
                    hidden_states = Tensor([batch_size, hidden_states.shape[1], self.llm_d_model], self.data_type)
                # P2P communication is modeled in compile_and_simulate
        
        # Final stage: Action head
        last_stage_tp = self.pp_stages[-1]['tp_size']
        # Use last token
        last_hidden = Tensor([batch_size, self.llm_d_model // last_stage_tp], self.data_type)
        
        # Action head MLP
        action_hidden = self.action_head_proj1(last_hidden, self.action_head_proj1_weight)
        action_hidden = self.action_head_gelu(action_hidden)
        action_logits = self.action_head_proj2(action_hidden, self.action_head_proj2_weight)
        
        # Reshape
        action_logits = Reshape(self.data_type)(action_logits, [batch_size, self.action_dim, self.n_action_bins // last_stage_tp])
        
        # AllReduce to gather full action predictions
        if last_stage_tp > 1:
            action_logits = self.allreduce_action(action_logits)
        
        return action_logits
    
    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU"):
        """
        Simulate the entire OpenVLA model with PP, TP, and DP.
        
        This simulates:
        1. Vision encoder processing (on designated stage)
        2. Vision-Language fusion
        3. LLM backbone processing across PP stages
        4. P2P communication between stages
        5. Action head prediction (on last stage)
        """
        device = system.device
        interconnect = system.interconnect
        
        stage_latencies = []
        
        # Process each PP stage
        for stage_idx, stage_cfg in enumerate(self.pp_stages):
            stage_tp = stage_cfg['tp_size']
            stage_latency = 0.0
            
            print(f"Simulating PP stage {stage_idx + 1}/{self.pp_size} (TP={stage_tp})...")
            
            # Vision encoder (if on this stage)
            if stage_idx == self.vision_stage_idx and self.vision_encoder is not None:
                print(f"  Simulating vision encoder...")
                vision_latency = self.vision_encoder.compile_and_simulate(system, compile_mode)
                stage_latency += vision_latency
                
                print(f"  Simulating vision projection...")
                vision_tokens_dummy = Tensor([1, self.vision_seq_len, self.vision_d_model], self.data_type)
                _ = self.vision_proj(vision_tokens_dummy, self.vision_proj_weight)
                vision_proj_latency = (
                    self.vision_proj.compile_and_simulate(device, compile_mode)
                    + device.compute_module.overhead.matmul
                )
                stage_latency += vision_proj_latency
            
            # LLM layers in this stage
            stage_layers = self.pp_stage_layers[stage_idx]
            start_layer_idx, end_layer_idx = stage_cfg['layer_range']
            print(f"  Simulating LLM layers {start_layer_idx} to {end_layer_idx-1}...")
            
            for layer_idx, layer in enumerate(stage_layers):
                global_layer_idx = start_layer_idx + layer_idx
                layer_type = "MHA" if global_layer_idx % 2 == 0 else "FFN"
                print(f"    Simulating {layer_type} layer {global_layer_idx}...")
                layer_latency = layer.compile_and_simulate(system, compile_mode)
                stage_latency += layer_latency
            
            # Action head (if last stage)
            if stage_idx == self.pp_size - 1:
                print(f"  Simulating action head...")
                last_hidden_dummy = Tensor([1, self.llm_d_model // stage_tp], self.data_type)
                _ = self.action_head_proj1(last_hidden_dummy, self.action_head_proj1_weight)
                action_proj1_latency = (
                    self.action_head_proj1.compile_and_simulate(device, compile_mode)
                    + device.compute_module.overhead.matmul
                )
                
                action_hidden_dummy = Tensor([1, self.llm_d_model // stage_tp], self.data_type)
                _ = self.action_head_gelu(action_hidden_dummy)
                action_gelu_latency = (
                    self.action_head_gelu.compile_and_simulate(device, compile_mode)
                    + device.compute_module.overhead.gelu
                )
                
                _ = self.action_head_proj2(action_hidden_dummy, self.action_head_proj2_weight)
                action_proj2_latency = (
                    self.action_head_proj2.compile_and_simulate(device, compile_mode)
                    + device.compute_module.overhead.matmul
                )
                
                action_head_latency = action_proj1_latency + action_gelu_latency + action_proj2_latency
                stage_latency += action_head_latency
                
                # Action head AllReduce (if TP > 1)
                if stage_tp > 1:
                    action_tensor = Tensor([1, self.action_dim, self.n_action_bins], self.data_type)
                    self.allreduce_action(action_tensor)
                    allreduce_action_latency = self.allreduce_action.simulate(interconnect)
                    stage_latency += allreduce_action_latency
            
            stage_latencies.append(stage_latency)
            
            # P2P communication to next stage (if not last stage)
            if stage_idx < self.pp_size - 1:
                print(f"  Simulating P2P communication to next stage...")
                # Create tensor representing hidden states being forwarded
                # Shape: [batch, seq_len, d_model]
                seq_len = self.vision_seq_len + self.text_seq_len  # Approximate
                hidden_states_tensor = Tensor([1, seq_len, self.llm_d_model], self.data_type)
                p2p = self.p2p_comm[stage_idx]
                p2p(hidden_states_tensor)
                p2p_latency = p2p.simulate(interconnect)
                print(f"    P2P latency: {p2p_latency:.6f}s")
        
        # Total latency with pipeline parallelism
        # Simple model: sum of stage latencies + pipeline overhead
        # More sophisticated models would account for pipelining efficiency
        total_latency = sum(stage_latencies)
        
        # Pipeline overhead: with PP, there's bubble overhead
        # For first micro-batch: all stages execute sequentially
        # For subsequent micro-batches: stages can overlap, but there's still overhead
        # Simplified model: add (pp_size - 1) * max_stage_latency * pipeline_efficiency_factor
        if self.pp_size > 1:
            max_stage_latency = max(stage_latencies)
            # Pipeline efficiency factor: accounts for bubble time
            # With ideal pipelining, overhead is ~0, but in practice there's overhead
            pipeline_efficiency = 0.3  # Empirically tuned factor
            pipeline_overhead = max_stage_latency * (self.pp_size - 1) * pipeline_efficiency
            total_latency += pipeline_overhead
        
        self.latency = total_latency
        self.stage_latencies = stage_latencies
        return self.latency
    
    def roofline_model(self, system: System):
        """
        Roofline model for OpenVLA with PP - provides upper bound on performance.
        """
        device = system.device
        
        total_roofline = 0.0
        stage_rooflines = []
        
        # Process each PP stage
        for stage_idx, stage_cfg in enumerate(self.pp_stages):
            stage_roofline = 0.0
            
            # Vision encoder (if on this stage)
            if stage_idx == self.vision_stage_idx and self.vision_encoder is not None:
                vision_roofline = self.vision_encoder.roofline_model(system)
                stage_roofline += vision_roofline
                
                # Set up vision_proj operator with dummy tensors
                vision_tokens_dummy = Tensor([1, self.vision_seq_len, self.vision_d_model], self.data_type)
                _ = self.vision_proj(vision_tokens_dummy, self.vision_proj_weight)
                
                vision_proj_roofline = self.vision_proj.roofline_model(device)
                stage_roofline += vision_proj_roofline
            
            # LLM layers in this stage
            stage_layers = self.pp_stage_layers[stage_idx]
            for layer in stage_layers:
                layer_roofline = layer.roofline_model(system)
                stage_roofline += layer_roofline
            
            # Action head (if last stage)
            if stage_idx == self.pp_size - 1:
                # Set up action head operators with dummy tensors
                last_stage_tp = self.pp_stages[-1]['tp_size']
                batch_size = 1
                last_hidden_dummy = Tensor([batch_size, self.llm_d_model // last_stage_tp], self.data_type)
                
                # Set up action_head_proj1
                _ = self.action_head_proj1(last_hidden_dummy, self.action_head_proj1_weight)
                
                # Set up action_head_gelu
                action_hidden_dummy = Tensor([batch_size, self.llm_d_model // last_stage_tp], self.data_type)
                _ = self.action_head_gelu(action_hidden_dummy)
                
                # Set up action_head_proj2
                _ = self.action_head_proj2(action_hidden_dummy, self.action_head_proj2_weight)
                
                action_proj1_roofline = self.action_head_proj1.roofline_model(device)
                action_gelu_roofline = self.action_head_gelu.roofline_model(device)
                action_proj2_roofline = self.action_head_proj2.roofline_model(device)
                stage_roofline += action_proj1_roofline + action_gelu_roofline + action_proj2_roofline
            
            stage_rooflines.append(stage_roofline)
            total_roofline += stage_roofline
        
        # For PP, roofline is sum of all stages (no pipelining benefits in roofline)
        self.roofline_latency = total_roofline
        self.stage_rooflines = stage_rooflines
        return self.roofline_latency
    
    def get_pp_configuration_summary(self) -> str:
        """Get a summary of the PP configuration"""
        lines = []
        lines.append(f"OpenVLA PP Configuration (PP size: {self.pp_size})")
        lines.append(f"Total LLM layers: {self.total_layers} ({self.llm_n_layers} transformer blocks)")
        lines.append("")
        for stage_idx, stage_cfg in enumerate(self.pp_stages):
            start_layer_idx, end_layer_idx = stage_cfg['layer_range']
            lines.append(f"PP Stage {stage_idx}:")
            lines.append(f"  Layers: [{start_layer_idx}, {end_layer_idx}) ({end_layer_idx - start_layer_idx} layers)")
            lines.append(f"  TP size: {stage_cfg['tp_size']}")
            if 'device_group' in stage_cfg:
                lines.append(f"  Device group: {stage_cfg['device_group']}")
            if stage_idx == self.vision_stage_idx:
                lines.append(f"  Contains: Vision Encoder + Vision Projection")
            if stage_idx == self.pp_size - 1:
                lines.append(f"  Contains: Action Head")
            lines.append("")
        return "\n".join(lines)

