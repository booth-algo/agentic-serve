"""Composer: enumerate per-layer kernel calls, sum predictions.

Given a model config, GPU, seq_len, and bs, produces a complete
latency breakdown per layer and total across all layers.
"""

from dataclasses import dataclass

from .configs.gpu_specs import get_gpu
from .configs.model_configs import ModelConfig
from .kernels.gemm import GemmPredictor
from .kernels.flash_attn import FlashAttnPredictor
from .kernels.elementwise import (
    RmsNormPredictor, SiluMulPredictor, RotaryEmbPredictor, ResidualAddPredictor,
)


@dataclass
class LayerBreakdown:
    rmsnorm_attn_us: float
    q_proj_us: float
    k_proj_us: float
    v_proj_us: float
    rotary_us: float
    flash_attn_us: float
    o_proj_us: float
    residual_attn_us: float
    rmsnorm_ffn_us: float
    gate_proj_us: float
    up_proj_us: float
    silu_mul_us: float
    down_proj_us: float
    residual_ffn_us: float

    @property
    def gemm_us(self) -> float:
        return (self.q_proj_us + self.k_proj_us + self.v_proj_us +
                self.o_proj_us + self.gate_proj_us + self.up_proj_us +
                self.down_proj_us)

    @property
    def attn_us(self) -> float:
        return self.flash_attn_us

    @property
    def elem_us(self) -> float:
        return (self.rmsnorm_attn_us + self.rotary_us + self.residual_attn_us +
                self.rmsnorm_ffn_us + self.silu_mul_us + self.residual_ffn_us)

    @property
    def total_us(self) -> float:
        return self.gemm_us + self.attn_us + self.elem_us


class Composer:
    def __init__(self, gpu: str):
        self.gpu = gpu
        self.gemm = GemmPredictor(gpu)
        self.flash = FlashAttnPredictor(gpu)
        self.rmsnorm = RmsNormPredictor(gpu)
        self.silu_mul = SiluMulPredictor(gpu)
        self.rotary = RotaryEmbPredictor(gpu)
        self.residual = ResidualAddPredictor(gpu)

    def predict_layer(self, cfg: ModelConfig, seq_len: int,
                      bs: int = 1, kv_len: int | None = None,
                      phase: str = "prefill",
                      tensor_parallel_size: int = 1) -> LayerBreakdown:
        if kv_len is None:
            kv_len = seq_len
        tp = max(1, int(tensor_parallel_size))

        M = seq_len * bs
        h = cfg.hidden_dim
        nh = max(1, cfg.n_heads // tp)
        nkv = max(1, cfg.n_kv_heads // tp)
        hd = cfg.head_dim
        ffn = max(1, cfg.intermediate_size // tp)

        if phase == "decode":
            M = bs

        seq = seq_len if phase == "prefill" else 1
        attention_phase = (
            "cached_prefill"
            if phase == "prefill" and kv_len > seq_len
            else phase
        )

        # MoE: each token routes to top_k experts. Each expert has its own
        # gate/up/down weights with the same (ffn, h) shape. The total FFN
        # cost is top_k * single_expert_cost, not n_experts * cost.
        # For dense models (n_experts=1, top_k=1), this is a no-op.
        ffn_scale = cfg.top_k  # how many expert FFN sets fire per token

        return LayerBreakdown(
            rmsnorm_attn_us=self.rmsnorm.predict_from_shape(h, seq, bs),
            q_proj_us=self.gemm.predict(M, nh * hd, h),
            k_proj_us=self.gemm.predict(M, nkv * hd, h),
            v_proj_us=self.gemm.predict(M, nkv * hd, h),
            rotary_us=self.rotary.predict_from_shape(nh, hd, seq, bs),
            flash_attn_us=self.flash.predict(
                seq_len if phase == "prefill" else 1,
                nh, hd, causal=True, kv_len=kv_len,
                n_kv_heads=nkv, batch=bs, phase=attention_phase),
            o_proj_us=self.gemm.predict(M, h, nh * hd),
            residual_attn_us=self.residual.predict_from_shape(h, seq, bs),
            rmsnorm_ffn_us=self.rmsnorm.predict_from_shape(h, seq, bs),
            gate_proj_us=self.gemm.predict(M, ffn, h) * ffn_scale,
            up_proj_us=self.gemm.predict(M, ffn, h) * ffn_scale,
            silu_mul_us=self.silu_mul.predict_from_shape(ffn, seq, bs) * ffn_scale,
            down_proj_us=self.gemm.predict(M, h, ffn) * ffn_scale,
            residual_ffn_us=self.residual.predict_from_shape(h, seq, bs),
        )

    def predict_ttft_us(self, cfg: ModelConfig, isl: int,
                        bs: int = 1, kv_len: int | None = None,
                        tensor_parallel_size: int = 1) -> float:
        layer = self.predict_layer(
            cfg, seq_len=isl, bs=bs, kv_len=kv_len, phase="prefill",
            tensor_parallel_size=tensor_parallel_size,
        )
        return layer.total_us * cfg.n_layers

    def predict_ttft_ms(self, cfg: ModelConfig, isl: int,
                        bs: int = 1, kv_len: int | None = None,
                        tensor_parallel_size: int = 1) -> float:
        return self.predict_ttft_us(cfg, isl, bs, kv_len,
                                     tensor_parallel_size=tensor_parallel_size) / 1000.0

    def predict_decode_step_us(self, cfg: ModelConfig, kv_len: int,
                               bs: int = 1,
                               tensor_parallel_size: int = 1) -> float:
        layer = self.predict_layer(
            cfg, seq_len=1, bs=bs, kv_len=kv_len, phase="decode",
            tensor_parallel_size=tensor_parallel_size,
        )
        kernel_us = layer.total_us * cfg.n_layers
        # Serving runtime overhead per scheduler step: paged-attention block
        # table indirection, KV cache block management, scheduler loop.
        # Scales with active decode requests. Placeholder values — calibrate
        # with D ∈ {1,2,4,8,16,32} NCU profiling sweep.
        gpu_spec = get_gpu(self.gpu)
        overhead_us = (gpu_spec.step_overhead_base_us
                       + bs * gpu_spec.step_overhead_per_req_us)
        return kernel_us + overhead_us

    def attribute_tpot(self, cfg: ModelConfig, isl: int, osl: int,
                       bs: int = 1) -> dict:
        """Per-component TPOT attribution at midpoint KV length."""
        kv_mid = isl + osl // 2
        layer = self.predict_layer(cfg, seq_len=1, bs=bs,
                                   kv_len=kv_mid, phase="decode")
        total = layer.total_us
        if total == 0:
            return {}
        return {
            "kv_len": kv_mid,
            "gemm_us": round(layer.gemm_us, 2),
            "attn_us": round(layer.attn_us, 2),
            "elem_us": round(layer.elem_us, 2),
            "total_per_layer_us": round(total, 2),
            "total_all_layers_us": round(total * cfg.n_layers, 2),
            "gemm_pct": round(layer.gemm_us / total * 100, 1),
            "attn_pct": round(layer.attn_us / total * 100, 1),
            "elem_pct": round(layer.elem_us / total * 100, 1),
        }
