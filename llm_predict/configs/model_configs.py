"""Model architecture configs for kernel shape enumeration.

Each config defines the per-layer structure. The composer uses these
to enumerate all kernel calls and their shapes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    n_layers: int
    vocab_size: int
    is_moe: bool = False
    n_experts: int = 1
    top_k: int = 1
    expert_intermediate_size: int | None = None
    expert_weight_bits: int | None = None
    active_params_b: float | None = None
    sliding_window: int | None = None
    full_attention_layers: int | None = None
    moe_overhead_us: float = 0.0  # fused kernel routing + all-reduce per decode step

    @property
    def ffn_intermediate_size(self) -> int:
        return self.expert_intermediate_size or self.intermediate_size


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Llama-3.1-8B": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        hidden_dim=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate_size=14336, n_layers=32, vocab_size=128256,
    ),
    "Llama-3.1-70B": ModelConfig(
        name="Llama-3.1-70B-Instruct",
        hidden_dim=8192, n_heads=64, n_kv_heads=8, head_dim=128,
        intermediate_size=28672, n_layers=80, vocab_size=128256,
    ),
    "Llama-3.3-70B": ModelConfig(
        name="Llama-3.3-70B-Instruct",
        hidden_dim=8192, n_heads=64, n_kv_heads=8, head_dim=128,
        intermediate_size=28672, n_layers=80, vocab_size=128256,
    ),
    "Qwen2.5-72B": ModelConfig(
        name="Qwen2.5-72B-Instruct",
        hidden_dim=8192, n_heads=64, n_kv_heads=8, head_dim=128,
        intermediate_size=29568, n_layers=80, vocab_size=152064,
    ),
    "Mixtral-8x7B": ModelConfig(
        name="Mixtral-8x7B-Instruct-v0.1",
        hidden_dim=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate_size=14336, n_layers=32, vocab_size=32000,
        is_moe=True, n_experts=8, top_k=2,
    ),
    "gpt-oss-20b": ModelConfig(
        name="gpt-oss-20b",
        hidden_dim=2880, n_heads=64, n_kv_heads=8, head_dim=64,
        intermediate_size=2880, n_layers=24, vocab_size=201088,
        is_moe=True, n_experts=32, top_k=4,
        expert_intermediate_size=2880, expert_weight_bits=4, active_params_b=3.6,
        sliding_window=128, full_attention_layers=12,
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        hidden_dim=2880, n_heads=64, n_kv_heads=8, head_dim=64,
        intermediate_size=2880, n_layers=36, vocab_size=201088,
        is_moe=True, n_experts=128, top_k=4,
        expert_intermediate_size=2880, expert_weight_bits=4, active_params_b=5.1,
        sliding_window=128, full_attention_layers=18,
    ),
    "Qwen3.5-9B": ModelConfig(
        name="Qwen3.5-9B",
        hidden_dim=3584, n_heads=28, n_kv_heads=4, head_dim=128,
        intermediate_size=18944, n_layers=36, vocab_size=151936,
    ),
    "Qwen3.5-27B": ModelConfig(
        name="Qwen3.5-27B",
        hidden_dim=5120, n_heads=40, n_kv_heads=8, head_dim=128,
        intermediate_size=27648, n_layers=64, vocab_size=151936,
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        hidden_dim=2880, n_heads=64, n_kv_heads=8, head_dim=64,
        intermediate_size=2880, n_layers=36, vocab_size=201088,
        is_moe=True, n_experts=128, top_k=4,
        expert_intermediate_size=2880, expert_weight_bits=4, active_params_b=5.1,
        sliding_window=128, full_attention_layers=18,
    ),
    "Gemma-2-9B": ModelConfig(
        name="gemma-2-9b-it",
        hidden_dim=3584, n_heads=16, n_kv_heads=8, head_dim=256,
        intermediate_size=14336, n_layers=42, vocab_size=256000,
    ),
    "Granite-3.0-8B": ModelConfig(
        name="granite-3.0-8b-instruct",
        hidden_dim=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate_size=12800, n_layers=40, vocab_size=49152,
    ),
    "Yi-1.5-34B": ModelConfig(
        name="Yi-1.5-34B-Chat",
        hidden_dim=7168, n_heads=56, n_kv_heads=8, head_dim=128,
        intermediate_size=20480, n_layers=60, vocab_size=64000,
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name]
