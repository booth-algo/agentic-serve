"""Model architecture registry for per-kernel training.

Provides `ModelConfig` (the shape a labeler/composer needs) and a legacy
name→config lookup keyed on the directory names used in old NCU profiling data.

Replaces the hardcoded `MODELS` dict at `per-kernel-rebuild/label_all_v3.py:56-62`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str                       # canonical short name (used for training labels)
    d: int                          # hidden size
    h: int                          # num attention heads
    kv: int                         # num KV heads (GQA-aware)
    ffn: int                        # intermediate size
    n_layers: int
    vocab: int = 128256
    E: int = 1                      # num local experts (1 = dense)
    k: int = 1                      # active experts per token (1 = dense)
    bs: int = 1
    seq: int = 128
    held_out: bool = False

    @property
    def head_dim(self) -> int:
        return self.d // self.h

    @property
    def is_moe(self) -> bool:
        return self.E > 1

    def to_dict(self) -> dict:
        return {
            "name": self.name, "d": self.d, "h": self.h, "kv": self.kv,
            "ffn": self.ffn, "n_layers": self.n_layers, "vocab": self.vocab,
            "E": self.E, "k": self.k, "bs": self.bs, "seq": self.seq,
            "held_out": self.held_out,
        }


# ───────── Legacy specs ─────────
_FALLBACK: dict[str, dict] = {
    "Llama-3.1-8B-Instruct": dict(
        name="Llama-8B", d=4096, h=32, kv=8, ffn=14336, n_layers=32, vocab=128256,
    ),
    "Llama-3.1-70B-Instruct": dict(
        name="Llama-70B", d=8192, h=64, kv=8, ffn=28672, n_layers=80, vocab=128256,
    ),
    # Shares architecture with Llama-3.1-70B.
    "Llama-3.3-70B-Instruct": dict(
        name="Llama-3.3-70B", d=8192, h=64, kv=8, ffn=28672, n_layers=80, vocab=128256,
    ),
    # Mixtral-8x7B-Instruct-v0.1: 8 experts, top-2.
    "Mixtral-8x7B-Instruct": dict(
        name="Mixtral-8x7B", d=4096, h=32, kv=8, ffn=14336, n_layers=32, vocab=32000, E=8, k=2,
    ),
    "Qwen2.5-72B-Instruct": dict(
        name="Qwen-72B", d=8192, h=64, kv=8, ffn=29568, n_layers=80, vocab=152064,
    ),
    "gpt-oss-20b": dict(
        name="gpt-oss-20b", d=6144, h=48, kv=8, ffn=16384, n_layers=44,
        vocab=100352, E=8, k=2,
    ),
    # Qwen3.5 values sourced from `config.json` on gpu-4 (/data/models/Qwen3.5-*).
    # Note: Qwen3.5 is a HYBRID-attention model — only 1/4 of layers are
    # `full_attention`; the other 3/4 are `linear_attention`. The composer in
    # this package assumes full attention at every layer and will over-predict
    # flash_attn time by 4×. Skip Qwen3.5 from composition validation until the
    # composer adds linear-attention support (see composer.predict_layer_ms).
    "Qwen3.5-9B": dict(
        name="Qwen3.5-9B", d=4096, h=16, kv=4, ffn=12288, n_layers=32, vocab=248320,
    ),
    "Qwen3.5-27B": dict(
        name="Qwen3.5-27B", d=5120, h=24, kv=4, ffn=17408, n_layers=64, vocab=248320,
    ),
    # Qwen2.5-7B-Instruct — added 2026-04-24 for RTX2080Ti per-op arch coverage.
    "Qwen2.5-7B-Instruct": dict(
        name="Qwen-7B", d=3584, h=28, kv=4, ffn=18944, n_layers=28, vocab=152064,
    ),
    "Yi-1.5-34B-Chat": dict(
        name="Yi-34B", d=7168, h=56, kv=8, ffn=20480, n_layers=60, vocab=64000,
    ),
    "granite-3.0-8b-instruct": dict(
        name="Granite-8B", d=4096, h=32, kv=8, ffn=12800, n_layers=40, vocab=49152,
    ),
    "gemma-2-9b-it": dict(
        name="Gemma-9B", d=3584, h=16, kv=8, ffn=14336, n_layers=42, vocab=256000,
    ),
}

# Canonical short name per directory (used as training-data `model` column).
_DIR_TO_SHORT: dict[str, str] = {
    "Llama-3.1-8B-Instruct":  "Llama-8B",
    "Llama-3.1-70B-Instruct": "Llama-70B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen2.5-72B-Instruct":   "Qwen-72B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "Qwen3.5-27B":            "Qwen3.5-27B",
    "gpt-oss-20b":            "gpt-oss-20b",
    "Mixtral-8x7B-Instruct":  "Mixtral",
    "Qwen2.5-7B-Instruct":    "Qwen-7B",
    "Yi-1.5-34B-Chat":       "Yi-34B",
    "granite-3.0-8b-instruct": "Granite-8B",
    "gemma-2-9b-it":         "Gemma-9B",
}


def get_model_config(dir_name: str, held_out: bool = False) -> Optional[ModelConfig]:
    """Resolve the ncu directory name to a ModelConfig, or None if unknown."""
    short = _DIR_TO_SHORT.get(dir_name)
    if short is None:
        return None

    fb = _FALLBACK.get(dir_name)
    if fb is not None:
        cfg = ModelConfig(**fb)
        cfg.held_out = held_out
        return cfg
    return None


# ───────── Train/held-out splits per GPU ─────────
HELD_OUT_BY_GPU: dict[str, set[str]] = {
    "A100":     {"Qwen2.5-72B-Instruct", "Llama-3.3-70B-Instruct"},  # Llama-3.1-70B moved to pool (2026-04-24)
    "RTX3090":  {"Qwen2.5-72B-Instruct"},  # Llama-70B moved to training pool (2026-04-24)
    "RTX2080Ti": {"Qwen2.5-7B-Instruct"},   # 3-model LOMO: hold out Qwen-7B (added 2026-04-24).
}


def is_held_out(dir_name: str, gpu: str) -> bool:
    return dir_name in HELD_OUT_BY_GPU.get(gpu, set())


__all__ = [
    "ModelConfig",
    "get_model_config",
    "is_held_out",
    "HELD_OUT_BY_GPU",
]
