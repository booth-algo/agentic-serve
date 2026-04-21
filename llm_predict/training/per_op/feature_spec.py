"""Feature specification for per-op XGBoost predictor.

Single source of truth for the 26 analytical features the per-op model
consumes. Extracted from two duplicated sites:

1. `experiment/train_perop_v4.py:compute_features_like_transformer()`
   (the legacy ad-hoc training script).
2. `llm_predict/predictors/per_op/features.py:compute_perop_features()`
   (the runtime feature computation).

The runtime feature module stays where it is for now — the predictor
loader in `predictors/per_op/predictor.py` imports from it at serve
time. A later refactor should make that module import this file (see
README). Until then, the two must stay lock-step; `audit_features()`
and the smoke tests catch ordering / count drift.

Layer-walker op mapping
-----------------------
`OP_MAP` mirrors the runtime layer walker in
`llm_predict/models/software/transformer.py` — it is how a labeler
(or transformer forward pass) turns a nested-module name
("self_attn", "mlp", "block_sparse_moe", ...) into one of the four
predictor-op categories: attn / ffn / norm_pre / norm_post.

Leak-guard
----------
`FORBIDDEN_SUBSTRINGS` is intentionally identical to the per-kernel
leak-guard list — any feature whose name contains a raw ncu metric
substring (cycles / throughput / launch / register / occupancy / ...)
indicates hardware-counter leakage the runtime predictor cannot
reproduce from shape alone.
"""
from __future__ import annotations

import math
from typing import Any


# ───────── Op mapping (module name → predictor op category) ─────────
# Must stay consistent with:
#   - llm_predict/models/software/transformer.py (the layer walker)
#   - llm_predict/predictors/per_op/features.py (runtime)
OP_MAP: dict[str, str] = {
    "self_attn":                 "attn",
    "mlp":                       "ffn",
    "block_sparse_moe":          "ffn",
    "input_layernorm":           "norm_pre",
    "post_attention_layernorm":  "norm_post",
}


# ───────── 26-feature vector (order matters — consumed by pkl) ─────────
# The index of a name in this list == its column index in the trained
# XGB model feature_cols. Do NOT reorder without retraining.
PEROP_FEATURES: list[str] = [
    # v3 core (20): tok / flops / weight bytes / arithmetic intensity / op one-hots / arch
    "tok", "log_tok",
    "flops", "log_flops",
    "wt_bytes", "log_wt",
    "ai", "log_ai",
    "is_attn", "is_ffn", "is_norm",
    "d", "ffn", "E", "k",
    "d2", "d_ffn",
    "eff_tok", "a_exp", "tot_wt",
    # v4 additions (6): bs / seq / attention-quadratic term
    "bs", "seq",
    "log_bs", "log_seq",
    "attn_quad", "log_attn_quad",
]


# ───────── Leak-guard (identical to per_kernel) ─────────
FORBIDDEN_SUBSTRINGS: list[str] = [
    "cycles", "throughput", "launch_", "dram_bytes_sum", "dram_bytes_read",
    "dram_bytes_write", "register", "occupancy", "sm_active",
    "compute_throughput", "memory_throughput",
]


def audit_features(feature_cols: list[str]) -> list[tuple[str, str]]:
    """Returns [(feat, matched_bad_substring), ...] if any feature name
    contains a forbidden substring. Empty list ⇒ no leaks."""
    leaks: list[tuple[str, str]] = []
    for feat in feature_cols:
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in feat.lower():
                leaks.append((feat, bad))
    return leaks


# ───────── compute_features(row) → list[float] of length 26 ─────────

def compute_features(row: dict[str, Any]) -> list[float]:
    """Compute the 26-dim analytical feature vector for a single (op, shape, arch) row.

    Parameters
    ----------
    row : dict
        Must contain keys (architecture + shape + op):
            op:           "attn" | "ffn" | "norm_pre" | "norm_post"
            n_tokens:     int — bs * seq during prefill; bs during decode
            bs:           int — batch size
            seq:          int — sequence length (per-batch)
            d:            int — hidden_size
            h:            int — num_attention_heads
            kv:           int — num_key_value_heads
            ffn:          int — intermediate_size
            E:            int — num_local_experts (0 or 1 = dense)
            k:            int — num_experts_per_tok (0 or 1 = dense)
            kv_cache_len: int — decode-phase KV cache length (0 for prefill)

    Returns
    -------
    list[float] of length `len(PEROP_FEATURES)`.
    """
    op = row["op"]
    tok = float(row["n_tokens"])
    d = float(row["d"])
    h = float(row["h"])
    kv = float(row["kv"])
    ffn = float(row["ffn"])
    E = float(row.get("E", 0))
    k = float(row.get("k", 0))
    bs_f = float(row.get("bs", 1))
    seq_f = float(row.get("seq", tok))
    kv_cache_len = float(row.get("kv_cache_len", 0))

    d_h = d / h if h > 0 else d

    is_a = float(op == "attn")
    is_f = float(op == "ffn")
    is_n = float(op in ("norm_pre", "norm_post"))

    # MoE bookkeeping (0 for dense models).
    tpe = max(1, tok * k / max(E, 1)) if E > 0 else 0
    aexp = min(tok * k, E) if E > 0 else 0

    # Decode: attention scans the KV cache; prefill: self-attention over seq.
    effective_kv = max(kv_cache_len, seq_f) if kv_cache_len > 0 else seq_f

    attn_fl = is_a * (2 * tok * d * (d + 2 * kv * d_h)
                      + 4 * bs_f * h * seq_f * effective_kv * d_h
                      + 2 * tok * d * d)
    dffn_fl = is_f * (1 - min(E, 1)) * 2 * tok * d * ffn * 3
    mffn_fl = is_f * (2 * tpe * d * ffn * 3 * aexp if E > 0 else 0)
    norm_fl = is_n * tok * d * 5
    total_fl = attn_fl + dffn_fl + mffn_fl + norm_fl

    wt = (is_a * d * (d + 2 * kv * d_h + d) * 2
          + is_f * (E * d * ffn * 3 * 2 if E > 0 else d * ffn * 3 * 2)
          + is_n * d * 2)
    ai = total_fl / wt if wt > 0 else 0

    attn_quad = is_a * bs_f * seq_f * effective_kv * h * d_h

    return [
        tok, math.log2(tok + 1),
        total_fl, math.log2(total_fl + 1),
        wt, math.log2(wt + 1),
        ai, math.log2(ai + 1),
        is_a, is_f, is_n,
        d, ffn, E, k,
        d * d, d * ffn,
        tpe if E > 0 else tok,
        aexp if E > 0 else 1,
        wt,
        bs_f, seq_f,
        math.log2(bs_f + 1), math.log2(seq_f + 1),
        attn_quad, math.log2(attn_quad + 1) if attn_quad > 0 else 0,
    ]


__all__ = [
    "OP_MAP",
    "PEROP_FEATURES",
    "FORBIDDEN_SUBSTRINGS",
    "audit_features",
    "compute_features",
]
