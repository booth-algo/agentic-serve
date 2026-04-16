"""
Feature computation for per-op XGBoost predictor.

These functions compute analytical features from transformer model parameters
that are fed into the per-op XGBoost model.
"""

import math
import numpy as np


def compute_perop_features(n_tokens, op, d_model, n_heads, n_kv_heads, intermediate_size, num_experts, top_k, batch_size=1, seq_len=None, kv_cache_len=0):
    """Compute analytical features for per-op XGBoost prediction.

    For decode (seq=1): kv_cache_len determines attention scan length.
    For prefill (seq>1): kv_cache_len=0, attention is self-attention over seq.
    """
    tok = float(n_tokens); d = float(d_model); h = float(n_heads)
    kv = float(n_kv_heads); ffn = float(intermediate_size)
    E = float(num_experts); k = float(top_k); d_h = d / h

    is_a = float(op == "attn"); is_f = float(op == "ffn"); is_n = float(op in ("norm_pre", "norm_post"))

    tpe = max(1, tok * k / max(E, 1)) if E > 0 else 0
    aexp = min(tok * k, E) if E > 0 else 0

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
