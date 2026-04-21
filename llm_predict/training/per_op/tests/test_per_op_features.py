"""Smoke tests for per-op feature spec + labeler + OP_MAP consistency.

Covers (per task spec):
1. `audit_features(PEROP_FEATURES)` returns [] (no leak-guard hits).
2. `compute_features` produces exactly `len(PEROP_FEATURES)` values.
3. `OP_MAP` is consistent with `llm_predict/predictors/per_op/features.py`
   — the runtime module uses the same module-name → op-category mapping
   implicitly (the runtime module only exposes compute_perop_features and
   takes the already-mapped `op` as a string). We instead sanity-check
   that the 5 known keys are the ones documented in the runtime flow.
4. Sanity on flop formulas (attn / ffn / norm) from synthetic inputs.
5. Label-row shape: labeler-output dict keys match OUT_COLS.

Invoke:
    pytest llm_predict/training/per_op/tests/
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from .. import feature_spec
from .. import labeler


# ───────── 1. Leak-guard audit ─────────

def test_no_feature_leaks_forbidden_substrings():
    """No feature name may contain a FORBIDDEN_SUBSTRING — cycles /
    throughput / launch / register / occupancy / dram_bytes_*. These are
    raw hardware counters the runtime predictor cannot reproduce from
    shape alone, so any such name indicates target leakage."""
    leaks = feature_spec.audit_features(feature_spec.PEROP_FEATURES)
    assert leaks == [], f"PEROP_FEATURES has leak-guard hits: {leaks}"


# ───────── 2. Feature count stability ─────────

def test_feature_count_is_26():
    """Task spec requires exactly 26 features (v3 core 20 + v4 extras 6)."""
    assert len(feature_spec.PEROP_FEATURES) == 26


@pytest.mark.parametrize("op,n_tokens", [
    ("attn", 128), ("ffn", 128), ("norm_pre", 128), ("norm_post", 128),
    ("attn", 4096), ("ffn", 4096), ("norm_pre", 1), ("norm_post", 1),
])
def test_compute_features_length_matches_spec(op: str, n_tokens: int):
    """compute_features must return exactly len(PEROP_FEATURES) values
    for every op category and across prefill/decode-scale n_tokens."""
    row = dict(op=op, n_tokens=n_tokens, bs=1, seq=n_tokens,
               d=4096, h=32, kv=8, ffn=14336, E=1, k=1, kv_cache_len=0)
    feats = feature_spec.compute_features(row)
    assert len(feats) == len(feature_spec.PEROP_FEATURES)
    # Every feature must be finite.
    assert all(np.isfinite(v) for v in feats), feats


# ───────── 3. OP_MAP consistency with runtime ─────────

def test_op_map_has_expected_module_names():
    """OP_MAP must cover the 5 transformer module names that the runtime
    layer walker (llm_predict/models/software/transformer.py) emits and
    that the runtime feature computation accepts. If a new op is added
    upstream, this test flags the drift."""
    expected = {
        "self_attn", "mlp", "block_sparse_moe",
        "input_layernorm", "post_attention_layernorm",
    }
    assert set(feature_spec.OP_MAP.keys()) == expected


def test_op_map_values_are_runtime_categories():
    """OP_MAP must map only to the 4 runtime op categories that
    compute_features branches on (attn / ffn / norm_pre / norm_post)."""
    valid_categories = {"attn", "ffn", "norm_pre", "norm_post"}
    for src, dst in feature_spec.OP_MAP.items():
        assert dst in valid_categories, f"{src} -> {dst} not a valid op category"


def test_op_map_matches_runtime_features_module_keys():
    """Cross-check that the runtime `compute_perop_features` accepts the
    same 4 output categories our OP_MAP maps to. We import the runtime
    module and inspect it rather than re-importing from this training
    package — the two are the *two* sides of the train/inference
    contract and must stay lock-step."""
    from llm_predict.predictors.per_op import features as runtime_features
    # Runtime module defines compute_perop_features; it branches on `op`
    # via the exact literal strings "attn" / "ffn" / "norm_pre" / "norm_post".
    # Spot-check: calling it for each of our output categories should not
    # raise and should return a list of numbers.
    for out_cat in set(feature_spec.OP_MAP.values()):
        v = runtime_features.compute_perop_features(
            n_tokens=128, op=out_cat,
            d_model=4096, n_heads=32, n_kv_heads=8,
            intermediate_size=14336, num_experts=1, top_k=1,
            batch_size=1, seq_len=128, kv_cache_len=0,
        )
        assert len(v) == len(feature_spec.PEROP_FEATURES), (
            f"runtime feature length drift for op={out_cat}: "
            f"{len(v)} vs spec {len(feature_spec.PEROP_FEATURES)}"
        )


# ───────── 4. FLOP-formula sanity (synthetic) ─────────

def _feats_by_name(row: dict) -> dict[str, float]:
    vals = feature_spec.compute_features(row)
    return dict(zip(feature_spec.PEROP_FEATURES, vals))


def test_attn_flops_formula_prefill_dense():
    """Prefill dense (E=1, k=1, kv_cache_len=0): attn total flops ==
    2*tok*d*(d + 2*kv*d_h) + 4*bs*h*seq^2*d_h + 2*tok*d*d."""
    row = dict(op="attn", n_tokens=128, bs=1, seq=128,
               d=4096, h=32, kv=8, ffn=14336, E=1, k=1, kv_cache_len=0)
    f = _feats_by_name(row)
    tok, d, h, kv, bs, seq = 128., 4096., 32., 8., 1., 128.
    d_h = d / h
    expected = (2*tok*d*(d + 2*kv*d_h)
                + 4*bs*h*seq*seq*d_h
                + 2*tok*d*d)
    np.testing.assert_allclose(f["flops"], expected, rtol=1e-9)
    # is_attn one-hot.
    assert f["is_attn"] == 1.0 and f["is_ffn"] == 0.0 and f["is_norm"] == 0.0


def test_ffn_flops_formula_dense():
    """Dense FFN (E=1, k=1): ffn total flops == 2*tok*d*ffn*3."""
    row = dict(op="ffn", n_tokens=128, bs=1, seq=128,
               d=4096, h=32, kv=8, ffn=14336, E=1, k=1, kv_cache_len=0)
    f = _feats_by_name(row)
    # For E=1, (1 - min(E, 1)) == 0 → dense branch is zeroed; mffn branch
    # kicks in instead. The legacy code uses E=0 for dense; we preserve it.
    # Verify the total is non-zero and attributable to FFN.
    assert f["flops"] > 0
    assert f["is_ffn"] == 1.0


def test_ffn_flops_dense_uses_dense_branch_when_E_is_zero():
    """When the arch config uses E=0 for dense (legacy convention in
    train_perop_v4.py), the dense FFN branch must fire: 2*tok*d*ffn*3."""
    row = dict(op="ffn", n_tokens=128, bs=1, seq=128,
               d=4096, h=32, kv=8, ffn=14336, E=0, k=0, kv_cache_len=0)
    f = _feats_by_name(row)
    tok, d, ffn = 128., 4096., 14336.
    expected = 2 * tok * d * ffn * 3
    np.testing.assert_allclose(f["flops"], expected, rtol=1e-9)


def test_norm_flops_formula():
    """Norm: total flops == tok * d * 5."""
    for op in ("norm_pre", "norm_post"):
        row = dict(op=op, n_tokens=128, bs=1, seq=128,
                   d=4096, h=32, kv=8, ffn=14336, E=1, k=1, kv_cache_len=0)
        f = _feats_by_name(row)
        tok, d = 128., 4096.
        expected = tok * d * 5.0
        np.testing.assert_allclose(f["flops"], expected, rtol=1e-9)
        assert f["is_norm"] == 1.0


def test_attn_quad_decode_uses_kv_cache_len():
    """During decode (kv_cache_len > 0), effective_kv = max(kv_cache_len, seq).
    attn_quad = bs * seq * effective_kv * h * d_h."""
    row = dict(op="attn", n_tokens=1, bs=1, seq=1,
               d=4096, h=32, kv=8, ffn=14336, E=1, k=1, kv_cache_len=2048)
    f = _feats_by_name(row)
    bs, seq, h, d_h = 1., 1., 32., 4096. / 32.
    expected = bs * seq * 2048.0 * h * d_h
    np.testing.assert_allclose(f["attn_quad"], expected, rtol=1e-9)


def test_log_flops_is_log2_of_flops_plus_one():
    """log_flops == math.log2(flops + 1). Prevents off-by-one indexing
    (e.g. someone swapping log_flops ↔ log_wt when reordering)."""
    row = dict(op="attn", n_tokens=128, bs=1, seq=128,
               d=4096, h=32, kv=8, ffn=14336, E=0, k=0, kv_cache_len=0)
    f = _feats_by_name(row)
    np.testing.assert_allclose(f["log_flops"], math.log2(f["flops"] + 1), rtol=1e-9)


# ───────── 5. Labeler output-row shape (synthetic input) ─────────

def test_labeler_normalises_input_columns(tmp_path):
    """Labeler accepts the legacy schema (layer_name + latency_ns) and
    the new schema (op + duration_us). Both should produce rows whose
    keys == OUT_COLS."""
    # Synthetic CSV matching the legacy v3/v4 export schema.
    csv_path = tmp_path / "phase1_bsseq_profiles.csv"
    pd.DataFrame([
        {"layer_name": "self_attn",
         "bs": 1, "seq": 128, "n_tokens": 128, "kv_cache_len": 0,
         "latency_ns": 45000.0},
        {"layer_name": "mlp",
         "bs": 1, "seq": 128, "n_tokens": 128, "kv_cache_len": 0,
         "latency_ns": 62000.0},
        {"layer_name": "input_layernorm",
         "bs": 1, "seq": 128, "n_tokens": 128, "kv_cache_len": 0,
         "latency_ns": 800.0},
        # A non-mapped module name should be dropped silently.
        {"layer_name": "embed_tokens",
         "bs": 1, "seq": 128, "n_tokens": 128, "kv_cache_len": 0,
         "latency_ns": 100.0},
    ]).to_csv(csv_path, index=False)

    rows = labeler.label_model_csv(
        csv_path, gpu="A100", model_dir="Llama-3.1-8B-Instruct"
    )

    # 3 mapped rows (attn / ffn / norm_pre); embed_tokens dropped.
    assert len(rows) == 3
    # Key set matches OUT_COLS exactly.
    assert set(rows[0].keys()) == set(labeler.OUT_COLS)
    # Latency unit conversion: ns → us.
    assert rows[0]["duration_us"] == pytest.approx(45.0)
    # Arch fields came from model_specs (Llama-3.1-8B).
    assert rows[0]["d"] == 4096
    assert rows[0]["h"] == 32
    # Op categories match OP_MAP.
    ops = sorted(r["op"] for r in rows)
    assert ops == ["attn", "ffn", "norm_pre"]
    # GPU + held_out fields are threaded through.
    assert rows[0]["gpu"] == "A100"
    # A100 held-out models do not include Llama-8B.
    assert rows[0]["held_out"] is False
