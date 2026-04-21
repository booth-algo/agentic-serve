"""Smoke tests for per-kernel feature builders + split invariants.

Two kinds of tests:

1. **Pure-builder unit tests** (always run) — drive
   `feature_spec.build_{gemm,flash_attn,elementwise,misc}_features` with
   synthetic DataFrames and assert the expected derived columns.

2. **Data-dependent integration tests** (skip if `data/per_family/` is
   missing) — load real per-family CSVs and assert downstream
   invariants: target positivity, Qwen3.5 absence from flash_attn,
   one-hot exclusivity on real rows.

Invoke:
    pytest llm_predict/training/per_kernel/tests/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .. import feature_spec
from .. import splits


PKG_DIR = Path(__file__).resolve().parent.parent
PER_FAMILY_DIR = PKG_DIR / "data" / "per_family"


# ───────── Feature-spec leak guards ─────────

def test_no_feature_leaks_forbidden_substrings():
    """No feature name across any family may contain a FORBIDDEN substring
    (cycles/throughput/launch_/dram_bytes_/register/occupancy/etc).
    Those are hardware counters the runtime predictor cannot reproduce
    from shape alone — including them would be target leakage."""
    for fam, cfg in feature_spec.FAMILY_CONFIG.items():
        leaks = feature_spec.audit_features(cfg["features"])
        assert leaks == [], f"{fam} leaks: {leaks}"


def test_raw_cols_contain_target():
    """Every family's raw column list must include target_us + the target
    source. Downstream consumers depend on this invariant."""
    for fam, cfg in feature_spec.FAMILY_CONFIG.items():
        assert "target_us" in cfg["raw_cols"], fam
        assert "gpu_time_duration_ms" in cfg["raw_cols"], fam


# ───────── Pure builder tests (synthetic data) ─────────

def test_build_gemm_features_log_flops_monotonic_in_mnk():
    """log_flops must be monotonic non-decreasing in M*N*K when sorted by
    that product. Sanity: it's log(2*M*N*K) in the limit."""
    df = pd.DataFrame({
        "M":     [64,  128, 256,  512, 1024, 2048],
        "N":     [64,  128, 256,  512, 1024, 2048],
        "K":     [64,  128, 256,  512, 1024, 2048],
        "dtype": ["bf16"] * 6,
    })
    out = feature_spec.build_gemm_features(df)
    flops = (out["M"] * out["N"] * out["K"]).astype(float).values
    log_flops = out["log_flops"].values
    order = np.argsort(flops)
    assert np.all(np.diff(log_flops[order]) >= 0), \
        f"log_flops not monotonic in M*N*K: {log_flops[order]}"


def test_build_gemm_features_analytical_flops_formula():
    """analytical_flops must equal 2 * M * N * K exactly."""
    df = pd.DataFrame({"M": [128, 256], "N": [4096, 8192], "K": [4096, 4096],
                       "dtype": ["bf16", "bf16"]})
    out = feature_spec.build_gemm_features(df)
    expected = 2.0 * df["M"] * df["N"] * df["K"]
    np.testing.assert_array_almost_equal(
        out["analytical_flops"].values, expected.values, decimal=1
    )


def test_build_gemm_features_dtype_onehot():
    """dtype_onehot_bf16 is 1 for bf16, 0 otherwise."""
    df = pd.DataFrame({"M": [128, 128], "N": [128, 128], "K": [128, 128],
                       "dtype": ["bf16", "fp16"]})
    out = feature_spec.build_gemm_features(df)
    assert out["dtype_onehot_bf16"].tolist() == [1, 0]


def test_build_gemm_features_drops_nan_shapes():
    """Rows with any of M/N/K as NaN (e.g. gemv rows) must be dropped."""
    df = pd.DataFrame({
        "M":     [128, np.nan, 256],
        "N":     [128, 128,    256],
        "K":     [128, 128,    np.nan],
        "dtype": ["bf16", "bf16", "bf16"],
    })
    out = feature_spec.build_gemm_features(df)
    assert len(out) == 1


def test_build_flash_attn_features_total_flops():
    """total_flops must equal 4 * bs * n_heads * seq^2 * head_dim."""
    df = pd.DataFrame({
        "bs":       [1, 1],
        "seq":      [128, 256],
        "n_heads":  [32, 64],
        "head_dim": [128, 128],
        "kv_heads": [8, 8],
    })
    out = feature_spec.build_flash_attn_features(df)
    expected = 4.0 * df["bs"] * df["n_heads"] * (df["seq"] ** 2) * df["head_dim"]
    np.testing.assert_array_almost_equal(
        out["total_flops"].values, expected.values, decimal=1
    )


def test_build_elementwise_features_onehot_exactly_one():
    """For each row, exactly one op_type_onehot_* is 1 (known op) OR the
    'other' bucket is 1 (unknown op). Known ops should NOT set 'other'=1."""
    df = pd.DataFrame({
        "op_type": ["rmsnorm", "silu", "mul", "totally_unknown_op"],
        "numel":   [1024, 1024, 1024, 1024],
    })
    out = feature_spec.build_elementwise_features(df)
    onehot_cols = [f"op_type_onehot_{op}" for op in feature_spec.ELEM_OPS]
    row_sums = out[onehot_cols].sum(axis=1).values
    # Every row should have exactly 1.0 active across the one-hot block.
    np.testing.assert_array_equal(row_sums, [1.0, 1.0, 1.0, 1.0])
    # Known-op rows set only their own onehot.
    assert out.loc[0, "op_type_onehot_rmsnorm"] == 1
    assert out.loc[1, "op_type_onehot_silu"] == 1
    # Unknown op is routed to 'other'.
    assert out.loc[3, "op_type_onehot_other"] == 1


def test_build_misc_features_kernel_family_onehot_exactly_one():
    """For each row, exactly one kernel_family_onehot_* is 1."""
    df = pd.DataFrame({
        "kernel_family": feature_spec.MISC_FAMILIES,
        "M": [128] * 4, "N": [128] * 4, "K": [128] * 4, "numel": [0] * 4,
    })
    out = feature_spec.build_misc_features(df)
    onehot_cols = [f"kernel_family_onehot_{fam}"
                   for fam in feature_spec.MISC_FAMILIES]
    row_sums = out[onehot_cols].sum(axis=1).values
    np.testing.assert_array_equal(row_sums, [1, 1, 1, 1])


def test_build_misc_features_size_proxy():
    """numel_or_shape_total uses numel when available, else M*N."""
    df = pd.DataFrame({
        "kernel_family": ["reduce", "splitk_reduce"],
        "M":     [0.0,  256.0],
        "N":     [0.0,  256.0],
        "K":     [0.0,  256.0],
        "numel": [4096, 0.0],
    })
    out = feature_spec.build_misc_features(df)
    # Row 0: numel=4096 wins.
    assert out.loc[0, "numel_or_shape_total"] == 4096
    # Row 1: numel=0 → fallback to M*N = 256*256.
    assert out.loc[1, "numel_or_shape_total"] == 256 * 256


# ───────── Data-dependent integration tests ─────────

REQUIRES_DATA = pytest.mark.skipif(
    not PER_FAMILY_DIR.is_dir(),
    reason=(
        f"per-family CSVs not found at {PER_FAMILY_DIR}. "
        "Run `python -m llm_predict.training.per_kernel.split_by_family` first."
    ),
)


@REQUIRES_DATA
@pytest.mark.parametrize("gpu,family", [
    ("A100", "gemm"), ("A100", "flash_attn"),
    ("A100", "elementwise"), ("A100", "misc"),
    ("RTX3090", "gemm"), ("RTX3090", "flash_attn"),
    ("RTX3090", "elementwise"), ("RTX3090", "misc"),
    ("RTX2080Ti", "gemm"), ("RTX2080Ti", "elementwise"), ("RTX2080Ti", "misc"),
])
def test_target_positive(gpu: str, family: str):
    """target_us > 0 for every row in every per-family CSV."""
    df = splits.load_per_family(PER_FAMILY_DIR, gpu, family)
    assert df is not None and len(df) > 0, f"missing {gpu}/{family}"
    assert (df["target_us"] > 0).all(), (
        f"non-positive target_us in {gpu}/{family}: "
        f"{df.loc[df['target_us'] <= 0].head().to_dict()}"
    )


@REQUIRES_DATA
@pytest.mark.parametrize("gpu", ["A100", "RTX3090"])
def test_qwen35_excluded_from_flash_attn(gpu: str):
    """Per .claude/paper/predictor_notes.md — Qwen3.5-{9B,27B} must not
    appear in any GPU's flash_attn.csv. They're excluded by the splitter
    because hybrid attention only emits flash_fwd on 1/4 of layers."""
    df = splits.load_per_family(PER_FAMILY_DIR, gpu, "flash_attn")
    assert df is not None and len(df) > 0
    banned = {"Qwen3.5-9B", "Qwen3.5-27B"}
    found = set(df["model"].dropna().unique()) & banned
    assert found == set(), f"{gpu}/flash_attn contains banned models: {found}"


@REQUIRES_DATA
@pytest.mark.parametrize("gpu,family", [
    ("A100", "gemm"), ("RTX3090", "gemm"),
])
def test_real_gemm_log_flops_monotonic(gpu: str, family: str):
    """After applying build_gemm_features to real rows, log_flops sorted by
    analytical_flops is monotonic non-decreasing."""
    df = splits.load_per_family(PER_FAMILY_DIR, gpu, family)
    out = feature_spec.build_gemm_features(df)
    order = np.argsort(out["analytical_flops"].values)
    log_flops = out["log_flops"].values[order]
    diffs = np.diff(log_flops)
    assert np.all(diffs >= -1e-9), (
        f"log_flops not monotonic in {gpu}/{family} after sort: "
        f"{diffs[diffs < -1e-9][:5]}"
    )


@REQUIRES_DATA
@pytest.mark.parametrize("gpu", ["A100", "RTX3090", "RTX2080Ti"])
def test_real_misc_onehot_exactly_one(gpu: str):
    """Every row in real misc.csv produces exactly one
    kernel_family_onehot_* = 1 after the builder."""
    df = splits.load_per_family(PER_FAMILY_DIR, gpu, "misc")
    assert df is not None and len(df) > 0
    out = feature_spec.build_misc_features(df)
    onehot_cols = [f"kernel_family_onehot_{fam}"
                   for fam in feature_spec.MISC_FAMILIES]
    row_sums = out[onehot_cols].sum(axis=1).values
    # Rows in misc.csv have kernel_family restricted to MISC_FAMILIES by
    # split_by_family.py, so every row should hit exactly one.
    assert np.all(row_sums == 1), (
        f"{gpu}/misc: rows with onehot sum != 1: "
        f"{sorted(set(row_sums.tolist()))}"
    )


# ───────── Split iterator invariants (synthetic) ─────────

def test_lomo_splits_disjoint():
    """LOMO train and test should be disjoint by model. Every non-held-out
    model should appear as a test-split exactly once."""
    df = pd.DataFrame({
        "model":    ["A", "A", "B", "B", "C", "C", "D"],
        "held_out": [False] * 6 + [True],
    })
    folds = list(splits.lomo_splits(df))
    assert len(folds) == 3  # A, B, C (not D — held out)
    seen = set()
    for hold_m, tr, te in folds:
        assert hold_m not in tr["model"].values, \
            f"hold_m {hold_m} leaked into train"
        assert set(te["model"].unique()) == {hold_m}
        assert "D" not in tr["model"].values, "held-out in train"
        assert "D" not in te["model"].values, "held-out in test"
        seen.add(hold_m)
    assert seen == {"A", "B", "C"}


def test_heldout_split_partition():
    """heldout_split must partition the input: len(train) + len(test) = len(df)."""
    df = pd.DataFrame({
        "model": ["X", "Y", "Z", "W"],
        "held_out": [False, False, True, True],
    })
    tr, te = splits.heldout_split(df)
    assert len(tr) == 2 and len(te) == 2
    assert set(tr["model"]) == {"X", "Y"}
    assert set(te["model"]) == {"Z", "W"}
