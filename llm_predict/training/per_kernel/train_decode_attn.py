"""Train decode_attn XGBoost pkl from per-op decode self_attn rows.

Reads per_op_labeled.csv, filters to (op=="attn", kv_cache_len > 0),
trains a small XGBoost model predicting log(duration_us), saves pkl
compatible with PerKernelPredictor's family format.

Usage:
    python train_decode_attn.py --gpu RTX2080Ti
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


DECODE_ATTN_FEATURES = [
    "bs", "log_bs",
    "kv_cache_len", "log_kv_cache_len",
    "n_heads", "kv_heads", "head_dim",
    "kv_bytes", "log_kv_bytes",
]


def compute_decode_attn_features(row: dict) -> dict:
    bs = float(row["bs"])
    kv_len = float(row["kv_cache_len"])
    n_heads = float(row["h"])
    kv_heads = float(row["kv"])
    head_dim = float(row.get("head_dim", row["d"] / row["h"]))
    kv_bytes = 2.0 * bs * kv_len * kv_heads * head_dim * 2  # K+V, bf16
    return {
        "bs": bs,
        "log_bs": math.log2(bs + 1),
        "kv_cache_len": kv_len,
        "log_kv_cache_len": math.log2(kv_len + 1),
        "n_heads": n_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_bytes": kv_bytes,
        "log_kv_bytes": math.log2(kv_bytes + 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    if pkg_dir.name != "per_kernel":
        pkg_dir = Path(os.environ.get("REPO_ROOT", ".")) / "llm_predict" / "training" / "per_kernel"
    repo_root = pkg_dir.parent.parent.parent

    data_path = Path(args.data) if args.data else repo_root / "llm_predict" / "training" / "per_op" / "data" / "per_op_labeled.csv"
    df = pd.read_csv(data_path)
    decode_attn = df[(df["gpu"] == args.gpu) & (df["op"] == "attn") & (df["kv_cache_len"] > 0)].copy()

    if len(decode_attn) == 0:
        print(f"[{args.gpu}] no decode attn rows — skipping")
        return

    print(f"[{args.gpu}] {len(decode_attn)} decode attn rows")

    feat_rows = []
    targets = []
    for _, row in decode_attn.iterrows():
        feats = compute_decode_attn_features(row.to_dict())
        feat_rows.append([feats[c] for c in DECODE_ATTN_FEATURES])
        targets.append(math.log(row["duration_us"]))

    X = np.array(feat_rows, dtype=float)
    y = np.array(targets, dtype=float)

    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    pred_us = np.exp(preds)
    actual_us = np.exp(y)
    mape = np.mean(np.abs(pred_us - actual_us) / actual_us) * 100
    print(f"[{args.gpu}] train MAPE: {mape:.2f}% ({len(decode_attn)} rows)")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = repo_root / "llm_predict" / "profiling" / "data" / args.gpu / "trained" / "per_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "perkernel_decode_attn_shape_v2.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": list(DECODE_ATTN_FEATURES),
            "n_training": len(decode_attn),
            "train_mape": mape,
            "version": "v2",
            "target": "log_duration_us",
            "gpu": args.gpu,
        }, f)
    print(f"[{args.gpu}] saved {pkl_path}")


if __name__ == "__main__":
    main()
