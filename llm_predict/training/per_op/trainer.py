"""trainer.py — per-GPU per-op XGBoost trainer with LOMO + held-out.

Consumes `data/per_op_labeled.csv` (see `labeler.py` for the schema)
and writes one pkl per GPU at:

    llm_predict/profiling/data/{gpu}/trained/per_op/perop_v5_shape.pkl

The runtime loader at `llm_predict/predictors/per_op/predictor.py`
currently hard-codes `perop_analytical_v4.pkl` — the new v5 path
deliberately does not clobber it. Migration is future work (see README).

Pkl payload (stable contract)
-----------------------------
    {
        "model":           xgb.XGBRegressor,
        "feature_cols":    list[str]   # == feature_spec.PEROP_FEATURES
        "target":          "log_duration_us",
        "gpu":             str,
        "n_training":      int,
        "heldout_mape":    float | None,
        "heldout_models":  list[str],
        "pool_models":     list[str],
        "version":         "per_op_v5",
    }

CLI
---
    python -m llm_predict.training.per_op.trainer \\
        --data    llm_predict/training/per_op/data/per_op_labeled.csv \\
        --out-dir llm_predict/profiling/data \\
        --gpus    A100 RTX3090 RTX2080Ti
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from . import feature_spec
from . import splits as perop_splits


warnings.filterwarnings("ignore")


XGB_PARAMS: dict[str, Any] = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Apply feature_spec.compute_features per row, return (n, 26) matrix."""
    rows = df.to_dict(orient="records")
    return np.array([feature_spec.compute_features(r) for r in rows], dtype=float)


def train_one(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    # Fit raw microseconds directly (matches v4 perop_analytical_v4 scheme).
    # Log-space fit was tried but consistently under-predicts by ~30% on
    # prefill_seq128 held-out; raw us with reg:squarederror is the proven
    # baseline.
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model


def predict_us(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    return np.maximum(0, model.predict(X))


def train_one_gpu(df: pd.DataFrame, gpu: str, out_dir: Path) -> dict:
    """Train a single pkl for one GPU. Returns a report dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_models = perop_splits.lomo_models(df)
    heldout_models = sorted(
        df[df["held_out"].astype(bool)]["model"].dropna().unique().tolist()
    )

    # ───── LOMO CV ─────
    cv_results: dict[str, dict[str, Any]] = {}
    if len(pool_models) >= 2:
        for hold_m, train_df, test_df in perop_splits.lomo_splits(df):
            n_tr, n_te = len(train_df), len(test_df)
            if n_tr < perop_splits.MIN_TRAIN_ROWS or n_te == 0:
                cv_results[hold_m] = {"mape": float("nan"), "n_train": n_tr, "n_test": n_te}
                continue
            X_tr = build_feature_matrix(train_df)
            X_te = build_feature_matrix(test_df)
            y_tr = train_df["duration_us"].values
            y_te = test_df["duration_us"].values
            m = train_one(X_tr, y_tr)
            y_pred = predict_us(m, X_te)
            cv_results[hold_m] = {
                "mape": mape(y_te, y_pred),
                "n_train": n_tr,
                "n_test": n_te,
            }

    # ───── Final model: train on full pool, evaluate on held-out ─────
    train_df, test_df = perop_splits.heldout_split(df, gpu=gpu)
    n_tr, n_te = len(train_df), len(test_df)
    if n_tr < perop_splits.MIN_TRAIN_ROWS:
        return {
            "gpu": gpu, "skipped": True,
            "reason": f"n_train={n_tr} < {perop_splits.MIN_TRAIN_ROWS}",
            "cv_results": cv_results,
            "pool_models": pool_models,
            "heldout_models": heldout_models,
        }

    X_tr = build_feature_matrix(train_df)
    y_tr = train_df["duration_us"].values
    final_model = train_one(X_tr, y_tr)

    heldout_mape: float | None = None
    heldout_per_model: dict[str, float] = {}
    if n_te > 0:
        X_te = build_feature_matrix(test_df)
        y_te = test_df["duration_us"].values
        y_pred = predict_us(final_model, X_te)
        heldout_mape = mape(y_te, y_pred)
        for mdl in heldout_models:
            mask = (test_df["model"] == mdl).values
            if mask.sum() == 0:
                heldout_per_model[mdl] = float("nan")
                continue
            heldout_per_model[mdl] = mape(y_te[mask], y_pred[mask])

    # ───── Persist pkl (stable payload contract) ─────
    pkl_path = out_dir / "perop_v5_shape.pkl"
    payload: dict[str, Any] = {
        "model":          final_model,
        "feature_cols":   list(feature_spec.PEROP_FEATURES),
        "target":         "duration_us",
        "gpu":            gpu,
        "n_training":     int(n_tr),
        "heldout_mape":   heldout_mape,
        "heldout_models": heldout_models,
        "pool_models":    pool_models,
        "version":        "per_op_v5",
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    return {
        "gpu": gpu,
        "skipped": False,
        "saved_pkl": str(pkl_path),
        "n_training": int(n_tr),
        "n_heldout": int(n_te),
        "heldout_mape": heldout_mape,
        "heldout_per_model": heldout_per_model,
        "cv_results": cv_results,
        "pool_models": pool_models,
        "heldout_models": heldout_models,
        "feature_audit": feature_spec.audit_features(feature_spec.PEROP_FEATURES),
    }


def write_report(results: list[dict], report_path: Path) -> None:
    lines: list[str] = ["# Per-Op Training Report (per_op_v5)", ""]

    feats = feature_spec.PEROP_FEATURES
    leaks_any = any(r.get("feature_audit") for r in results)
    lines.append(
        f"- features ({len(feats)}): " + ", ".join(f"`{f}`" for f in feats)
        + (" **LEAK**" if leaks_any else " ✓")
    )
    lines.append("")

    lines.append("## Headline: held-out MAPE per GPU")
    lines.append("")
    lines.append("| GPU | n_train | n_heldout | pool | heldout | MAPE |")
    lines.append("|---|---:|---:|---|---|---:|")
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['gpu']} | — | — | — | — | skipped ({r['reason']}) |")
            continue
        hm = r.get("heldout_mape")
        mape_s = f"{hm:.2f}%" if hm is not None and not np.isnan(hm) else "—"
        lines.append(
            f"| {r['gpu']} | {r['n_training']} | {r['n_heldout']} | "
            f"{', '.join(r['pool_models'])} | "
            f"{', '.join(r['heldout_models']) or '—'} | {mape_s} |"
        )
    lines.append("")

    lines.append("## Per-heldout-model MAPE")
    lines.append("")
    for r in results:
        if r.get("skipped") or not r.get("heldout_per_model"):
            continue
        lines.append(f"### {r['gpu']}")
        lines.append("")
        lines.append("| held-out model | MAPE |")
        lines.append("|---|---:|")
        for mdl, v in r["heldout_per_model"].items():
            if np.isnan(v):
                lines.append(f"| {mdl} | n/a |")
            else:
                lines.append(f"| {mdl} | {v:.2f}% |")
        lines.append("")

    lines.append("## LOMO cross-validation")
    lines.append("")
    for r in results:
        if r.get("skipped") or not r.get("cv_results"):
            lines.append(f"### {r['gpu']} — skipped (<2 pool models)")
            lines.append("")
            continue
        lines.append(f"### {r['gpu']}")
        lines.append("")
        lines.append("| held-out | n_train | n_test | MAPE |")
        lines.append("|---|---:|---:|---:|")
        for hold_m, info in r["cv_results"].items():
            v = info.get("mape", float("nan"))
            mape_s = "n/a" if np.isnan(v) else f"{v:.2f}%"
            lines.append(
                f"| {hold_m} | {info.get('n_train', 0)} | "
                f"{info.get('n_test', 0)} | {mape_s} |"
            )
        lines.append("")

    lines.append("## Saved pkls")
    for r in results:
        if not r.get("skipped"):
            lines.append(f"- **{r['gpu']}**: `{r['saved_pkl']}`")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def run(data_csv: Path, out_dir: Path, report_path: Path, gpus: list[str]) -> list[dict]:
    print(f"[*] Loading {data_csv}")
    df = pd.read_csv(data_csv)
    print(f"[*] {len(df)} rows; GPUs present = {sorted(df['gpu'].unique())}")

    results: list[dict] = []
    for gpu in gpus:
        sub = df[df["gpu"] == gpu].copy()
        if len(sub) == 0:
            print(f"[-] {gpu}: no rows, skipping")
            continue
        print(f"[*] {gpu}: {len(sub)} rows — training...")
        gpu_out_dir = out_dir / gpu / "trained" / "per_op"
        r = train_one_gpu(sub, gpu, gpu_out_dir)
        if r.get("skipped"):
            print(f"    [skip] {r['reason']}")
        else:
            hm = r.get("heldout_mape")
            mape_s = f"{hm:.2f}%" if hm is not None and not np.isnan(hm) else "—"
            print(f"    n_train={r['n_training']} n_heldout={r['n_heldout']} MAPE={mape_s}")
        results.append(r)

    write_report(results, report_path)
    summary_path = report_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump([{
            "gpu": r["gpu"],
            "skipped": r.get("skipped", False),
            "heldout_mape": r.get("heldout_mape"),
            "saved_pkl": r.get("saved_pkl"),
            "pool_models": r.get("pool_models", []),
            "heldout_models": r.get("heldout_models", []),
        } for r in results], f, indent=2, default=str)
    print(f"[*] Report: {report_path}")
    print(f"[*] Summary: {summary_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    llm_predict_dir = pkg_dir.parents[1]
    data_csv = Path(args.data) if args.data else pkg_dir / "data" / "per_op_labeled.csv"
    out_dir = Path(args.out_dir) if args.out_dir else llm_predict_dir / "profiling" / "data"
    report = Path(args.report) if args.report else pkg_dir / "reports" / "training_report.md"

    run(data_csv, out_dir, report, args.gpus)


if __name__ == "__main__":
    main()
