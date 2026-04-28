"""Phase A2+A3 trainer — per-(GPU × family) shape-only XGBoost.

Port of `/root/per-kernel-rebuild/train_shape_v2.py`. Changes:
- Outer loop over GPUs (A100 / RTX3090 / RTX2080Ti); one pkl per
  (GPU × family) lands at `llm_predict/profiling/data/{gpu}/trained/per_kernel/`.
- Held-out model list driven by `model_specs.HELD_OUT_BY_GPU`.
- Feature columns match `llm_predict/predictors/per_kernel/predictor.py` so the
  runtime API loads the trained pkls without adaptation.
- Skip-family guard for sparse training pools (e.g. 2080Ti misc).

CLI
---
    python -m llm_predict.training.per_kernel.trainer \\
        --data    llm_predict/training/per_kernel/data/kernels_labeled.csv \\
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

from . import model_specs  # noqa: F401  (kept for parity with labeler; used indirectly)
from llm_predict.training.per_kernel.ensure_data import ensure_kernels_csv
from .feature_spec import (
    FAMILY_CONFIG,
    FAMILY_EXCLUDED_MODELS,
    MISC_FAMILIES,
    audit_features,  # noqa: F401  (re-exported for legacy callers)
)

warnings.filterwarnings("ignore")


XGB_PARAMS: dict[str, Any] = dict(
    n_estimators=800, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=4,
    tree_method="hist", verbosity=0,
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ───────── Core train / predict ─────────

FAMILY_XGB_OVERRIDES: dict[str, dict[str, Any]] = {
    # misc is heterogeneous (4 subfamilies) and has 24K rows — more capacity.
    "misc": dict(n_estimators=1200, max_depth=10),
    # flash_attn is data-poor (~112 rows) — smaller model to avoid overfit.
    "flash_attn": dict(n_estimators=400, max_depth=5),
}


def train_one(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str],
                family: str | None = None) -> xgb.XGBRegressor:
    y_log = np.log(np.maximum(y, 1e-6))
    params = dict(XGB_PARAMS)
    if family and family in FAMILY_XGB_OVERRIDES:
        params.update(FAMILY_XGB_OVERRIDES[family])
    model = xgb.XGBRegressor(**params)
    model.fit(X[feature_cols].values, y_log)
    return model


def predict_ms(model: xgb.XGBRegressor, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    y_log_pred = model.predict(X[feature_cols].values)
    return np.exp(y_log_pred)


MIN_TRAIN_ROWS = 5


# ───────── Per-GPU orchestrator ─────────

def train_one_gpu(df: pd.DataFrame, gpu: str, out_dir: Path) -> dict:
    """Train 4 pkls for a single GPU. Returns a report dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype_for_pkl = "fp16" if gpu == "RTX2080Ti" else "bf16"

    family_frames: dict[str, pd.DataFrame] = {}
    for fam, cfg in FAMILY_CONFIG.items():
        sub = df[df["kernel_family"].isin(cfg["kernel_families"])].copy()
        # Per-family model exclusions per .claude/paper/predictor_notes.md
        # (e.g. Qwen3.5 dropped from flash_attn — hybrid attention).
        excluded = FAMILY_EXCLUDED_MODELS.get(fam, set())
        if excluded:
            sub = sub[~sub["model"].isin(excluded)]
        family_frames[fam] = cfg["builder"](sub)

    training_models = sorted(set(df[~df["held_out"].astype(bool)]["model"].unique()))
    cv_results: dict[str, dict[str, Any]] = {}
    if len(training_models) >= 2:
        for hold_mdl in training_models:
            cv_results[hold_mdl] = {}
            for fam, cfg in FAMILY_CONFIG.items():
                data = family_frames[fam]
                train_data = data[(~data["held_out"].astype(bool)) & (data["model"] != hold_mdl)].copy()
                test_data  = data[(~data["held_out"].astype(bool)) & (data["model"] == hold_mdl)].copy()
                n_tr, n_te = len(train_data), len(test_data)
                if n_tr < MIN_TRAIN_ROWS or n_te == 0:
                    cv_results[hold_mdl][fam] = {"mape": float("nan"), "n_train": n_tr, "n_test": n_te}
                    continue
                m = train_one(train_data, train_data["gpu_time_duration_ms"].values, cfg["features"], family=fam)
                y_pred = predict_ms(m, test_data, cfg["features"])
                cv_results[hold_mdl][fam] = {"mape": mape(test_data["gpu_time_duration_ms"].values, y_pred),
                                              "n_train": n_tr, "n_test": n_te}

    final_models: dict[str, xgb.XGBRegressor] = {}
    heldout: dict[str, dict[str, Any]] = {}
    heldout_per_model: dict[str, dict[str, float]] = {}
    heldout_list = sorted(set(df[df["held_out"].astype(bool)]["model"].unique()))

    for fam, cfg in FAMILY_CONFIG.items():
        data = family_frames[fam]
        train_data = data[~data["held_out"].astype(bool)].copy()
        test_data  = data[data["held_out"].astype(bool)].copy()
        n_tr, n_te = len(train_data), len(test_data)
        if n_tr < MIN_TRAIN_ROWS:
            heldout[fam] = {"skipped": True, "reason": f"n_train={n_tr} < {MIN_TRAIN_ROWS}",
                            "n_train": n_tr, "n_test": n_te}
            continue
        model = train_one(train_data, train_data["gpu_time_duration_ms"].values, cfg["features"], family=fam)
        final_models[fam] = model

        if n_te > 0:
            y_pred = predict_ms(model, test_data, cfg["features"])
            heldout[fam] = {"mape": mape(test_data["gpu_time_duration_ms"].values, y_pred),
                            "n_train": n_tr, "n_test": n_te,
                            "y_test_sum": float(test_data["gpu_time_duration_ms"].sum()),
                            "y_pred_sum": float(y_pred.sum())}
            per_m = {}
            for mdl in heldout_list:
                mask = (test_data["model"] == mdl).values
                if mask.sum() == 0:
                    per_m[mdl] = float("nan")
                    continue
                per_m[mdl] = mape(test_data["gpu_time_duration_ms"].values[mask], y_pred[mask])
            heldout_per_model[fam] = per_m
        else:
            heldout[fam] = {"mape": float("nan"), "n_train": n_tr, "n_test": 0, "no_holdout": True}

    saved: list[str] = []
    for fam, model in final_models.items():
        cfg = FAMILY_CONFIG[fam]
        pkl_path = out_dir / f"perkernel_{fam}_shape_v2.pkl"

        if fam == "misc":
            # Train one sub-model per misc subfamily for more accurate per-shape
            # predictions. Payload shape: `{models: {subfamily: xgb}, ...}`.
            sub_models: dict[str, xgb.XGBRegressor] = {}
            sub_train_counts: dict[str, int] = {}
            data = family_frames[fam]
            sub_train_data = data[~data["held_out"].astype(bool)]
            for sub in MISC_FAMILIES:
                sub_df = sub_train_data[sub_train_data["kernel_family"] == sub]
                if len(sub_df) < MIN_TRAIN_ROWS:
                    continue
                sub_m = train_one(sub_df,
                                    sub_df["gpu_time_duration_ms"].values,
                                    cfg["features"], family="misc")
                sub_models[sub] = sub_m
                sub_train_counts[sub] = len(sub_df)
            payload = {
                "model": model,                          # back-compat
                "models": sub_models,                    # per-subfamily
                "per_subfamily_n_training": sub_train_counts,
                "feature_cols": list(cfg["features"]),
                "target": "log_gpu_time_duration_ms",
                "kernel_family": fam,
                "gpu": gpu,
                "dtype": dtype_for_pkl,
                "n_training": heldout[fam].get("n_train", 0),
                "heldout_mape": heldout[fam].get("mape"),
                "version": "shape_v3_per_subfamily",
            }
        else:
            payload = {
                "model": model,
                "feature_cols": list(cfg["features"]),
                "target": "log_gpu_time_duration_ms",
                "kernel_family": fam,
                "gpu": gpu,
                "dtype": dtype_for_pkl,
                "n_training": heldout[fam].get("n_train", 0),
                "heldout_mape": heldout[fam].get("mape"),
                "version": "shape_v2",
            }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        saved.append(str(pkl_path))

    agg_rows: list[tuple] = []
    for mdl in heldout_list:
        total_pred = 0.0
        total_meas = 0.0
        per_fam_detail: dict[str, Any] = {}
        for fam, cfg in FAMILY_CONFIG.items():
            data = family_frames[fam]
            test = data[(data["held_out"].astype(bool)) & (data["model"] == mdl)].copy()
            if len(test) == 0:
                continue
            model = final_models.get(fam)
            if model is None:
                continue
            y_pred = predict_ms(model, test, cfg["features"])
            y_meas = test["gpu_time_duration_ms"].values
            total_pred += y_pred.sum()
            total_meas += y_meas.sum()
            per_fam_detail[fam] = (float(y_pred.sum()), float(y_meas.sum()), len(test))
        err_pct = abs(total_pred - total_meas) / max(total_meas, 1e-9) * 100.0
        agg_rows.append((mdl, total_pred, total_meas, err_pct, per_fam_detail))

    return {
        "gpu": gpu,
        "cv_results": cv_results,
        "heldout": heldout,
        "heldout_per_model": heldout_per_model,
        "agg": agg_rows,
        "saved_pkls": saved,
        "feature_audit": {fam: audit_features(cfg["features"]) for fam, cfg in FAMILY_CONFIG.items()},
        "training_models": training_models,
        "heldout_list": heldout_list,
    }


# ───────── Report writer ─────────

def write_report(results: list[dict], report_path: Path) -> None:
    lines: list[str] = ["# Phase A3 Training Report (shape_v2)", ""]

    for fam in FAMILY_CONFIG:
        feats = FAMILY_CONFIG[fam]["features"]
        leaks_any = any(r["feature_audit"].get(fam) for r in results)
        lines.append(f"- **{fam}** features ({len(feats)}): " +
                     (", ".join(f"`{f}`" for f in feats)) +
                     (" **LEAK**" if leaks_any else " ✓"))
    lines.append("")

    lines.append("## Headline: aggregate layer-total error per (GPU × held-out model)")
    lines.append("")
    lines.append("| GPU | held-out model | sum(pred ms) | sum(meas ms) | abs err % |")
    lines.append("|---|---|---:|---:|---:|")
    for r in results:
        for mdl, tp, tm, ep, _ in r["agg"]:
            lines.append(f"| {r['gpu']} | {mdl} | {tp:.2f} | {tm:.2f} | {ep:.2f}% |")
    lines.append("")

    lines.append("## Held-out per-family MAPE (full-train)")
    lines.append("")
    lines.append("| GPU | family | n_train | n_test | MAPE |")
    lines.append("|---|---|---:|---:|---:|")
    for r in results:
        for fam in FAMILY_CONFIG:
            entry = r["heldout"].get(fam, {})
            if entry.get("skipped"):
                lines.append(f"| {r['gpu']} | {fam} | {entry.get('n_train',0)} | {entry.get('n_test',0)} | skipped ({entry.get('reason','')}) |")
            elif entry.get("no_holdout"):
                lines.append(f"| {r['gpu']} | {fam} | {entry.get('n_train',0)} | 0 | no-heldout (train only) |")
            else:
                lines.append(f"| {r['gpu']} | {fam} | {entry.get('n_train',0)} | {entry.get('n_test',0)} | {entry.get('mape', float('nan')):.2f}% |")
    lines.append("")

    lines.append("## Leave-one-model-out CV")
    lines.append("")
    for r in results:
        if not r["cv_results"]:
            lines.append(f"### {r['gpu']} — skipped (<2 training models)")
            lines.append("")
            continue
        lines.append(f"### {r['gpu']}")
        lines.append("")
        lines.append("| held-out | gemm | flash_attn | elementwise | misc |")
        lines.append("|---|---:|---:|---:|---:|")
        for hold_mdl, per_fam in r["cv_results"].items():
            row = f"| {hold_mdl} |"
            for fam in FAMILY_CONFIG:
                entry = per_fam.get(fam, {})
                v = entry.get("mape", float("nan"))
                if np.isnan(v):
                    row += " n/a |"
                else:
                    row += f" {v:.1f}% (n={entry.get('n_test', 0)}) |"
            lines.append(row)
        lines.append("")

    lines.append("## Saved pkls")
    for r in results:
        lines.append(f"- **{r['gpu']}**")
        for p in r["saved_pkls"]:
            lines.append(f"  - `{p}`")
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
        gpu_out_dir = out_dir / gpu / "trained" / "per_kernel"
        r = train_one_gpu(sub, gpu, gpu_out_dir)
        for fam, entry in r["heldout"].items():
            if entry.get("skipped"):
                print(f"    [skip] {fam}: {entry.get('reason')}")
            else:
                mape_val = entry.get("mape", float("nan"))
                print(f"    {fam}: n_train={entry.get('n_train',0)} n_test={entry.get('n_test',0)} MAPE={mape_val:.2f}%")
        results.append(r)

    write_report(results, report_path)
    summary_path = report_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({r["gpu"]: {
            "heldout_mape": {fam: entry.get("mape") for fam, entry in r["heldout"].items()},
            "aggregate_err_pct": {mdl: ep for mdl, _, _, ep, _ in r["agg"]},
            "saved_pkls": r["saved_pkls"],
        } for r in results}, f, indent=2, default=str)
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
    llm_predict_dir = pkg_dir.parents[1]   # llm_predict/ — matches PerKernelPredictor._pkl_dir()
    data_csv = Path(args.data) if args.data else pkg_dir / "data" / "kernels_labeled.csv"
    data_csv = ensure_kernels_csv(data_csv)
    out_dir  = Path(args.out_dir) if args.out_dir else llm_predict_dir / "profiling" / "data"
    report   = Path(args.report) if args.report else pkg_dir / "reports" / "training_report.md"

    run(data_csv, out_dir, report, args.gpus)


if __name__ == "__main__":
    main()
