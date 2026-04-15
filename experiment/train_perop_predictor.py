"""
Train RF, XGBoost, and GradientBoosting models to predict per-component
transformer layer latencies from in-context profiling data.
"""

import sys
import os

sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")

# Add custom python packages path for xgboost etc.
sys.path.insert(0, "/data/kevinlau/python-packages")

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATHS = {
    "Llama-3.1-8B-Instruct": "/data/LLMServingSim/llm_profile/perf_models/A100/data/models/Llama-3.1-8B-Instruct/tp1/layers.csv",
    "Mixtral-8x7B-Instruct":  "/data/LLMServingSim/llm_profile/perf_models/A100/Mixtral-8x7B-Instruct/tp1/layers.csv",
    "Qwen2_5-72B-Instruct":   "/data/LLMServingSim/llm_profile/perf_models/A100/Qwen2_5-72B-Instruct/tp1/layers.csv",
    "gpt-oss-20b":             "/data/LLMServingSim/llm_profile/perf_models/A100/gpt-oss-20b/tp1/layers.csv",
}

MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "d_model": 4096, "n_heads": 32, "n_kv_heads": 8,
        "intermediate_size": 14336, "num_experts": 0, "top_k": 0, "is_moe": False,
    },
    "Mixtral-8x7B-Instruct": {
        "d_model": 4096, "n_heads": 32, "n_kv_heads": 8,
        "intermediate_size": 14336, "num_experts": 8, "top_k": 2, "is_moe": True,
    },
    "Qwen2_5-72B-Instruct": {
        "d_model": 8192, "n_heads": 64, "n_kv_heads": 8,
        "intermediate_size": 29568, "num_experts": 0, "top_k": 0, "is_moe": False,
    },
    "gpt-oss-20b": {
        "d_model": 2880, "n_heads": 64, "n_kv_heads": 8,
        "intermediate_size": 2880, "num_experts": 32, "top_k": 4, "is_moe": True,
    },
}

# Llama fine-grained ops that map to each unified category
LLAMA_ATTENTION_OPS = {"q_proj", "k_proj", "v_proj", "rope", "o_proj"}
LLAMA_FFN_OPS       = {"gate_proj", "up_proj", "act_fn", "down_proj"}
LLAMA_NORM_PRE_OPS  = {"input_layernorm"}
LLAMA_NORM_POST_OPS = {"post_layernorm"}

MODULE_ATTENTION_OPS  = {"self_attn"}
MODULE_FFN_OPS        = {"mlp", "block_sparse_moe"}
MODULE_NORM_PRE_OPS   = {"input_layernorm"}
MODULE_NORM_POST_OPS  = {"post_attention_layernorm"}

ISOLATED_LAYER_PATH = (
    "/home/kevinlau/llmserve/llm_predict/profiling/data/A100/"
    "layer_latency_training_data.json"
)
SAVE_MODEL_PATH = (
    "/home/kevinlau/llmserve/llm_predict/profiling/data/A100/"
    "perop_predictor.pkl"
)

OP_CATEGORIES = ["attention", "ffn", "norm_pre", "norm_post"]

# ---------------------------------------------------------------------------
# Step 1: Load and normalise raw CSVs
# ---------------------------------------------------------------------------

def classify_op_llama(op_name: str) -> str | None:
    """Map a Llama fine-grained op name to a unified category (or None to skip)."""
    if op_name in LLAMA_ATTENTION_OPS:
        return "attention"
    if op_name in LLAMA_FFN_OPS:
        return "ffn"
    if op_name in LLAMA_NORM_PRE_OPS:
        return "norm_pre"
    if op_name in LLAMA_NORM_POST_OPS:
        return "norm_post"
    return None  # embedding, final_layernorm, lm_head — skip


def classify_op_module(op_name: str) -> str | None:
    """Map a module-level op name to a unified category."""
    if op_name in MODULE_ATTENTION_OPS:
        return "attention"
    if op_name in MODULE_FFN_OPS:
        return "ffn"
    if op_name in MODULE_NORM_PRE_OPS:
        return "norm_pre"
    if op_name in MODULE_NORM_POST_OPS:
        return "norm_post"
    return None


def load_and_normalise(model_name: str, csv_path: str) -> pd.DataFrame:
    """
    Load one model's CSV, aggregate fine-grained ops into 4 unified categories,
    and return a DataFrame with columns:
        model_name, op_category, n_tokens, latency_ns
    """
    df = pd.read_csv(csv_path)

    # Rename columns defensively
    df.columns = [c.strip() for c in df.columns]
    # Normalise column names: LLMServingSim uses 'latency(ns)', ours uses 'latency_ns'
    df = df.rename(columns={"latency(ns)": "latency_ns", "input": "n_tokens", "layer_name": "op_name"})

    is_llama = model_name == "Llama-3.1-8B-Instruct"
    classify_fn = classify_op_llama if is_llama else classify_op_module

    df["op_category"] = df["op_name"].apply(classify_fn)
    df = df.dropna(subset=["op_category"])

    # Aggregate: sum latency for ops that map to the same category at the same
    # token count. (e.g. q_proj + k_proj + v_proj + rope + o_proj → attention)
    grouped = (
        df.groupby(["n_tokens", "op_category"], as_index=False)["latency_ns"]
        .sum()
    )
    grouped["model_name"] = model_name
    return grouped[["model_name", "op_category", "n_tokens", "latency_ns"]]


def load_all_models() -> pd.DataFrame:
    frames = []
    for model_name, path in DATA_PATHS.items():
        print(f"  Loading {model_name} from {path}")
        df = load_and_normalise(model_name, path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Also compute total_block per (model, n_tokens)
    total = (
        combined.groupby(["model_name", "n_tokens"], as_index=False)["latency_ns"]
        .sum()
        .rename(columns={"latency_ns": "latency_ns"})
    )
    total["op_category"] = "total_block"
    combined = pd.concat([combined, total], ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Step 2: Feature engineering
# ---------------------------------------------------------------------------

def compute_flops(row: pd.Series) -> float:
    """Approximate FLOPs for one forward pass through the op."""
    cfg = MODEL_CONFIGS[row["model_name"]]
    n  = row["n_tokens"]
    d  = cfg["d_model"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv_heads"]
    ff  = cfg["intermediate_size"]
    tk  = max(cfg["top_k"], 1)
    cat = row["op_category"]

    if cat == "attention":
        head_dim = d / nh
        # QKV projections + attention scores + output projection
        flops = 4.0 * n * (nh / nkv) * (head_dim ** 2)
    elif cat == "ffn":
        # gate_proj + up_proj + down_proj  (×3 matmuls), scaled by top_k for MoE
        flops = 2.0 * n * d * ff * 3.0
        if cfg["is_moe"]:
            # Each token routes to top_k experts; expert hidden dim unchanged
            # but only top_k / num_experts fraction of experts fire per token
            flops *= tk
    elif cat in ("norm_pre", "norm_post"):
        # LayerNorm: ~2 * n * d
        flops = 2.0 * n * d
    else:  # total_block
        flops = 0.0
    return float(flops)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach architectural and derived features to the normalised DataFrame."""
    rows = []
    for _, row in df.iterrows():
        cfg = MODEL_CONFIGS[row["model_name"]]
        n   = row["n_tokens"]
        cat = row["op_category"]

        feat = {
            "model_name":        row["model_name"],
            "op_category":       cat,
            "n_tokens":          n,
            "latency_ns":        row["latency_ns"],
            # Architectural
            "d_model":           cfg["d_model"],
            "n_heads":           cfg["n_heads"],
            "n_kv_heads":        cfg["n_kv_heads"],
            "intermediate_size": cfg["intermediate_size"],
            "num_experts":       cfg["num_experts"],
            "top_k":             cfg["top_k"],
            "is_moe":            int(cfg["is_moe"]),
            # Derived
            "log_n_tokens":      np.log2(n + 1),
            "ffn_ratio":         cfg["intermediate_size"] / cfg["d_model"],
            "gqa_ratio":         cfg["n_kv_heads"] / cfg["n_heads"],
            # One-hot op type
            "op_is_attention":   int(cat == "attention"),
            "op_is_ffn":         int(cat in ("ffn",)),
            "op_is_norm":        int(cat in ("norm_pre", "norm_post")),
            "op_is_total":       int(cat == "total_block"),
            # FLOPs
            "compute_flops":     compute_flops(row),
            "log_flops":         np.log2(compute_flops(row) + 1),
        }
        rows.append(feat)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "n_tokens", "d_model", "n_heads", "n_kv_heads", "intermediate_size",
    "num_experts", "top_k", "is_moe",
    "log_n_tokens", "ffn_ratio", "gqa_ratio",
    "op_is_attention", "op_is_ffn", "op_is_norm", "op_is_total",
    "compute_flops", "log_flops",
]

TARGET_COL = "latency_ns"


# ---------------------------------------------------------------------------
# Step 3: Model definitions
# ---------------------------------------------------------------------------

def build_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda",
            random_state=42, verbosity=0,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Step 4: Evaluation helpers
# ---------------------------------------------------------------------------

def within_band(y_true, y_pred, lo=0.8, hi=1.2) -> float:
    """Fraction of predictions within [lo*true, hi*true]."""
    ratio = y_pred / (y_true + 1e-9)
    return float(np.mean((ratio >= lo) & (ratio <= hi)))


def eval_model(model, X_train, y_train, X_test, y_test, label=""):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape  = mean_absolute_percentage_error(y_test, preds) * 100
    wb    = within_band(y_test.values, preds) * 100
    median_ratio = float(np.median(preds / (y_test.values + 1e-9)))
    return {"label": label, "MAPE_%": mape, "Within20%": wb, "MedianRatio": median_ratio, "preds": preds}


# ---------------------------------------------------------------------------
# Step 5: Cross-validation
# ---------------------------------------------------------------------------

def run_cv(feat_df: pd.DataFrame, models: dict, n_splits: int = 5):
    print("\n" + "="*70)
    print(f"5-Fold Cross-Validation (all ops combined, n={len(feat_df)})")
    print("="*70)

    X = feat_df[FEATURE_COLS].values
    y = feat_df[TARGET_COL].values

    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=1,
        )
        mape_cv = -scores.mean() * 100
        results[name] = mape_cv
        print(f"  {name:20s}  CV MAPE = {mape_cv:.2f}%")
    return results


# ---------------------------------------------------------------------------
# Step 6: Leave-One-Model-Out (LOMO)
# ---------------------------------------------------------------------------

def run_lomo(feat_df: pd.DataFrame, models: dict):
    print("\n" + "="*70)
    print("Leave-One-Model-Out (LOMO) Evaluation")
    print("="*70)

    model_names = list(DATA_PATHS.keys())
    lomo_results = {}

    for held_out in model_names:
        train_df = feat_df[feat_df["model_name"] != held_out]
        test_df  = feat_df[feat_df["model_name"] == held_out]

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df[TARGET_COL].values

        print(f"\n  Held-out: {held_out}  (test={len(test_df)}, train={len(train_df)})")
        model_res = {}
        for name, model_template in models.items():
            import copy
            m = copy.deepcopy(model_template)
            res = eval_model(m, X_train, y_train, X_test, y_test, label=name)
            model_res[name] = res
            print(f"    {name:20s}  MAPE={res['MAPE_%']:6.2f}%  "
                  f"Within20%={res['Within20%']:5.1f}%  "
                  f"MedianRatio={res['MedianRatio']:.3f}")

        # Per-op breakdown on the best model (lowest MAPE)
        best_name = min(model_res, key=lambda k: model_res[k]["MAPE_%"])
        best_preds = model_res[best_name]["preds"]
        test_df_copy = test_df.copy()
        test_df_copy["pred_ns"] = best_preds

        print(f"\n    Per-op breakdown ({best_name}):")
        for op in OP_CATEGORIES + ["total_block"]:
            sub = test_df_copy[test_df_copy["op_category"] == op]
            if sub.empty:
                continue
            op_mape = mean_absolute_percentage_error(sub[TARGET_COL], sub["pred_ns"]) * 100
            op_wb   = within_band(sub[TARGET_COL].values, sub["pred_ns"].values) * 100
            print(f"      {op:15s}  MAPE={op_mape:6.2f}%  Within20%={op_wb:5.1f}%")

        lomo_results[held_out] = model_res
    return lomo_results


# ---------------------------------------------------------------------------
# Step 7: Detailed per-op predictions at specific token counts
# ---------------------------------------------------------------------------

def detailed_predictions(feat_df: pd.DataFrame, best_model, token_counts=(1, 64, 256, 512)):
    print("\n" + "="*70)
    print("Detailed Per-Op Predictions vs Measured (best model, full-data fit)")
    print("="*70)

    X = feat_df[FEATURE_COLS].values
    y = feat_df[TARGET_COL].values
    best_model.fit(X, y)
    feat_df = feat_df.copy()
    feat_df["pred_ns"] = best_model.predict(X)

    for model_name in DATA_PATHS:
        print(f"\n  Model: {model_name}")
        sub = feat_df[feat_df["model_name"] == model_name]
        print(f"  {'op_category':15s} {'tok':>6} {'measured_us':>12} {'predicted_us':>12} {'ratio':>7}")
        print("  " + "-"*57)
        for tok in token_counts:
            t_sub = sub[sub["n_tokens"] == tok]
            for op in OP_CATEGORIES + ["total_block"]:
                op_sub = t_sub[t_sub["op_category"] == op]
                if op_sub.empty:
                    continue
                meas = op_sub[TARGET_COL].values[0]
                pred = op_sub["pred_ns"].values[0]
                ratio = pred / (meas + 1e-9)
                print(f"  {op:15s} {tok:>6} {meas/1e3:>12.2f} {pred/1e3:>12.2f} {ratio:>7.3f}")
    return best_model


# ---------------------------------------------------------------------------
# Step 8: Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model, model_name: str):
    print(f"\n{'='*70}")
    print(f"Feature Importance: {model_name}")
    print("="*70)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])
        for feat, score in pairs:
            bar = "#" * int(score * 60)
            print(f"  {feat:25s} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Step 9: Compare against isolated layer measurements
# ---------------------------------------------------------------------------

def compare_isolated_layers(feat_df: pd.DataFrame, best_model):
    if not os.path.exists(ISOLATED_LAYER_PATH):
        print(f"\nIsolated layer file not found: {ISOLATED_LAYER_PATH} — skipping.")
        return

    print("\n" + "="*70)
    print("Comparison vs Isolated Layer Measurements (LLMCompass)")
    print("="*70)

    with open(ISOLATED_LAYER_PATH) as f:
        iso_data = json.load(f)

    X_all = feat_df[FEATURE_COLS].values
    feat_df = feat_df.copy()
    feat_df["pred_ns"] = best_model.predict(X_all)

    # iso_data expected structure: list of dicts with at minimum
    # "model", "n_tokens", "latency_ns" (total layer latency)
    # Adapt as needed based on actual file format.
    if isinstance(iso_data, list):
        iso_df = pd.DataFrame(iso_data)
    elif isinstance(iso_data, dict):
        rows = []
        for model_key, entries in iso_data.items():
            if isinstance(entries, list):
                for e in entries:
                    e["model"] = model_key
                    rows.append(e)
        iso_df = pd.DataFrame(rows)
    else:
        print("  Unrecognised isolated layer data format — skipping.")
        return

    # Try to map column names
    col_map = {}
    for col in iso_df.columns:
        cl = col.lower()
        if "model" in cl:
            col_map[col] = "model"
        elif "token" in cl or col == "n" or col == "seq_len":
            col_map[col] = "n_tokens"
        elif "latency" in cl or "time" in cl:
            col_map[col] = "latency_ns"
    iso_df = iso_df.rename(columns=col_map)

    required = {"model", "n_tokens", "latency_ns"}
    if not required.issubset(iso_df.columns):
        print(f"  Isolated layer data missing columns (have: {list(iso_df.columns)}) — skipping.")
        return

    print(f"  {'model':30s} {'tok':>6} {'measured_us':>12} {'pred_sum_us':>12} {'ratio':>7}")
    print("  " + "-"*67)
    for _, iso_row in iso_df.iterrows():
        mname  = str(iso_row["model"])
        ntok   = int(iso_row["n_tokens"])
        meas   = float(iso_row["latency_ns"])

        # Sum predicted per-op latencies for this (model, tok)
        sub = feat_df[
            (feat_df["model_name"] == mname) &
            (feat_df["n_tokens"] == ntok) &
            (feat_df["op_category"].isin(OP_CATEGORIES))
        ]
        if sub.empty:
            continue
        pred_sum = sub["pred_ns"].sum()
        ratio = pred_sum / (meas + 1e-9)
        print(f"  {mname:30s} {ntok:>6} {meas/1e3:>12.2f} {pred_sum/1e3:>12.2f} {ratio:>7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading profiling data ...")
    raw_df = load_all_models()
    print(f"  Total rows (incl. total_block): {len(raw_df)}")

    print("\nBuilding features ...")
    feat_df = build_features(raw_df)
    feat_df = feat_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"  Feature matrix: {feat_df.shape}")

    # Separate total_block rows for later use; train on per-op rows
    perop_df  = feat_df[feat_df["op_category"] != "total_block"].copy()
    total_df  = feat_df[feat_df["op_category"] == "total_block"].copy()
    print(f"  Per-op rows: {len(perop_df)}, Total-block rows: {len(total_df)}")

    models = build_models()

    # 5-fold CV
    cv_results = run_cv(perop_df, models)

    # LOMO
    lomo_results = run_lomo(perop_df, models)

    # Determine best model by average LOMO MAPE
    avg_lomo = {name: 0.0 for name in models}
    for held_out, model_res in lomo_results.items():
        for name, res in model_res.items():
            avg_lomo[name] += res["MAPE_%"]
    avg_lomo = {k: v / len(lomo_results) for k, v in avg_lomo.items()}

    print("\n" + "="*70)
    print("Model Comparison Summary")
    print("="*70)
    print(f"  {'Model':20s}  {'5-fold CV MAPE':>16}  {'Avg LOMO MAPE':>14}")
    print("  " + "-"*55)
    for name in models:
        print(f"  {name:20s}  {cv_results[name]:>14.2f}%  {avg_lomo[name]:>13.2f}%")

    best_model_name = min(avg_lomo, key=avg_lomo.get)
    print(f"\n  Best model by LOMO: {best_model_name}")

    import copy
    best_model = copy.deepcopy(models[best_model_name])

    # Detailed predictions on full dataset fit
    best_model = detailed_predictions(perop_df, best_model)

    # Feature importance
    print_feature_importance(best_model, best_model_name)

    # Compare against isolated layer measurements
    compare_isolated_layers(perop_df, best_model)

    # Save best model
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    with open(SAVE_MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": FEATURE_COLS,
                     "model_name": best_model_name}, f)
    print(f"\nBest model saved to {SAVE_MODEL_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
