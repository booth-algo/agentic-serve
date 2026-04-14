"""
Train ML model to predict correction factor: measured / predicted.
Uses LLMCompass prediction as primary feature + architecture params.
"""
import json, os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import make_scorer
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")

# Load training data
with open("llmcompass/profiler/profiles/A100/layer_latency_training_data.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Get LLMCompass predictions for each data point
import llmcompass.software_model.transformer as tmod
tmod._kernel_predictor = None
from llmcompass.profiler.ml_predictor import KernelPredictor
p = KernelPredictor("llmcompass/profiler/profiles/A100")
p.train_all()
tmod._kernel_predictor = p
from llmcompass.software_model.transformer import TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP
from llmcompass.software_model.utils import data_type_dict, Tensor
from llmcompass.design_space_exploration.dse import template_to_system, read_architecture_template
tmod._kernel_predictor = p
arch = read_architecture_template("device_configs/GA100.json")
system = template_to_system(arch)
tmod._kernel_predictor = p

print("Collecting LLMCompass predictions for training data...", flush=True)
predictions = []
for i, row in df.iterrows():
    tmod._kernel_predictor = p
    if row["phase"] == "prefill":
        block = TransformerBlockInitComputationTP(
            d_model=int(row["d_model"]), n_heads=int(row["n_heads"]), device_count=1,
            data_type=data_type_dict["fp16"], intermediate_size=int(row["intermediate_size"]),
            n_kv_heads=int(row["n_kv_heads"]), use_flash_attention=True,
            activation_type="silu", use_ml_predictor=True, use_cuda_graph=False)
        X = Tensor([int(row["batch_size"]), int(row["seq_len"]), int(row["d_model"])], data_type_dict["fp16"])
        _ = block(X)
    else:
        block = TransformerBlockAutoRegressionTP(
            d_model=int(row["d_model"]), n_heads=int(row["n_heads"]), device_count=1,
            data_type=data_type_dict["fp16"], intermediate_size=int(row["intermediate_size"]),
            n_kv_heads=int(row["n_kv_heads"]), use_flash_attention=True,
            activation_type="silu", use_ml_predictor=True, use_cuda_graph=False)
        x = Tensor([int(row["batch_size"]), 1, int(row["d_model"])], data_type_dict["fp16"])
        _ = block(x, 512)
    tmod._kernel_predictor = p
    pred = block.compile_and_simulate(system, "heuristic-GPU") * 1e3
    predictions.append(pred)

df["llmcompass_pred_ms"] = predictions
df["correction_factor"] = df["latency_ms"] / df["llmcompass_pred_ms"]

print(f"Correction factor stats:")
print(f"  Mean: {df['correction_factor'].mean():.2f}")
print(f"  Std:  {df['correction_factor'].std():.2f}")
print(f"  Min:  {df['correction_factor'].min():.2f}")
print(f"  Max:  {df['correction_factor'].max():.2f}")

# Features: LLMCompass prediction + architecture context
df["log_pred"] = np.log2(df["llmcompass_pred_ms"] + 0.01)
df["is_decode"] = (df["phase"] == "decode").astype(int)
df["log_n_tokens"] = np.log2(df["n_tokens"] + 1)
df["ffn_ratio"] = df["intermediate_size"] / df["d_model"]
df["gqa_ratio"] = df["n_kv_heads"] / df["n_heads"]

feature_cols = [
    "llmcompass_pred_ms", "log_pred",
    "d_model", "intermediate_size", "n_heads", "n_kv_heads",
    "num_experts", "top_k", "is_decode",
    "batch_size", "seq_len", "n_tokens", "log_n_tokens",
    "ffn_ratio", "gqa_ratio",
]

X = df[feature_cols].values
y = df["correction_factor"].values
groups = df["model_name"].values

def mape_cf(y_true, y_pred):
    """MAPE on correction factor -> translates to MAPE on final latency."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"\nTraining correction models on {len(X)} samples...")
print("=" * 70)

models = {
    "RF": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "GBR": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, min_samples_leaf=3, random_state=42),
}

for name, model in models.items():
    model.fit(X, y)
    train_preds = model.predict(X)
    
    # Correction factor accuracy
    cf_mape = mape_cf(y, train_preds)
    
    # End-to-end latency accuracy
    corrected_latency = df["llmcompass_pred_ms"].values * train_preds
    latency_mape = np.mean(np.abs((df["latency_ms"].values - corrected_latency) / df["latency_ms"].values)) * 100
    
    # CV
    cv_scores = cross_val_score(model, X, y, cv=5, 
        scoring=make_scorer(lambda yt, yp: -mape_cf(yt, yp)))
    cv_mape = -cv_scores.mean()
    
    # LOMO
    logo = LeaveOneGroupOut()
    lomo_preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        model.fit(X[tr], y[tr])
        lomo_preds[te] = model.predict(X[te])
    lomo_corrected = df["llmcompass_pred_ms"].values * lomo_preds
    lomo_latency_mape = np.mean(np.abs((df["latency_ms"].values - lomo_corrected) / df["latency_ms"].values)) * 100
    
    # Refit on all data
    model.fit(X, y)
    
    print(f"\n{name}:")
    print(f"  Train CF MAPE:     {cf_mape:.1f}%")
    print(f"  Train latency MAPE: {latency_mape:.1f}%")
    print(f"  CV CF MAPE:        {cv_mape:.1f}%")
    print(f"  LOMO latency MAPE: {lomo_latency_mape:.1f}% (unseen model generalization)")

# Compare: raw LLMCompass vs corrected
raw_mape = np.mean(np.abs((df["latency_ms"].values - df["llmcompass_pred_ms"].values) / df["latency_ms"].values)) * 100
print(f"\n{'='*70}")
print(f"Raw LLMCompass MAPE: {raw_mape:.1f}%")
print(f"(This is WITHOUT the 1.45x calibration — using per-kernel ML predictions)")

# Use best model for detailed results
best = models["RF"]
best.fit(X, y)
final_preds = best.predict(X)
corrected = df["llmcompass_pred_ms"].values * final_preds

print(f"\n{'='*70}")
print(f"Detailed: RF-corrected predictions")
print(f"{'='*70}")
print(f"{'Model':>20s} {'Phase':>7s} {'BS':>3s} {'Seq':>5s} {'Meas':>8s} {'Raw':>8s} {'Corr':>8s} {'RawE%':>7s} {'CorrE%':>7s}")
print("-" * 80)
for i, row in df.iterrows():
    raw_err = abs(row["llmcompass_pred_ms"] / row["latency_ms"] - 1) * 100
    corr_err = abs(corrected[i] / row["latency_ms"] - 1) * 100
    print(f"{row['model_name']:>20s} {row['phase']:>7s} {row['batch_size']:>3d} {row['seq_len']:>5d} {row['latency_ms']:>8.3f} {row['llmcompass_pred_ms']:>8.3f} {corrected[i]:>8.3f} {raw_err:>6.1f}% {corr_err:>6.1f}%")

# Feature importance
print(f"\nFeature importance (RF correction model):")
for feat, imp in sorted(zip(feature_cols, best.feature_importances_), key=lambda x: -x[1]):
    if imp > 0.01:
        print(f"  {feat:<25s}: {imp:.3f}")

# Save
with open("llmcompass/profiler/profiles/A100/layer_correction_model.pkl", "wb") as f:
    pickle.dump({"model": best, "features": feature_cols, "type": "correction_factor"}, f)
print(f"\nSaved correction model")
