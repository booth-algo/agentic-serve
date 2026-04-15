"""Lean per-op predictor training. No nested loops, vectorized predictions."""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")
os.environ["PYTHONPATH"] = "/data/kevinlau/python-packages:" + os.environ.get("PYTHONPATH","")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

MODEL_CONFIGS = {
    "Llama-8B": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 0, "k": 0},
    "Mixtral": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 8, "k": 2},
    "Qwen-72B": {"d": 8192, "h": 64, "kv": 8, "ffn": 29568, "E": 0, "k": 0},
    "gpt-oss-20b": {"d": 2880, "h": 64, "kv": 8, "ffn": 2880, "E": 32, "k": 4},
}

OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
           "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post", "total_block": "total"}

paths = {
    "Llama-8B": "perf_models/A100/Llama-3_1-8B-Instruct/tp1/layers_v3.csv",
    "Mixtral": "perf_models/A100/Mixtral-8x7B-Instruct/tp1/layers_v3.csv",
    "Qwen-72B": "perf_models/A100/Qwen2_5-72B-Instruct/tp1/layers_v3.csv",
    "gpt-oss-20b": "perf_models/A100/gpt-oss-20b/tp1/layers_v3.csv",
}

# Load + normalize
frames = []
for name, rel_path in paths.items():
    path = f"/data/LLMServingSim/llm_profile/{rel_path}"
    df = pd.read_csv(path)
    df["model"] = name
    df["op"] = df["layer_name"].map(OP_MAP)
    df = df.dropna(subset=["op"])
    cfg = MODEL_CONFIGS[name]
    for key, val in cfg.items(): df[key] = val
    df["lat_us"] = df["latency_ns"] / 1000
    frames.append(df[["model","op","input","lat_us","d","h","kv","ffn","E","k"]])

all_df = pd.concat(frames)

# Use ONLY reliable per-op (Llama + Mixtral) for training
# Use total_block from ALL models for layer-level eval
train_df = all_df[(all_df.model.isin(["Llama-8B","Mixtral"])) & (all_df.op != "total")]
total_df = all_df[all_df.op == "total"]

# Features
def featurize(df):
    X = pd.DataFrame()
    X["tok"] = df["input"].values
    X["log_tok"] = np.log2(df["input"].values + 1)
    X["d"] = df["d"].values
    X["h"] = df["h"].values
    X["kv"] = df["kv"].values
    X["ffn"] = df["ffn"].values
    X["E"] = df["E"].values
    X["k"] = df["k"].values
    X["is_moe"] = (df["E"].values > 0).astype(int)
    X["is_attn"] = (df["op"].values == "attn").astype(int)
    X["is_ffn"] = (df["op"].values == "ffn").astype(int)
    X["is_norm"] = df["op"].isin(["norm_pre","norm_post"]).values.astype(int)
    X["ffn_ratio"] = df["ffn"].values / df["d"].values
    X["gqa_ratio"] = df["kv"].values / df["h"].values
    return X.values

X_train = featurize(train_df)
y_train = train_df["lat_us"].values
groups = train_df["model"].values

print(f"Training data: {len(X_train)} rows (Llama+Mixtral per-op)")

# Train
models_dict = {
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGB": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
}

for name, mdl in models_dict.items():
    mdl.fit(X_train, y_train)
    train_pred = mdl.predict(X_train)
    train_mape = np.mean(np.abs((y_train - train_pred) / np.maximum(y_train, 1))) * 100
    
    # LOMO
    logo = LeaveOneGroupOut()
    lomo_pred = np.zeros(len(y_train))
    for tr, te in logo.split(X_train, y_train, groups):
        mdl.fit(X_train[tr], y_train[tr])
        lomo_pred[te] = mdl.predict(X_train[te])
    lomo_mape = np.mean(np.abs((y_train - lomo_pred) / np.maximum(y_train, 1))) * 100
    
    mdl.fit(X_train, y_train)
    print(f"{name}: train MAPE={train_mape:.1f}%, LOMO MAPE={lomo_mape:.1f}%")

# Use XGB for predictions
best = models_dict["XGB"]

# Per-op accuracy at key token counts (Llama + Mixtral)
print(f"\nPer-op at tok=256:")
print(f"{'Model':>12} {'Op':>10} {'Meas(us)':>10} {'Pred(us)':>10} {'Ratio':>8}")
for model_name in ["Llama-8B", "Mixtral"]:
    sub = train_df[(train_df.model == model_name) & (train_df.input == 256)]
    X_sub = featurize(sub)
    p_sub = best.predict(X_sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        r = p_sub[i] / max(row.lat_us, 0.1)
        print(f"{model_name:>12} {row.op:>10} {row.lat_us:>10.1f} {p_sub[i]:>10.1f} {r:>7.2f}x")

# Layer-level: predict per-op for ALL models, sum, compare to total_block
print(f"\nLayer-level (sum of per-op predictions vs measured total_block):")
print(f"{'Model':>12} {'Tok':>5} {'PredSum(us)':>12} {'Meas(us)':>10} {'Ratio':>8} {'Rating':>6}")
print("-"*58)
for model_name in ["Llama-8B", "Mixtral", "Qwen-72B", "gpt-oss-20b"]:
    cfg = MODEL_CONFIGS[model_name]
    for tok in [1, 64, 128, 256, 512]:
        pred_sum = 0
        for op in ["attn", "ffn", "norm_pre", "norm_post"]:
            row = pd.DataFrame([{"input": tok, "op": op, "model": model_name,
                "d": cfg["d"], "h": cfg["h"], "kv": cfg["kv"], "ffn": cfg["ffn"],
                "E": cfg["E"], "k": cfg["k"]}])
            pred_sum += best.predict(featurize(row))[0]
        
        meas_row = total_df[(total_df.model == model_name) & (total_df.input == tok)]
        if len(meas_row) > 0:
            meas = meas_row.lat_us.values[0]
            ratio = pred_sum / meas
            rating = "GOOD" if abs(ratio-1) <= 0.2 else ("OK" if abs(ratio-1) <= 0.3 else "BAD")
            print(f"{model_name:>12} {tok:>5} {pred_sum:>12.1f} {meas:>10.1f} {ratio:>7.2f}x {rating:>6}")

# Feature importance
print(f"\nFeature importance (XGB):")
feat_names = ["tok","log_tok","d","h","kv","ffn","E","k","is_moe","is_attn","is_ffn","is_norm","ffn_ratio","gqa_ratio"]
for f, imp in sorted(zip(feat_names, best.feature_importances_), key=lambda x: -x[1])[:8]:
    print(f"  {f:<15}: {imp:.3f}")

with open("llm_predict/profiling/data/A100/perop_predictor_v3.pkl", "wb") as f:
    pickle.dump({"model": best, "features": feat_names}, f)
print("\nSaved model")
