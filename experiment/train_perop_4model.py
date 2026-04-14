"""Train per-op predictor with 3 models: Llama-8B, Llama-70B, Mixtral."""
import os, sys, pickle, numpy as np, pandas as pd
os.environ["PYTHONPATH"] = "/data/kevinlau/python-packages"
sys.path.insert(0, "/home/kevinlau/llmserve"); os.chdir("/home/kevinlau/llmserve")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

MODEL_CONFIGS = {
    "Llama-8B": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 0, "k": 0},
    "Llama-70B": {"d": 8192, "h": 64, "kv": 8, "ffn": 28672, "E": 0, "k": 0},
    "Mixtral": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 8, "k": 2},
    "Qwen-72B": {"d": 8192, "h": 64, "kv": 8, "ffn": 29568, "E": 0, "k": 0},
    "gpt-oss-20b": {"d": 2880, "h": 64, "kv": 8, "ffn": 2880, "E": 32, "k": 4},
}
OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
           "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post", "total_block": "total"}

paths = {
    "Llama-8B": "/data/LLMServingSim/llm_profile/perf_models/A100/Llama-3_1-8B-Instruct/tp1/layers_v3.csv",
    "Llama-70B": "/data/LLMServingSim/llm_profile/perf_models/A100/Llama-3_1-70B-Instruct/tp1/layers_v3.csv",
    "Mixtral": "/data/LLMServingSim/llm_profile/perf_models/A100/Mixtral-8x7B-Instruct/tp1/layers_v3.csv",
    "Qwen-72B": "/data/LLMServingSim/llm_profile/perf_models/A100/Qwen2_5-72B-Instruct/tp1/layers_v3.csv",
    "gpt-oss-20b": "/data/LLMServingSim/llm_profile/perf_models/A100/gpt-oss-20b/tp1/layers_v3.csv",
}

frames = []
for name, path in paths.items():
    df = pd.read_csv(path); df["model"] = name
    df["op"] = df["layer_name"].map(OP_MAP); df = df.dropna(subset=["op"])
    cfg = MODEL_CONFIGS[name]
    for key, val in cfg.items(): df[key] = val
    df["lat_us"] = df["latency_ns"] / 1000
    frames.append(df[["model","op","input","lat_us","d","h","kv","ffn","E","k"]])
all_df = pd.concat(frames)

# Train on 3 reliable models
train_models = ["Llama-8B", "Llama-70B", "Mixtral"]
train_df = all_df[(all_df.model.isin(train_models)) & (all_df.op != "total")]
total_df = all_df[all_df.op == "total"]

def compute_features(df):
    tok = df["input"].values.astype(float)
    d = df["d"].values.astype(float); h = df["h"].values.astype(float)
    kv = df["kv"].values.astype(float); ffn = df["ffn"].values.astype(float)
    E = df["E"].values.astype(float); k = df["k"].values.astype(float)
    is_a = (df["op"].values == "attn").astype(float)
    is_f = (df["op"].values == "ffn").astype(float)
    is_n = df["op"].isin(["norm_pre","norm_post"]).values.astype(float)
    d_h = d / h
    # Compute-aware features
    attn_flops = is_a * (2*tok*d*(d + 2*kv*d_h) + 4*tok*tok*h*d_h + 2*tok*d*d)
    dense_ffn_flops = is_f * (1 - np.minimum(E, 1)) * 2*tok*d*ffn*3
    tpe = np.where(E > 0, np.maximum(1, tok*k/np.maximum(E,1)), 0)
    aexp = np.where(E > 0, np.minimum(tok*k, E), 0)
    moe_ffn_flops = is_f * np.where(E > 0, 2*tpe*d*ffn*3*aexp, 0)
    norm_flops = is_n * tok * d * 5
    total_flops = attn_flops + dense_ffn_flops + moe_ffn_flops + norm_flops
    # Memory: weight bytes
    attn_wt = is_a * d * (d + 2*kv*d_h + d) * 2
    ffn_wt = is_f * np.where(E > 0, E*d*ffn*3*2, d*ffn*3*2)
    norm_wt = is_n * d * 2
    total_wt = attn_wt + ffn_wt + norm_wt
    ai = np.where(total_wt > 0, total_flops / total_wt, 0)
    return np.column_stack([
        tok, np.log2(tok+1), total_flops, np.log2(total_flops+1),
        total_wt, np.log2(total_wt+1), ai, np.log2(ai+1),
        is_a, is_f, is_n, d, ffn, E, k,
        d*d, d*ffn, np.where(E>0,tpe,tok), np.where(E>0,aexp,1), 
        np.where(E>0,E*d*ffn*3*2,d*ffn*3*2),  # total weight load
    ])

feat_names = ["tok","log_tok","flops","log_flops","wt_bytes","log_wt","ai","log_ai",
    "is_attn","is_ffn","is_norm","d","ffn","E","k","d2","d_ffn","eff_tok","a_exp","tot_wt"]

X = compute_features(train_df); y = train_df["lat_us"].values; groups = train_df["model"].values
print(f"Training: {len(X)} rows from {train_models}")

models = {
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGB": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
}
for name, mdl in models.items():
    mdl.fit(X, y); p = mdl.predict(X)
    t_mape = np.mean(np.abs((y-p)/np.maximum(y,1)))*100
    lp = np.zeros(len(y)); logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        mdl.fit(X[tr], y[tr]); lp[te] = mdl.predict(X[te])
    l_mape = np.mean(np.abs((y-lp)/np.maximum(y,1)))*100
    mdl.fit(X, y)
    print(f"  {name}: train={t_mape:.1f}%, LOMO={l_mape:.1f}%")

best = models["XGB"]
# Layer-level predictions for ALL 5 models
print(f"\nLayer predictions (sum per-op → total_block):")
print(f"{'Model':>12} {'Tok':>5} {'Pred':>10} {'Meas':>10} {'Ratio':>8} {'Rating':>6}")
print("-"*55)
good_count = 0; total_count = 0
for mn in ["Llama-8B", "Llama-70B", "Mixtral", "Qwen-72B", "gpt-oss-20b"]:
    cfg = MODEL_CONFIGS[mn]
    for tok in [1, 64, 128, 256, 512]:
        pred_sum = 0
        for op in ["attn", "ffn", "norm_pre", "norm_post"]:
            row = pd.DataFrame([{"input":tok,"op":op,"model":mn,"d":cfg["d"],"h":cfg["h"],
                "kv":cfg["kv"],"ffn":cfg["ffn"],"E":cfg["E"],"k":cfg["k"]}])
            pred_sum += best.predict(compute_features(row))[0]
        mr = total_df[(total_df.model==mn)&(total_df.input==tok)]
        if len(mr) > 0:
            m = mr.lat_us.values[0]; ratio = pred_sum/m
            rating = "GOOD" if abs(ratio-1)<=0.2 else ("OK" if abs(ratio-1)<=0.3 else "BAD")
            if abs(ratio-1)<=0.2: good_count += 1
            total_count += 1
            print(f"{mn:>12} {tok:>5} {pred_sum:>10.1f} {m:>10.1f} {ratio:>7.2f}x {rating:>6}")

print(f"\nGOOD: {good_count}/{total_count} ({good_count/total_count*100:.0f}%)")
print(f"\nFeature importance:")
for f, imp in sorted(zip(feat_names, best.feature_importances_), key=lambda x:-x[1])[:8]:
    print(f"  {f:<15}: {imp:.3f}")
