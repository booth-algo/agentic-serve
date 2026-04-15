"""Per-op predictor using analytical compute features (FLOPs, bytes, GEMM shapes)."""
import os, sys, pickle, numpy as np, pandas as pd
os.environ["PYTHONPATH"] = "/data/kevinlau/python-packages"
sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")

from sklearn.ensemble import RandomForestRegressor
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
    "Llama-8B": "/data/LLMServingSim/llm_profile/perf_models/A100/Llama-3_1-8B-Instruct/tp1/layers_v3.csv",
    "Mixtral": "/data/LLMServingSim/llm_profile/perf_models/A100/Mixtral-8x7B-Instruct/tp1/layers_v3.csv",
    "Qwen-72B": "/data/LLMServingSim/llm_profile/perf_models/A100/Qwen2_5-72B-Instruct/tp1/layers_v3.csv",
    "gpt-oss-20b": "/data/LLMServingSim/llm_profile/perf_models/A100/gpt-oss-20b/tp1/layers_v3.csv",
}

frames = []
for name, path in paths.items():
    df = pd.read_csv(path)
    df["model"] = name; df["op"] = df["layer_name"].map(OP_MAP)
    df = df.dropna(subset=["op"])
    cfg = MODEL_CONFIGS[name]
    for key, val in cfg.items(): df[key] = val
    df["lat_us"] = df["latency_ns"] / 1000
    frames.append(df[["model","op","input","lat_us","d","h","kv","ffn","E","k"]])
all_df = pd.concat(frames)

train_df = all_df[(all_df.model.isin(["Llama-8B","Mixtral"])) & (all_df.op != "total")]
total_df = all_df[all_df.op == "total"]

def compute_analytical_features(df):
    """Compute actual FLOPs and bytes for each op based on architecture."""
    tok = df["input"].values.astype(float)
    d = df["d"].values.astype(float)
    h = df["h"].values.astype(float)
    kv = df["kv"].values.astype(float)
    ffn = df["ffn"].values.astype(float)
    E = df["E"].values.astype(float)
    k = df["k"].values.astype(float)
    is_attn = (df["op"].values == "attn").astype(float)
    is_ffn = (df["op"].values == "ffn").astype(float)
    is_norm = df["op"].isin(["norm_pre","norm_post"]).values.astype(float)
    d_h = d / h
    
    # Attention FLOPs: QKV projections + attention matmul + O projection
    # QKV: 3 GEMMs [tok, d] x [d, d_qkv] ~ 2*tok*d*(d + 2*kv*d_h) 
    # Attention: 2*tok*tok*h*d_h (Q@K + A@V)
    # O: 2*tok*d*d
    attn_qkv_flops = 2 * tok * d * (d + 2 * kv * d_h)
    attn_sdpa_flops = 4 * tok * tok * h * d_h  # approximate, ignores flash
    attn_o_flops = 2 * tok * d * d
    attn_total_flops = is_attn * (attn_qkv_flops + attn_sdpa_flops + attn_o_flops)
    
    # Attention bytes: read Q,K,V,O weights + activations
    attn_weight_bytes = is_attn * (d * (d + 2*kv*d_h + d)) * 2  # bf16
    attn_act_bytes = is_attn * tok * d * 4 * 2  # Q,K,V,O activations, bf16
    
    # FFN FLOPs: gate_proj + up_proj + down_proj = 3 GEMMs
    # Dense: 2*tok*d*ffn*3
    # MoE: 2*tpe*d*ffn*3*active_experts (tpe = tok*k/E)
    dense_ffn_flops = is_ffn * (1 - np.minimum(E, 1)) * 2 * tok * d * ffn * 3
    tpe = np.where(E > 0, np.maximum(1, tok * k / E), 0)
    active_exp = np.where(E > 0, np.minimum(tok * k, E), 0)
    moe_ffn_flops = is_ffn * np.minimum(E, 1) * 2 * tpe * d * ffn * 3 * active_exp
    ffn_total_flops = dense_ffn_flops + moe_ffn_flops
    
    # FFN bytes
    ffn_weight_bytes = is_ffn * d * ffn * 3 * 2  # 3 weight matrices, bf16
    moe_weight_bytes = np.where(E > 0, is_ffn * E * d * ffn * 3 * 2, 0)  # all expert weights
    
    # Norm FLOPs/bytes: elementwise on [tok, d]
    norm_flops = is_norm * tok * d * 5
    norm_bytes = is_norm * tok * d * 2 * 3  # read + write + intermediate
    
    total_flops = attn_total_flops + ffn_total_flops + norm_flops
    total_bytes = attn_weight_bytes + attn_act_bytes + ffn_weight_bytes + norm_bytes
    
    # Arithmetic intensity
    arith_intensity = np.where(total_bytes > 0, total_flops / total_bytes, 0)
    
    return np.column_stack([
        tok, np.log2(tok + 1),
        total_flops, np.log2(total_flops + 1),
        total_bytes, np.log2(total_bytes + 1),
        arith_intensity, np.log2(arith_intensity + 1),
        is_attn, is_ffn, is_norm,
        d, ffn, E, k,
        d * d,  # d^2 (attention compute proxy)
        d * ffn,  # d*ffn (FFN compute proxy)
        np.where(E > 0, tpe, tok),  # effective tokens per compute unit
        np.where(E > 0, active_exp, 1),  # active expert count
        np.where(E > 0, moe_weight_bytes, ffn_weight_bytes),  # weight memory pressure
    ])

feat_names = ["tok", "log_tok", "total_flops", "log_flops", "total_bytes", "log_bytes",
    "arith_intensity", "log_ai", "is_attn", "is_ffn", "is_norm",
    "d_model", "ffn_size", "num_experts", "top_k",
    "d_squared", "d_times_ffn", "eff_tokens", "active_experts", "weight_bytes"]

X_train = compute_analytical_features(train_df)
y_train = train_df["lat_us"].values
groups = train_df["model"].values

print(f"Training: {len(X_train)} rows, {X_train.shape[1]} features")

models = {
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGB": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    p = mdl.predict(X_train)
    train_mape = np.mean(np.abs((y_train - p) / np.maximum(y_train, 1))) * 100
    
    logo = LeaveOneGroupOut()
    lp = np.zeros(len(y_train))
    for tr, te in logo.split(X_train, y_train, groups):
        mdl.fit(X_train[tr], y_train[tr]); lp[te] = mdl.predict(X_train[te])
    lomo_mape = np.mean(np.abs((y_train - lp) / np.maximum(y_train, 1))) * 100
    mdl.fit(X_train, y_train)
    print(f"{name}: train={train_mape:.1f}%, LOMO={lomo_mape:.1f}%")

best = models["XGB"]

# Per-op at tok=256
print(f"\nPer-op at tok=256 (trained models):")
for mn in ["Llama-8B", "Mixtral"]:
    sub = train_df[(train_df.model == mn) & (train_df.input == 256)]
    Xs = compute_analytical_features(sub); ps = best.predict(Xs)
    for i, (_, r) in enumerate(sub.iterrows()):
        print(f"  {mn:>10} {r.op:>10}: meas={r.lat_us:>8.1f} pred={ps[i]:>8.1f} ratio={ps[i]/max(r.lat_us,0.1):.2f}x")

# Layer-level for ALL models
print(f"\nLayer predictions (all models):")
print(f"{'Model':>12} {'Tok':>5} {'Pred(us)':>10} {'Meas(us)':>10} {'Ratio':>8} {'Rating':>6}")
print("-"*55)
for mn in ["Llama-8B", "Mixtral", "Qwen-72B", "gpt-oss-20b"]:
    cfg = MODEL_CONFIGS[mn]
    for tok in [1, 64, 256, 512]:
        pred_sum = 0
        for op in ["attn", "ffn", "norm_pre", "norm_post"]:
            row = pd.DataFrame([{"input": tok, "op": op, "model": mn,
                "d": cfg["d"], "h": cfg["h"], "kv": cfg["kv"], "ffn": cfg["ffn"],
                "E": cfg["E"], "k": cfg["k"]}])
            pred_sum += best.predict(compute_analytical_features(row))[0]
        meas_row = total_df[(total_df.model == mn) & (total_df.input == tok)]
        if len(meas_row) > 0:
            m = meas_row.lat_us.values[0]
            ratio = pred_sum / m
            rating = "GOOD" if abs(ratio-1) <= 0.2 else ("OK" if abs(ratio-1) <= 0.3 else "BAD")
            print(f"{mn:>12} {tok:>5} {pred_sum:>10.1f} {m:>10.1f} {ratio:>7.2f}x {rating:>6}")

print(f"\nFeature importance:")
for f, imp in sorted(zip(feat_names, best.feature_importances_), key=lambda x: -x[1])[:10]:
    print(f"  {f:<20}: {imp:.3f}")

with open("llm_predict/profiling/data/A100/perop_analytical_v3.pkl", "wb") as f:
    pickle.dump({"model": best, "features": feat_names, "compute_fn": "compute_analytical_features"}, f)
print("\nSaved")
