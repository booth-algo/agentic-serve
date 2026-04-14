"""Per-op predictor v4: v3 analytical approach + Phase 1 data (long seq + BS>1).
Same features as v3 but trained on wider token range. Adds bs/seq as extra features."""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/data/kevinlau/python-packages")
sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")

from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor
import math

MODEL_CONFIGS = {
    "Llama-8B": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 0, "k": 0},
    "Mixtral": {"d": 4096, "h": 32, "kv": 8, "ffn": 14336, "E": 8, "k": 2},
    "Qwen-72B": {"d": 8192, "h": 64, "kv": 8, "ffn": 29568, "E": 0, "k": 0},
    "gpt-oss-20b": {"d": 2880, "h": 64, "kv": 8, "ffn": 2880, "E": 32, "k": 4},
}

OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
           "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post"}

V3_PATHS = {
    "Llama-8B": "/data/LLMServingSim/llm_profile/perf_models/A100/Llama-3_1-8B-Instruct/tp1/layers_v3.csv",
    "Mixtral": "/data/LLMServingSim/llm_profile/perf_models/A100/Mixtral-8x7B-Instruct/tp1/layers_v3.csv",
    "Qwen-72B": "/data/LLMServingSim/llm_profile/perf_models/A100/Qwen2_5-72B-Instruct/tp1/layers_v3.csv",
    "gpt-oss-20b": "/data/LLMServingSim/llm_profile/perf_models/A100/gpt-oss-20b/tp1/layers_v3.csv",
}

PHASE1_PATHS = {
    "Llama-8B": "/data/LLMServingSim/llm_profile/perf_models/A100/Llama-3_1-8B-Instruct/tp1/phase1_bsseq_profiles.csv",
    "Mixtral": "/data/LLMServingSim/llm_profile/perf_models/A100/Mixtral-8x7B-Instruct/tp1/phase1_bsseq_profiles.csv",
    "Qwen-72B": "/data/LLMServingSim/llm_profile/perf_models/A100/Qwen2_5-72B-Instruct/tp1/phase1_bsseq_profiles.csv",
    "gpt-oss-20b": "/data/LLMServingSim/llm_profile/perf_models/A100/gpt-oss-20b/tp1/phase1_bsseq_profiles.csv",
}

# Use the SAME feature computation as _compute_perop_features in transformer.py
# This ensures training/inference feature consistency
def compute_features_like_transformer(tok, op, d, h, kv, ffn, E, k, bs=1, seq=None):
    """Replicate _compute_perop_features from transformer.py, plus bs/seq."""
    if seq is None:
        seq = tok  # BS=1 default
    tok = float(tok); d = float(d); h = float(h)
    kv = float(kv); ffn = float(ffn)
    E_f = float(E); k_f = float(k); d_h = d / h
    bs = float(bs); seq = float(seq)

    is_a = float(op == "attn")
    is_f = float(op == "ffn")
    is_n = float(op in ("norm_pre", "norm_post"))

    tpe = max(1, tok * k_f / max(E_f, 1)) if E_f > 0 else 0
    aexp = min(tok * k_f, E_f) if E_f > 0 else 0

    attn_fl = is_a * (2*tok*d*(d + 2*kv*d_h) + 4*tok*tok*h*d_h + 2*tok*d*d)
    dffn_fl = is_f * (1 - min(E_f, 1)) * 2*tok*d*ffn*3
    mffn_fl = is_f * (2*tpe*d*ffn*3*aexp if E_f > 0 else 0)
    norm_fl = is_n * tok * d * 5
    total_fl = attn_fl + dffn_fl + mffn_fl + norm_fl

    wt = (is_a * d * (d + 2*kv*d_h + d) * 2 +
          is_f * (E_f*d*ffn*3*2 if E_f > 0 else d*ffn*3*2) +
          is_n * d * 2)
    ai = total_fl / wt if wt > 0 else 0

    # v3 features (20)
    v3 = [tok, math.log2(tok+1), total_fl, math.log2(total_fl+1),
          wt, math.log2(wt+1), ai, math.log2(ai+1),
          is_a, is_f, is_n, d, ffn, E_f, k_f,
          d*d, d*ffn, tpe if E_f > 0 else tok, aexp if E_f > 0 else 1, wt]

    # v4 additions: bs, seq, attention quadratic (bs*seq^2 vs tok^2)
    attn_quad = is_a * bs * seq * seq * h * d_h
    v4_extra = [bs, seq, math.log2(bs+1), math.log2(seq+1),
                attn_quad, math.log2(attn_quad+1) if attn_quad > 0 else 0]

    return v3 + v4_extra


FEAT_NAMES = ["tok", "log_tok", "flops", "log_flops", "wt_bytes", "log_wt",
              "ai", "log_ai", "is_attn", "is_ffn", "is_norm",
              "d", "ffn", "E", "k", "d2", "d_ffn", "eff_tok", "a_exp", "tot_wt",
              # v4 additions
              "bs", "seq", "log_bs", "log_seq", "attn_quad", "log_attn_quad"]


def load_v3_data():
    rows = []
    for name, path in V3_PATHS.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"latency(ns)": "latency_ns", "input": "n_tokens", "layer_name": "op_name"})
        df["op"] = df["op_name"].map(OP_MAP)
        df = df.dropna(subset=["op"])
        cfg = MODEL_CONFIGS[name]
        for _, r in df.iterrows():
            tok = int(r["n_tokens"])
            feats = compute_features_like_transformer(
                tok, r["op"], cfg["d"], cfg["h"], cfg["kv"], cfg["ffn"], cfg["E"], cfg["k"],
                bs=1, seq=tok)
            rows.append({
                "model": name, "op": r["op"], "bs": 1, "seq": tok,
                "lat_us": r["latency_ns"] / 1000,
                "features": feats,
            })
        print("  v3 %s: %d rows" % (name, len(df)))
    return rows


def load_phase1_data():
    rows = []
    for name, path in PHASE1_PATHS.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"layer_name": "op_name"})
        df["op"] = df["op_name"].map(OP_MAP)
        df = df.dropna(subset=["op"])
        cfg = MODEL_CONFIGS[name]
        for _, r in df.iterrows():
            bs = int(r["bs"]); seq = int(r["seq"]); tok = int(r["n_tokens"])
            feats = compute_features_like_transformer(
                tok, r["op"], cfg["d"], cfg["h"], cfg["kv"], cfg["ffn"], cfg["E"], cfg["k"],
                bs=bs, seq=seq)
            rows.append({
                "model": name, "op": r["op"], "bs": bs, "seq": seq,
                "lat_us": r["latency_ns"] / 1000,
                "features": feats,
            })
        print("  Phase1 %s: %d rows" % (name, len(df)))
    return rows


def main():
    print("=" * 70)
    print("Per-Op Predictor v4 (v3 features + bs/seq + Phase 1 data)")
    print("=" * 70)

    print("\nLoading data...")
    v3_rows = load_v3_data()
    p1_rows = load_phase1_data()
    all_rows = v3_rows + p1_rows

    models_arr = np.array([r["model"] for r in all_rows])
    X = np.array([r["features"] for r in all_rows])
    y = np.array([r["lat_us"] for r in all_rows])
    bs_arr = np.array([r["bs"] for r in all_rows])
    seq_arr = np.array([r["seq"] for r in all_rows])
    ops_arr = np.array([r["op"] for r in all_rows])

    print("Total: %d rows, %d features" % (X.shape[0], X.shape[1]))
    print("Models: %s" % sorted(set(models_arr)))
    print("BS range: %s" % sorted(set(bs_arr)))
    print("Seq range: %d-%d" % (seq_arr.min(), seq_arr.max()))

    # Train
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X, y)
    train_preds = xgb.predict(X)
    train_errs = np.abs((y - train_preds) / np.maximum(y, 0.01))
    print("\nTrain: MAPE=%.1f%%, GOOD=%d/%d" % (train_errs.mean()*100, (train_errs<=0.2).sum(), len(y)))

    # LOMO
    print("\n" + "=" * 70)
    print("LOMO (Leave-One-Model-Out)")
    print("=" * 70)
    logo = LeaveOneGroupOut()
    lomo_preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, models_arr):
        m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
        m.fit(X[tr], y[tr])
        lomo_preds[te] = m.predict(X[te])

    lomo_errs = np.abs((y - lomo_preds) / np.maximum(y, 0.01))
    lomo_good = (lomo_errs <= 0.2).sum()
    print("Overall LOMO: MAPE=%.1f%%, GOOD=%d/%d (%.1f%%)" % (
        lomo_errs.mean()*100, lomo_good, len(y), 100*lomo_good/len(y)))

    for mn in sorted(set(models_arr)):
        mask = models_arr == mn
        errs_m = lomo_errs[mask]
        good_m = (errs_m <= 0.2).sum()
        print("  %s: MAPE=%.1f%%, GOOD=%d/%d (%.1f%%)" % (
            mn, errs_m.mean()*100, good_m, mask.sum(), 100*good_m/mask.sum()))
        for op in ["attn", "ffn", "norm_pre", "norm_post"]:
            op_mask = mask & (ops_arr == op)
            if op_mask.sum() == 0: continue
            op_errs = lomo_errs[op_mask]
            print("    %10s: MAPE=%5.1f%%, GOOD=%d/%d" % (op, op_errs.mean()*100, (op_errs<=0.2).sum(), op_mask.sum()))

    # Layer-level LOMO: sum per-op at each (model, bs, seq)
    print("\n" + "=" * 70)
    print("LOMO Layer-Level (BS>1 + long seq)")
    print("=" * 70)
    layer_results = {}
    for i in range(len(all_rows)):
        key = (all_rows[i]["model"], all_rows[i]["bs"], all_rows[i]["seq"])
        if key not in layer_results:
            layer_results[key] = {"meas": 0.0, "pred": 0.0}
        layer_results[key]["meas"] += y[i]
        layer_results[key]["pred"] += lomo_preds[i]

    total_good = 0
    total_count = 0
    for (mn, bs, seq), v in sorted(layer_results.items()):
        err = (v["pred"] - v["meas"]) / max(v["meas"], 0.01) * 100
        good = abs(err) <= 20
        total_good += int(good)
        total_count += 1
        # Only print interesting ones
        if bs > 1 or seq in [64, 256, 512, 1024, 2048]:
            mark = "OK" if good else "!!"
            print("  %12s bs=%2d seq=%5d: meas=%8.1fus pred=%8.1fus err=%+6.1f%% %s" % (
                mn, bs, seq, v["meas"], v["pred"], err, mark))

    print("\nLayer LOMO: GOOD=%d/%d (%.1f%%)" % (total_good, total_count, 100*total_good/total_count))

    # Feature importance
    print("\nFeature importance:")
    imp = xgb.feature_importances_
    for f, i in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])[:12]:
        print("  %20s: %.4f" % (f, i))

    # Save
    save_path = "llmcompass/profiler/profiles/A100/perop_analytical_v4.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "model": xgb, "features": FEAT_NAMES, "version": "v4",
            "n_training": len(X), "models": sorted(set(models_arr)),
        }, f)
    print("\nSaved to %s" % save_path)

    # Also test: what does inference look like for BS>1?
    print("\n" + "=" * 70)
    print("Inference test: Llama-8B bs=8 seq=512")
    print("=" * 70)
    cfg = MODEL_CONFIGS["Llama-8B"]
    total_pred = 0
    for op in ["attn", "ffn", "norm_pre", "norm_post"]:
        feats = compute_features_like_transformer(
            4096, op, cfg["d"], cfg["h"], cfg["kv"], cfg["ffn"], cfg["E"], cfg["k"],
            bs=8, seq=512)
        pred = xgb.predict(np.array([feats]))[0]
        total_pred += pred
        print("  %10s: %.1f us" % (op, pred))
    print("  Total: %.1f us (measured ~9189 us)" % total_pred)


if __name__ == "__main__":
    main()
