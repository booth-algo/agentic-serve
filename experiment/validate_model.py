"""Validate v4 predictor against a model: profile prefill+decode, compare predictions.
Usage: python validate_model.py <cuda_device> <model_path> [--decode-only] [--prefill-only]
"""
import torch, os, sys, csv, json, statistics, pickle, math
import numpy as np
sys.path.insert(0, "/data/kevinlau/python-packages")
sys.path.insert(0, "/home/kevinlau/llmserve")
os.chdir("/home/kevinlau/llmserve")
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Llama-3.1-8B-Instruct"
MODE = sys.argv[3] if len(sys.argv) > 3 else "both"
model_name = os.path.basename(MODEL_PATH)
WARMUP = 5; REPEAT = 15

# Load predictor
with open("llmcompass/profiler/profiles/A100/perop_analytical_v4.pkl", "rb") as f:
    pred_data = pickle.load(f)
xgb_model = pred_data["model"]

OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
          "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post"}

def compute_feats(tok, op, d, h, kv, ffn, E, k, bs=1, seq=None, kvc=0):
    if seq is None: seq = tok
    tok=float(tok);d=float(d);h=float(h);kv_h=float(kv);ffn=float(ffn)
    E=float(E);k=float(k);dh=d/h;bs=float(bs);seq=float(seq)
    ia=float(op=="attn");iff=float(op=="ffn");inn=float(op in("norm_pre","norm_post"))
    tpe=max(1,tok*k/max(E,1))if E>0 else 0;ae=min(tok*k,E)if E>0 else 0
    ekv=max(float(kvc),seq)if kvc>0 else seq
    af=ia*(2*tok*d*(d+2*kv_h*dh)+4*bs*h*seq*ekv*dh+2*tok*d*d)
    df=iff*(1-min(E,1))*2*tok*d*ffn*3;mf=iff*(2*tpe*d*ffn*3*ae if E>0 else 0)
    nf=inn*tok*d*5;tf=af+df+mf+nf
    wt=ia*d*(d+2*kv_h*dh+d)*2+iff*(E*d*ffn*3*2 if E>0 else d*ffn*3*2)+inn*d*2
    ai=tf/wt if wt>0 else 0;aq=ia*bs*seq*ekv*h*dh
    return [tok,math.log2(tok+1),tf,math.log2(tf+1),wt,math.log2(wt+1),ai,math.log2(ai+1),
            ia,iff,inn,d,ffn,E,k,d*d,d*ffn,tpe if E>0 else tok,ae if E>0 else 1,wt,
            bs,seq,math.log2(bs+1),math.log2(seq+1),aq,math.log2(aq+1) if aq>0 else 0]

def predict_layer(d, h, kv, ffn, E, k, bs, seq, kvc=0):
    tok = bs * seq
    ps = 0
    for op in ["attn", "ffn", "norm_pre", "norm_post"]:
        f = compute_feats(tok, op, d, h, kv, ffn, E, k, bs, seq, kvc)
        ps += max(0, xgb_model.predict(np.array([f]))[0])
    # MoE decode correction
    is_decode = (seq <= 1 and tok <= 8)
    if is_decode and E >= 8:
        moe_scale = 1.0 + 0.15 * E
        ps = ps * min(moe_scale, 4.0)
    return ps  # microseconds

# Load model
print("Loading %s (2 layers)..." % model_name)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
d = config.hidden_size
h = config.num_attention_heads
kv = getattr(config, "num_key_value_heads", h)
ffn = config.intermediate_size
n_layers = config.num_hidden_layers
E = getattr(config, "num_local_experts", 0)
k = getattr(config, "num_experts_per_tok", 0)
print("Arch: d=%d h=%d kv=%d ffn=%d layers=%d E=%d k=%d" % (d, h, kv, ffn, n_layers, E, k))

config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layer = model.model.layers[0]
elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    layer = model.transformer.h[0]
else:
    print("ERROR: Cannot find transformer layers")
    sys.exit(1)

sub_names = [n for n, _ in layer.named_children()]
print("Sub-modules: %s" % sub_names)

def time_layer_full(layer_args, layer_kwargs):
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = layer(*layer_args, **layer_kwargs)
        torch.cuda.synchronize()
    lats = []
    with torch.no_grad():
        for _ in range(REPEAT):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record()
            _ = layer(*layer_args, **layer_kwargs)
            e.record(); torch.cuda.synchronize()
            lats.append(s.elapsed_time(e) * 1e6)
    lats.sort()
    return int(lats[len(lats)//2])

results = []

# === PREFILL ===
if MODE in ("both", "prefill", "--prefill-only"):
    print("\n=== PREFILL VALIDATION ===")
    prefill_grid = [(1, 64), (1, 128), (1, 256), (1, 512), (1, 1024),
                    (4, 128), (4, 512), (8, 256), (8, 512)]

    for bs, seq in prefill_grid:
        try:
            input_ids = torch.randint(0, 1000, (bs, seq), device=DEVICE)
            captured = {}
            def hook_fn(module, args, kwargs):
                captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
                captured["kwargs"] = {kk: v.detach() if hasattr(v, "detach") else v for kk, v in kwargs.items()}
            hk = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            hk.remove()
            layer_args = captured["args"]
            layer_kwargs = {kk: v for kk, v in captured["kwargs"].items() if kk != "past_key_values"}
            layer_kwargs["use_cache"] = False

            meas_ns = time_layer_full(layer_args, layer_kwargs)
            meas_us = meas_ns / 1000 if meas_ns > 10000 else meas_ns  # handle ns vs us
            pred_us = predict_layer(d, h, kv, ffn, E, k, bs, seq)
            meas_ms = meas_ns / 1e6
            pred_ms = pred_us / 1e3
            err = (pred_ms - meas_ms) / meas_ms * 100
            ok = abs(err) <= 20
            mark = "GOOD" if ok else "BAD"

            results.append({
                "model": model_name, "phase": "prefill", "bs": bs, "seq": seq,
                "measured_ms": round(meas_ms, 3), "predicted_ms": round(pred_ms, 3),
                "error_pct": round(err, 1), "rating": mark
            })
            print("  bs=%d seq=%4d: meas=%.3fms pred=%.3fms err=%+.1f%% %s" % (bs, seq, meas_ms, pred_ms, err, mark))
        except torch.cuda.OutOfMemoryError:
            print("  bs=%d seq=%4d: OOM" % (bs, seq))
            torch.cuda.empty_cache()
        except Exception as ex:
            print("  bs=%d seq=%4d: ERROR %s" % (bs, seq, ex))

# === DECODE ===
if MODE in ("both", "decode", "--decode-only"):
    print("\n=== DECODE VALIDATION ===")
    decode_grid = [(1, 256), (1, 512), (1, 1024), (4, 512), (8, 512)]

    for bs, kv_len in decode_grid:
        try:
            prefill_ids = torch.randint(0, 1000, (bs, kv_len), device=DEVICE)
            with torch.no_grad():
                pf_out = model(input_ids=prefill_ids, use_cache=True)
            past_kv = pf_out.past_key_values
            decode_ids = torch.randint(0, 1000, (bs, 1), device=DEVICE)

            captured = {}
            def hook_fn(module, args, kwargs):
                captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
                captured["kwargs"] = {kk: (v.detach() if hasattr(v, "detach") else v) for kk, v in kwargs.items()}
            hk = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
            with torch.no_grad():
                _ = model(input_ids=decode_ids, past_key_values=past_kv, use_cache=True)
            hk.remove()
            layer_args = captured["args"]
            layer_kwargs = dict(captured["kwargs"])

            meas_ns = time_layer_full(layer_args, layer_kwargs)
            pred_us = predict_layer(d, h, kv, ffn, E, k, bs, 1, kv_len)
            meas_ms = meas_ns / 1e6
            pred_ms = pred_us / 1e3
            err = (pred_ms - meas_ms) / meas_ms * 100
            ok = abs(err) <= 20
            mark = "GOOD" if ok else "BAD"

            results.append({
                "model": model_name, "phase": "decode", "bs": bs, "kv_cache": kv_len,
                "measured_ms": round(meas_ms, 3), "predicted_ms": round(pred_ms, 3),
                "error_pct": round(err, 1), "rating": mark
            })
            print("  bs=%d kv=%4d: meas=%.3fms pred=%.3fms err=%+.1f%% %s" % (bs, kv_len, meas_ms, pred_ms, err, mark))

            del past_kv, pf_out; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("  bs=%d kv=%4d: OOM" % (bs, kv_len))
            torch.cuda.empty_cache()
        except Exception as ex:
            print("  bs=%d kv=%4d: ERROR %s" % (bs, kv_len, ex))

# Summary
good = sum(1 for r in results if r["rating"] == "GOOD")
total = len(results)
print("\n=== SUMMARY: %s ===" % model_name)
print("  GOOD: %d/%d (%.0f%%)" % (good, total, 100*good/total if total else 0))
for phase in ["prefill", "decode"]:
    pr = [r for r in results if r["phase"] == phase]
    pg = sum(1 for r in pr if r["rating"] == "GOOD")
    if pr:
        print("  %s: %d/%d (%.0f%%)" % (phase, pg, len(pr), 100*pg/len(pr)))

# Save
out_path = "experiment/validation_%s.json" % model_name.replace(".", "_").replace("-", "_")
with open(out_path, "w") as f:
    json.dump({"model": model_name, "arch": {"d": d, "h": h, "kv": kv, "ffn": ffn, "E": E, "k": k, "n_layers": n_layers}, "results": results}, f, indent=2)
print("Saved to %s" % out_path)
