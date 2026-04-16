"""Profile real-world serving configs: high-BS decode + long-seq prefill."""
import torch, os, sys, csv
sys.path.insert(0, "/data/kevinlau/python-packages")
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Llama-3.1-8B-Instruct"
model_name = os.path.basename(MODEL_PATH)
WARMUP = 5; REPEAT = 15

OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
          "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post"}

# Real-world gaps:
# 1. High-BS decode (continuous batching): BS=16,32,64 at KV=256,512,1024
# 2. Long-seq prefill: BS=1-4 at seq=2048,4096
DECODE_GRID = [(bs, kv) for bs in [16, 32, 64] for kv in [256, 512, 1024]]
PREFILL_GRID = [(bs, seq) for bs in [1, 2, 4] for seq in [2048, 4096]]

print("Loading %s (2 layers)..." % model_name)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layer = model.model.layers[0]
else:
    raise RuntimeError("Cannot find layers")
sub_names = [n for n, _ in layer.named_children()]

def time_layer(layer_args, layer_kwargs):
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

def time_submodule(sub_mod, layer_args, layer_kwargs):
    lats = []
    with torch.no_grad():
        for _ in range(REPEAT):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            def pre_h(m, a, se=se): se.record()
            def post_h(m, a, o, ee=ee): ee.record()
            h1 = sub_mod.register_forward_pre_hook(pre_h)
            h2 = sub_mod.register_forward_hook(post_h)
            torch.cuda.synchronize()
            _ = layer(*layer_args, **layer_kwargs)
            torch.cuda.synchronize()
            h1.remove(); h2.remove()
            lats.append(se.elapsed_time(ee) * 1e6)
    lats.sort()
    return int(lats[len(lats)//2])

results = []

# === HIGH-BS DECODE ===
print("\n=== High-BS Decode ===")
for bs, kv_len in DECODE_GRID:
    print("  decode bs=%d kv=%d..." % (bs, kv_len), end="", flush=True)
    try:
        prefill_ids = torch.randint(0, 1000, (bs, kv_len), device=DEVICE)
        with torch.no_grad():
            pf_out = model(input_ids=prefill_ids, use_cache=True)
        past_kv = pf_out.past_key_values
        decode_ids = torch.randint(0, 1000, (bs, 1), device=DEVICE)
        captured = {}
        def hook_fn(module, args, kwargs):
            captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
            captured["kwargs"] = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in kwargs.items()}
        h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        with torch.no_grad():
            _ = model(input_ids=decode_ids, past_key_values=past_kv, use_cache=True)
        h.remove()
        layer_args = captured["args"]
        layer_kwargs = dict(captured["kwargs"])

        total_ns = time_layer(layer_args, layer_kwargs)
        results.append({"model_name": model_name, "layer_name": "total_block",
            "bs": bs, "seq": 1, "kv_cache": kv_len, "n_tokens": bs,
            "tp_size": 1, "latency_ns": total_ns, "phase": "decode"})
        for sn in sub_names:
            sub_ns = time_submodule(getattr(layer, sn), layer_args, layer_kwargs)
            results.append({"model_name": model_name, "layer_name": sn,
                "bs": bs, "seq": 1, "kv_cache": kv_len, "n_tokens": bs,
                "tp_size": 1, "latency_ns": sub_ns, "phase": "decode"})
        print(" total=%.0fus" % (total_ns/1000))
        del past_kv, pf_out; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(" OOM"); torch.cuda.empty_cache()
    except Exception as ex:
        print(" ERROR: %s" % ex)

# === LONG-SEQ PREFILL ===
print("\n=== Long-Seq Prefill ===")
for bs, seq in PREFILL_GRID:
    print("  prefill bs=%d seq=%d (ntok=%d)..." % (bs, seq, bs*seq), end="", flush=True)
    try:
        input_ids = torch.randint(0, 1000, (bs, seq), device=DEVICE)
        captured = {}
        def hook_fn(module, args, kwargs):
            captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
            captured["kwargs"] = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in kwargs.items()}
        h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        h.remove()
        layer_args = captured["args"]
        layer_kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
        layer_kwargs["use_cache"] = False

        total_ns = time_layer(layer_args, layer_kwargs)
        results.append({"model_name": model_name, "layer_name": "total_block",
            "bs": bs, "seq": seq, "kv_cache": 0, "n_tokens": bs*seq,
            "tp_size": 1, "latency_ns": total_ns, "phase": "prefill"})
        for sn in sub_names:
            sub_ns = time_submodule(getattr(layer, sn), layer_args, layer_kwargs)
            results.append({"model_name": model_name, "layer_name": sn,
                "bs": bs, "seq": seq, "kv_cache": 0, "n_tokens": bs*seq,
                "tp_size": 1, "latency_ns": sub_ns, "phase": "prefill"})
        print(" total=%.0fus" % (total_ns/1000))
    except torch.cuda.OutOfMemoryError:
        print(" OOM"); torch.cuda.empty_cache()
    except Exception as ex:
        print(" ERROR: %s" % ex)

# Save
out_dir = "/data/LLMServingSim/llm_profile/perf_models/A100/%s/tp1" % model_name.replace(".", "_").replace(" ", "_")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "realworld_profiles.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name","layer_name","bs","seq","kv_cache","n_tokens","tp_size","latency_ns","phase"])
    w.writeheader(); w.writerows(results)
print("\nSaved %d rows to %s" % (len(results), csv_path))
