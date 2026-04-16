"""Phase 1: Profile sub-modules at long sequences AND BS>1 for per-op predictor training.
Combines Phase 1.1 (long seq) and Phase 1.2 (BS>1) into one script.
Uses v4-style CUDA events (no sync in pre-hook)."""
import torch, csv, os, sys, json
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Llama-3.1-8B-Instruct"
model_name = os.path.basename(MODEL_PATH)

WARMUP = 5
REPEAT = 15

# Profiling grid: (bs, seq) combinations
# Phase 1.1: long sequences at BS=1
# Phase 1.2: BS>1 at various seq lengths
GRID = [
    # Long seq (BS=1) - fills gap at seq>=1024
    (1, 768), (1, 1024), (1, 1536), (1, 2048),
    # BS>1 - new dimension for per-op predictor
    (2, 128), (2, 256), (2, 512),
    (4, 128), (4, 256), (4, 512),
    (8, 128), (8, 256), (8, 512),
    (16, 128), (16, 256), (16, 512),
    # Cross combinations
    (4, 1024), (8, 1024),
    (2, 64), (4, 64), (8, 64),
]

sys.path.insert(0, "/data/kevinlau/python-packages")
print("Loading %s (2 layers)..." % model_name)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()

# Find layers
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layer = model.model.layers[0]
elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    layer = model.transformer.h[0]
else:
    raise RuntimeError("Cannot find transformer layers in model")

sub_names = [n for n, _ in layer.named_children()]
print("Sub-modules: %s" % sub_names)

# Get model config for metadata
arch = {
    "d_model": getattr(config, "hidden_size", 0),
    "n_heads": getattr(config, "num_attention_heads", 0),
    "n_kv_heads": getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 0)),
    "intermediate_size": getattr(config, "intermediate_size", 0),
    "num_experts": getattr(config, "num_local_experts", 0),
    "top_k": getattr(config, "num_experts_per_tok", 0),
}
print("Arch: %s" % arch)

results = []
for bs, seq in GRID:
    n_tok = bs * seq
    print("  bs=%d seq=%d (ntok=%d)..." % (bs, seq, n_tok), end="", flush=True)

    try:
        input_ids = torch.randint(0, 1000, (bs, seq), device=DEVICE)

        # Capture layer inputs
        captured = {}
        def hook_fn(module, args, kwargs):
            captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
            captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
        h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        h.remove()
        layer_args = captured["args"]
        layer_kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
        layer_kwargs["use_cache"] = False

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = layer(*layer_args, **layer_kwargs)
            torch.cuda.synchronize()

        # Time full layer
        lats = []
        with torch.no_grad():
            for _ in range(REPEAT):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                _ = layer(*layer_args, **layer_kwargs)
                e.record()
                torch.cuda.synchronize()
                lats.append(s.elapsed_time(e) * 1e6)  # ns
        lats.sort()
        total_ns = int(lats[len(lats)//2])
        results.append({
            "model_name": model_name, "layer_name": "total_block",
            "bs": bs, "seq": seq, "n_tokens": n_tok,
            "kv_cache": 0, "tp_size": 1, "latency_ns": total_ns
        })

        # Time each sub-module
        for sub_name in sub_names:
            sub_mod = getattr(layer, sub_name)
            lats = []
            with torch.no_grad():
                for _ in range(REPEAT):
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)

                    def pre_h(module, args, se=start_evt):
                        se.record()
                    def post_h(module, args, output, ee=end_evt):
                        ee.record()

                    h1 = sub_mod.register_forward_pre_hook(pre_h)
                    h2 = sub_mod.register_forward_hook(post_h)

                    torch.cuda.synchronize()
                    _ = layer(*layer_args, **layer_kwargs)
                    torch.cuda.synchronize()

                    h1.remove()
                    h2.remove()
                    lats.append(start_evt.elapsed_time(end_evt) * 1e6)

            lats.sort()
            sub_ns = int(lats[len(lats)//2])
            results.append({
                "model_name": model_name, "layer_name": sub_name,
                "bs": bs, "seq": seq, "n_tokens": n_tok,
                "kv_cache": 0, "tp_size": 1, "latency_ns": sub_ns
            })

        print(" total=%.0fus" % (total_ns / 1000))

    except torch.cuda.OutOfMemoryError:
        print(" OOM, skipping")
        torch.cuda.empty_cache()
    except Exception as ex:
        print(" ERROR: %s" % ex)

# Save results
out_dir = "/data/LLMServingSim/llm_profile/perf_models/A100/%s/tp1" % model_name.replace(".", "_").replace("-", "-")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "phase1_bsseq_profiles.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "layer_name", "bs", "seq", "n_tokens", "kv_cache", "tp_size", "latency_ns"])
    w.writeheader()
    w.writerows(results)
print("\nSaved %d rows to %s" % (len(results), csv_path))

# Also save as JSON with metadata
json_path = os.path.join(out_dir, "phase1_bsseq_profiles.json")
with open(json_path, "w") as f:
    json.dump({"model_name": model_name, "arch": arch, "results": results}, f, indent=2)
print("Saved JSON to %s" % json_path)
