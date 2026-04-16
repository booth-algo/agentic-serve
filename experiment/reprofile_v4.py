"""Re-profile specific models at tok ranges where v3 data was garbage.
Uses v4 CUDA events profiler (no sync in pre-hook)."""
import torch, csv, os, sys
sys.path.insert(0, "/data/kevinlau/python-packages")
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Llama-3.1-8B-Instruct"
# Token range to profile (start, end inclusive)
TOK_START = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TOK_END = int(sys.argv[4]) if len(sys.argv) > 4 else 512
model_name = os.path.basename(MODEL_PATH)

WARMUP = 5
REPEAT = 15

print("Loading %s (2 layers) for re-profiling tok=%d-%d..." % (model_name, TOK_START, TOK_END))
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
print("Sub-modules: %s" % sub_names)

results = []
for n_tok in range(TOK_START, TOK_END + 1):
    if n_tok % 50 == 0:
        print("  tok=%d/%d..." % (n_tok, TOK_END), flush=True)

    input_ids = torch.randint(0, 1000, (1, n_tok), device=DEVICE)

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
            lats.append(s.elapsed_time(e) * 1e6)
    lats.sort()
    results.append({"layer_name": "total_block", "input": n_tok, "kv_cache": 0, "tp_size": 1, "latency_ns": int(lats[len(lats)//2])})

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
        results.append({"layer_name": sub_name, "input": n_tok, "kv_cache": 0, "tp_size": 1, "latency_ns": int(lats[len(lats)//2])})

# Save as v4 CSV (same format as v3 for compatibility)
out_dir = "/data/LLMServingSim/llm_profile/perf_models/A100/%s/tp1" % model_name.replace(".", "_").replace("-", "-")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "layers_v4.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency_ns"])
    w.writeheader()
    w.writerows(results)
print("\nSaved %d rows to %s" % (len(results), csv_path))
