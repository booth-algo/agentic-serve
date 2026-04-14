"""Profile each sub-module INDIVIDUALLY to avoid nested timing issues."""
import torch, csv, os, sys, statistics
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = sys.argv[1]
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
model_name = os.path.basename(MODEL_PATH)
WARMUP = 5; REPEAT = 15

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()
layer = model.model.layers[0]

# Get sub-module names
sub_names = [n for n, _ in layer.named_children()]
print(f"Model: {model_name}, Sub-modules: {sub_names}")

results = []
for n_tok in range(1, 513):
    input_ids = torch.randint(0, 1000, (1, n_tok), device=DEVICE)
    
    # Capture layer inputs
    captured = {}
    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
        captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
    h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
    with torch.no_grad(): _ = model(input_ids=input_ids)
    h.remove()
    
    layer_args = captured["args"]
    layer_kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    layer_kwargs["use_cache"] = False
    
    # Time full layer
    with torch.no_grad():
        for _ in range(WARMUP): _ = layer(*layer_args, **layer_kwargs)
        torch.cuda.synchronize()
        lats = []
        for _ in range(REPEAT):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record()
            _ = layer(*layer_args, **layer_kwargs)
            e.record(); torch.cuda.synchronize()
            lats.append(s.elapsed_time(e) * 1e6)  # ns
        lats.sort()
        results.append({"layer_name": "total_block", "input": n_tok, "kv_cache": 0, 
                        "tp_size": 1, "latency_ns": int(lats[len(lats)//2])})
    
    # Time each sub-module by replacing with timed wrapper
    # Run full layer but only time ONE sub-module at a time
    for sub_name in sub_names:
        sub_mod = getattr(layer, sub_name)
        with torch.no_grad():
            for _ in range(WARMUP): _ = layer(*layer_args, **layer_kwargs)
            torch.cuda.synchronize()
            lats = []
            for _ in range(REPEAT):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                # Hook only this one sub-module
                def pre_h(module, args, start=start):
                    torch.cuda.synchronize()
                    start.record()
                def post_h(module, args, output, end=end):
                    end.record()
                
                h1 = sub_mod.register_forward_pre_hook(pre_h)
                h2 = sub_mod.register_forward_hook(post_h)
                _ = layer(*layer_args, **layer_kwargs)
                torch.cuda.synchronize()
                h1.remove(); h2.remove()
                lats.append(start.elapsed_time(end) * 1e6)  # ns
            
            lats.sort()
            results.append({"layer_name": sub_name, "input": n_tok, "kv_cache": 0,
                           "tp_size": 1, "latency_ns": int(lats[len(lats)//2])})
    
    if n_tok % 100 == 0:
        print(f"  {n_tok}/512", flush=True)

# Save
safe_name = model_name.replace(".", "_")
out_dir = f"/data/LLMServingSim/llm_profile/perf_models/A100/{safe_name}/tp1"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/layers_v3.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency_ns"])
    w.writeheader()
    for r in results: w.writerow(r)
print(f"Saved {len(results)} rows to {out_path}")
print("DONE")
