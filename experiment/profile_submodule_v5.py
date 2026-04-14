"""v5: Profile sub-modules using CUDA events only — no sync between modules.
Sync ONCE at end of forward pass, then read all event timestamps."""
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
sub_names = [n for n, _ in layer.named_children()]
print(f"Model: {model_name}, Sub-modules: {sub_names}")

# Register ALL hooks simultaneously — no sync in hooks
timings = {}
hooks = []

def make_hooks(name, mod):
    def pre_hook(module, inputs):
        module._evt_start = torch.cuda.Event(enable_timing=True)
        module._evt_end = torch.cuda.Event(enable_timing=True)
        module._evt_start.record()
    def post_hook(module, inputs, output):
        module._evt_end.record()
        # NO sync here — collect events, sync once at end
    hooks.append(mod.register_forward_pre_hook(pre_hook))
    hooks.append(mod.register_forward_hook(post_hook))

for name in sub_names:
    make_hooks(name, getattr(layer, name))

# Also hook the full layer
layer_start = [None]
layer_end = [None]
def layer_pre(module, inputs):
    layer_start[0] = torch.cuda.Event(enable_timing=True)
    layer_end[0] = torch.cuda.Event(enable_timing=True)
    layer_start[0].record()
def layer_post(module, inputs, output):
    layer_end[0].record()
hooks.append(layer.register_forward_pre_hook(layer_pre))
hooks.append(layer.register_forward_hook(layer_post))

results = []
for n_tok in range(1, 513):
    input_ids = torch.randint(0, 1000, (1, n_tok), device=DEVICE)
    
    # Capture layer inputs
    captured = {}
    def cap_hook(module, args, kwargs):
        captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
        captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
    h = layer.register_forward_pre_hook(cap_hook, with_kwargs=True)
    with torch.no_grad(): _ = model(input_ids=input_ids)
    h.remove()
    layer_args = captured["args"]
    layer_kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    layer_kwargs["use_cache"] = False
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP): _ = layer(*layer_args, **layer_kwargs)
        torch.cuda.synchronize()
    
    # Measure: run REPEAT times, collect all events, sync ONCE per run
    all_timings = {name: [] for name in sub_names}
    all_timings["total_block"] = []
    
    with torch.no_grad():
        for _ in range(REPEAT):
            _ = layer(*layer_args, **layer_kwargs)
            torch.cuda.synchronize()  # ONE sync after full forward
            
            # Read all sub-module timings
            for name in sub_names:
                mod = getattr(layer, name)
                if hasattr(mod, "_evt_start") and hasattr(mod, "_evt_end"):
                    elapsed_us = mod._evt_start.elapsed_time(mod._evt_end) * 1000  # ms -> us
                    all_timings[name].append(elapsed_us)
            
            # Read layer timing
            if layer_start[0] is not None and layer_end[0] is not None:
                all_timings["total_block"].append(
                    layer_start[0].elapsed_time(layer_end[0]) * 1000)
    
    # Store medians
    for name in list(sub_names) + ["total_block"]:
        lats = all_timings[name]
        if lats:
            lats.sort()
            median_us = lats[len(lats)//2]
            results.append({"layer_name": name, "input": n_tok, "kv_cache": 0,
                           "tp_size": 1, "latency_ns": int(median_us * 1000)})  # us -> ns
    
    if n_tok % 100 == 0: print(f"  {n_tok}/512", flush=True)

# Cleanup hooks
for h in hooks: h.remove()

# Save
safe_name = model_name.replace(".", "_")
out_dir = f"/data/LLMServingSim/llm_profile/perf_models/A100/{safe_name}/tp1"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/layers_v5.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency_ns"])
    w.writeheader()
    for r in results: w.writerow(r)
print(f"Saved {len(results)} rows to {out_path}")

# Quick sanity check at tok=256
import pandas as pd
df = pd.DataFrame(results)
t = df[df["input"]==256]
tb = t[t.layer_name=="total_block"].latency_ns.values[0]/1000 if len(t[t.layer_name=="total_block"]) > 0 else 0
parts = t[t.layer_name!="total_block"].latency_ns.sum()/1000
print(f"\ntok=256 sanity check:")
for _, r in t.iterrows():
    print(f"  {r.layer_name:>25}: {r.latency_ns/1000:.1f} us")
print(f"  parts/total = {parts/tb:.2f}" if tb > 0 else "")
print("DONE")
