#!/usr/bin/env python3
"""Profile MoE models at small token counts (M=1-256) on H100.
Measures isolated layer + sub-module latencies via CUDA events.

Usage:
    python experiment/profile_moe_h100.py cuda:0 /workspace/models/Mixtral-8x7B-Instruct-v0.1 1 256
    python experiment/profile_moe_h100.py cuda:0 /workspace/models/gpt-oss-20b 1 256
"""
import torch, csv, os, sys, json
from pathlib import Path

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
TOK_START = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TOK_END = int(sys.argv[4]) if len(sys.argv) > 4 else 256
model_name = os.path.basename(MODEL_PATH)

WARMUP = 5
REPEAT = 20

from transformers import AutoConfig, AutoModelForCausalLM

print(f"Loading {model_name} (2 layers) for profiling tok={TOK_START}-{TOK_END} on {DEVICE}...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Extract arch info
d = config.hidden_size
h = config.num_attention_heads
kv = getattr(config, "num_key_value_heads", h)
ffn = config.intermediate_size
E = getattr(config, "num_local_experts", 0)
k = getattr(config, "num_experts_per_tok", 0)
n_layers = config.num_hidden_layers
print(f"Arch: d={d} h={h} kv={kv} ffn={ffn} E={E} k={k} layers={n_layers}")

config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters: {params_m:.1f}M")

layer = model.model.layers[0]
sub_names = [n for n, _ in layer.named_children()]
print(f"Sub-modules: {sub_names}")

results = []
# Token counts: 1-16 by 1, then 32, 64, 128, 256
tok_range = list(range(TOK_START, min(17, TOK_END + 1)))
for t in [32, 64, 128, 256]:
    if TOK_START <= t <= TOK_END and t not in tok_range:
        tok_range.append(t)
tok_range.sort()

for n_tok in tok_range:
    print(f"  tok={n_tok}...", end=" ", flush=True)

    input_ids = torch.randint(0, 1000, (1, n_tok), device=DEVICE)

    # Capture layer inputs
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
            lats.append(s.elapsed_time(e) * 1e3)  # microseconds
    lats.sort()
    total_us = lats[len(lats) // 2]
    results.append({"layer_name": "total_block", "n_tokens": n_tok, "latency_us": round(total_us, 2)})

    # Time each sub-module
    sub_lats = {}
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
                lats.append(start_evt.elapsed_time(end_evt) * 1e3)  # us

        lats.sort()
        sub_us = lats[len(lats) // 2]
        sub_lats[sub_name] = round(sub_us, 2)
        results.append({"layer_name": sub_name, "n_tokens": n_tok, "latency_us": round(sub_us, 2)})

    print(f"total={total_us:.1f}us  " + "  ".join(f"{n}={v:.1f}" for n, v in sub_lats.items()))

# Save results
safe_name = model_name.replace(".", "_").replace("-", "_")
out_dir = Path(f"experiment/profiles_h100")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / f"{safe_name}_moe_profile.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer_name", "n_tokens", "latency_us"])
    w.writeheader()
    w.writerows(results)

# Also save metadata
meta_path = out_dir / f"{safe_name}_meta.json"
with open(meta_path, "w") as f:
    json.dump({
        "model": model_name, "device": DEVICE, "hardware": "H100-SXM5-80GB",
        "arch": {"d": d, "h": h, "kv": kv, "ffn": ffn, "E": E, "k": k, "n_layers": n_layers},
        "sub_modules": sub_names, "tok_range": tok_range,
        "warmup": WARMUP, "repeat": REPEAT,
    }, f, indent=2)

print(f"\nSaved {len(results)} rows to {csv_path}")
print(f"Metadata: {meta_path}")
