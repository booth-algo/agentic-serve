"""Validate XGBoost generalization at unseen seq lengths using isolated layer measurement."""
import os, sys, json, torch, statistics
sys.path.insert(0, "/data/kevinlau/python-packages")
sys.path.insert(0, "/home/kevinlau/llmserve")
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = "cuda:0"
WARMUP = 5
REPEAT = 20
# Unseen seq lengths (not in training: 64,128,256,512,1024)
VAL_GRID = [(1, 96), (1, 192), (1, 384), (1, 768), (3, 96), (3, 192), (3, 384)]

def capture_and_measure(model, layer, bs, seq_len, device):
    captured = {}
    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
        captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
    handle = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
    input_ids = torch.randint(0, 1000, (bs, seq_len), device=device)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    args = captured["args"]
    kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    kwargs["use_cache"] = False
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = layer(*args, **kwargs)
        torch.cuda.synchronize()
    times = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        with torch.no_grad():
            _ = layer(*args, **kwargs)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return statistics.median(times)

model_path = "/data/models/Llama-3.1-8B-Instruct"
print("Loading %s..." % os.path.basename(model_path))
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True)
model.eval()
config = AutoConfig.from_pretrained(model_path)
layer = model.model.layers[16]  # middle layer

results = []
for bs, seq in VAL_GRID:
    try:
        lat = capture_and_measure(model, layer, bs, seq, DEVICE)
        results.append({"batch_size": bs, "seq_len": seq, "latency_ms": lat})
        print("  bs=%d seq=%d: %.3f ms (isolated layer)" % (bs, seq, lat))
    except Exception as e:
        print("  bs=%d seq=%d: ERROR %s" % (bs, seq, e))

with open("experiment/validation_generalization.json", "w") as f:
    json.dump({"model": "Llama-3.1-8B-Instruct", "method": "isolated_layer", "results": results}, f, indent=2)
print("Saved %d results" % len(results))
