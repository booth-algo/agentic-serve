"""Profile decode phase: single token with variable KV cache lengths.
Uses CUDA events on isolated layer for accurate sub-module timing."""
import torch, os, sys, csv
sys.path.insert(0, "/data/kevinlau/python-packages")
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Llama-3.1-8B-Instruct"
model_name = os.path.basename(MODEL_PATH)
WARMUP = 5
REPEAT = 15

# KV cache lengths to profile (simulating different decode positions)
KV_LENGTHS = [64, 128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 4, 8]

print("Loading %s (2 layers) for decode profiling..." % model_name)
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

OP_MAP = {"self_attn": "attn", "mlp": "ffn", "block_sparse_moe": "ffn",
          "input_layernorm": "norm_pre", "post_attention_layernorm": "norm_post"}

results = []
for bs in BATCH_SIZES:
    for kv_len in KV_LENGTHS:
        print("  bs=%d kv_len=%d..." % (bs, kv_len), end="", flush=True)
        try:
            # Build prefill to populate KV cache
            prefill_ids = torch.randint(0, 1000, (bs, kv_len), device=DEVICE)
            with torch.no_grad():
                prefill_out = model(input_ids=prefill_ids, use_cache=True)
            past_kv = prefill_out.past_key_values

            # Decode: single new token
            decode_ids = torch.randint(0, 1000, (bs, 1), device=DEVICE)

            # Capture layer inputs during decode
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
                "bs": bs, "seq": 1, "kv_cache": kv_len, "n_tokens": bs,
                "tp_size": 1, "latency_ns": total_ns, "phase": "decode"
            })

            # Time each sub-module
            for sub_name in sub_names:
                sub_mod = getattr(layer, sub_name)
                lats = []
                with torch.no_grad():
                    for _ in range(REPEAT):
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        def pre_h(module, args, se=start_evt): se.record()
                        def post_h(module, args, output, ee=end_evt): ee.record()
                        h1 = sub_mod.register_forward_pre_hook(pre_h)
                        h2 = sub_mod.register_forward_hook(post_h)
                        torch.cuda.synchronize()
                        _ = layer(*layer_args, **layer_kwargs)
                        torch.cuda.synchronize()
                        h1.remove(); h2.remove()
                        lats.append(start_evt.elapsed_time(end_evt) * 1e6)
                lats.sort()
                sub_ns = int(lats[len(lats)//2])
                results.append({
                    "model_name": model_name, "layer_name": sub_name,
                    "bs": bs, "seq": 1, "kv_cache": kv_len, "n_tokens": bs,
                    "tp_size": 1, "latency_ns": sub_ns, "phase": "decode"
                })

            print(" total=%.0fus" % (total_ns / 1000))

            # Clean up KV cache memory
            del past_kv, prefill_out
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(" OOM, skipping")
            torch.cuda.empty_cache()
        except Exception as ex:
            print(" ERROR: %s" % ex)
            import traceback; traceback.print_exc()

# Save
out_dir = "/data/LLMServingSim/llm_profile/perf_models/A100/%s/tp1" % model_name.replace(".", "_").replace(" ", "_")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "decode_phase1_profiles.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "layer_name", "bs", "seq", "kv_cache", "n_tokens", "tp_size", "latency_ns", "phase"])
    w.writeheader()
    w.writerows(results)
print("\nSaved %d rows to %s" % (len(results), csv_path))
