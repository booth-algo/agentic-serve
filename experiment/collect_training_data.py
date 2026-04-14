"""
Collect end-to-end layer latency training data for ML predictor.
Runs isolated layer measurements across models × batch sizes × seq lengths.
"""
import os, sys, json, torch, statistics
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM

DEVICE = "cuda:0"
WARMUP = 5
REPEAT = 20

def capture_and_measure(model, layer, bs, seq_len, device):
    """Capture layer inputs via hook, then measure isolated layer latency."""
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
        lats = []
        for _ in range(REPEAT):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            _ = layer(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e))
    lats.sort()
    return lats[len(lats)//2]

def collect_model_data(model_path, device, batch_sizes, seq_lens, is_moe=False):
    """Collect measurements for one model across a grid of configs."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Extract architecture params
    arch = {
        "model_name": os.path.basename(model_path),
        "d_model": config.hidden_size,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "intermediate_size": config.intermediate_size,
        "num_experts": getattr(config, "num_local_experts", 0),
        "top_k": getattr(config, "num_experts_per_tok", 0),
        "is_moe": is_moe,
    }
    
    config.num_hidden_layers = 2
    print(f"\nLoading {arch['model_name']} (2 layers)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
        torch_dtype=torch.bfloat16, device_map=device,
        ignore_mismatched_sizes=True, low_cpu_mem_usage=True,
        trust_remote_code=True)
    model.eval()
    layer = model.model.layers[0]
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)
    
    results = []
    for bs in batch_sizes:
        for seq_len in seq_lens:
            try:
                lat_ms = capture_and_measure(model, layer, bs, seq_len, device)
                row = {**arch, "batch_size": bs, "seq_len": seq_len, 
                       "n_tokens": bs * seq_len, "latency_ms": lat_ms}
                results.append(row)
                print(f"  bs={bs} seq={seq_len}: {lat_ms:.3f} ms", flush=True)
            except Exception as e:
                print(f"  bs={bs} seq={seq_len}: ERROR {str(e)[:50]}", flush=True)
    
    del model
    torch.cuda.empty_cache()
    return results

def main():
    all_results = []
    
    # Dense models
    dense_models = [
        "/data/models/Llama-3.1-8B-Instruct",
        "/data/models/Qwen2.5-72B-Instruct",
    ]
    # More batch sizes and seq lengths for denser coverage
    prefill_grid = [(bs, seq) for bs in [1, 2, 4, 8, 16] for seq in [64, 128, 256, 512, 1024]]
    decode_grid = [(bs, 1) for bs in [1, 2, 4, 8, 16, 32]]
    
    for model_path in dense_models:
        print(f"\n{'='*60}")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"{'='*60}")
        
        # Prefill
        for bs, seq in prefill_grid:
            try:
                config = AutoConfig.from_pretrained(model_path)
                config.num_hidden_layers = 2
                model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                    torch_dtype=torch.bfloat16, device_map=DEVICE,
                    ignore_mismatched_sizes=True, low_cpu_mem_usage=True)
                model.eval()
                layer = model.model.layers[0]
                lat = capture_and_measure(model, layer, bs, seq, DEVICE)
                
                arch = {
                    "model_name": os.path.basename(model_path),
                    "d_model": config.hidden_size,
                    "n_heads": config.num_attention_heads,
                    "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                    "intermediate_size": config.intermediate_size,
                    "num_experts": 0, "top_k": 0, "is_moe": False,
                }
                all_results.append({**arch, "batch_size": bs, "seq_len": seq,
                    "phase": "prefill", "n_tokens": bs*seq, "latency_ms": lat})
                print(f"  prefill bs={bs} seq={seq}: {lat:.3f} ms", flush=True)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  prefill bs={bs} seq={seq}: SKIP ({str(e)[:40]})", flush=True)
                try: del model; torch.cuda.empty_cache()
                except: pass
        
        # Decode
        for bs, seq in decode_grid:
            try:
                config = AutoConfig.from_pretrained(model_path)
                config.num_hidden_layers = 2
                model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                    torch_dtype=torch.bfloat16, device_map=DEVICE,
                    ignore_mismatched_sizes=True, low_cpu_mem_usage=True)
                model.eval()
                layer = model.model.layers[0]
                lat = capture_and_measure(model, layer, bs, seq, DEVICE)
                
                arch = {
                    "model_name": os.path.basename(model_path),
                    "d_model": config.hidden_size,
                    "n_heads": config.num_attention_heads,
                    "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                    "intermediate_size": config.intermediate_size,
                    "num_experts": 0, "top_k": 0, "is_moe": False,
                }
                all_results.append({**arch, "batch_size": bs, "seq_len": seq,
                    "phase": "decode", "n_tokens": bs*seq, "latency_ms": lat})
                print(f"  decode bs={bs} seq={seq}: {lat:.3f} ms", flush=True)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  decode bs={bs} seq={seq}: SKIP ({str(e)[:40]})", flush=True)
                try: del model; torch.cuda.empty_cache()
                except: pass
    
    # MoE model
    print(f"\n{'='*60}")
    print(f"Model: gpt-oss-20b (MoE)")
    print(f"{'='*60}")
    moe_prefill = [(bs, seq) for bs in [1, 2, 4] for seq in [64, 128, 256, 512]]
    moe_decode = [(bs, 1) for bs in [1, 2, 4, 8]]
    
    for phase, grid in [("prefill", moe_prefill), ("decode", moe_decode)]:
        for bs, seq in grid:
            try:
                config = AutoConfig.from_pretrained("/data/models/gpt-oss-20b")
                config.num_hidden_layers = 2
                model = AutoModelForCausalLM.from_pretrained("/data/models/gpt-oss-20b", config=config,
                    torch_dtype=torch.bfloat16, device_map=DEVICE,
                    ignore_mismatched_sizes=True, low_cpu_mem_usage=True)
                model.eval()
                layer = model.model.layers[0]
                lat = capture_and_measure(model, layer, bs, seq, DEVICE)
                
                all_results.append({
                    "model_name": "gpt-oss-20b", "d_model": 2880,
                    "n_heads": 64, "n_kv_heads": 8, "intermediate_size": 2880,
                    "num_experts": 32, "top_k": 4, "is_moe": True,
                    "batch_size": bs, "seq_len": seq, "phase": phase,
                    "n_tokens": bs*seq, "latency_ms": lat})
                print(f"  {phase} bs={bs} seq={seq}: {lat:.3f} ms", flush=True)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {phase} bs={bs} seq={seq}: SKIP ({str(e)[:40]})", flush=True)
                try: del model; torch.cuda.empty_cache()
                except: pass
    
    # Save
    output_path = "/home/kevinlau/llmserve/llmcompass/profiler/profiles/A100/layer_latency_training_data.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} data points to {output_path}")

if __name__ == "__main__":
    main()
