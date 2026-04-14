"""Profile models at seq lengths NOT in training data to validate XGBoost generalization."""
import torch, time, sys, os, json
sys.path.insert(0, "/data/kevinlau/python-packages")
sys.path.insert(0, "/home/kevinlau/llmserve")

DEVICE = "cuda:0"
WARMUP = 3
REPEAT = 10

# Validation grid: seq lengths NOT in training data
VAL_SEQS = [96, 192, 384, 768]
VAL_BS = [1, 3]  # bs=3 not in training

def profile_model(model_path, model_name, val_seqs, val_bs):
    from transformers import AutoModelForCausalLM, AutoConfig
    
    config = AutoConfig.from_pretrained(model_path)
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True)
    model.eval()
    
    n_layers = config.num_hidden_layers
    results = []
    
    for bs in val_bs:
        for seq in val_seqs:
            # Skip if too large for memory
            try:
                input_ids = torch.randint(100, 30000, (bs, seq), device=DEVICE)
                
                # Warmup
                for _ in range(WARMUP):
                    with torch.no_grad():
                        _ = model(input_ids=input_ids)
                    torch.cuda.synchronize()
                
                # Measure
                times = []
                for _ in range(REPEAT):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = model(input_ids=input_ids)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)  # ms
                
                median_ms = sorted(times)[len(times)//2]
                per_layer_ms = median_ms / n_layers
                
                results.append({
                    "model_name": model_name,
                    "d_model": config.hidden_size,
                    "n_heads": config.num_attention_heads,
                    "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                    "intermediate_size": config.intermediate_size,
                    "num_experts": getattr(config, "num_local_experts", 0),
                    "top_k": getattr(config, "num_experts_per_tok", 0),
                    "batch_size": bs,
                    "seq_len": seq,
                    "phase": "prefill",
                    "n_tokens": bs * seq,
                    "e2e_ms": median_ms,
                    "per_layer_ms": per_layer_ms,
                    "n_layers": n_layers,
                })
                print(f"  bs={bs} seq={seq}: {median_ms:.1f}ms total, {per_layer_ms:.3f}ms/layer")
                
            except torch.cuda.OutOfMemoryError:
                print(f"  bs={bs} seq={seq}: OOM, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  bs={bs} seq={seq}: ERROR {e}")
    
    del model
    torch.cuda.empty_cache()
    return results

# Profile Llama-8B (fits easily)
all_results = []
all_results.extend(profile_model(
    "/data/models/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", VAL_SEQS, VAL_BS))

# Save results
out_path = "experiment/validation_grid_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {len(all_results)} results to {out_path}")
