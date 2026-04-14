"""
Standalone profiling script using PyTorch Profiler.

Loads a slice of the model, runs warmup, then profiles one forward pass
with torch.profiler to capture per-kernel timing, FLOPs, and memory.
Can be run standalone or invoked by run_ncu.py.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _try_resolve(obj, *candidates):
    """Try multiple dotted attribute paths, return first that works or None."""
    for path in candidates:
        try:
            parts = path.split(".")
            o = obj
            for p in parts:
                o = getattr(o, p)
            return o
        except AttributeError:
            continue
    return None


def load_model_slice(model_path: str, num_layers: int, device: str, trust_remote_code: bool):
    """Load a model and extract a slice of layers for profiling."""
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # Find architecture components
    layers = _try_resolve(model, "model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")
    embed = _try_resolve(model, "model.embed_tokens", "transformer.wte", "gpt_neox.embed_in")
    norm = _try_resolve(model, "model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm")
    rotary_emb = _try_resolve(model, "model.rotary_emb", "transformer.rotary_emb")

    if layers is None or embed is None or norm is None:
        print("WARNING: Could not find model components, profiling full model", file=sys.stderr)
        return model, config

    # Select layers to profile
    start_layer = 1
    end_layer = min(start_layer + num_layers, len(layers))
    selected = [layers[i] for i in range(start_layer, end_layer)]

    class LayerSlice(nn.Module):
        def __init__(self, embed_tokens, selected_layers, final_norm, rotary):
            super().__init__()
            self.embed_tokens = embed_tokens
            self.layers = nn.ModuleList(selected_layers)
            self.norm = final_norm
            self.rotary_emb = rotary

        def forward(self, input_ids, past_key_values=None):
            hidden = self.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape

            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

            position_embeddings = None
            if self.rotary_emb is not None:
                position_embeddings = self.rotary_emb(hidden, position_ids)

            new_past = []
            for i, layer in enumerate(self.layers):
                pkv = past_key_values[i] if past_key_values else None
                kwargs = {"position_ids": position_ids, "use_cache": True}
                if pkv is not None:
                    kwargs["past_key_value"] = pkv
                if position_embeddings is not None:
                    kwargs["position_embeddings"] = position_embeddings

                out = layer(hidden, **kwargs)
                if isinstance(out, tuple):
                    hidden = out[0]
                    new_past.append(out[1] if len(out) > 1 else None)
                else:
                    hidden = out
                    new_past.append(None)

            hidden = self.norm(hidden)
            return hidden, new_past

    slim = LayerSlice(embed, selected, norm, rotary_emb)
    slim.eval()
    del model
    torch.cuda.empty_cache()

    print(f"Loaded LayerSlice: layers [{start_layer}:{end_layer}] on {device}", file=sys.stderr)
    return slim, config


def build_kv_cache(model, input_ids):
    """Run a prefill pass to build KV cache for decode profiling."""
    with torch.no_grad():
        _, past = model(input_ids)
    return past


def profile_with_torch(model, input_ids, phase, config, output_path, past_key_values=None, model_name=None):
    """Profile using torch.profiler and export results."""
    from torch.profiler import profile as torch_profile, ProfilerActivity

    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            if phase == "decode" and past_key_values is not None:
                vocab_size = config.vocab_size
                decode_ids = torch.randint(0, vocab_size, (input_ids.shape[0], 1), device=input_ids.device)
                model(decode_ids, past_key_values=past_key_values)
            else:
                model(input_ids)
        torch.cuda.synchronize()

    # Export chrome trace
    trace_path = str(output_path) + ".trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"  Chrome trace: {trace_path}", file=sys.stderr)

    # Extract per-kernel data with shapes for AI calculation
    kernels = []
    for evt in prof.key_averages(group_by_input_shape=True):
        device_time = evt.self_device_time_total if hasattr(evt, 'self_device_time_total') else 0
        if device_time > 0:
            kernels.append({
                "name": evt.key,
                "cuda_time_us": device_time,
                "cpu_time_us": evt.self_cpu_time_total,
                "calls": evt.count,
                "flops": evt.flops if evt.flops else 0,
                "input_shapes": str(evt.input_shapes) if evt.input_shapes else "",
            })

    # Extract model architecture info for analytical FLOP computation
    model_info = {
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_attention_heads": getattr(config, "num_attention_heads", 0),
        "num_key_value_heads": getattr(config, "num_key_value_heads", 0),
        "head_dim": getattr(config, "head_dim", 0) or (getattr(config, "hidden_size", 0) // max(getattr(config, "num_attention_heads", 1), 1)),
        "intermediate_size": getattr(config, "intermediate_size", 0),
        "vocab_size": getattr(config, "vocab_size", 0),
    }

    result = {
        "model": model_name or os.path.basename(os.environ.get("ROOFLINE_MODEL_PATH", "unknown")),
        "phase": phase,
        "batch_size": input_ids.shape[0],
        "seq_len": input_ids.shape[1],
        "num_layers_profiled": int(os.environ.get("ROOFLINE_LAYERS", "2")),
        "profiler": "torch",
        "model_info": model_info,
        "kernel_summary": sorted(kernels, key=lambda x: -x["cuda_time_us"]),
        "total_cuda_time_us": sum(k["cuda_time_us"] for k in kernels),
        "total_flops": sum(k["flops"] for k in kernels),
    }

    json_path = str(output_path) + ".json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON output: {json_path}", file=sys.stderr)

    # Print summary table
    print(f"\n  Top kernels by CUDA time:", file=sys.stderr)
    print(f"  {'Kernel':<60s} {'CUDA μs':>10s} {'Calls':>6s} {'FLOPs':>15s}", file=sys.stderr)
    print(f"  {'-'*91}", file=sys.stderr)
    for k in sorted(kernels, key=lambda x: -x["cuda_time_us"])[:20]:
        name = k["name"][:60]
        flops_str = f"{k['flops']:.2e}" if k["flops"] > 0 else "-"
        print(f"  {name:<60s} {k['cuda_time_us']:>10.1f} {k['calls']:>6d} {flops_str:>15s}", file=sys.stderr)

    return result


def main():
    model_path = os.environ["ROOFLINE_MODEL_PATH"]
    num_layers = int(os.environ.get("ROOFLINE_LAYERS", "2"))
    batch_size = int(os.environ["ROOFLINE_BATCH_SIZE"])
    seq_len = int(os.environ.get("ROOFLINE_SEQ_LEN", "512"))
    phase = os.environ.get("ROOFLINE_PHASE", "prefill")
    device = os.environ.get("ROOFLINE_DEVICE", "cuda:0")
    trust_remote_code = os.environ.get("ROOFLINE_TRUST_REMOTE_CODE", "false").lower() == "true"
    output_dir = os.environ.get("ROOFLINE_OUTPUT_DIR", "results/roofline/raw")

    print(f"Config: model={model_path}, layers={num_layers}, bs={batch_size}, "
          f"seq_len={seq_len}, phase={phase}, device={device}", file=sys.stderr)

    model, config = load_model_slice(model_path, num_layers, device, trust_remote_code)

    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print("Warming up...", file=sys.stderr)
    with torch.no_grad():
        for _ in range(3):
            if phase == "prefill":
                model(input_ids)
            else:
                _ = build_kv_cache(model, input_ids)
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling {phase}...", file=sys.stderr)
    model_name = os.path.basename(model_path)
    output_path = Path(output_dir) / f"{model_name}_{phase}_bs{batch_size}"

    past = None
    if phase == "decode":
        past = build_kv_cache(model, input_ids)
        torch.cuda.synchronize()

    profile_with_torch(model, input_ids, phase, config, output_path, past_key_values=past)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
