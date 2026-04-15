#!/usr/bin/env python3
"""
Validate LLMCompass MoE predictions on H100 against real isolated-layer measurements.
Runs gpt-oss-20b and Mixtral-8x7B with H100 ML predictor.

Usage:
    python experiment/validate_moe_h100.py --model mixtral --device cuda:0
    python experiment/validate_moe_h100.py --model gpt-oss --device cuda:0
    python experiment/validate_moe_h100.py --model both --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = {
    "gpt-oss-20b": {
        "path": "/workspace/models/gpt-oss-20b",
        "hidden_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "num_hidden_layers": 24,
        "intermediate_size": 2880,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "hidden_act": "silu",
    },
    "mixtral-8x7b": {
        "path": "/workspace/models/Mixtral-8x7B-Instruct-v0.1",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
    },
}

BATCH_SIZES = [1, 4, 8]
PREFILL_SEQ_LENS = [64, 128, 256, 512]
DECODE_KV_LENS = [256, 512, 1024]
WARMUP_ITERS = 5
MEASURE_ITERS = 20


# ---------------------------------------------------------------------------
# LLMCompass predictions (H100)
# ---------------------------------------------------------------------------

def load_h100_predictor():
    """Load ML predictor from H100 profiles."""
    import llmcompass.software_model.transformer as transformer_mod
    from llmcompass.profiler.ml_predictor import KernelPredictor

    profiles_dir = "llmcompass/profiler/profiles/H100"
    print(f"[LLMCompass] Loading H100 ML predictor from {profiles_dir}")
    transformer_mod._kernel_predictor = None
    predictor = KernelPredictor(profiles_dir)
    predictor.train_all(force_retrain=False)
    transformer_mod._kernel_predictor = predictor
    return predictor


def build_llmcompass_block(cfg, tp_size=1):
    """Build TransformerBlockMoETP for a given model config."""
    from llmcompass.software_model.transformer import TransformerBlockMoETP
    from llmcompass.software_model.utils import data_type_dict

    block = TransformerBlockMoETP(
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        device_count=tp_size,
        data_type=data_type_dict["fp16"],  # bf16 same size as fp16
        intermediate_size=cfg["intermediate_size"],
        n_kv_heads=cfg["num_key_value_heads"],
        num_experts=cfg["num_local_experts"],
        top_k=cfg["num_experts_per_tok"],
        expert_intermediate_size=cfg["intermediate_size"],
        use_flash_attention=True,
        activation_type=cfg["hidden_act"],
        use_ml_predictor=True,
        use_cuda_graph=False,
    )
    return block


def run_predictions(cfg, tp_size=1):
    """Run LLMCompass predictions for all (phase, bs, seq) combos."""
    from llmcompass.software_model.utils import data_type_dict, Tensor
    from llmcompass.design_space_exploration.dse import template_to_system, read_architecture_template
    import llmcompass.software_model.transformer as _tmod

    predictor = load_h100_predictor()

    # Load H100 system config
    system_config = os.path.join(str(REPO_ROOT), "device_configs", "GH100.json")
    if not os.path.isfile(system_config):
        print(f"[LLMCompass] ERROR: GH100.json not found")
        return {}

    arch_specs = read_architecture_template(system_config)
    system = template_to_system(arch_specs)

    results = {}

    # Prefill
    for bs in BATCH_SIZES:
        for seq in PREFILL_SEQ_LENS:
            try:
                block = build_llmcompass_block(cfg, tp_size)
                X = Tensor([bs, seq, cfg["hidden_size"]], data_type_dict["fp16"])
                _ = block(X)
                _tmod._kernel_predictor = predictor
                latency_s = block.compile_and_simulate(system, "heuristic-GPU")
                latency_ms = latency_s * 1e3
                results[("prefill", bs, seq)] = latency_ms
            except Exception as e:
                print(f"  prefill bs={bs} seq={seq}: ERROR - {e}")
                results[("prefill", bs, seq)] = None

    # Decode
    for bs in BATCH_SIZES:
        for kv_len in DECODE_KV_LENS:
            try:
                block = build_llmcompass_block(cfg, tp_size)
                X = Tensor([bs, 1, cfg["hidden_size"]], data_type_dict["fp16"])
                _ = block(X)
                _tmod._kernel_predictor = predictor
                latency_s = block.compile_and_simulate(system, "heuristic-GPU")
                latency_ms = latency_s * 1e3
                results[("decode", bs, kv_len)] = latency_ms
            except Exception as e:
                print(f"  decode bs={bs} kv={kv_len}: ERROR - {e}")
                results[("decode", bs, kv_len)] = None

    return results


# ---------------------------------------------------------------------------
# Real measurements (isolated layer)
# ---------------------------------------------------------------------------

def load_model_2layers(model_path, device):
    """Load model with only 2 decoder layers."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"\n[Measure] Loading 2-layer model from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_hidden_layers = 2

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [Measure] Parameters: {params_m:.1f}M")
    return model


def _capture_layer_inputs(model, batch_size, seq_len, device):
    """Run full model once to capture exact inputs to layer 0."""
    captured = {}
    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
        captured["kwargs"] = {k: v.detach() if hasattr(v, "detach") else v for k, v in kwargs.items()}
    handle = model.model.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    return captured


def measure_layer(model, device, batch_size, seq_len, use_cache=False, past_kv=None):
    """Measure isolated single-layer forward pass with CUDA events. Returns median ms."""
    layer = model.model.layers[0]

    if past_kv is not None:
        # Decode: run with past KV
        decode_ids = torch.randint(0, 1000, (batch_size, 1), device=device)
        captured = {}
        def hook_fn(module, args, kwargs):
            captured["args"] = tuple(a.detach() if hasattr(a, "detach") else a for a in args)
            captured["kwargs"] = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in kwargs.items()}
        h = model.model.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
        with torch.no_grad():
            _ = model(input_ids=decode_ids, past_key_values=past_kv, use_cache=True)
        h.remove()
        args = captured["args"]
        kwargs = dict(captured["kwargs"])
    else:
        # Prefill
        captured = _capture_layer_inputs(model, batch_size, seq_len, device)
        args = captured["args"]
        kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
        kwargs["use_cache"] = False

    dev = torch.device(device)
    latencies = []

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = layer(*args, **kwargs)
        torch.cuda.synchronize(dev)

        for _ in range(MEASURE_ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(dev)
            s.record()
            _ = layer(*args, **kwargs)
            e.record()
            torch.cuda.synchronize(dev)
            latencies.append(s.elapsed_time(e))

    latencies.sort()
    return latencies[len(latencies) // 2]


def run_measurements(model_path, device):
    """Run real measurements for all combos. Returns dict."""
    print(f"\n[Measure] Running real measurements on {device}...")
    results = {}

    try:
        model = load_model_2layers(model_path, device)
    except Exception as e:
        print(f"[Measure] ERROR loading model: {e}")
        import traceback; traceback.print_exc()
        return results

    # Prefill
    for bs in BATCH_SIZES:
        for seq in PREFILL_SEQ_LENS:
            try:
                lat = measure_layer(model, device, bs, seq)
                results[("prefill", bs, seq)] = lat
                print(f"  prefill bs={bs} seq={seq:4d}: {lat:.3f} ms")
            except torch.cuda.OutOfMemoryError:
                print(f"  prefill bs={bs} seq={seq:4d}: OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  prefill bs={bs} seq={seq:4d}: ERROR {e}")

    # Decode (need KV cache)
    for bs in BATCH_SIZES:
        for kv_len in DECODE_KV_LENS:
            try:
                # Build KV cache via prefill
                prefill_ids = torch.randint(0, 1000, (bs, kv_len), device=device)
                with torch.no_grad():
                    pf_out = model(input_ids=prefill_ids, use_cache=True)
                past_kv = pf_out.past_key_values

                lat = measure_layer(model, device, bs, 1, use_cache=True, past_kv=past_kv)
                results[("decode", bs, kv_len)] = lat
                print(f"  decode  bs={bs} kv={kv_len:4d}: {lat:.3f} ms")

                del past_kv, pf_out
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  decode  bs={bs} kv={kv_len:4d}: OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  decode  bs={bs} kv={kv_len:4d}: ERROR {e}")

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def rate(ratio):
    dev = abs(ratio - 1.0)
    if dev <= 0.20: return "GOOD"
    if dev <= 0.30: return "OK"
    return "BAD"


def compare_and_save(model_name, cfg, predictions, measurements, output_path):
    """Print comparison table and save results."""
    print(f"\n{'=' * 85}")
    print(f"  {model_name} — MoE: {cfg['num_local_experts']} experts, top-{cfg['num_experts_per_tok']}")
    print(f"  Hardware: H100-SXM5-80GB  |  Predictor: H100 ML (RF)")
    print(f"{'=' * 85}")
    print(f"{'Phase':<10} {'BS':>4} {'Seq/KV':>8} {'Predicted':>12} {'Measured':>12} {'Ratio':>8} {'Rating':>7}")
    print("-" * 85)

    rows = []
    good = ok = bad = skip = 0

    for phase in ["prefill", "decode"]:
        lens = PREFILL_SEQ_LENS if phase == "prefill" else DECODE_KV_LENS
        for bs in BATCH_SIZES:
            for sl in lens:
                key = (phase, bs, sl)
                pred = predictions.get(key)
                meas = measurements.get(key)

                if pred is not None and meas is not None and meas > 0:
                    ratio = pred / meas
                    r = rate(ratio)
                    if r == "GOOD": good += 1
                    elif r == "OK": ok += 1
                    else: bad += 1
                    print(f"{phase:<10} {bs:>4} {sl:>8} {pred:>11.3f}ms {meas:>11.3f}ms {ratio:>7.3f}x {r:>7}")
                    rows.append({"phase": phase, "batch_size": bs, "seq_or_kv": sl,
                                 "predicted_ms": round(float(pred), 3), "measured_ms": round(float(meas), 3),
                                 "ratio": round(float(ratio), 3), "rating": r})
                else:
                    skip += 1

    total = good + ok + bad
    print(f"{'=' * 85}")
    print(f"  GOOD: {good}/{total}  OK: {ok}/{total}  BAD: {bad}/{total}  (skipped: {skip})")
    print(f"{'=' * 85}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_name,
        "hardware": "H100-SXM5-80GB",
        "predictor": "H100 ML (RandomForest)",
        "architecture": cfg,
        "results": rows,
        "summary": {"good": good, "ok": ok, "bad": bad, "total": total},
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {out}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["gpt-oss", "mixtral", "both"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    models_to_run = []
    if args.model in ("gpt-oss", "both"):
        models_to_run.append(("gpt-oss-20b", MODELS["gpt-oss-20b"]))
    if args.model in ("mixtral", "both"):
        models_to_run.append(("mixtral-8x7b", MODELS["mixtral-8x7b"]))

    for model_name, cfg in models_to_run:
        print(f"\n{'#' * 85}")
        print(f"  Validating: {model_name}")
        print(f"{'#' * 85}")

        predictions = {}
        measurements = {}

        if not args.skip_predict:
            predictions = run_predictions(cfg)

        if not args.skip_real:
            measurements = run_measurements(cfg["path"], args.device)

        output_path = f"experiment/moe_validation_{model_name.replace('-', '_')}_h100.json"
        compare_and_save(model_name, cfg, predictions, measurements, output_path)


if __name__ == "__main__":
    main()
