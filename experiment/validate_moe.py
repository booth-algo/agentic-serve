#!/usr/bin/env python3
"""
Validation script for LLMCompass MoE simulator against real gpt-oss-20b measurements.
Run on gpu-4 (8x A100-SXM4-40GB).

Usage:
    python validate_moe.py [--device cuda:0] [--model-path /data/models/gpt-oss-20b] [--tp-size 1]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — adjust to wherever llm_predict lives relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # ~/llmserve
sys.path.insert(0, str(REPO_ROOT))

import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Validate LLMCompass MoE predictions vs real measurements")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for real measurements (default: cuda:0)")
    parser.add_argument("--model-path", default="/data/models/gpt-oss-20b", help="Path to gpt-oss-20b weights")
    parser.add_argument("--tp-size", type=int, default=1, choices=[1, 2], help="Tensor parallel size (default: 1)")
    parser.add_argument("--output", default="experiment/moe_validation_results.json", help="Output JSON path")
    parser.add_argument("--skip-real", action="store_true", help="Skip real measurements (predictions only)")
    parser.add_argument("--skip-predict", action="store_true", help="Skip LLMCompass predictions (real only)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# gpt-oss-20b architecture constants
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "hidden_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "num_hidden_layers": 24,
    "intermediate_size": 2880,
    "num_local_experts": 32,
    "num_experts_per_tok": 4,
    "hidden_act": "silu",
}

BATCH_SIZES = [1, 4, 8]
PREFILL_SEQ_LEN = 512
WARMUP_ITERS = 5
MEASURE_ITERS = 20


# ---------------------------------------------------------------------------
# LLMCompass predictions
# ---------------------------------------------------------------------------

def load_a100_predictor():
    """Load ML predictor pointed at A100 profiles. Falls back to analytical if unavailable."""
    try:
        import llm_predict.models.software.transformer as transformer_mod
        from llm_predict.predictors.per_category.predictor import CategoryPredictor

        # Use relative path (matching how the predictor was trained) for cache key consistency
        # Must be called from REPO_ROOT working directory
        profiles_dir = "llm_predict/profiling/data/A100"
        abs_profiles = os.path.join(str(REPO_ROOT), profiles_dir)
        if not os.path.isdir(abs_profiles):
            profiles_dir = abs_profiles  # fall back to absolute

        if os.path.isdir(profiles_dir) or os.path.isdir(abs_profiles):
            print(f"[LLMCompass] Loading A100 ML predictor from {profiles_dir}")
            transformer_mod._kernel_predictor = None  # clear any stale singleton
            predictor = CategoryPredictor(profiles_dir)
            predictor.train_all(force_retrain=False)
            transformer_mod._kernel_predictor = predictor
            return True, "ml"
        else:
            print(f"[LLMCompass] WARNING: A100 profiles not found, falling back to analytical")
            return False, "analytical"
    except Exception as e:
        print(f"[LLMCompass] WARNING: Could not load ML predictor: {e}, falling back to analytical")
        return False, "analytical"


def build_predictor_block(tp_size: int, use_ml_predictor: bool):
    """Instantiate TransformerBlockMoETP for gpt-oss-20b."""
    from llm_predict.models.software.transformer import TransformerBlockMoETP
    from llm_predict.models.software.utils import data_type_dict

    cfg = MODEL_CONFIG
    block = TransformerBlockMoETP(
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        device_count=tp_size,
        data_type=data_type_dict["fp16"],
        intermediate_size=cfg["intermediate_size"],
        n_kv_heads=cfg["num_key_value_heads"],
        num_experts=cfg["num_local_experts"],
        top_k=cfg["num_experts_per_tok"],
        expert_intermediate_size=cfg["intermediate_size"],
        use_flash_attention=True,
        activation_type=cfg["hidden_act"],
        use_ml_predictor=use_ml_predictor,
        use_cuda_graph=False,
    )
    return block


def run_predictions(tp_size: int):
    """
    Run LLMCompass predictions for all (phase, batch_size) combos.
    Returns dict: {(phase, bs): latency_ms}
    """
    print("\n[LLMCompass] Running predictions...")

    try:
        from llm_predict.models.software.utils import data_type_dict, Tensor
        from llm_predict.dse.dse import template_to_system, read_architecture_template
    except ImportError as e:
        print(f"[LLMCompass] ERROR: Could not import LLMCompass: {e}")
        return {}, "unavailable"

    ml_available, predictor_mode = load_a100_predictor()

    # Force-load A100 predictor directly (bypass caching issues)
    import llm_predict.models.software.transformer as _tmod
    from llm_predict.predictors.per_category.predictor import CategoryPredictor as _KP
    _tmod._kernel_predictor = None
    _a100_pred = _KP("llm_predict/profiling/data/A100")
    _a100_pred.train_all(force_retrain=False)
    _tmod._kernel_predictor = _a100_pred
    ml_available = True

    # Load A100 system config
    # Try a few common locations for the device config
    config_candidates = [
        os.path.join(str(REPO_ROOT), "device_configs", "GA100.json"),
        os.path.join(str(REPO_ROOT), "device_configs", "GA100.json"),
    ]
    try:
        import llm_predict
        config_candidates.append(
            os.path.join(os.path.dirname(llm_predict.__file__), "device_configs", "GA100.json")
        )
    except Exception:
        pass

    system_config = None
    for c in config_candidates:
        if os.path.isfile(c):
            system_config = c
            break

    if system_config is None:
        print("[LLMCompass] ERROR: Could not find GA100.json device config")
        return {}, predictor_mode

    print(f"[LLMCompass] Using device config: {system_config}")
    arch_specs = read_architecture_template(system_config)
    system = template_to_system(arch_specs)

    # Store predictor reference to force-set before each simulation
    import llm_predict.models.software.transformer as _tmod
    _a100_predictor = _tmod._kernel_predictor

    results = {}

    for bs in BATCH_SIZES:
        # --- Prefill ---
        try:
            import llm_predict.models.software.transformer as _tcheck
            if _tcheck._kernel_predictor is not None:
                _pd = getattr(_tcheck._kernel_predictor, 'profiles_dir', 'unknown')
                if 'A100' not in _pd:
                    print(f"  WARNING: predictor is {_pd}, forcing A100")
                    load_a100_predictor()
            block = build_predictor_block(tp_size, ml_available)
            X = Tensor([bs, PREFILL_SEQ_LEN, MODEL_CONFIG["hidden_size"]], data_type_dict["fp16"])
            _ = block(X)
            _tmod._kernel_predictor = _a100_predictor  # ensure A100 predictor
            latency_s = block.compile_and_simulate(system, "heuristic-GPU")
            latency_ms = latency_s * 1e3
            results[("prefill", bs)] = latency_ms
            print(f"  prefill bs={bs}: {latency_ms:.3f} ms")
        except Exception as e:
            print(f"  prefill bs={bs}: ERROR - {e}")
            results[("prefill", bs)] = None

        # --- Decode (seq_len=1) ---
        try:
            block = build_predictor_block(tp_size, ml_available)
            X = Tensor([bs, 1, MODEL_CONFIG["hidden_size"]], data_type_dict["fp16"])
            _ = block(X)
            _tmod._kernel_predictor = _a100_predictor  # ensure A100 predictor
            latency_s = block.compile_and_simulate(system, "heuristic-GPU")
            latency_ms = latency_s * 1e3
            results[("decode", bs)] = latency_ms
            print(f"  decode  bs={bs}: {latency_ms:.3f} ms")
        except Exception as e:
            print(f"  decode  bs={bs}: ERROR - {e}")
            results[("decode", bs)] = None

    return results, predictor_mode


# ---------------------------------------------------------------------------
# Real measurements
# ---------------------------------------------------------------------------

def load_model_2layers(model_path: str, device: str):
    """
    Load gpt-oss-20b with only 2 decoder layers to save GPU memory.
    Returns (model, tokenizer_or_None).
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"\n[Measure] Loading 2-layer model from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path)

    # Override to 2 layers only
    config.num_hidden_layers = 2

    # Some MoE configs use different field names — patch what we can
    for field in ["num_local_experts", "num_experts", "moe_num_experts"]:
        if hasattr(config, field):
            print(f"  [Measure] config.{field} = {getattr(config, field)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  [Measure] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def _capture_layer_inputs(model, batch_size: int, seq_len: int, device: str):
    """Run full model once to capture exact inputs to layer 0 (avoids shape mismatches)."""
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


def measure_latency_cuda_events(model, device: str, batch_size: int, seq_len: int):
    """
    Measure ISOLATED per-layer forward pass latency using CUDA events.
    Profiles just the decoder layer (no embedding, no lm_head, no final norm).
    Uses hooks to capture real layer inputs, then times just the layer forward.
    Returns median latency in ms across MEASURE_ITERS.
    """
    dev = torch.device(device)
    layer = model.model.layers[0]

    # Capture real inputs by running full model once
    captured = _capture_layer_inputs(model, batch_size, seq_len, device)
    args = captured["args"]
    kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    kwargs["use_cache"] = False

    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = layer(*args, **kwargs)
        torch.cuda.synchronize(dev)

        # Measure
        for _ in range(MEASURE_ITERS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize(dev)
            start_event.record()
            _ = layer(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize(dev)
            elapsed_ms = start_event.elapsed_time(end_event)
            latencies.append(elapsed_ms)  # single layer, no division needed

    latencies_sorted = sorted(latencies)
    median_ms = latencies_sorted[len(latencies_sorted) // 2]
    return median_ms


def run_real_measurements(model_path: str, device: str):
    """
    Run real latency measurements for all (phase, batch_size) combos.
    Returns dict: {(phase, bs): latency_ms}
    """
    print("\n[Measure] Running real PyTorch measurements...")

    results = {}

    try:
        model = load_model_2layers(model_path, device)
    except Exception as e:
        print(f"[Measure] ERROR: Could not load model: {e}")
        return results

    # Prefill measurements
    for bs in BATCH_SIZES:
        try:
            print(f"  [Measure] prefill bs={bs} seq={PREFILL_SEQ_LEN} ...", end=" ", flush=True)
            latency_ms = measure_latency_cuda_events(model, device, bs, PREFILL_SEQ_LEN)
            results[("prefill", bs)] = latency_ms
            print(f"{latency_ms:.3f} ms")
        except Exception as e:
            print(f"ERROR - {e}")
            results[("prefill", bs)] = None

    # Decode measurements (seq_len=1)
    for bs in BATCH_SIZES:
        try:
            print(f"  [Measure] decode  bs={bs} seq=1 ...", end=" ", flush=True)
            latency_ms = measure_latency_cuda_events(model, device, bs, 1)
            results[("decode", bs)] = latency_ms
            print(f"{latency_ms:.3f} ms")
        except Exception as e:
            print(f"ERROR - {e}")
            results[("decode", bs)] = None

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

def rate_accuracy(ratio: float) -> str:
    """Return GOOD/OK/BAD based on how close ratio is to 1.0."""
    deviation = abs(ratio - 1.0)
    if deviation <= 0.20:
        return "GOOD"
    elif deviation <= 0.30:
        return "OK"
    else:
        return "BAD"


def print_comparison_table(predictions: dict, measurements: dict, tp_size: int, predictor_mode: str):
    """Print a formatted comparison table."""
    cfg = MODEL_CONFIG
    print("\n" + "=" * 75)
    print(f"  Model: gpt-oss-20b (MoE: {cfg['num_local_experts']} experts, top-{cfg['num_experts_per_tok']})")
    print(f"  Hardware: A100-SXM4-40GB  |  TP={tp_size}  |  Predictor: {predictor_mode}")
    print("=" * 75)
    header = f"{'Phase':<10} {'BS':>4} {'Predicted (ms)':>16} {'Measured (ms)':>15} {'Ratio':>8} {'Rating':>7}"
    print(header)
    print("-" * 75)

    rows = []
    for phase in ["prefill", "decode"]:
        for bs in BATCH_SIZES:
            key = (phase, bs)
            pred = predictions.get(key)
            meas = measurements.get(key)

            pred_str = f"{pred:.3f}" if pred is not None else "N/A"
            meas_str = f"{meas:.3f}" if meas is not None else "N/A"

            if pred is not None and meas is not None:
                ratio = pred / meas
                rating = rate_accuracy(ratio)
                ratio_str = f"{ratio:.3f}"
            else:
                ratio = None
                rating = "N/A"
                ratio_str = "N/A"

            print(f"{phase:<10} {bs:>4} {pred_str:>16} {meas_str:>15} {ratio_str:>8} {rating:>7}")
            rows.append({
                "phase": phase,
                "batch_size": bs,
                "predicted_ms": pred,
                "measured_ms": meas,
                "ratio": ratio,
                "rating": rating,
            })

    print("=" * 75)
    print("  Ratio = Predicted / Measured  |  GOOD: ±20%  OK: ±30%  BAD: >±30%")
    print("=" * 75)
    return rows


def save_results(output_path: str, rows: list, tp_size: int, predictor_mode: str, device: str):
    """Save results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": "gpt-oss-20b",
        "architecture": MODEL_CONFIG,
        "hardware": "A100-SXM4-40GB",
        "tp_size": tp_size,
        "predictor_mode": predictor_mode,
        "device": device,
        "prefill_seq_len": PREFILL_SEQ_LEN,
        "results": rows,
    }

    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Output] Results saved to {out.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 75)
    print("  LLMCompass MoE Validation — gpt-oss-20b vs A100 measurements")
    print("=" * 75)
    print(f"  Device:     {args.device}")
    print(f"  Model path: {args.model_path}")
    print(f"  TP size:    {args.tp_size}")
    print(f"  Output:     {args.output}")

    predictions = {}
    predictor_mode = "skipped"
    measurements = {}

    if not args.skip_predict:
        predictions, predictor_mode = run_predictions(args.tp_size)

    if not args.skip_real:
        measurements = run_real_measurements(args.model_path, args.device)

    # Print comparison
    rows = print_comparison_table(predictions, measurements, args.tp_size, predictor_mode)

    # Save results
    save_results(args.output, rows, args.tp_size, predictor_mode, args.device)


if __name__ == "__main__":
    main()
