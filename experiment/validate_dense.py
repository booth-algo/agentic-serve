"""
validate_dense.py - Validate LLMCompass dense transformer block predictions against
real A100 measurements.

Usage:
    python validate_dense.py --model-path /data/models/Llama-3.1-8B-Instruct \
        [--device cuda:0] [--tp-size 1]

This script lives at ~/llmserve/experiment/validate_dense.py.
REPO_ROOT is set to SCRIPT_DIR.parent (i.e., ~/llmserve).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Path setup: script is in experiment/, repo root is one level up
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# LLMCompass imports (must come after sys.path setup)
# ---------------------------------------------------------------------------
from llm_predict.models.software.transformer import (
    TransformerBlockAutoRegressionTP,
    TransformerBlockInitComputationTP,
)
from llm_predict.models.software.utils import Tensor, data_type_dict
import llm_predict.models.software.transformer as tmod
from llm_predict.dse.dse import (
    read_architecture_template,
    template_to_system,
)
from llm_predict.predictors.per_kernel.predictor import KernelPredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREFILL_SEQ_LEN = 512
DECODE_KV_LEN = 512  # KV cache length used during decode
BATCH_SIZES = [1, 4, 8]
WARMUP_ITERS = 5
REPEAT_ITERS = 20


# ---------------------------------------------------------------------------
# A100 predictor (module-level singleton with explicit reload helper)
# ---------------------------------------------------------------------------
_PREDICTOR: KernelPredictor | None = None
_SYSTEM = None
_ARCH = None


def _load_predictor() -> tuple:
    """Load (or reload) the A100 KernelPredictor and associated system."""
    global _PREDICTOR, _SYSTEM, _ARCH
    if _PREDICTOR is None:
        # Use relative path so the profiler's cache keys are stable
        profile_dir = "llm_predict/profiling/data/A100"
        tmod._kernel_predictor = None
        _PREDICTOR = KernelPredictor(profile_dir)
        _PREDICTOR.train_all()
        _ARCH = read_architecture_template("device_configs/GA100.json")
        _SYSTEM = template_to_system(_ARCH)
    return _PREDICTOR, _SYSTEM


def _force_predictor() -> None:
    """Re-inject the predictor into the transformer module (singleton workaround)."""
    predictor, _ = _load_predictor()
    tmod._kernel_predictor = predictor


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------
def load_model_config(model_path: str) -> dict:
    """Read HuggingFace config and extract dense transformer arch params."""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    params = {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_act": getattr(cfg, "hidden_act", "silu"),
    }
    return params, cfg


def load_model_2layers(model_path: str, device: str, config):
    """Load model with only 2 hidden layers to reduce GPU memory."""
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def get_layers(model):
    """Return the list of transformer decoder layers from a model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(
        "Cannot locate decoder layers. Tried model.model.layers and model.transformer.h."
    )


# ---------------------------------------------------------------------------
# Real measurement helpers
# ---------------------------------------------------------------------------
def _capture_layer_inputs(model, batch_size: int, seq_len: int, device: str) -> dict:
    """Run one forward pass and capture the inputs to layers[0] via a hook."""
    captured: dict = {}

    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(
            a.detach() if hasattr(a, "detach") else a for a in args
        )
        captured["kwargs"] = {
            k: v.detach() if hasattr(v, "detach") else v
            for k, v in kwargs.items()
        }

    layers = get_layers(model)
    handle = layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    return captured


def measure_isolated_layer(layer, captured: dict, warmup: int = WARMUP_ITERS, repeat: int = REPEAT_ITERS) -> float:
    """Return median latency (ms) for a single isolated layer forward pass."""
    args = captured["args"]
    kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    kwargs["use_cache"] = False

    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(*args, **kwargs)
        torch.cuda.synchronize()

        lats = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            _ = layer(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            lats.append(start.elapsed_time(end))

    lats.sort()
    return lats[len(lats) // 2]


# ---------------------------------------------------------------------------
# LLMCompass prediction helpers
# ---------------------------------------------------------------------------
def predict_prefill(
    hidden_size: int,
    n_heads: int,
    n_kv_heads: int,
    intermediate_size: int,
    activation_type: str,
    tp_size: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """Return predicted prefill latency (ms) for one transformer block."""
    _force_predictor()
    _, system = _load_predictor()

    block = TransformerBlockInitComputationTP(
        d_model=hidden_size,
        n_heads=n_heads,
        device_count=tp_size,
        data_type=data_type_dict["fp16"],
        intermediate_size=intermediate_size,
        n_kv_heads=n_kv_heads,
        use_flash_attention=True,
        activation_type=activation_type,
        use_ml_predictor=True,
        use_cuda_graph=False,
    )
    X = Tensor([batch_size, seq_len, hidden_size], data_type_dict["fp16"])
    _ = block(X)

    _force_predictor()
    lat_s = block.compile_and_simulate(system, "heuristic-GPU")
    return lat_s * 1000.0  # convert s -> ms


def predict_decode(
    hidden_size: int,
    n_heads: int,
    n_kv_heads: int,
    intermediate_size: int,
    activation_type: str,
    tp_size: int,
    batch_size: int,
    kv_cache_len: int,
) -> float:
    """Return predicted decode latency (ms) for one transformer block."""
    _force_predictor()
    _, system = _load_predictor()

    block = TransformerBlockAutoRegressionTP(
        d_model=hidden_size,
        n_heads=n_heads,
        device_count=tp_size,
        data_type=data_type_dict["fp16"],
        intermediate_size=intermediate_size,
        n_kv_heads=n_kv_heads,
        use_flash_attention=True,
        activation_type=activation_type,
        use_ml_predictor=True,
        use_cuda_graph=False,
    )
    x = Tensor([batch_size, 1, hidden_size], data_type_dict["fp16"])
    _ = block(x, kv_cache_len)

    _force_predictor()
    lat_s = block.compile_and_simulate(system, "heuristic-GPU")
    return lat_s * 1000.0


# ---------------------------------------------------------------------------
# Rating helper
# ---------------------------------------------------------------------------
def rate(predicted: float, measured: float) -> str:
    if measured == 0:
        return "N/A"
    err = abs(predicted - measured) / measured
    if err <= 0.20:
        return "GOOD"
    if err <= 0.30:
        return "OK"
    return "BAD"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate LLMCompass dense block predictions vs real A100 measurements"
    )
    p.add_argument("--model-path", required=True, help="Path to HuggingFace model dir")
    p.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    model_name = Path(args.model_path).name
    device = args.device
    tp_size = args.tp_size

    print(f"\n=== Dense Block Validation: {model_name} ===")
    print(f"Device : {device}  |  TP size: {tp_size}")

    # ------------------------------------------------------------------
    # 1. Load model config
    # ------------------------------------------------------------------
    print("\n[1/4] Loading model config...")
    arch_params, hf_config = load_model_config(args.model_path)
    hidden_size = arch_params["hidden_size"]
    intermediate_size = arch_params["intermediate_size"]
    n_heads = arch_params["num_attention_heads"]
    n_kv_heads = arch_params["num_key_value_heads"]
    activation_type = arch_params["hidden_act"]
    num_layers_orig = arch_params["num_hidden_layers"]

    print(f"  hidden_size       = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  num_attention_heads = {n_heads}")
    print(f"  num_key_value_heads = {n_kv_heads}")
    print(f"  num_hidden_layers   = {num_layers_orig}")
    print(f"  hidden_act          = {activation_type}")

    # ------------------------------------------------------------------
    # 2. Load model (2 layers only)
    # ------------------------------------------------------------------
    print("\n[2/4] Loading model (2 layers)...")
    model = load_model_2layers(args.model_path, device, hf_config)
    layers = get_layers(model)
    layer = layers[0]
    print(f"  Loaded. Layer type: {type(layer).__name__}")

    # ------------------------------------------------------------------
    # 3. LLMCompass predictions
    # ------------------------------------------------------------------
    print("\n[3/4] Running LLMCompass predictions (A100)...")
    # Change working directory to REPO_ROOT so relative paths resolve
    os.chdir(REPO_ROOT)
    _load_predictor()

    predictions: dict = {}
    for bs in BATCH_SIZES:
        print(f"  Predicting prefill bs={bs}, seq_len={PREFILL_SEQ_LEN}...")
        prefill_pred = predict_prefill(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            tp_size=tp_size,
            batch_size=bs,
            seq_len=PREFILL_SEQ_LEN,
        )
        print(f"    -> {prefill_pred:.3f} ms")

        print(f"  Predicting decode bs={bs}, kv_cache_len={DECODE_KV_LEN}...")
        decode_pred = predict_decode(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            tp_size=tp_size,
            batch_size=bs,
            kv_cache_len=DECODE_KV_LEN,
        )
        print(f"    -> {decode_pred:.3f} ms")

        predictions[bs] = {
            "prefill_pred_ms": prefill_pred,
            "decode_pred_ms": decode_pred,
        }

    # ------------------------------------------------------------------
    # 4. Real measurements
    # ------------------------------------------------------------------
    print("\n[4/4] Running real measurements on GPU...")
    measurements: dict = {}
    for bs in BATCH_SIZES:
        print(f"  Capturing layer inputs (prefill) bs={bs}, seq_len={PREFILL_SEQ_LEN}...")
        captured_prefill = _capture_layer_inputs(model, bs, PREFILL_SEQ_LEN, device)
        prefill_meas = measure_isolated_layer(layer, captured_prefill)
        print(f"    -> {prefill_meas:.3f} ms")

        print(f"  Capturing layer inputs (decode)  bs={bs}, seq_len=1...")
        captured_decode = _capture_layer_inputs(model, bs, 1, device)
        decode_meas = measure_isolated_layer(layer, captured_decode)
        print(f"    -> {decode_meas:.3f} ms")

        measurements[bs] = {
            "prefill_meas_ms": prefill_meas,
            "decode_meas_ms": decode_meas,
        }

    # ------------------------------------------------------------------
    # 5. Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Model':<30}  {model_name}")
    print(f"{'Prefill seq_len':<30}  {PREFILL_SEQ_LEN}   Decode KV len: {DECODE_KV_LEN}")
    print("=" * 90)
    header = f"{'BS':>4} {'Mode':<8} {'Predicted(ms)':>14} {'Measured(ms)':>14} {'Error%':>8} {'Rating':>6}"
    print(header)
    print("-" * 90)

    results = []
    for bs in BATCH_SIZES:
        pp = predictions[bs]["prefill_pred_ms"]
        pm = measurements[bs]["prefill_meas_ms"]
        dp = predictions[bs]["decode_pred_ms"]
        dm = measurements[bs]["decode_meas_ms"]

        p_err = (pp - pm) / pm * 100 if pm else float("nan")
        d_err = (dp - dm) / dm * 100 if dm else float("nan")

        p_rating = rate(pp, pm)
        d_rating = rate(dp, dm)

        print(f"{bs:>4} {'prefill':<8} {pp:>14.3f} {pm:>14.3f} {p_err:>+8.1f}% {p_rating:>6}")
        print(f"{bs:>4} {'decode':<8} {dp:>14.3f} {dm:>14.3f} {d_err:>+8.1f}% {d_rating:>6}")

        results.append({
            "batch_size": bs,
            "prefill": {
                "predicted_ms": pp,
                "measured_ms": pm,
                "error_pct": round(p_err, 2),
                "rating": p_rating,
            },
            "decode": {
                "predicted_ms": dp,
                "measured_ms": dm,
                "error_pct": round(d_err, 2),
                "rating": d_rating,
            },
        })

    print("=" * 90)
    print("Rating: GOOD = ±20%, OK = ±30%, BAD = >±30%")

    # ------------------------------------------------------------------
    # 6. Save JSON results
    # ------------------------------------------------------------------
    output_dir = SCRIPT_DIR  # experiment/
    output_path = output_dir / f"dense_validation_{model_name}.json"
    output_data = {
        "model_path": args.model_path,
        "model_name": model_name,
        "device": device,
        "tp_size": tp_size,
        "prefill_seq_len": PREFILL_SEQ_LEN,
        "decode_kv_len": DECODE_KV_LEN,
        "arch_params": arch_params,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
