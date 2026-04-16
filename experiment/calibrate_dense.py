#!/usr/bin/env python3
"""
Auto-calibrate LLMCompass dense transformer block overhead constants by comparing
predictions against real isolated layer measurements.

Usage:
    python calibrate_dense.py --model-path /data/models/Llama-3.1-8B-Instruct
    python calibrate_dense.py --model-path /data/models/Llama-3.1-8B-Instruct \
        --model-path /data/models/Qwen2.5-72B-Instruct \
        --device cuda:0 \
        --output llm_predict/profiling/data/A100/dense_calibration.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — must happen before any llm_predict imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_config(model_path: str):
    """Return (model, config) with only 2 transformer layers loaded."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Limit to 2 layers so we load fast and only measure one real layer
    cfg.num_hidden_layers = 2

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    return model, cfg


# ---------------------------------------------------------------------------
# Isolated layer measurement (mirrors validate_moe.py pattern)
# ---------------------------------------------------------------------------

def _capture_layer_inputs(model, batch_size: int, seq_len: int, device: str) -> dict:
    """Run a forward pass and capture the inputs to layer[0] via a pre-hook."""
    captured: dict = {}

    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(
            a.detach() if hasattr(a, "detach") else a for a in args
        )
        captured["kwargs"] = {
            k: v.detach() if hasattr(v, "detach") else v
            for k, v in kwargs.items()
        }

    handle = model.model.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()
    return captured


def measure_isolated_layer(
    layer,
    captured: dict,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Return median latency (ms) of the isolated layer forward pass."""
    args = captured["args"]
    kwargs = {k: v for k, v in captured["kwargs"].items() if k != "past_key_values"}
    kwargs["use_cache"] = False

    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(*args, **kwargs)
        torch.cuda.synchronize()

        lats = []
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            _ = layer(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e))  # ms

    lats.sort()
    return lats[len(lats) // 2]


# ---------------------------------------------------------------------------
# LLMCompass prediction helpers
# ---------------------------------------------------------------------------

def _init_predictor(profiles_dir: str = "llm_predict/profiling/data/A100"):
    """Train and return the ML predictor. Call once."""
    import llm_predict.models.software.transformer as tmod
    from llm_predict.predictors.per_category.predictor import CategoryPredictor

    tmod._kernel_predictor = None
    p = CategoryPredictor(profiles_dir)
    p.train_all()
    tmod._kernel_predictor = p
    return p


def _force_predictor(p, tmod):
    """Re-inject predictor into the module (works around singleton bug)."""
    tmod._kernel_predictor = p


def predict_prefill(
    p,
    tmod,
    system,
    hidden: int,
    heads: int,
    kv_heads: int,
    ffn: int,
    bs: int,
    seq: int,
) -> float:
    """Return LLMCompass prefill prediction in ms."""
    from llm_predict.models.software.transformer import TransformerBlockInitComputationTP
    from llm_predict.models.software.utils import Tensor, data_type_dict

    _force_predictor(p, tmod)
    block = TransformerBlockInitComputationTP(
        d_model=hidden,
        n_heads=heads,
        device_count=1,
        data_type=data_type_dict["fp16"],
        intermediate_size=ffn,
        n_kv_heads=kv_heads,
        use_flash_attention=True,
        activation_type="silu",
        use_ml_predictor=True,
        use_cuda_graph=False,
    )
    X = Tensor([bs, seq, hidden], data_type_dict["fp16"])
    _ = block(X)
    _force_predictor(p, tmod)
    pred_s = block.compile_and_simulate(system, "heuristic-GPU")
    return pred_s * 1e3  # convert to ms


def predict_decode(
    p,
    tmod,
    system,
    hidden: int,
    heads: int,
    kv_heads: int,
    ffn: int,
    bs: int,
    kv_len: int,
) -> float:
    """Return LLMCompass decode prediction in ms."""
    from llm_predict.models.software.transformer import TransformerBlockAutoRegressionTP
    from llm_predict.models.software.utils import Tensor, data_type_dict

    _force_predictor(p, tmod)
    dblock = TransformerBlockAutoRegressionTP(
        d_model=hidden,
        n_heads=heads,
        device_count=1,
        data_type=data_type_dict["fp16"],
        intermediate_size=ffn,
        n_kv_heads=kv_heads,
        use_flash_attention=True,
        activation_type="silu",
        use_ml_predictor=True,
        use_cuda_graph=False,
    )
    x = Tensor([bs, 1, hidden], data_type_dict["fp16"])
    _ = dblock(x, kv_len)
    _force_predictor(p, tmod)
    pred_s = dblock.compile_and_simulate(system, "heuristic-GPU")
    return pred_s * 1e3  # convert to ms


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

DataPoint = Dict  # keys: model_name, mode, bs, seq_or_kv, measured_ms, predicted_ms


def collect_measurements(
    model_path: str,
    device: str,
    p,
    tmod,
    system,
    prefill_grid: Tuple[List[int], List[int]],
    decode_grid: Tuple[List[int], List[int]],
) -> List[DataPoint]:
    """Load model, measure isolated layers, collect LLMCompass predictions."""
    model_name = Path(model_path).name
    print(f"\n{'='*70}")
    print(f"Collecting data for: {model_name}")
    print(f"{'='*70}")

    # Load model (CPU first, then move layer to GPU)
    print("  Loading model (2-layer stub)...")
    model, cfg = load_model_config(model_path)

    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    ffn = getattr(cfg, "intermediate_size", 4 * hidden)

    print(f"  Architecture: hidden={hidden}, heads={heads}, kv_heads={kv_heads}, ffn={ffn}")

    # Move entire model to GPU for capture, then isolate layer
    model = model.to(device).eval()
    layer = model.model.layers[0]

    batch_sizes, seq_lens = prefill_grid
    _, kv_lens = decode_grid

    data_points: List[DataPoint] = []

    # --- Prefill measurements ---
    print(f"\n  Prefill measurements:")
    print(f"  {'BS':>4}  {'SeqLen':>6}  {'Measured(ms)':>13}  {'Predicted(ms)':>14}  {'Error%':>7}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*13}  {'-'*14}  {'-'*7}")

    for bs in batch_sizes:
        for seq in seq_lens:
            try:
                captured = _capture_layer_inputs(model, bs, seq, device)
                measured_ms = measure_isolated_layer(layer, captured)

                os.chdir(REPO_ROOT)
                predicted_ms = predict_prefill(
                    p, tmod, system, hidden, heads, kv_heads, ffn, bs, seq
                )

                err_pct = (measured_ms - predicted_ms) / measured_ms * 100
                print(
                    f"  {bs:>4}  {seq:>6}  {measured_ms:>13.3f}  "
                    f"{predicted_ms:>14.3f}  {err_pct:>+7.1f}%"
                )

                data_points.append(
                    dict(
                        model_name=model_name,
                        mode="prefill",
                        bs=bs,
                        seq_or_kv=seq,
                        hidden=hidden,
                        heads=heads,
                        kv_heads=kv_heads,
                        ffn=ffn,
                        measured_ms=measured_ms,
                        predicted_ms=predicted_ms,
                    )
                )
            except Exception as exc:
                print(f"  {bs:>4}  {seq:>6}  SKIPPED ({exc})")

    # --- Decode measurements ---
    # For decode we need the model to generate KV cache first.
    # Strategy: capture layer input for a single decode step by running prefill
    # with seq_len=kv_len then checking the next token step.
    print(f"\n  Decode measurements (bs=1, varying kv_len):")
    print(f"  {'BS':>4}  {'KV_Len':>6}  {'Measured(ms)':>13}  {'Predicted(ms)':>14}  {'Error%':>7}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*13}  {'-'*14}  {'-'*7}")

    for kv_len in kv_lens:
        try:
            # Prime KV cache with kv_len tokens, then measure one decode step
            bs = 1
            measured_ms = _measure_decode_step(model, layer, bs, kv_len, device)

            os.chdir(REPO_ROOT)
            predicted_ms = predict_decode(
                p, tmod, system, hidden, heads, kv_heads, ffn, bs, kv_len
            )

            err_pct = (measured_ms - predicted_ms) / measured_ms * 100
            print(
                f"  {bs:>4}  {kv_len:>6}  {measured_ms:>13.3f}  "
                f"{predicted_ms:>14.3f}  {err_pct:>+7.1f}%"
            )

            data_points.append(
                dict(
                    model_name=model_name,
                    mode="decode",
                    bs=bs,
                    seq_or_kv=kv_len,
                    hidden=hidden,
                    heads=heads,
                    kv_heads=kv_heads,
                    ffn=ffn,
                    measured_ms=measured_ms,
                    predicted_ms=predicted_ms,
                )
            )
        except Exception as exc:
            print(f"  {1:>4}  {kv_len:>6}  SKIPPED ({exc})")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return data_points


def _measure_decode_step(model, layer, bs: int, kv_len: int, device: str) -> float:
    """
    Measure isolated layer latency for a single decode step with kv_len context.
    Uses past_key_values to simulate real KV cache state.
    """
    # Run prefill to get KV cache
    input_ids = torch.randint(0, 1000, (bs, kv_len), device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values

    # Capture layer input for decode step (single new token)
    captured: dict = {}

    def hook_fn(module, args, kwargs):
        captured["args"] = tuple(
            a.detach() if hasattr(a, "detach") else a for a in args
        )
        captured["kwargs"] = {
            k: v.detach() if hasattr(v, "detach") else v
            for k, v in kwargs.items()
        }

    handle = model.model.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
    next_token = torch.randint(0, 1000, (bs, 1), device=device)
    with torch.no_grad():
        _ = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
    handle.remove()

    # Measure the isolated layer (strip past_key_values from kwargs, use_cache=False)
    return measure_isolated_layer(layer, captured)


# ---------------------------------------------------------------------------
# Calibration / fitting
# ---------------------------------------------------------------------------

def fit_overhead_constants(data_points: List[DataPoint]) -> Dict:
    """
    Fit (framework_overhead_us, kernel_overhead_us, elementwise_scale) to
    minimise prediction error across all data points.

    Model:
        corrected_ms = predicted_ms
                       + framework_overhead_us * 1e-3
                       + (N_KERNELS_DENSE * kernel_overhead_us) * 1e-3
                       + elementwise_scale * elementwise_estimate_ms

    We treat N_KERNELS_DENSE = 14 (from LLMCompass source) as fixed.
    elementwise_estimate_ms is approximated from the residual structure.

    For simplicity we absorb all three into two effective parameters:
        fixed_overhead_ms  = framework_overhead_us * 1e-3 + N_KERNELS * kernel_overhead_us * 1e-3
        elementwise_scale  (applied to the analytical elementwise estimate)

    Since we don't directly have the elementwise component broken out, we
    estimate it as a fraction of the predicted time and fit that fraction.
    """
    from scipy.optimize import minimize

    N_KERNELS_DENSE = 14

    measured = np.array([dp["measured_ms"] for dp in data_points])
    predicted = np.array([dp["predicted_ms"] for dp in data_points])

    # Residuals without any correction
    residuals_ms = measured - predicted

    print(f"\n{'='*70}")
    print("Fitting overhead constants...")
    print(f"  Data points: {len(data_points)}")
    print(f"  Raw residuals: mean={residuals_ms.mean():.3f}ms, "
          f"std={residuals_ms.std():.3f}ms, "
          f"max={residuals_ms.max():.3f}ms")

    # We model the correction as:
    #   correction_ms = framework_us * 1e-3
    #                   + N_KERNELS * kernel_us * 1e-3
    #                   + elementwise_scale * (predicted * elem_fraction)
    # where elem_fraction ~ 0.1 (heuristic: elementwise ops ~10% of total)
    ELEM_FRACTION = 0.10

    def objective(params):
        framework_us, kernel_us, elem_scale = params
        if framework_us < 0 or kernel_us < 0 or elem_scale < 0:
            return 1e9
        correction_ms = (
            framework_us * 1e-3
            + N_KERNELS_DENSE * kernel_us * 1e-3
            + (elem_scale - 1.0) * ELEM_FRACTION * predicted
        )
        corrected = predicted + correction_ms
        errors_pct = np.abs((corrected - measured) / measured) * 100
        return errors_pct.mean()

    # Initial guess based on expected ranges
    x0 = [350.0, 20.0, 2.0]  # framework_us, kernel_us, elem_scale
    bounds = [(0, 2000), (0, 100), (0.1, 10.0)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    framework_us, kernel_us, elem_scale = result.x

    # Compute final corrected predictions
    correction_ms = (
        framework_us * 1e-3
        + N_KERNELS_DENSE * kernel_us * 1e-3
        + (elem_scale - 1.0) * ELEM_FRACTION * predicted
    )
    corrected = predicted + correction_ms
    errors_pct = np.abs((corrected - measured) / measured) * 100
    raw_errors_pct = np.abs((predicted - measured) / measured) * 100

    return dict(
        framework_overhead_us=float(framework_us),
        kernel_overhead_us=float(kernel_us),
        elementwise_scale=float(elem_scale),
        n_kernels_dense=N_KERNELS_DENSE,
        n_data_points=len(data_points),
        mean_error_pct=float(errors_pct.mean()),
        max_error_pct=float(errors_pct.max()),
        raw_mean_error_pct=float(raw_errors_pct.mean()),
        raw_max_error_pct=float(raw_errors_pct.max()),
        corrected_ms=corrected.tolist(),
        measured_ms=measured.tolist(),
        predicted_ms=predicted.tolist(),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(data_points: List[DataPoint], calibration: Dict):
    corrected_ms = calibration["corrected_ms"]
    measured_ms = calibration["measured_ms"]
    predicted_ms = calibration["predicted_ms"]

    print(f"\n{'='*70}")
    print("Final Results: Measured vs Predicted vs Corrected")
    print(f"{'='*70}")
    hdr = (
        f"  {'Model':<25}  {'Mode':>7}  {'BS':>4}  {'Seq/KV':>6}  "
        f"{'Meas(ms)':>9}  {'Pred(ms)':>9}  {'Corr(ms)':>9}  "
        f"{'RawErr%':>8}  {'CorrErr%':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, dp in enumerate(data_points):
        m = measured_ms[i]
        p = predicted_ms[i]
        c = corrected_ms[i]
        raw_err = (p - m) / m * 100
        corr_err = (c - m) / m * 100
        print(
            f"  {dp['model_name']:<25}  {dp['mode']:>7}  {dp['bs']:>4}  "
            f"{dp['seq_or_kv']:>6}  {m:>9.3f}  {p:>9.3f}  {c:>9.3f}  "
            f"{raw_err:>+8.1f}%  {corr_err:>+9.1f}%"
        )

    print(f"\n  Raw prediction  — mean error: {calibration['raw_mean_error_pct']:.1f}%  "
          f"max: {calibration['raw_max_error_pct']:.1f}%")
    print(f"  After correction — mean error: {calibration['mean_error_pct']:.1f}%  "
          f"max: {calibration['max_error_pct']:.1f}%")


def print_constants(calibration: Dict):
    print(f"\n{'='*70}")
    print("Optimal Overhead Constants")
    print(f"{'='*70}")
    print(f"  framework_overhead_us : {calibration['framework_overhead_us']:.1f} µs")
    print(f"  kernel_overhead_us    : {calibration['kernel_overhead_us']:.1f} µs  "
          f"(x{calibration['n_kernels_dense']} kernels = "
          f"{calibration['n_kernels_dense'] * calibration['kernel_overhead_us']:.1f} µs total)")
    print(f"  elementwise_scale     : {calibration['elementwise_scale']:.3f}x")
    print(f"  n_data_points         : {calibration['n_data_points']}")
    print(f"  corrected MAPE        : {calibration['mean_error_pct']:.2f}%")
    print(f"  corrected max error   : {calibration['max_error_pct']:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate LLMCompass dense transformer block overhead constants."
    )
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        required=True,
        metavar="PATH",
        help="Path to a HuggingFace model. Repeat flag for multiple models.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--output",
        default="llm_predict/profiling/data/A100/dense_calibration.json",
        help="Output JSON path for calibration constants.",
    )
    parser.add_argument(
        "--profiles-dir",
        default="llm_predict/profiling/data/A100",
        help="LLMCompass profiles directory (default: llm_predict/profiling/data/A100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="GPU warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measurement repeat count (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = args.device
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # Change to repo root so relative paths work for LLMCompass
    os.chdir(REPO_ROOT)

    # Initialise LLMCompass predictor once
    print(f"\nInitialising LLMCompass predictor from: {args.profiles_dir}")
    import llm_predict.models.software.transformer as tmod
    p = _init_predictor(args.profiles_dir)

    from llm_predict.dse.dse import (
        read_architecture_template,
        template_to_system,
    )
    arch = read_architecture_template("device_configs/GA100.json")
    system = template_to_system(arch)
    # Re-inject after system load (singleton bug)
    tmod._kernel_predictor = p

    # Measurement grids
    prefill_batch_sizes = [1, 2, 4, 8, 16]
    prefill_seq_lens = [128, 256, 512, 1024]
    decode_kv_lens = [128, 256, 512, 1024]

    all_data_points: List[DataPoint] = []

    for model_path in args.model_paths:
        dps = collect_measurements(
            model_path=model_path,
            device=device,
            p=p,
            tmod=tmod,
            system=system,
            prefill_grid=(prefill_batch_sizes, prefill_seq_lens),
            decode_grid=(prefill_batch_sizes, decode_kv_lens),
        )
        all_data_points.extend(dps)

    if not all_data_points:
        print("ERROR: No valid data points collected. Exiting.")
        sys.exit(1)

    print(f"\nTotal data points collected: {len(all_data_points)}")

    # Fit calibration constants
    calibration = fit_overhead_constants(all_data_points)

    # Report
    print_results_table(all_data_points, calibration)
    print_constants(calibration)

    # Build output record
    hardware_name = torch.cuda.get_device_name(device)
    calibration_models = list({dp["model_name"] for dp in all_data_points})

    output_record = {
        "hardware": hardware_name,
        "calibration_models": calibration_models,
        "framework_overhead_us": round(calibration["framework_overhead_us"], 2),
        "kernel_overhead_us": round(calibration["kernel_overhead_us"], 2),
        "elementwise_scale": round(calibration["elementwise_scale"], 4),
        "n_kernels_dense": calibration["n_kernels_dense"],
        "n_data_points": calibration["n_data_points"],
        "mean_error_pct": round(calibration["mean_error_pct"], 2),
        "max_error_pct": round(calibration["max_error_pct"], 2),
        "raw_mean_error_pct": round(calibration["raw_mean_error_pct"], 2),
        "raw_max_error_pct": round(calibration["raw_max_error_pct"], 2),
        "timestamp": time.strftime("%Y-%m-%d"),
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_record, f, indent=4)

    print(f"\nCalibration constants saved to: {output_path.resolve()}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
