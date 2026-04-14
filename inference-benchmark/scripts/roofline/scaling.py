"""
Virtual GPU scaling analysis using LLMCompass.

1. Layer scaling curves: predict latency as layers increase from 32→1000
2. TP/DP parallelism heatmaps: throughput across all valid configurations
3. Optimal parallelism recommendations per model × device count
"""

import json
import sys
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.environ.get("LLMSERVE_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "llmserve")))

from llmcompass.design_space_exploration.dse import template_to_system, read_architecture_template
from llmcompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from llmcompass.software_model.utils import Tensor, data_type_dict

# ── Config ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "roofline" / "scaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_CONFIG = os.environ.get(
    "DEVICE_CONFIG",
    str(Path(os.environ.get("LLMSERVE_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "llmserve"))) / "device_configs" / "H100.json")
)

MODELS = {
    "Llama-3.1-8B": {"d_model": 4096, "n_heads": 32, "n_layers": 32},
    "Qwen3-32B":    {"d_model": 5120, "n_heads": 40, "n_layers": 64},
    "Llama-3.1-70B":{"d_model": 8192, "n_heads": 64, "n_layers": 80},
    "Qwen2.5-72B":  {"d_model": 8192, "n_heads": 64, "n_layers": 80},
}

LAYER_COUNTS = [32, 64, 128, 256, 512, 1000]
DEVICE_COUNTS = [1, 2, 4, 8]
BATCH_SIZE = 8
SEQ_LEN = 512


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def predict_per_layer_latency(system, d_model, n_heads, batch_size, seq_len, phase, tp_size=1):
    """Predict latency for a single transformer layer."""
    if phase == "prefill":
        block = TransformerBlockInitComputationTP(
            d_model=d_model, n_heads=n_heads,
            device_count=tp_size, data_type=data_type_dict["fp16"],
        )
        X = Tensor([batch_size, seq_len, d_model], data_type_dict["fp16"])
        _ = block(X)
    else:
        block = TransformerBlockAutoRegressionTP(
            d_model=d_model, n_heads=n_heads,
            device_count=tp_size, data_type=data_type_dict["fp16"],
        )
        X = Tensor([batch_size, 1, d_model], data_type_dict["fp16"])
        _ = block(X, seq_len=seq_len)

    latency = block.compile_and_simulate(system, compile_mode="heuristic-GPU")
    return latency


# ── Step 1: Layer Scaling Curves ──────────────────────────────────────

def run_layer_scaling(system):
    """Predict latency as layer count scales from 32 to 1000."""
    print("=" * 70)
    print("STEP 1: Layer Scaling Curves")
    print("=" * 70)

    results = {}

    for model_name, cfg in MODELS.items():
        results[model_name] = {"prefill": {}, "decode": {}}

        for phase in ["prefill", "decode"]:
            print(f"\n  {model_name} {phase}...", end=" ", flush=True)
            t0 = time.time()

            per_layer = predict_per_layer_latency(
                system, cfg["d_model"], cfg["n_heads"],
                BATCH_SIZE, SEQ_LEN, phase,
            )

            for n_layers in LAYER_COUNTS:
                total_ms = per_layer * n_layers * 1e3  # seconds → ms
                results[model_name][phase][str(n_layers)] = {
                    "per_layer_ms": per_layer * 1e3,
                    "total_ms": total_ms,
                    "n_layers": n_layers,
                }

            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) | per_layer={per_layer*1e6:.1f}us")

    # Save JSON
    out_path = OUTPUT_DIR / "layer_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Plot
    setup_style()
    MODEL_COLORS = {
        "Llama-3.1-8B": "#0072B2",
        "Qwen3-32B": "#E69F00",
        "Llama-3.1-70B": "#009E73",
        "Qwen2.5-72B": "#CC79A7",
    }

    for phase in ["prefill", "decode"]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name in MODELS:
            layers = []
            latencies = []
            for nl_str, data in results[model_name][phase].items():
                layers.append(data["n_layers"])
                latencies.append(data["total_ms"])

            color = MODEL_COLORS.get(model_name, "#666666")
            ax.plot(layers, latencies, "o-", color=color, linewidth=2,
                    markersize=6, label=model_name)

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Total Latency (ms)")
        ax.set_title(f"Layer Scaling — {phase} (BS={BATCH_SIZE}, seq={SEQ_LEN})")
        ax.legend(fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        for fmt in ("pdf", "png"):
            fig.savefig(OUTPUT_DIR / f"layer_scaling_{phase}.{fmt}")
        plt.close(fig)
        print(f"  Saved: layer_scaling_{phase}.{{pdf,png}}")

    return results


# ── Step 2: TP/DP Parallelism Search ─────────────────────────────────

def run_parallelism_search(system):
    """Search all TP×DP configurations for each model × device count."""
    print("\n" + "=" * 70)
    print("STEP 2: TP/DP Parallelism Search")
    print("=" * 70)

    results = {}

    for model_name, cfg in MODELS.items():
        results[model_name] = {}

        for n_devices in DEVICE_COUNTS:
            print(f"\n  {model_name} | {n_devices} GPUs...", flush=True)
            configs = []

            for dp in range(1, n_devices + 1):
                if n_devices % dp != 0:
                    continue
                tp = n_devices // dp

                # Skip if TP > n_heads (can't split heads further)
                if tp > cfg["n_heads"]:
                    continue

                try:
                    t0 = time.time()
                    per_layer = predict_per_layer_latency(
                        system, cfg["d_model"], cfg["n_heads"],
                        BATCH_SIZE, SEQ_LEN, "prefill", tp_size=tp,
                    )
                    total_latency = per_layer * cfg["n_layers"]
                    # Throughput: dp replicas process in parallel
                    throughput = dp * BATCH_SIZE / total_latency  # samples/sec
                    elapsed = time.time() - t0

                    config_result = {
                        "dp": dp, "tp": tp,
                        "per_layer_ms": per_layer * 1e3,
                        "total_latency_ms": total_latency * 1e3,
                        "throughput_samples_per_sec": throughput,
                        "valid": True,
                    }
                    configs.append(config_result)
                    print(f"    DP={dp} TP={tp}: {throughput:.1f} samples/s "
                          f"({total_latency*1e3:.1f}ms) [{elapsed:.1f}s]")
                except Exception as e:
                    configs.append({
                        "dp": dp, "tp": tp, "valid": False,
                        "error": str(e)[:80],
                    })
                    print(f"    DP={dp} TP={tp}: FAILED ({str(e)[:40]})")

            results[model_name][str(n_devices)] = configs

    # Save JSON
    out_path = OUTPUT_DIR / "parallelism_search.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Plot heatmaps
    setup_style()
    for model_name in MODELS:
        fig, axes = plt.subplots(1, len(DEVICE_COUNTS), figsize=(4 * len(DEVICE_COUNTS), 4),
                                  squeeze=False)

        for idx, n_devices in enumerate(DEVICE_COUNTS):
            ax = axes[0][idx]
            configs = results[model_name].get(str(n_devices), [])
            valid = [c for c in configs if c.get("valid", False)]

            if not valid:
                ax.text(0.5, 0.5, "No valid\nconfigs", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_title(f"{n_devices} GPUs")
                continue

            dps = [c["dp"] for c in valid]
            tps = [c["tp"] for c in valid]
            throughputs = [c["throughput_samples_per_sec"] for c in valid]

            scatter = ax.scatter(tps, dps, c=throughputs, cmap="YlOrRd", s=200,
                                 edgecolors="black", linewidths=0.5, zorder=5)
            for c in valid:
                ax.annotate(f"{c['throughput_samples_per_sec']:.0f}",
                            (c["tp"], c["dp"]), ha="center", va="center",
                            fontsize=8, fontweight="bold")

            ax.set_xlabel("TP size")
            ax.set_ylabel("DP size")
            ax.set_title(f"{n_devices} GPUs")
            ax.set_xticks(sorted(set(tps)))
            ax.set_yticks(sorted(set(dps)))

        fig.suptitle(f"{model_name} — Throughput (samples/s) by TP×DP", fontsize=13)
        fig.tight_layout()

        for fmt in ("pdf", "png"):
            fig.savefig(OUTPUT_DIR / f"heatmap_{model_name}.{fmt}")
        plt.close(fig)
        print(f"  Saved: heatmap_{model_name}.{{pdf,png}}")

    return results


# ── Step 3: Optimal Config Summary ───────────────────────────────────

def run_optimal_summary(search_results):
    """Extract best config per model × device count and plot comparison."""
    print("\n" + "=" * 70)
    print("STEP 3: Optimal Parallelism Recommendations")
    print("=" * 70)

    optimal = {}
    print(f"\n{'Model':<18s} {'GPUs':<6s} {'Best DP':<8s} {'Best TP':<8s} {'Throughput':<15s} {'Latency'}")
    print("-" * 75)

    for model_name, device_results in search_results.items():
        optimal[model_name] = {}
        for n_devices_str, configs in device_results.items():
            valid = [c for c in configs if c.get("valid", False)]
            if not valid:
                continue
            best = max(valid, key=lambda c: c["throughput_samples_per_sec"])
            optimal[model_name][n_devices_str] = best
            print(f"{model_name:<18s} {n_devices_str:<6s} {best['dp']:<8d} {best['tp']:<8d} "
                  f"{best['throughput_samples_per_sec']:<15.1f} {best['total_latency_ms']:.1f}ms")

    # Save JSON
    out_path = OUTPUT_DIR / "optimal_configs.json"
    with open(out_path, "w") as f:
        json.dump(optimal, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Plot: throughput vs GPU count for each model
    setup_style()
    MODEL_COLORS = {
        "Llama-3.1-8B": "#0072B2",
        "Qwen3-32B": "#E69F00",
        "Llama-3.1-70B": "#009E73",
        "Qwen2.5-72B": "#CC79A7",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, configs in optimal.items():
        gpus = []
        throughputs = []
        for n_str, best in sorted(configs.items(), key=lambda x: int(x[0])):
            gpus.append(int(n_str))
            throughputs.append(best["throughput_samples_per_sec"])

        color = MODEL_COLORS.get(model_name, "#666666")
        ax.plot(gpus, throughputs, "o-", color=color, linewidth=2,
                markersize=8, label=model_name)

        # Annotate best TP/DP
        for n_str, best in configs.items():
            ax.annotate(f"DP{best['dp']}×TP{best['tp']}",
                        (int(n_str), best["throughput_samples_per_sec"]),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=7, color=color)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(f"Optimal Throughput Scaling (BS={BATCH_SIZE}, seq={SEQ_LEN})")
    ax.legend(fontsize=9)
    ax.set_xticks(DEVICE_COUNTS)

    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"optimal_configs_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved: optimal_configs_summary.{{pdf,png}}")

    return optimal


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading H100 system config...")
    arch_specs = read_architecture_template(DEVICE_CONFIG)
    system = template_to_system(arch_specs)
    print(f"System: {system.device.compute_module.core_count} cores, "
          f"{system.interconnect.device_count} devices\n")

    # Step 1
    layer_results = run_layer_scaling(system)

    # Step 2
    search_results = run_parallelism_search(system)

    # Step 3
    optimal = run_optimal_summary(search_results)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files: {', '.join(f.name for f in OUTPUT_DIR.iterdir())}")


if __name__ == "__main__":
    main()
