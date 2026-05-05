#!/usr/bin/env python3
"""Per-component OI/CF roofline — clean publication-quality figure.

Panel (a): Per-component OI on H100 roofline. Selected components shown
at C=1 (colored circles) and C=80 (smaller, lighter). Ridge point, HW
roofline curve, formula annotation. Style matches roofline_multiturn_8b.py.

Panel (b): Capacity footprint vs throughput ceiling. CF sweep from C=1..80
with horizontal compute ceiling, vertical weight/HBM lines, and KV cache
decomposition annotation.

Usage:
    python scripts/roofline/plot_per_layer_roofline.py \
        --c1-json results/roofline/raw/llama8b_analytical_prefill_B1.json \
        --c80-json results/roofline/raw/llama8b_analytical_prefill_B80.json \
        --output-pdf figures/per_layer_oi_cf.pdf \
        --output-latex figures/per_layer_table.tex
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ── Constants ───────────────────────────────────────────────────────────

P_PEAK = 989.0; BW = 3352.0; HBM_GB = 80.0
RIDGE_OI = P_PEAK * 1000.0 / BW

MODEL_M = 8e9; MODEL_D = 4096; MODEL_TP = 1
BPP = 2; BPKV = 2; N_KV = 8; N_H = 32; N_LAYERS = 32
HEAD_DIM = MODEL_D // N_H
W_TOTAL_BYTES = MODEL_M * BPP / MODEL_TP
C_SWEEP = [1, 5, 10, 20, 40, 80]
C_ANNOT = "#555555"

# ── Paper formulas ──────────────────────────────────────────────────────

def gemm_ai(M, d=MODEL_D):
    return M * d / (2.0 * M + d)

def kernel_ceiling_at_B(B, avg_ISL=640, avg_OSL=256, d=MODEL_D, cache_hit_rate=0.0):
    eff_ISL = avg_ISL * (1.0 - cache_hit_rate)
    decode_ai = gemm_ai(max(B, 1), d)
    decode_peak = min(P_PEAK, BW * decode_ai / 1000.0)
    prefill_ai = gemm_ai(max(eff_ISL, 1), d)
    prefill_peak = min(P_PEAK, BW * prefill_ai / 1000.0)
    prefill_time = eff_ISL / prefill_peak
    decode_time = avg_OSL / decode_peak
    return (avg_ISL + avg_OSL) / (prefill_time + decode_time)

def kv_per_seq_bytes(ISL, OSL):
    L = ISL + OSL / 2.0
    return N_LAYERS * 2.0 * HEAD_DIM * N_KV * BPKV * L / MODEL_TP

def cf_per_gpu(C, ISL, OSL):
    L = ISL + OSL / 2.0
    wt = (MODEL_M * BPP) / MODEL_TP / 1e9
    kv = (N_LAYERS * 2.0 * MODEL_D * L * (N_KV / N_H) * BPKV) / MODEL_TP / 1e9
    return wt + C * kv

def component_oi_ceiling(oi):
    return min(P_PEAK, BW * oi / 1000.0)

# ── Component registry ──────────────────────────────────────────────────

COMP_WEIGHTS = {
    "q_proj": MODEL_D * MODEL_D,
    "k_proj": MODEL_D * MODEL_D * N_KV // N_H,
    "v_proj": MODEL_D * MODEL_D * N_KV // N_H,
    "o_proj": MODEL_D * MODEL_D,
    "gate_proj": MODEL_D * 14336,
    "up_proj": MODEL_D * 14336,
    "down_proj": 14336 * MODEL_D,
}

COMP_COLORS = {
    "q_proj": "#E69F00", "k_proj": "#56B4E9", "v_proj": "#56B4E9",
    "o_proj": "#009E73", "gate_proj": "#CC79A7", "up_proj": "#CC79A7",
    "down_proj": "#882255",
}

COMP_LABELS = {
    "q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O",
    "gate_proj": "Gate", "up_proj": "Up", "down_proj": "Down",
    "attention": "Attn", "rmsnorm_in": "RMS", "rope": "RoPE",
    "silu": "SiLU", "gate_mul": "Gate×Up",
    "rmsnorm_post": "RMS",
}

# Only show these components on the plot (the interesting ones)
SHOW_COMPONENTS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "attention", "rmsnorm_in", "rope", "silu",
]

# ── Plotting ────────────────────────────────────────────────────────────

def plot_oi_panel(ax, comps_c1, comps_c80, oi_c1, oi_c80):
    """Panel (a): Per-component OI on H100 roofline."""
    OI_MIN, OI_MAX = 0.5, 6000
    OI_grid = np.logspace(np.log10(OI_MIN), np.log10(OI_MAX), 800)
    T_hw = np.minimum(P_PEAK, BW * OI_grid / 1000.0)
    ax.plot(OI_grid, T_hw, "-", color="black", lw=1.8, zorder=3)

    # Ridge
    ax.axvline(RIDGE_OI, color="#bbbbbb", lw=0.5, ls=":", zorder=2)
    ax.text(RIDGE_OI * 1.12, 1.7, f"ridge\n({RIDGE_OI:.0f})",
            fontsize=6, color="#999999", ha="left", va="bottom")

    # Build lookup
    c1_by_name = {c["name"]: c for c in comps_c1}
    c80_by_name = {c["name"]: c for c in comps_c80}

    # Plot C=1 points (larger, labeled) and C=80 (smaller, muted)
    plotted_labels = set()
    for name in SHOW_COMPONENTS:
        c1 = c1_by_name.get(name)
        c80 = c80_by_name.get(name)
        if not c1 or c1["oi"] < 0.5:
            continue
        clr = COMP_COLORS.get(name, "#999999")
        lbl = COMP_LABELS.get(name, name)

        # C=1 — prominent
        ceil1 = component_oi_ceiling(c1["oi"])
        ax.plot(c1["oi"], ceil1, "o", color=clr, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, zorder=6, alpha=0.9)

        # C=80 — smaller, same color
        if c80:
            ceil80 = component_oi_ceiling(c80["oi"])
            ax.plot(c80["oi"], ceil80, "D", color=clr, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.3, zorder=5, alpha=0.45)

        # Label C=1 points with careful positioning to avoid overlaps
        label_offsets = {
            # name: (ox, oy) offset in points
            "Q": (5, 5), "K": (-24, -8), "V": (5, -10), "O": (5, -5),
            "Gate": (5, 5), "Up": (5, -10), "Down": (-24, 6),
            "Attn": (5, -10), "RMS": (-22, 6), "RoPE": (-22, -6),
            "SiLU": (5, -10),
        }
        ox, oy = label_offsets.get(lbl, (5, 4))
        ax.annotate(lbl, (c1["oi"], ceil1), textcoords="offset points",
                    xytext=(ox, oy), fontsize=5.5, color="#333333",
                    fontweight="bold" if lbl in ("Q","K","V","O","Gate","Up","Down") else "normal")

    # Legend: C=1 vs C=80 vs layer total
    lyr_ceil1 = component_oi_ceiling(oi_c1)
    lyr_ceil80 = component_oi_ceiling(oi_c80)
    ax.plot(oi_c1, lyr_ceil1, "s", color="black", markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=9)
    ax.annotate(f"layer\n(C=1)", (oi_c1, lyr_ceil1),
                textcoords="offset points", xytext=(10, -15),
                fontsize=6.5, color="black",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7))

    # Formula
    ax.text(0.97, 0.04,
            r"$\mathrm{OI} = \frac{\sum \mathrm{FLOPs}}{\sum \mathrm{DRAM\_bytes}}$",
            transform=ax.transAxes, fontsize=7, color="#666666",
            ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(OI_MIN, OI_MAX); ax.set_ylim(0.5, 1500)
    ax.set_xlabel("Operational intensity [FLOP / byte]")
    ax.set_ylabel("Throughput ceiling [TFLOP/s]")
    ax.grid(True, which="both", alpha=0.18, ls=":")
    ax.text(0.02, 0.94, "(a)", transform=ax.transAxes,
            fontsize=11, fontweight="bold")


def plot_cf_panel(ax, avg_ISL=640, avg_OSL=256):
    """Panel (b): Capacity footprint vs throughput (paper style)."""
    wt_total = MODEL_M * BPP / MODEL_TP / 1e9

    C_points = []
    for C in C_SWEEP:
        cf = cf_per_gpu(C, avg_ISL, avg_OSL)
        tput = kernel_ceiling_at_B(C, avg_ISL, avg_OSL)
        C_points.append((cf, tput, C))

    cf_vals = [p[0] for p in C_points]
    tput_vals = [p[1] for p in C_points]
    cf_max = cf_vals[-1] + 5

    # Compute ceiling
    ax.axhline(P_PEAK, color="black", lw=1.8, zorder=3)

    # Weight vertical
    ax.axvline(wt_total, color="#aaaaaa", lw=0.7, ls="--", zorder=2)
    ax.text(wt_total + 0.3, 1.8, f"weights\n({wt_total:.0f} GB)",
            rotation=90, color="#aaaaaa", fontsize=6, va="bottom")

    # HBM limit + hatch
    ax.axvline(HBM_GB, color="#aaaaaa", lw=0.7, ls="--", zorder=2)
    ax.axvspan(HBM_GB, cf_max, facecolor="#cc4444", alpha=0.04, zorder=1)
    ax.text(HBM_GB + 0.3, 10, f"HBM\n({HBM_GB:.0f} GB)",
            rotation=90, color="#aaaaaa", fontsize=6, va="bottom")

    # CF sweep as a line with points
    ax.plot(cf_vals, tput_vals, "-", color="#0072B2", lw=1.5, zorder=5)
    ax.plot(cf_vals, tput_vals, "o", color="#0072B2", markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=6)

    # C annotations
    offsets = {1: (8, -10), 5: (-20, -8), 10: (-20, -8),
               20: (-20, 8), 40: (-20, 10), 80: (8, -10)}
    for i, (cf, tput, C) in enumerate(C_points):
        ox, oy = offsets.get(C, (5, 5))
        ax.annotate(f"C={C}", (cf, tput), textcoords="offset points",
                    xytext=(ox, oy), fontsize=6, color=C_ANNOT)

    # KV decomposition annotation at C=80
    cf_80 = cf_vals[-1]
    kv_80 = cf_80 - wt_total
    ax.annotate(f"weights {wt_total:.0f} GB\n+ KV {kv_80:.1f} GB",
                xy=(cf_80, tput_vals[-1]), textcoords="offset points",
                xytext=(15, 15), fontsize=6.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7),
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85, pad=3))

    # Formula
    ax.text(0.97, 0.04,
            r"$\mathrm{CF} = m\,b_p + C \cdot \mathrm{KV}_{\mathrm{seq}}$",
            transform=ax.transAxes, fontsize=7, color="#666666",
            ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    ax.set_yscale("log")
    ax.set_xlim(wt_total - 3, cf_max)
    ax.set_ylim(0.5, 1500)
    ax.set_xlabel("Capacity footprint [GB]")
    ax.set_ylabel("Throughput ceiling [TFLOP/s]")
    ax.grid(True, which="both", alpha=0.18, ls=":")
    ax.text(0.02, 0.94, "(b)", transform=ax.transAxes,
            fontsize=11, fontweight="bold")


def render_figure(c1, c80, out_pdf):
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.titlesize": 11,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    fig.subplots_adjust(wspace=0.34)

    comps_c1 = c1["components"]
    comps_c80 = c80["components"]
    oi_c1 = c1["oi"]
    oi_c80 = c80["oi"]

    plot_oi_panel(ax_a, comps_c1, comps_c80, oi_c1, oi_c80)
    plot_cf_panel(ax_b, avg_ISL=640, avg_OSL=256)

    # Legend
    legend_elems = [
        Line2D([0], [0], color="black", lw=1.8,
               label=f"H100 HW roofline ({P_PEAK:.0f} TFLOP/s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00",
               markersize=7, markeredgecolor="black", markeredgewidth=0.5,
               label=r"$C{=}1$ (bandwidth-limited)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888",
               markersize=4, markeredgecolor="black", markeredgewidth=0.3,
               label=r"$C{=}80$ (compute-limited)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markersize=8, markeredgecolor="white", markeredgewidth=0.8,
               label="Layer aggregate"),
    ]
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"PDF: {out_pdf}")


# ── Table printing ──────────────────────────────────────────────────────

def classify_bound(oi):
    return "compute" if oi > RIDGE_OI else "bandwidth"

def print_table(c1, c80):
    comps = c1["components"]
    wt_layer = sum(COMP_WEIGHTS.values()) * BPP / 1e9

    print(f"\n{'='*95}")
    print(f"Per-component bound analysis — Llama-3.1-8B, H100, bf16, tp=1")
    print(f"Ridge: {RIDGE_OI:.0f} FLOP/byte  |  HBM: {HBM_GB:.0f} GB  |  "
          f"Per-layer weight: {wt_layer:.2f} GB")
    print(f"{'='*95}")
    hdr = f"{'Component':<14s} {'Wt':>5s} {'OI(C1)':>7s} {'OI(C80)':>7s} " \
          f"{'Bound(C1)':>10s} {'Bound(C80)':>10s}  {'Why'}"
    print(hdr); print("-" * 95)

    for c1_item, c80_item in zip(comps, c80["components"]):
        name = c1_item["name"]
        label = COMP_LABELS.get(name, name)
        wt_mb = COMP_WEIGHTS.get(name, 0) * BPP / 1e6
        wt_str = f"{wt_mb:>3.0f}" if wt_mb > 0 else "  —"
        b1 = classify_bound(c1_item["oi"])
        b80 = classify_bound(c80_item["oi"])
        why = ("GEMM" if name in COMP_WEIGHTS else
               "Flash attn" if name == "attention" else
               "Elementwise" if "norm" in name or name in ("rope","silu","gate_mul") else "")
        print(f"{label:<14s} {wt_str:>3s}MB {c1_item['oi']:>7.0f} {c80_item['oi']:>7.0f} "
              f"{b1:>10s} {b80:>10s}  {why}")

    print("-" * 95)
    print(f"{'LAYER':<14s} {wt_layer*1000:>3.0f}MB {c1['oi']:>7.0f} {c80['oi']:>7.0f} "
          f"{classify_bound(c1['oi']):>10s} {classify_bound(c80['oi']):>10s}")

    # Capacity thresholds
    print(f"\n{'─'*95}")
    print("KV cache capacity thresholds (CF > 80 GB HBM → attention capacity-bound):")
    print(f"{'Context':<12s} {'KV/tok/layer':>14s} {'CF=80GB at C=':>16s}")
    print("-" * 45)
    for ISL, OSL, label in [(512,128,"512 tok"),(2048,512,"2K tok"),
                              (8192,1024,"8K tok"),(32768,2048,"32K tok"),
                              (131072,4096,"128K tok")]:
        L = ISL + OSL/2.0
        kv_l = 2.0 * HEAD_DIM * N_KV * BPKV * L / MODEL_TP / 1024
        kv_all = kv_l * N_LAYERS / 1024 / 1024  # GB
        if kv_all > 0:
            C_at_hbm = int((HBM_GB - wt_layer * N_LAYERS) / kv_all)
        else:
            C_at_hbm = 9999
        bound_str = f"C={C_at_hbm}" if C_at_hbm > 0 else "always"
        print(f"{label:<12s} {kv_l:>11.1f} KB {bound_str:>16s}")


def generate_latex_table(c1, c80, output_path):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-component operational intensity for a Llama-3.1-8B "
        r"decoder layer on H100 (bf16, TP=1). Ridge = " + f"{RIDGE_OI:.0f}" +
        r" FLOP/byte. C = compute-bound, BW = memory-bandwidth-bound. "
        r"Capacity-bound arises at 8K+ contexts when KV cache exceeds HBM.}",
        r"\label{tab:per-component-oi}",
        r"\small", r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Component & Wt (MB) & OI (C=1) & OI (C=80) & Bound (C=1) & Bound (C=80) \\",
        r"\midrule",
    ]
    for c1_item, c80_item in zip(c1["components"], c80["components"]):
        name = c1_item["name"]
        label = COMP_LABELS.get(name, name)
        wt_mb = COMP_WEIGHTS.get(name, 0) * BPP / 1e6
        wt_str = f"{wt_mb:.0f}" if wt_mb > 0 else r"\textemdash"
        b1 = "C" if c1_item["oi"] > RIDGE_OI else "BW"
        b80 = "C" if c80_item["oi"] > RIDGE_OI else "BW"
        lines.append(f"{label} & {wt_str} & {c1_item['oi']:.0f} & "
                     f"{c80_item['oi']:.0f} & {b1} & {b80} \\\\")
    wt_layer = sum(COMP_WEIGHTS.values()) * BPP / 1e6
    layer_b1 = "C" if c1["oi"] > RIDGE_OI else "BW"
    layer_b80 = "C" if c80["oi"] > RIDGE_OI else "BW"
    lines.extend([
        r"\midrule",
        r"\textbf{Layer total} & " +
        f"{wt_layer:.0f} & {c1['oi']:.0f} & {c80['oi']:.0f} & "
        f"{layer_b1} & {layer_b80} \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\vspace{3pt}",
        r"{\footnotesize KV capacity thresholds: 8K tok $\rightarrow$ C$\ge$57; "
        r"32K tok $\rightarrow$ C$\ge$14; 128K tok $\rightarrow$ C$\ge$3.}",
        r"\end{table}",
    ])
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-component OI/CF roofline")
    parser.add_argument("--c1-json", type=str, required=True)
    parser.add_argument("--c80-json", type=str, required=True)
    parser.add_argument("--output-pdf", type=str,
                        default="results/roofline/figures/per_layer_oi_cf.pdf")
    parser.add_argument("--output-latex", type=str, default=None)
    args = parser.parse_args()

    with open(args.c1_json) as f:
        c1 = json.load(f)
    with open(args.c80_json) as f:
        c80 = json.load(f)

    print_table(c1, c80)

    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    render_figure(c1, c80, out_pdf)

    if args.output_latex:
        generate_latex_table(c1, c80, Path(args.output_latex))

    return 0

if __name__ == "__main__":
    sys.exit(main())
