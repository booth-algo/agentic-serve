#!/usr/bin/env python3
"""Generate the single-turn roofline figure (OI + CF panels) for Llama-3.1-70B on H100."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_JSON = REPO_ROOT / "inference-benchmark" / "dashboard" / "public" / "data.json"
FIGURE_PDF = REPO_ROOT / "roofline_oi_cf_h100.pdf"
CSV_OUT = REPO_ROOT / "roofline_singleturn_data.csv"

P_PEAK = 989.0
BW = 3352.0
HBM_GB = 80.0

MODEL_M = 70e9
MODEL_D = 8192
MODEL_TP = 2
BPP = 2
BPKV = 2
N_KV = 8
N_H = 64
N_LAYERS = 80

B_SWEEP = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320]

WORKLOADS = [
    {"label": "70B/cl", "model": "Llama-3.1-70B", "profile": "chat-long",
     "hardware": "H100x2", "emphasis": True},
    {"label": "70B/cs", "model": "Llama-3.1-70B", "profile": "chat-short",
     "hardware": "H100x2", "emphasis": False},
    {"label": "70B/cm", "model": "Llama-3.1-70B", "profile": "chat-medium",
     "hardware": "H100x2", "emphasis": False},
]

C_GREEN = "#009E73"
C_RED = "#D55E00"
C_ARROW = "#555555"
C_ANNOT = "#808080"


def hw_roofline(OI):
    return min(P_PEAK, BW * OI / 1000.0)


def gemm_ai(M, d=MODEL_D):
    return M * d / (2.0 * M + d)


def kernel_ceiling_at_B(B, avg_ISL=162.0, avg_OSL=297.0, d=MODEL_D):
    # Harmonic blend of prefill (GEMM at M=ISL) and decode (GEMM at M=B)
    decode_ai = gemm_ai(max(B, 1), d)
    decode_peak = min(P_PEAK, BW * decode_ai / 1000.0)
    prefill_ai = gemm_ai(max(avg_ISL, 1), d)
    prefill_peak = min(P_PEAK, BW * prefill_ai / 1000.0)
    prefill_time = avg_ISL / prefill_peak
    decode_time = avg_OSL / decode_peak
    return (avg_ISL + avg_OSL) / (prefill_time + decode_time)


def oi_of_workload(B, m, d, ISL, OSL):
    L = ISL + OSL / 2.0
    num = 2.0 * m * d * L * B
    den = BPP * (m * d + d * L * B + m * L * B)
    return num / den


def cf_per_gpu(B, m, d, ISL, OSL, n_kv, n_h, TP):
    L = ISL + OSL / 2.0
    wt = (m * BPP) / TP / 1e9
    kv = (N_LAYERS * 2.0 * d * L * (n_kv / n_h) * BPKV) / TP / 1e9
    return wt + B * kv


def kv_bytes_per_request(d, ISL, OSL, n_kv, n_h, TP):
    L = ISL + OSL / 2.0
    return (N_LAYERS * 2.0 * d * L * (n_kv / n_h) * BPKV) / TP / 1e9


def cf_roofline_at(CF, weight_floor, kv_per_req, avg_ISL, avg_OSL):
    # CF -> B -> kernel_ceiling_at_B
    B = max((CF - weight_floor) / kv_per_req, 0.01)
    return kernel_ceiling_at_B(B, avg_ISL, avg_OSL)


def load_r2_records(path, model, profile, hardware):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        cfg = r.get("config") or {}
        if (r.get("modelShort") == model
                and cfg.get("profile") == profile
                and r.get("hardware") == hardware
                and cfg.get("backend") == "vllm"
                and r.get("quant", "BF16") == "BF16"
                and cfg.get("concurrency") in B_SWEEP):
            rows.append(r)
    rows.sort(key=lambda x: x["config"]["concurrency"])
    return rows


def build_workload_df(workload, records):
    out = []
    for rec in records:
        B = rec["config"]["concurrency"]
        s = rec["summary"]
        n_ok = s["successful_requests"]
        if n_ok == 0:
            continue
        avg_isl = s["total_input_tokens"] / n_ok
        avg_osl = s["total_output_tokens"] / n_ok
        ttt = s["total_token_throughput"]
        rps = s["request_throughput"]
        t_obs = 2.0 * MODEL_M * ttt / 1e12 / MODEL_TP
        oi = oi_of_workload(B, MODEL_M, MODEL_D, avg_isl, avg_osl)
        cf = cf_per_gpu(B, MODEL_M, MODEL_D, avg_isl, avg_osl,
                        N_KV, N_H, MODEL_TP)
        out.append({
            "workload_label": workload["label"],
            "model": workload["model"],
            "profile": workload["profile"],
            "hardware": workload["hardware"],
            "B": B,
            "OI": oi,
            "T_obs": t_obs,
            "CF_per_GPU_GB": cf,
            "R_obs": rps,
            "avg_ISL": avg_isl,
            "avg_OSL": avg_osl,
            "total_tok_s": ttt,
        })
    return pd.DataFrame(out)


def build_all_workloads(path):
    out = {}
    for w in WORKLOADS:
        recs = load_r2_records(path, w["model"], w["profile"], w["hardware"])
        out[w["label"]] = build_workload_df(w, recs)
    return out


def render_figure(workload_dfs, out_pdf):
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.titlesize": 11,
        "font.family": "serif",
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.subplots_adjust(wspace=0.35)

    PROFILE_STYLE = {
        "chat-long":   {"marker": "o", "s_emph": 50, "color": "#D55E00",
                        "edgelw": 0.5, "zorder": 7},
        "chat-short":  {"marker": "^", "s_emph": 30, "color": "#0072B2",
                        "edgelw": 0.3, "zorder": 6},
        "chat-medium": {"marker": "s", "s_emph": 30, "color": "#CC79A7",
                        "edgelw": 0.3, "zorder": 6},
    }
    B_MAX_ALPHA = 320.0

    df_anchor = workload_dfs["70B/cl"]
    anchor_row40 = df_anchor[df_anchor["B"] == 40]
    if not anchor_row40.empty:
        a_isl = float(anchor_row40.iloc[0]["avg_ISL"])
        a_osl = float(anchor_row40.iloc[0]["avg_OSL"])
        t_obs_40 = float(anchor_row40.iloc[0]["T_obs"])
    else:
        a_isl, a_osl, t_obs_40 = 162.0, 297.0, 100.0

    t_green_40 = kernel_ceiling_at_B(40, a_isl, a_osl)
    weight_per_gpu = MODEL_M * BPP / MODEL_TP / 1e9

    OI_MIN, OI_MAX = 1, 50000
    OI_grid = np.logspace(np.log10(OI_MIN), np.log10(OI_MAX), 800)

    T_hw = np.minimum(P_PEAK, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_hw, "-", color="black", lw=1.8, zorder=3)

    T_green = np.minimum(t_green_40, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_green, "-", color=C_GREEN, lw=1.5, zorder=4)

    for w in WORKLOADS:
        if w["emphasis"]:
            continue
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        p_isl = df["avg_ISL"].mean()
        p_osl = df["avg_OSL"].mean()
        if abs(p_isl - a_isl) < 10 and abs(p_osl - a_osl) < 30:
            continue
        t_green_profile = kernel_ceiling_at_B(40, p_isl, p_osl)
        T_green_2 = np.minimum(t_green_profile, BW * OI_grid / 1000.0)
        ax_a.plot(OI_grid, T_green_2, "--", color=C_GREEN, lw=1.2,
                  alpha=0.55, zorder=3.5)

    T_serve_a = np.minimum(t_obs_40, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_serve_a, "-", color=C_RED, lw=1.5, zorder=4.5)

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        sty = PROFILE_STYLE[w["profile"]]
        alphas = 0.4 + 0.6 * (df["B"].values / B_MAX_ALPHA)
        alphas = np.clip(alphas, 0.4, 1.0)
        ax_a.scatter(df["OI"], df["T_obs"],
                     marker=sty["marker"], s=sty["s_emph"],
                     c=sty["color"], edgecolors="black",
                     linewidth=sty["edgelw"], alpha=alphas,
                     zorder=sty["zorder"])

    for _, row in df_anchor.iterrows():
        b_val = int(row["B"])
        if b_val == 1:
            offset = (6, 4)
        elif b_val == 40:
            offset = (6, -10)
        else:
            continue
        ax_a.annotate(f"B={b_val}", (row["OI"], row["T_obs"]),
                      textcoords="offset points", xytext=offset,
                      fontsize=7, color="#555555")

    if not anchor_row40.empty:
        oi_40 = float(anchor_row40.iloc[0]["OI"])
        t_hw_40 = hw_roofline(oi_40)
        ratio_hw_green = t_hw_40 / t_green_40
        ratio_green_obs = t_green_40 / t_obs_40

        arr1 = FancyArrowPatch((oi_40, t_green_40), (oi_40, t_hw_40),
                               arrowstyle="<->", color=C_ARROW, lw=1.0,
                               mutation_scale=8, zorder=8)
        ax_a.add_patch(arr1)
        ax_a.text(oi_40 * 1.15, np.sqrt(t_hw_40 * t_green_40),
                  f"${ratio_hw_green:.1f}\\times$",
                  color=C_ARROW, fontsize=8.5, style="italic", va="center")

        arr2 = FancyArrowPatch((oi_40, t_obs_40), (oi_40, t_green_40),
                               arrowstyle="<->", color=C_ARROW, lw=1.0,
                               mutation_scale=8, zorder=8)
        ax_a.add_patch(arr2)
        ax_a.text(oi_40 * 0.85, np.sqrt(t_green_40 * t_obs_40),
                  f"${ratio_green_obs:.1f}\\times$",
                  color=C_ARROW, fontsize=8.5, style="italic",
                  va="center", ha="right")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(OI_MIN, OI_MAX)
    ax_a.set_ylim(0.5, 2000)
    ax_a.set_xlabel("Operational intensity [FLOP/byte]")
    ax_a.set_ylabel("Performance [TFLOP/s per GPU]")
    ax_a.grid(True, which="both", alpha=0.10, ls=":")
    ax_a.text(0.02, 0.94, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold")

    CF_MIN, CF_MAX = 68, 100
    kv_anchor = kv_bytes_per_request(MODEL_D, a_isl, a_osl, N_KV, N_H, MODEL_TP)

    ax_b.axhline(P_PEAK, color="black", lw=1.8, zorder=3)

    CF_grid = np.linspace(weight_per_gpu + 0.01, CF_MAX, 500)
    T_cf_roof = np.array([cf_roofline_at(cf, weight_per_gpu, kv_anchor,
                                          a_isl, a_osl) for cf in CF_grid])
    ax_b.plot(CF_grid, T_cf_roof, "-", color=C_GREEN, lw=1.5, zorder=4)

    for w in WORKLOADS:
        if w["emphasis"]:
            continue
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        p_isl = df["avg_ISL"].mean()
        p_osl = df["avg_OSL"].mean()
        kv_p = kv_bytes_per_request(MODEL_D, p_isl, p_osl, N_KV, N_H, MODEL_TP)
        if abs(kv_p - kv_anchor) / kv_anchor < 0.05:
            continue
        T_cf_2 = np.array([cf_roofline_at(cf, weight_per_gpu, kv_p,
                                           p_isl, p_osl) for cf in CF_grid])
        ax_b.plot(CF_grid, T_cf_2, "--", color=C_GREEN, lw=1.2,
                  alpha=0.55, zorder=3.5)

    ax_b.axhline(t_obs_40, color=C_RED, lw=1.5, zorder=5)

    ax_b.axvline(weight_per_gpu, color=C_ANNOT, lw=0.8, ls="--", zorder=2)
    ax_b.text(weight_per_gpu + 0.3, 1.5, "weights",
              rotation=90, color=C_ANNOT, fontsize=6.5, va="center")

    ax_b.axvline(HBM_GB, color=C_ANNOT, lw=0.8, ls="--", zorder=2)
    ax_b.axvspan(HBM_GB, CF_MAX, facecolor=C_ANNOT,
                 hatch="///", alpha=0.06, zorder=1)
    ax_b.text(HBM_GB + 0.3, 1.5, "80 GB HBM",
              rotation=90, color=C_ANNOT, fontsize=6.5, va="center")

    ax_b.annotate("", xy=(weight_per_gpu, 0.7), xytext=(HBM_GB, 0.7),
                  arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=0.8))
    ax_b.text((weight_per_gpu + HBM_GB) / 2, 0.58, "KV headroom",
              color=C_ANNOT, fontsize=6, ha="center", va="top")

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        sty = PROFILE_STYLE[w["profile"]]
        alphas = 0.4 + 0.6 * (df["B"].values / B_MAX_ALPHA)
        alphas = np.clip(alphas, 0.4, 1.0)
        ax_b.scatter(df["CF_per_GPU_GB"], df["T_obs"],
                     marker=sty["marker"], s=sty["s_emph"],
                     c=sty["color"], edgecolors="black",
                     linewidth=sty["edgelw"], alpha=alphas,
                     zorder=sty["zorder"])

    ax_b.annotate("B=1", xy=(70.05, 4.5),
                  textcoords="offset points", xytext=(-12, -8),
                  fontsize=6.5, color="#555555")

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        past_hbm = df[df["CF_per_GPU_GB"] > HBM_GB]
        if not past_hbm.empty:
            min_b = int(past_hbm["B"].min())
            ax_b.text(HBM_GB + 0.5, 45, f"B$\\geq${min_b}",
                      fontsize=6.5, color="#555555", va="center")
            break

    ax_b.set_yscale("log")
    ax_b.set_xlim(CF_MIN, CF_MAX)
    ax_b.set_ylim(0.5, 2000)
    ax_b.set_xlabel("Capacity footprint [GB per GPU]")
    ax_b.set_ylabel("Performance [TFLOP/s per GPU]")
    ax_b.grid(True, which="both", alpha=0.10, ls=":")
    ax_b.text(0.02, 0.94, "(b)", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold")

    legend_elems = [
        Line2D([0], [0], color="black", lw=1.8,
               label="H100 peak (989 TFLOP/s)"),
        Line2D([0], [0], color=C_GREEN, lw=1.5,
               label="GEMM ceiling"),
        Line2D([0], [0], color=C_GREEN, lw=1.2, ls="--", alpha=0.55,
               label="GEMM ceiling (chat-short)"),
        Line2D([0], [0], color=C_RED, lw=1.5,
               label="Serving ceiling"),
        Line2D([0], [0], marker="o", color="#D55E00", ls="",
               markersize=5, label="chat-long"),
        Line2D([0], [0], marker="^", color="#0072B2", ls="",
               markersize=5, label="chat-short"),
        Line2D([0], [0], marker="s", color="#CC79A7", ls="",
               markersize=5, label="chat-medium"),
    ]
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
               fontsize=7)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    print("Loading from:", DATA_JSON)
    workload_dfs = build_all_workloads(DATA_JSON)

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            print(f"  {w['label']:7s}  NO RECORDS")
            continue
        print(f"  {w['label']:7s}  N={len(df)}  "
              f"OI=[{df['OI'].min():.0f}, {df['OI'].max():.0f}]  "
              f"T_obs=[{df['T_obs'].min():.1f}, {df['T_obs'].max():.1f}] TFLOPS/GPU")

    all_rows = pd.concat(workload_dfs.values(), ignore_index=True)
    all_rows.to_csv(CSV_OUT, index=False, float_format="%.4f")
    print(f"CSV written to: {CSV_OUT}")

    df_cl = workload_dfs["70B/cl"]
    row40 = df_cl[df_cl["B"] == 40]
    if not row40.empty:
        oi40 = float(row40.iloc[0]["OI"])
        t_obs40 = float(row40.iloc[0]["T_obs"])
        t_hw40 = hw_roofline(oi40)
        a_isl40 = float(row40.iloc[0]["avg_ISL"])
        a_osl40 = float(row40.iloc[0]["avg_OSL"])
        t_green40 = kernel_ceiling_at_B(40, a_isl40, a_osl40)
        print(f"\nCascade at B=40 (per GPU):")
        print(f"  Hardware roofline:  {t_hw40:.1f} TFLOPS")
        print(f"  Kernel ceiling:     {t_green40:.1f} TFLOPS")
        print(f"  Observed serving:   {t_obs40:.1f} TFLOPS")
        print(f"  HW / kernel:        {t_hw40/t_green40:.1f}x")
        print(f"  Kernel / serving:   {t_green40/t_obs40:.1f}x")

    render_figure(workload_dfs, FIGURE_PDF)
    print(f"\nPDF written to: {FIGURE_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
