#!/usr/bin/env python3
"""Generate the multi-turn roofline figure (OI + CF panels) for Llama-3.1-70B on H100."""
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
DATA_JSON = REPO_ROOT / ".claude" / "r2_data" / "data.json"
FIGURE_PDF = REPO_ROOT / "figures" / "roofline_multiturn_h100.pdf"
CSV_OUT = REPO_ROOT / ".claude" / "figure_task" / "multiturn_roofline_data_v1.csv"

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

C_SWEEP = [1, 5, 10, 20, 40, 80]

WORKLOADS = [
    {"label": "70B/cmt-m", "model": "Llama-3.1-70B",
     "profile": "chat-multiturn-medium", "hardware": "H100x2",
     "tier": 2, "emphasis": True},
    {"label": "70B/cmt-s", "model": "Llama-3.1-70B",
     "profile": "chat-multiturn-short", "hardware": "H100x2",
     "tier": 2, "emphasis": False},
    {"label": "70B/tb-s", "model": "Llama-3.1-70B",
     "profile": "terminalbench-multiturn-short", "hardware": "H100x2",
     "tier": 1, "emphasis": False},
    {"label": "70B/osw-m", "model": "Llama-3.1-70B",
     "profile": "osworld-multiturn-medium", "hardware": "H100x2",
     "tier": 1, "emphasis": False},
]

C_GREEN = "#009E73"
C_RED = "#D55E00"
C_ARROW = "#555555"
C_ANNOT = "#808080"


def hw_roofline(OI):
    return min(P_PEAK, BW * OI / 1000.0)


def gemm_ai(M, d=MODEL_D):
    return M * d / (2.0 * M + d)


def kernel_ceiling_at_B(B, avg_ISL, avg_OSL, d=MODEL_D):
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
                and r.get("quant") == "BF16"
                and cfg.get("concurrency") in C_SWEEP):
            rows.append(r)
    rows.sort(key=lambda x: x["config"]["concurrency"])
    return rows


def build_workload_df(workload, records):
    out = []
    for rec in records:
        C = rec["config"]["concurrency"]
        s = rec["summary"]
        n_ok = s["successful_requests"]
        if n_ok == 0:
            continue
        avg_isl = s["total_input_tokens"] / n_ok
        avg_osl = s["total_output_tokens"] / n_ok
        ttt = s["total_token_throughput"]
        rps = s["request_throughput"]
        ttft = s["mean_ttft_ms"]
        t_obs = 2.0 * MODEL_M * ttt / 1e12 / MODEL_TP
        oi = oi_of_workload(C, MODEL_M, MODEL_D, avg_isl, avg_osl)
        cf = cf_per_gpu(C, MODEL_M, MODEL_D, avg_isl, avg_osl,
                        N_KV, N_H, MODEL_TP)
        out.append({
            "workload_label": workload["label"],
            "model": workload["model"],
            "profile": workload["profile"],
            "hardware": workload["hardware"],
            "tier": workload["tier"],
            "C": C,
            "OI": oi,
            "T_obs": t_obs,
            "CF_per_GPU_GB": cf,
            "R_obs": rps,
            "avg_ISL": avg_isl,
            "avg_OSL": avg_osl,
            "total_tok_s": ttt,
            "mean_ttft_ms": ttft,
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
        "chat-multiturn-medium":         {"marker": "o", "s": 50, "color": "#D55E00",
                                          "edgelw": 0.4, "zorder": 7},
        "chat-multiturn-short":          {"marker": "^", "s": 35, "color": "#0072B2",
                                          "edgelw": 0.4, "zorder": 6},
        "terminalbench-multiturn-short": {"marker": "D", "s": 45, "color": "#009E73",
                                          "edgelw": 1.0, "zorder": 8},
        "osworld-multiturn-medium":      {"marker": "s", "s": 45, "color": "#882255",
                                          "edgelw": 1.0, "zorder": 8},
    }
    C_MAX_ALPHA = 80.0

    df_anchor = workload_dfs["70B/cmt-m"]
    anchor_c40 = df_anchor[df_anchor["C"] == 40]
    if not anchor_c40.empty:
        a_isl = float(anchor_c40.iloc[0]["avg_ISL"])
        a_osl = float(anchor_c40.iloc[0]["avg_OSL"])
        t_obs_40 = float(anchor_c40.iloc[0]["T_obs"])
    else:
        a_isl, a_osl, t_obs_40 = 833.0, 244.0, 82.0

    t_green_chat = kernel_ceiling_at_B(40, a_isl, a_osl)

    df_tb = workload_dfs["70B/tb-s"]
    tb_c40 = df_tb[df_tb["C"] == 40]
    if not tb_c40.empty:
        tb_isl = float(tb_c40.iloc[0]["avg_ISL"])
        tb_osl = float(tb_c40.iloc[0]["avg_OSL"])
        t_obs_tb40 = float(tb_c40.iloc[0]["T_obs"])
    else:
        tb_isl, tb_osl, t_obs_tb40 = 2148.0, 33.0, 466.0

    weight_per_gpu = MODEL_M * BPP / MODEL_TP / 1e9

    OI_MIN, OI_MAX = 1, 50000
    OI_grid = np.logspace(np.log10(OI_MIN), np.log10(OI_MAX), 800)

    T_hw = np.minimum(P_PEAK, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_hw, "-", color="black", lw=1.8, zorder=3)

    T_green_chat_curve = np.minimum(t_green_chat, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_green_chat_curve, "-", color=C_GREEN, lw=1.5, zorder=4)

    T_serve = np.minimum(t_obs_40, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_serve, "-", color=C_RED, lw=1.5, zorder=4.5)

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        sty = PROFILE_STYLE[w["profile"]]
        alphas = 0.55 + 0.45 * (df["C"].values / C_MAX_ALPHA)
        alphas = np.clip(alphas, 0.55, 1.0)
        ax_a.scatter(df["OI"], df["T_obs"],
                     marker=sty["marker"], s=sty["s"],
                     c=sty["color"], edgecolors="black",
                     linewidth=sty["edgelw"], alpha=alphas,
                     zorder=sty["zorder"])

    for _, row in df_anchor.iterrows():
        c_val = int(row["C"])
        if c_val == 1:
            offset = (6, 4)
        elif c_val == 40:
            offset = (6, -10)
        else:
            continue
        ax_a.annotate(f"C={c_val}", (row["OI"], row["T_obs"]),
                      textcoords="offset points", xytext=offset,
                      fontsize=7, color="#555555")

    if not anchor_c40.empty:
        oi_40 = float(anchor_c40.iloc[0]["OI"])
        t_hw_40 = hw_roofline(oi_40)
        ratio_hw_green = t_hw_40 / t_green_chat
        ratio_green_obs = t_green_chat / t_obs_40

        arr1 = FancyArrowPatch((oi_40, t_green_chat), (oi_40, t_hw_40),
                               arrowstyle="<->", color=C_ARROW, lw=1.0,
                               mutation_scale=8, zorder=8)
        ax_a.add_patch(arr1)
        ax_a.text(oi_40 * 1.15, np.sqrt(t_hw_40 * t_green_chat),
                  f"${ratio_hw_green:.1f}\\times$",
                  color=C_ARROW, fontsize=8.5, style="italic", va="center")

        arr2 = FancyArrowPatch((oi_40, t_obs_40), (oi_40, t_green_chat),
                               arrowstyle="<->", color=C_ARROW, lw=1.0,
                               mutation_scale=8, zorder=8)
        ax_a.add_patch(arr2)
        ax_a.text(oi_40 * 0.85, np.sqrt(t_green_chat * t_obs_40),
                  f"${ratio_green_obs:.1f}\\times$",
                  color=C_ARROW, fontsize=8.5, style="italic",
                  va="center", ha="right")

    if not tb_c40.empty:
        oi_tb40 = float(tb_c40.iloc[0]["OI"])
        t_hw_tb = hw_roofline(oi_tb40)
        ratio_ag_total = t_hw_tb / t_obs_tb40

        ax_a.annotate(
            f"agentic: ${ratio_ag_total:.1f}\\times$ total",
            xy=(oi_tb40, t_obs_tb40),
            xytext=(-60, 22), textcoords="offset points",
            fontsize=7, color=C_ARROW, style="italic",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
            arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.6),
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(OI_MIN, OI_MAX)
    ax_a.set_ylim(0.5, 2000)
    ax_a.set_xlabel("Operational intensity [FLOP/byte]")
    ax_a.set_ylabel("Performance [TFLOP/s per GPU]")
    ax_a.grid(True, which="both", alpha=0.10, ls=":")
    ax_a.text(0.02, 0.94, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold")

    CF_MIN, CF_MAX = 68, 115
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
    ax_b.text(weight_per_gpu + 0.4, 1.5, "weights",
              rotation=90, color=C_ANNOT, fontsize=6.5, va="center")

    ax_b.axvline(HBM_GB, color=C_ANNOT, lw=0.8, ls="--", zorder=2)
    ax_b.axvspan(HBM_GB, CF_MAX, facecolor=C_ANNOT,
                 hatch="///", alpha=0.06, zorder=1)
    ax_b.text(HBM_GB + 0.4, 1.5, "80 GB HBM",
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
        alphas = 0.55 + 0.45 * (df["C"].values / C_MAX_ALPHA)
        alphas = np.clip(alphas, 0.55, 1.0)
        ax_b.scatter(df["CF_per_GPU_GB"], df["T_obs"],
                     marker=sty["marker"], s=sty["s"],
                     c=sty["color"], edgecolors="black",
                     linewidth=sty["edgelw"], alpha=alphas,
                     zorder=sty["zorder"])

    for w in WORKLOADS:
        if not w["emphasis"]:
            continue
        df = workload_dfs[w["label"]]
        c1 = df[df["C"] == 1]
        if not c1.empty:
            ax_b.annotate("C=1", xy=(float(c1.iloc[0]["CF_per_GPU_GB"]),
                                      float(c1.iloc[0]["T_obs"])),
                          textcoords="offset points", xytext=(6, -10),
                          fontsize=6.5, color="#555555")
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
               label="GEMM ceiling (chat-mt)"),
        Line2D([0], [0], color=C_GREEN, lw=1.2, ls="--", alpha=0.55,
               label="GEMM ceiling (other)"),
        Line2D([0], [0], color=C_RED, lw=1.5,
               label="Serving ceiling (chat-mt)"),
        Line2D([0], [0], marker="o", color="#D55E00", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="chat-mt-medium"),
        Line2D([0], [0], marker="^", color="#0072B2", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="chat-mt-short"),
        Line2D([0], [0], marker="D", color="#009E73", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=1.0,
               label="terminalbench-mt"),
        Line2D([0], [0], marker="s", color="#882255", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=1.0,
               label="osworld-mt"),
    ]
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
               fontsize=7)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading from:", DATA_JSON)
    workload_dfs = build_all_workloads(DATA_JSON)

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            print(f"  {w['label']:12s}  NO RECORDS")
            continue
        print(f"  {w['label']:12s}  N={len(df)}  "
              f"OI=[{df['OI'].min():.0f}, {df['OI'].max():.0f}]  "
              f"T_obs=[{df['T_obs'].min():.1f}, {df['T_obs'].max():.1f}] TFLOPS/GPU  "
              f"CF=[{df['CF_per_GPU_GB'].min():.1f}, {df['CF_per_GPU_GB'].max():.1f}] GB")

    all_rows = pd.concat(workload_dfs.values(), ignore_index=True)
    all_rows.to_csv(CSV_OUT, index=False, float_format="%.4f")
    print(f"CSV written to: {CSV_OUT}")

    df_anchor = workload_dfs["70B/cmt-m"]
    row40 = df_anchor[df_anchor["C"] == 40]
    if not row40.empty:
        oi40 = float(row40.iloc[0]["OI"])
        t_obs40 = float(row40.iloc[0]["T_obs"])
        t_hw40 = hw_roofline(oi40)
        a_isl40 = float(row40.iloc[0]["avg_ISL"])
        a_osl40 = float(row40.iloc[0]["avg_OSL"])
        t_green40 = kernel_ceiling_at_B(40, a_isl40, a_osl40)

        df_tb = workload_dfs["70B/tb-s"]
        tb40 = df_tb[df_tb["C"] == 40]

        print(f"\nCascade at C=40 (per GPU):")
        print(f"  Chat-multiturn-medium:")
        print(f"    Hardware roofline:  {t_hw40:.1f} TFLOPS")
        print(f"    GEMM ceiling:       {t_green40:.1f} TFLOPS")
        print(f"    Observed serving:   {t_obs40:.1f} TFLOPS")
        print(f"    HW / GEMM:          {t_hw40/t_green40:.1f}x")
        print(f"    GEMM / Serving:     {t_green40/t_obs40:.1f}x")

        if not tb40.empty:
            tb_isl = float(tb40.iloc[0]["avg_ISL"])
            tb_osl = float(tb40.iloc[0]["avg_OSL"])
            t_green_ag = kernel_ceiling_at_B(40, tb_isl, tb_osl)
            t_obs_tb = float(tb40.iloc[0]["T_obs"])
            print(f"  TerminalBench-multiturn-short:")
            print(f"    GEMM ceiling:       {t_green_ag:.1f} TFLOPS")
            print(f"    Observed serving:   {t_obs_tb:.1f} TFLOPS")
            print(f"    HW / GEMM:          {P_PEAK/t_green_ag:.1f}x")
            print(f"    GEMM / Serving:     {t_green_ag/t_obs_tb:.1f}x")

    render_figure(workload_dfs, FIGURE_PDF)
    print(f"\nPDF written to: {FIGURE_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
