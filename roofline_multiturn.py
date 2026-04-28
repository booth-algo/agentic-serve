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
C_ANNOT = "#808080"


def gemm_ai(M, d=MODEL_D):
    return M * d / (2.0 * M + d)


def kernel_ceiling_at_B(B, avg_ISL, avg_OSL, d=MODEL_D, cache_hit_rate=0.0):
    effective_ISL = avg_ISL * (1.0 - cache_hit_rate)
    decode_ai = gemm_ai(max(B, 1), d)
    decode_peak = min(P_PEAK, BW * decode_ai / 1000.0)
    prefill_ai = gemm_ai(max(effective_ISL, 1), d)
    prefill_peak = min(P_PEAK, BW * prefill_ai / 1000.0)
    prefill_time = effective_ISL / prefill_peak
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


def cf_roofline_at(CF, weight_floor, kv_per_req, avg_ISL, avg_OSL, cache_hit_rate=0.0):
    B = max((CF - weight_floor) / kv_per_req, 0.01)
    return kernel_ceiling_at_B(B, avg_ISL, avg_OSL, cache_hit_rate=cache_hit_rate)


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
            "CF_GB": cf,
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


def _workload_stats_at_saturation(workload_dfs, sat_C=40):
    stats = {}
    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        row = df[df["C"] == sat_C]
        if not row.empty:
            isl = float(row.iloc[0]["avg_ISL"])
            osl = float(row.iloc[0]["avg_OSL"])
            t_obs = float(row.iloc[0]["T_obs"])
        else:
            isl = df["avg_ISL"].mean()
            osl = df["avg_OSL"].mean()
            t_obs = df["T_obs"].max()
        stats[w["label"]] = {"ISL": isl, "OSL": osl, "T_obs": t_obs,
                             "profile": w["profile"], "emphasis": w["emphasis"],
                             "tier": w["tier"]}
    return stats


def render_figure(workload_dfs, out_pdf):
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.titlesize": 11,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    PROFILE_STYLE = {
        "chat-multiturn-medium":         {"marker": "o", "s": 55, "color": "#CC79A7",
                                          "edgelw": 0.4, "zorder": 7},
        "chat-multiturn-short":          {"marker": "^", "s": 45, "color": "#0072B2",
                                          "edgelw": 0.4, "zorder": 6},
        "terminalbench-multiturn-short": {"marker": "D", "s": 45, "color": "#56B4E9",
                                          "edgelw": 1.0, "zorder": 8},
        "osworld-multiturn-medium":      {"marker": "s", "s": 45, "color": "#882255",
                                          "edgelw": 1.0, "zorder": 8},
    }
    C_MAX_ALPHA = 80.0
    SAT_C = 40

    wl_stats = _workload_stats_at_saturation(workload_dfs, SAT_C)
    a_isl = wl_stats["70B/cmt-m"]["ISL"]
    a_osl = wl_stats["70B/cmt-m"]["OSL"]
    weight_per_gpu = MODEL_M * BPP / MODEL_TP / 1e9

    OI_MIN, OI_MAX = 100, 10000
    OI_grid = np.logspace(np.log10(OI_MIN), np.log10(OI_MAX), 800)

    T_hw = np.minimum(P_PEAK, BW * OI_grid / 1000.0)
    ax_a.plot(OI_grid, T_hw, "-", color="black", lw=1.8, zorder=3)

    chat_dfs_a = [workload_dfs[w["label"]] for w in WORKLOADS if w["tier"] == 2]
    chat_all_a = pd.concat(chat_dfs_a, ignore_index=True)
    log_oi = np.log10(chat_all_a["OI"].values)
    t_obs_chat = chat_all_a["T_obs"].values
    A_mat = np.vstack([log_oi, np.ones(len(log_oi))]).T
    serv_coeff = np.linalg.lstsq(A_mat, t_obs_chat, rcond=None)[0]
    oi_zero = 10 ** (-serv_coeff[1] / serv_coeff[0])
    OI_serv = np.logspace(np.log10(max(OI_MIN, oi_zero * 1.05)),
                          np.log10(OI_MAX), 500)
    T_serv = serv_coeff[0] * np.log10(OI_serv) + serv_coeff[1]
    ax_a.plot(OI_serv[T_serv > 0], T_serv[T_serv > 0], "-", color=C_GREEN,
              lw=1.5, zorder=4)

    df_anchor = workload_dfs["70B/cmt-m"]
    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        sty = PROFILE_STYLE[w["profile"]]
        alphas = np.clip(0.55 + 0.45 * (df["C"].values / C_MAX_ALPHA), 0.55, 1.0)
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

    df_tb = workload_dfs["70B/tb-s"]
    tb_c1 = df_tb[df_tb["C"] == 1]
    if not tb_c1.empty:
        ax_a.annotate("C=1", (float(tb_c1.iloc[0]["OI"]), float(tb_c1.iloc[0]["T_obs"])),
                      textcoords="offset points", xytext=(6, -8),
                      fontsize=6.5, color="#555555")

    bbox_kw = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5)
    OI_at_sat = oi_of_workload(SAT_C, MODEL_M, MODEL_D, a_isl, a_osl)
    T_serv_sat = serv_coeff[0] * np.log10(OI_at_sat) + serv_coeff[1]
    ax_a.text(OI_at_sat * 0.35, T_serv_sat * 1.15,
              f"serving ceiling\n({T_serv_sat:.0f} TFLOP/s @ C*={SAT_C})",
              fontsize=6.5, color=C_GREEN, ha="center", va="bottom", bbox=bbox_kw)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(OI_MIN, OI_MAX)
    ax_a.set_ylim(0.5, 2000)
    ax_a.set_xlabel("Operational intensity (OI) [FLOP/byte]")
    ax_a.set_ylabel("Performance [TFLOP/s per GPU]")
    ax_a.grid(True, which="both", alpha=0.20, ls=":")
    ax_a.text(0.02, 0.94, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold")
    ax_a.text(200, 1.0, "bandwidth\nbound",
              fontsize=6, color="#888888", ha="center", va="bottom", style="italic")
    ax_a.text(5000, 1.0, "compute\nbound",
              fontsize=6, color="#888888", ha="center", va="bottom", style="italic")

    CF_MIN, CF_MAX = 68, 115

    ax_b.axhline(P_PEAK, color="black", lw=1.8, zorder=3)

    kv_anchor = kv_bytes_per_request(MODEL_D, a_isl, a_osl, N_KV, N_H, MODEL_TP)
    CF_grid = np.linspace(weight_per_gpu + 0.01, CF_MAX, 500)
    T_cf = np.array([cf_roofline_at(cf, weight_per_gpu, kv_anchor,
                                     a_isl, a_osl, cache_hit_rate=0.0)
                     for cf in CF_grid])
    ax_b.plot(CF_grid, T_cf, "-", color=C_GREEN, lw=1.5, zorder=4)

    ax_b.axvline(weight_per_gpu, color=C_ANNOT, lw=0.8, ls="--", zorder=2)
    ax_b.text(weight_per_gpu + 0.4, 2.0, "weights",
              rotation=90, color=C_ANNOT, fontsize=7, va="center")

    ax_b.axvline(HBM_GB, color=C_ANNOT, lw=0.8, ls="--", zorder=2)
    ax_b.axvspan(HBM_GB, CF_MAX, facecolor=C_ANNOT,
                 hatch="///", alpha=0.06, zorder=1)
    ax_b.text(HBM_GB + 0.4, 2.0, "80 GB HBM",
              rotation=90, color=C_ANNOT, fontsize=7, va="center")

    ax_b.annotate("", xy=(weight_per_gpu, 1.2), xytext=(HBM_GB, 1.2),
                  arrowprops=dict(arrowstyle="<->", color=C_ANNOT, lw=0.8))
    ax_b.text((weight_per_gpu + HBM_GB) / 2, 0.95, "KV headroom",
              color=C_ANNOT, fontsize=6.5, ha="center", va="top")

    for w in WORKLOADS:
        df = workload_dfs[w["label"]]
        if df.empty:
            continue
        sty = PROFILE_STYLE[w["profile"]]
        alphas = np.clip(0.55 + 0.45 * (df["C"].values / C_MAX_ALPHA), 0.55, 1.0)
        ax_b.scatter(df["CF_GB"], df["T_obs"],
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
            ax_b.annotate("C=1", xy=(float(c1.iloc[0]["CF_GB"]),
                                      float(c1.iloc[0]["T_obs"])),
                          textcoords="offset points", xytext=(6, -10),
                          fontsize=6.5, color="#555555")
            break

    ax_b.set_xlim(CF_MIN, CF_MAX)
    ax_b.set_xlabel("Capacity footprint (CF) [GB]")
    ax_b.grid(True, which="both", alpha=0.20, ls=":")
    ax_b.text(0.98, 0.02, "ceilings at C*=40",
              transform=ax_b.transAxes, fontsize=5.5, color="#888888",
              ha="right", va="bottom", style="italic")
    ax_b.text(0.02, 0.94, "(b)", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold")

    legend_elems = [
        Line2D([0], [0], color="black", lw=1.8,
               label="H100 peak (989 TFLOP/s)"),
        Line2D([0], [0], color=C_GREEN, lw=1.5,
               label="serving ceiling"),
        Line2D([0], [0], marker="o", color="#CC79A7", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="chat-mt-medium"),
        Line2D([0], [0], marker="^", color="#0072B2", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="chat-mt-short"),
        Line2D([0], [0], marker="D", color="#56B4E9", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=1.0,
               label="terminalbench-mt"),
        Line2D([0], [0], marker="s", color="#882255", ls="",
               markersize=5, markeredgecolor="black", markeredgewidth=1.0,
               label="osworld-mt"),
    ]
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
               fontsize=7)

    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.05)
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
              f"CF=[{df['CF_GB'].min():.1f}, {df['CF_GB'].max():.1f}] GB")

    all_rows = pd.concat(workload_dfs.values(), ignore_index=True)
    all_rows.to_csv(CSV_OUT, index=False, float_format="%.4f")
    print(f"CSV written to: {CSV_OUT}")

    SAT_C = 40
    wl_stats = _workload_stats_at_saturation(workload_dfs, SAT_C)

    print(f"\nGEMM ceilings at C={SAT_C} (per GPU):")
    print(f"  {'Workload':30s}  {'ISL':>6s}  {'OSL':>6s}  {'T_obs':>8s}  "
          f"{'Ceil 0%':>8s}  {'Ceil 85%':>9s}  {'Overhead':>8s}")
    for w in WORKLOADS:
        if w["label"] not in wl_stats:
            continue
        ws = wl_stats[w["label"]]
        c0 = kernel_ceiling_at_B(SAT_C, ws["ISL"], ws["OSL"], cache_hit_rate=0.0)
        c85 = kernel_ceiling_at_B(SAT_C, ws["ISL"], ws["OSL"], cache_hit_rate=0.85)
        overhead = c0 / ws["T_obs"]
        print(f"  {w['profile']:30s}  {ws['ISL']:6.0f}  {ws['OSL']:6.0f}  "
              f"{ws['T_obs']:8.1f}  {c0:8.1f}  {c85:9.1f}  {overhead:7.1f}x")

    t_ceil = kernel_ceiling_at_B(SAT_C, wl_stats["70B/cmt-m"]["ISL"],
                                  wl_stats["70B/cmt-m"]["OSL"], cache_hit_rate=0.0)
    t_obs = wl_stats["70B/cmt-m"]["T_obs"]
    print(f"\nServing overhead @ C*={SAT_C}:")
    print(f"  GEMM ceiling: {t_ceil:.1f} TFLOP/s")
    print(f"  Observed:     {t_obs:.1f} TFLOP/s")
    print(f"  Overhead:     {t_ceil / t_obs:.1f}x")

    render_figure(workload_dfs, FIGURE_PDF)
    print(f"\nPDF written to: {FIGURE_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
