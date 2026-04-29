#!/usr/bin/env python3
"""
OI + CF roofline figures for LLM serving on H100.

Panel (a) — Effective system OI roofline:
  x = effective_OI = total_FLOP/s / total_memory_bytes/s
  y = total observed TFLOP/s per GPU

  Memory traffic model (continuous batching at concurrency C):
    fwd_passes/s   = rps + (OSL * rps) / C
    weight_bytes/s  = fwd_passes/s * weight_bytes_per_gpu
    kv_bytes/s      = OSL * rps * kv_per_seq
    total_bytes/s   = weight_bytes/s + kv_bytes/s
    effective_OI    = T_obs * 1e12 / total_bytes/s

  All points sit below the hardware roofline by construction.
  Prefill-heavy workloads (high ISL, low OSL) achieve high effective OI
  because their forward passes are dominated by prefill, which processes
  many tokens per weight read. They cross the ridge into compute-bound.

Panel (b) — Capacity footprint:
  x = weight GB + KV cache GB per GPU
  y = total observed TFLOP/s per GPU

Generates:
  roofline_{singleturn,multiturn}_h100.{pdf,png,csv}
"""
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

REPO = Path(__file__).resolve().parent
DATA_JSON = REPO / "inference-benchmark" / "dashboard" / "public" / "data.json"

# ── H100 SXM5 80 GB ─────────────────────────────────────────────────
P_PEAK = 989.0   # BF16 TFLOP/s
BW     = 3352.0  # GB/s HBM bandwidth
HBM_GB = 80.0

# ── Llama-3.1-70B on H100×2 (TP=2) ─────────────────────────────────
M_PAR    = 70e9
D        = 8192
TP       = 2
BPP      = 2
BPKV     = 2
N_KV     = 8
N_H      = 64
N_LAYERS = 80
HEAD_DIM = D // N_H

W_BYTES  = M_PAR * BPP / TP              # weight bytes per GPU
W_GB     = W_BYTES / 1e9                  # 70 GB
RIDGE_AI = P_PEAK * 1000.0 / BW          # ~295 FLOP/byte

# ── workload configs ─────────────────────────────────────────────────
SINGLETURN = [
    dict(key="chat-st", label="chat-singleturn", profile="chat-singleturn", color="#D55E00", marker="o"),
]
MULTITURN = [
    dict(key="cmt-m", label="chat-mt-medium",  profile="chat-multiturn-medium",        color="#D55E00", marker="o"),
    dict(key="cmt-s", label="chat-mt-short",   profile="chat-multiturn-short",          color="#0072B2", marker="^"),
    dict(key="tb-s",  label="terminalbench-mt", profile="terminalbench-multiturn-short", color="#009E73", marker="D"),
    dict(key="osw-m", label="osworld-mt",       profile="osworld-multiturn-medium",      color="#882255", marker="s"),
]
ST_HW, MT_HW = "H100x2", "H100x2"
ST_C = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320]
MT_C = [1, 5, 10, 20, 40, 80]


# ── helpers ──────────────────────────────────────────────────────────

def hw_roof(oi):
    return min(P_PEAK, BW / 1000.0 * oi)


def kv_per_seq_bytes(isl, osl):
    L = isl + osl / 2.0
    return N_LAYERS * 2.0 * HEAD_DIM * N_KV * BPKV * L / TP


def effective_oi(T_obs_tflops, rps, avg_ISL, avg_OSL, C):
    out_tok_rate = avg_OSL * rps
    fwd_passes_s = rps + out_tok_rate / max(C, 1)
    kv_seq = kv_per_seq_bytes(avg_ISL, avg_OSL)
    total_bytes_s = fwd_passes_s * W_BYTES + out_tok_rate * kv_seq
    if total_bytes_s < 1:
        return 0.0
    return T_obs_tflops * 1e12 / total_bytes_s


def cf_gb(C, isl, osl):
    """Capacity footprint per GPU."""
    L = isl + osl / 2.0
    kv_per_gpu = N_LAYERS * 2.0 * HEAD_DIM * N_KV * BPKV * L / TP / 1e9
    return W_GB + C * kv_per_gpu


# ── data loading ─────────────────────────────────────────────────────

def load_json(path=DATA_JSON):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_df(data, profile, hardware, c_vals, model="Llama-3.1-70B"):
    rows = []
    for r in data:
        cfg = r.get("config") or {}
        if (r.get("modelShort") == model
                and cfg.get("profile") == profile
                and r.get("hardware") == hardware
                and r.get("quant", "BF16") == "BF16"
                and cfg.get("concurrency") in c_vals):
            s = r["summary"]
            n = s["successful_requests"]
            if n == 0:
                continue
            C = cfg["concurrency"]
            isl = s["total_input_tokens"] / n
            osl = s["total_output_tokens"] / n
            ttt = s["total_token_throughput"]
            rps = s["request_throughput"]
            T = 2.0 * M_PAR * ttt / 1e12 / TP

            rows.append(dict(
                C=C, avg_ISL=isl, avg_OSL=osl,
                eff_OI=effective_oi(T, rps, isl, osl, C),
                T_obs=T,
                CF_GB=cf_gb(C, isl, osl),
                tok_s=ttt, rps=rps,
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.loc[df.groupby("C")["T_obs"].idxmax()]
              .sort_values("C").reset_index(drop=True))


# ── rendering ────────────────────────────────────────────────────────

def render(workloads, dfs, out_pdf):
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.titlesize": 11,
        "font.family": "serif",
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.8))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel (a): Effective OI roofline ─────────────────────────────
    AI_LO, AI_HI = 0.8, 3000
    ai = np.logspace(np.log10(AI_LO), np.log10(AI_HI), 800)

    # hardware roofline
    ax_a.plot(ai, np.minimum(P_PEAK, BW / 1000.0 * ai),
              "-", color="black", lw=1.8, zorder=3)

    for w in workloads:
        df = dfs[w["key"]]
        if df.empty:
            continue

        ax_a.plot(df["eff_OI"], df["T_obs"],
                  "-", color=w["color"], lw=1.5, alpha=0.7, zorder=5)
        ax_a.scatter(df["eff_OI"], df["T_obs"],
                     marker=w["marker"], s=46, zorder=7,
                     c=w["color"], edgecolors="black", linewidth=0.4)

        # C labels on first and last
        for idx in [0, len(df) - 1]:
            row = df.iloc[idx]
            oy = -10 if idx == 0 else 6
            ax_a.annotate(f"C={int(row['C'])}", (row["eff_OI"], row["T_obs"]),
                          textcoords="offset points", xytext=(5, oy),
                          fontsize=6, color="#555555")

    # ridge-point marker
    ax_a.axvline(RIDGE_AI, color="#bbbbbb", lw=0.5, ls=":", zorder=2)
    ax_a.text(RIDGE_AI * 0.75, 0.55, f"ridge ({RIDGE_AI:.0f})",
              fontsize=5.5, color="#aaaaaa", va="bottom", ha="right")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(AI_LO, AI_HI)
    ax_a.set_ylim(0.5, 1500)
    ax_a.set_xlabel("Effective operational intensity [FLOP/byte]")
    ax_a.set_ylabel("Throughput [TFLOP/s per GPU]")
    ax_a.grid(True, which="both", alpha=0.10, ls=":")
    ax_a.text(0.02, 0.94, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold")

    # ── Panel (b): CF roofline ───────────────────────────────────────
    cf_max_data = max(
        (df["CF_GB"].max() for df in dfs.values() if not df.empty), default=85)
    hbm_total_gb = HBM_GB * TP
    CF_LO = W_GB - 3
    CF_HI = max(cf_max_data + 10, hbm_total_gb / TP + 20)

    ax_b.axhline(P_PEAK, color="black", lw=1.8, zorder=3)

    for w in workloads:
        df = dfs[w["key"]]
        if df.empty:
            continue

        ax_b.plot(df["CF_GB"], df["T_obs"],
                  "-", color=w["color"], lw=1.5, alpha=0.7, zorder=6)
        ax_b.scatter(df["CF_GB"], df["T_obs"],
                     marker=w["marker"], s=48, zorder=7,
                     c=w["color"], edgecolors="black", linewidth=0.4)

    ax_b.axvline(W_GB, color="#aaaaaa", lw=0.7, ls="--", zorder=2)
    ax_b.text(W_GB + 0.3, 0.7, "weights", rotation=90,
              color="#aaaaaa", fontsize=6, va="center")
    ax_b.axvline(HBM_GB, color="#aaaaaa", lw=0.7, ls="--", zorder=2)
    ax_b.axvspan(HBM_GB, CF_HI, facecolor="#aaaaaa", hatch="///", alpha=0.04, zorder=1)
    ax_b.text(HBM_GB + 0.3, 0.7, f"{HBM_GB:.0f} GB HBM",
              rotation=90, color="#aaaaaa", fontsize=6, va="center")
    # total system annotation
    ax_b.text(0.98, 0.02, f"Total: {TP}×{HBM_GB:.0f} = {HBM_GB*TP:.0f} GB",
              transform=ax_b.transAxes, fontsize=6, color="#aaaaaa",
              ha="right", va="bottom")

    ax_b.set_yscale("log")
    ax_b.set_xlim(CF_LO, CF_HI)
    ax_b.set_ylim(0.5, 1500)
    ax_b.set_xlabel("Capacity footprint [GB per GPU]")
    ax_b.set_ylabel("Throughput [TFLOP/s per GPU]")
    ax_b.grid(True, which="both", alpha=0.10, ls=":")
    ax_b.text(0.02, 0.94, "(b)", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold")

    # ── legend ───────────────────────────────────────────────────────
    legs = [Line2D([0], [0], color="black", lw=1.8,
                   label=f"H100 HW roofline ({P_PEAK:.0f} TFLOP/s)")]
    for w in workloads:
        if dfs[w["key"]].empty:
            continue
        legs.append(Line2D([0], [0], marker=w["marker"], color=w["color"],
                           ls="-", lw=1.4, alpha=0.75,
                           markersize=5, markeredgecolor="black",
                           markeredgewidth=0.3, label=w["label"]))
    fig.legend(handles=legs, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=min(len(legs), 5),
               frameon=False, fontsize=7)

    for ext in (".pdf", ".png"):
        p = out_pdf.with_suffix(ext)
        fig.savefig(p, dpi=300 if ext == ".pdf" else 150,
                    bbox_inches="tight", pad_inches=0.03)
        print(f"  -> {p}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

def run_set(label, workloads, hw, c_vals, data, out_stem):
    print(f"\n{label}:")
    dfs = {}
    for w in workloads:
        dfs[w["key"]] = build_df(data, w["profile"], hw, c_vals)
        df = dfs[w["key"]]
        if df.empty:
            print(f"  {w['label']:22s}  NO DATA")
        else:
            print(f"  {w['label']:22s}  N={len(df):2d}  "
                  f"OI=[{df['eff_OI'].min():.1f}, {df['eff_OI'].max():.1f}]  "
                  f"T=[{df['T_obs'].min():.1f}, {df['T_obs'].max():.1f}] TFLOP/s  "
                  f"CF=[{df['CF_GB'].min():.1f}, {df['CF_GB'].max():.1f}] GB")

    all_rows = pd.concat(
        [df.assign(workload=w["label"]) for w, df in zip(workloads, dfs.values())
         if not df.empty],
        ignore_index=True)
    csv_path = REPO / f"{out_stem}_data.csv"
    all_rows.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV -> {csv_path}")

    render(workloads, dfs, REPO / f"{out_stem}.pdf")
    return dfs


def main():
    data = load_json()
    run_set("Single-turn", SINGLETURN, ST_HW, ST_C, data, "roofline_singleturn_h100")
    run_set("Multi-turn", MULTITURN, MT_HW, MT_C, data, "roofline_multiturn_h100")


if __name__ == "__main__":
    sys.exit(main())
