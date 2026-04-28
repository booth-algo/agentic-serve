"""Paper-facing validation — per-GPU MAPE + aggregate layer error.

Three comparison modes:

1. **ncu-self-consistency** (default):
   `composer.predict_ttft_ms(model, seq=128)` vs sum-of-kernel-times from
   `kernels_labeled.csv` for every `(gpu, model)`. Proves the kernel
   predictor reproduces the ncu numbers it was trained on.

2. **microbench_ttft** (`--mode microbench_ttft` or legacy `--vs-measured`):
   `composer.predict_ttft_ms(model, seq=avg_input_tokens, bs=concurrency)`
   vs `summary.median_ttft_ms` from the inference-benchmark `data.json`
   dump. Adds a third column that captures launch overhead + stream
   gaps + CPU dispatch — everything the ncu-self-consistency view
   misses.

3. **serving_e2e** (`--mode serving_e2e --profile <name>`):
   `serving_e2e.predict_serving_e2e(model, isl, osl, bs)` vs
   `summary.{median_ttft_ms, median_tpot_ms, median_e2el_ms}` from
   `data.json`, filtered to the specified workload profile (e.g.
   chat-short, chat-medium, chat-long). Validates the full ISL/OSL →
   TTFT + TPOT + E2EL prediction pipeline.

The two-column "overhead %" on the microbench_ttft report is
`(wall_clock_ms - kernel_sum_ms) / wall_clock_ms * 100` and represents
the fraction of real TTFT we don't model with kernels.

CLI
---
    python -m llm_predict.training.per_kernel.validate --gpu A100
    python -m llm_predict.training.per_kernel.validate --vs-measured
    python -m llm_predict.training.per_kernel.validate --vs-measured \\
        --target-seq 128 --seq-tolerance 16
    python -m llm_predict.training.per_kernel.validate \\
        --mode serving_e2e --profile chat-medium --gpu A100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

from . import composer, model_specs, serving_e2e
from llm_predict.training.per_kernel.ensure_data import ensure_kernels_csv


# Inverse of _DIR_TO_SHORT — CSV `model` column (short) → dir_name for ModelConfig.
_SHORT_TO_DIR = {
    "Llama-8B":       "Llama-3.1-8B-Instruct",
    "Llama-70B":      "Llama-3.1-70B-Instruct",
    "Llama-3.3-70B":  "Llama-3.3-70B-Instruct",
    "Qwen-72B":       "Qwen2.5-72B-Instruct",
    "Qwen3.5-9B":     "Qwen3.5-9B",
    "Qwen3.5-27B":    "Qwen3.5-27B",
    "gpt-oss-20b":    "gpt-oss-20b",
    "Mixtral-8x7B":   "Mixtral-8x7B-Instruct",
}


# ── data.json label mapping ─────────────────────────────────────────────────
# `hardware` in data.json uses slightly different labels than the predictor.
# TP>1 (e.g. "A100-40GBx2", "H100x4") is out-of-scope for TP=1 paper runs.
_HARDWARE_TO_GPU = {
    "2080Ti":    "RTX2080Ti",
    "3090":      "RTX3090",
    "A100-40GB": "A100",
    "H100":      "H100",
}


# `modelShort` in data.json → dir_name for ModelConfig. Models not in this
# map (e.g. gpt-oss-120b) are outside our predictor coverage and skipped.
_MODELSHORT_TO_DIR = {
    "Llama-3.1-8B":   "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B":  "Llama-3.1-70B-Instruct",
    "Llama-3.3-70B":  "Llama-3.3-70B-Instruct",
    "Qwen2.5-72B":    "Qwen2.5-72B-Instruct",
    "Qwen3.5-9B":     "Qwen3.5-9B",
    "Qwen3.5-27B":    "Qwen3.5-27B",
    "gpt-oss-20b":    "gpt-oss-20b",
    "Mixtral-8x7B":   "Mixtral-8x7B-Instruct",
}


# ── Composer scope ──────────────────────────────────────────────────────────
_HYBRID_ATTN_MODELS: set[str] = {"Qwen3.5-9B", "Qwen3.5-27B"}


def _arch_class(short: str, cfg) -> str:
    """Return 'supported' | 'moe' | 'hybrid_attn' for scope-aware MAPE."""
    if short in _HYBRID_ATTN_MODELS:
        return "hybrid_attn"
    if getattr(cfg, "is_moe", False):
        return "moe"
    return "supported"


def validate_gpu(df: pd.DataFrame, gpu: str, report_path: Path, seq: int = 128) -> None:
    pred = PerKernelPredictor(gpu=gpu)
    if not pred.load():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"# {gpu} validation\n\nNo pkls loaded; run trainer first.\n")
        print(f"[{gpu}] no pkls — wrote placeholder report")
        return

    df_gpu = df[df["gpu"] == gpu].copy()
    measured_per_model = df_gpu.groupby("model")["gpu_time_duration_ms"].sum()

    rows: list[tuple[str, str, float, float, float, int]] = []
    for short, dir_name in _SHORT_TO_DIR.items():
        if short not in measured_per_model.index:
            continue
        cfg = model_specs.get_model_config(dir_name,
                                            held_out=model_specs.is_held_out(dir_name, gpu))
        if cfg is None:
            continue
        pred_ms = composer.predict_ttft_ms(pred, cfg, seq=seq)
        meas_ms = float(measured_per_model[short])
        err_pct = abs(pred_ms - meas_ms) / max(meas_ms, 1e-9) * 100.0
        n_ker = int((df_gpu["model"] == short).sum())
        rows.append((short, _arch_class(short, cfg), pred_ms, meas_ms, err_pct, n_ker))

    lines: list[str] = [
        f"# {gpu} — Per-Kernel Composition Validation",
        "",
        f"- Predictor: {gpu} pkls ({sorted(pred.families_loaded())})",
        f"- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv",
        f"- Input: bs=1, seq={seq}, tp=1",
        "- Headline MAPE = supported architectures only (dense, non-hybrid-attn).",
        "  MoE and hybrid-attn rows are known composer gaps and listed separately.",
        "",
        "## Per-model",
        "",
        "| Model | arch | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    supported_pred = 0.0
    supported_meas = 0.0
    supported_errs: list[float] = []
    oos_counts = {"moe": 0, "hybrid_attn": 0}
    for short, arch, pred_ms, meas_ms, err_pct, n in rows:
        marker = " _(held-out)_" if model_specs.is_held_out(_SHORT_TO_DIR[short], gpu) else ""
        lines.append(
            f"| {short}{marker} | {arch} | {pred_ms:.2f} | {meas_ms:.2f} | {err_pct:.2f}% | {n} |"
        )
        if arch == "supported":
            supported_pred += pred_ms
            supported_meas += meas_ms
            supported_errs.append(err_pct)
        else:
            oos_counts[arch] += 1
    total_err = abs(supported_pred - supported_meas) / max(supported_meas, 1e-9) * 100.0
    mape_supported = (sum(supported_errs) / len(supported_errs)) if supported_errs else 0.0
    lines.append(
        f"| **supported aggregate** ({len(supported_errs)} rows) | | "
        f"**{supported_pred:.2f}** | **{supported_meas:.2f}** | "
        f"**Σ-err {total_err:.2f}% · MAPE {mape_supported:.2f}%** | |"
    )
    if any(oos_counts.values()):
        oos_desc = ", ".join(f"{v} {k}" for k, v in oos_counts.items() if v)
        lines.append(f"| _out-of-scope_ | | | | _{oos_desc} — excluded from headline_ | |")
    lines.append("")

    per_fam = (df_gpu.groupby(["model", "kernel_family"])["gpu_time_duration_ms"]
               .sum().unstack(fill_value=0.0))
    if not per_fam.empty:
        lines.append("## Measured time breakdown by family (ms)")
        lines.append("")
        lines.append(per_fam.round(3).to_markdown())
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(
        f"[{gpu}] supported MAPE {mape_supported:.2f}% "
        f"(Σ-err {total_err:.2f}%; {len(supported_errs)}/{len(rows)} models in scope) "
        f"— wrote {report_path}"
    )


# ── microbench_ttft (wall-clock vs measured TTFT) validation ─────────────────

def _load_measured_rows(data_json_path: Path,
                         gpu: str,
                         concurrency: int = 1,
                         target_seq: int | None = None,
                         seq_tolerance: int = 16,
                         profile_filter: str | None = None) -> list[dict]:
    """Return measured rows from data.json matching a given GPU/concurrency.

    If `target_seq` is given, rows are restricted to profiles whose
    per-request `avg_seq` lies within ±`seq_tolerance` tokens of it.

    If `profile_filter` is given, only rows matching that profile name
    are returned (used by serving_e2e mode).

    Each returned dict has:
        modelShort, backend, profile, concurrency,
        avg_isl, avg_osl, measured_ttft_ms, median_tpot_ms,
        median_e2el_ms, successful_requests
    """
    with open(data_json_path) as f:
        raw = json.load(f)

    wanted_hardware = {hw for hw, g in _HARDWARE_TO_GPU.items() if g == gpu}
    rows: list[dict] = []
    for r in raw:
        hw = r.get("hardware", "")
        if hw not in wanted_hardware:
            continue
        cfg = r.get("config", {}) or {}
        summ = r.get("summary", {}) or {}
        if int(cfg.get("concurrency", 0)) != concurrency:
            continue
        ms = r.get("modelShort", "")
        if ms not in _MODELSHORT_TO_DIR:
            continue
        if profile_filter is not None and cfg.get("profile", "") != profile_filter:
            continue
        succ = int(summ.get("successful_requests", 0) or 0)
        if succ <= 0:
            continue
        total_in = float(summ.get("total_input_tokens", 0) or 0)
        total_out = float(summ.get("total_output_tokens", 0) or 0)
        median_ttft = summ.get("median_ttft_ms")
        if median_ttft is None or total_in <= 0:
            continue
        avg_isl = total_in / succ
        avg_osl = total_out / succ if total_out > 0 else 0.0
        if target_seq is not None and abs(avg_isl - target_seq) > seq_tolerance:
            continue
        rows.append({
            "modelShort": ms,
            "backend": cfg.get("backend", ""),
            "profile": cfg.get("profile", ""),
            "concurrency": concurrency,
            "avg_isl": avg_isl,
            "avg_osl": avg_osl,
            # Backwards compat alias
            "avg_seq": avg_isl,
            "measured_ttft_ms": float(median_ttft),
            "median_tpot_ms": (
                float(summ["median_tpot_ms"]) if summ.get("median_tpot_ms") is not None else None
            ),
            "median_e2el_ms": (
                float(summ["median_e2el_ms"]) if summ.get("median_e2el_ms") is not None else None
            ),
            "successful_requests": succ,
        })
    return rows


def validate_vs_measured_gpu(df: pd.DataFrame,
                              gpu: str,
                              data_json_path: Path,
                              report_path: Path,
                              concurrency: int = 1,
                              target_seq: int | None = None,
                              seq_tolerance: int = 16) -> None:
    pred = PerKernelPredictor(gpu=gpu)
    if not pred.load():
        print(f"[{gpu}][microbench_ttft] no pkls — skipping")
        return

    measured_rows = _load_measured_rows(
        data_json_path, gpu, concurrency=concurrency,
        target_seq=target_seq, seq_tolerance=seq_tolerance,
    )
    if not measured_rows:
        print(f"[{gpu}][microbench_ttft] no matching rows in data.json — skipping")
        return

    df_gpu = df[df["gpu"] == gpu].copy()
    kernel_sum_per_model = df_gpu.groupby("model")["gpu_time_duration_ms"].sum().to_dict()

    out_rows: list[tuple] = []
    for row in measured_rows:
        dir_name = _MODELSHORT_TO_DIR[row["modelShort"]]
        cfg = model_specs.get_model_config(
            dir_name, held_out=model_specs.is_held_out(dir_name, gpu))
        if cfg is None:
            continue
        seq = max(1, int(round(row["avg_isl"])))
        pred_ms = composer.predict_ttft_ms(pred, cfg, seq=seq, bs=row["concurrency"])
        meas_ms = row["measured_ttft_ms"]
        err_pct = abs(pred_ms - meas_ms) / max(meas_ms, 1e-9) * 100.0

        short = next((s for s, d in _SHORT_TO_DIR.items() if d == dir_name), row["modelShort"])
        arch = _arch_class(short, cfg)
        ncu_sum_ms = kernel_sum_per_model.get(short)
        overhead_pct = (
            (meas_ms - ncu_sum_ms) / max(meas_ms, 1e-9) * 100.0
            if ncu_sum_ms is not None else None
        )
        out_rows.append((
            short, arch, row["backend"], row["profile"], seq, row["concurrency"],
            pred_ms, meas_ms, err_pct, ncu_sum_ms, overhead_pct,
            row.get("median_tpot_ms"),
        ))

    seq_filter_note = (
        f", avg_seq within ±{seq_tolerance} of {target_seq}"
        if target_seq is not None else ""
    )
    lines: list[str] = [
        f"# {gpu} — microbench_ttft Validation (vs inference-benchmark data.json)",
        "",
        f"- Predictor track: **microbench_ttft** (prefill-only TTFT)",
        f"- Predictor: {gpu} pkls ({sorted(pred.families_loaded())})",
        f"- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)",
        f"- Filter: concurrency={concurrency}, TP=1{seq_filter_note}",
        f"- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.",
        "- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.",
        "",
        "## Per-row",
        "",
        "| Model | arch | backend | profile | avg seq | bs | predicted TTFT (ms) | measured TTFT p50 (ms) | abs err % | ncu Σ (ms) | overhead % | median TPOT (ms) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    supported_errs: list[float] = []
    supported_overheads: list[float] = []
    oos_counts = {"moe": 0, "hybrid_attn": 0}
    for (short, arch, backend, profile, seq, bs, pred_ms, meas_ms, err_pct,
         ncu_sum_ms, overhead_pct, tpot_ms) in out_rows:
        marker = " _(held-out)_" if model_specs.is_held_out(_SHORT_TO_DIR.get(short, ""), gpu) else ""
        ncu_cell = f"{ncu_sum_ms:.2f}" if ncu_sum_ms is not None else "\u2014"
        ov_cell = f"{overhead_pct:.1f}%" if overhead_pct is not None else "\u2014"
        tpot_cell = f"{tpot_ms:.2f}" if tpot_ms is not None else "\u2014"
        lines.append(
            f"| {short}{marker} | {arch} | {backend} | {profile} | {seq} | {bs} | "
            f"{pred_ms:.2f} | {meas_ms:.2f} | {err_pct:.2f}% | {ncu_cell} | {ov_cell} | {tpot_cell} |"
        )
        if arch == "supported":
            supported_errs.append(err_pct)
            if overhead_pct is not None:
                supported_overheads.append(overhead_pct)
        else:
            oos_counts[arch] += 1

    if supported_errs:
        mape = sum(supported_errs) / len(supported_errs)
        lines.append(
            f"| **supported MAPE** ({len(supported_errs)} rows) | | | | | | | | "
            f"**{mape:.2f}%** | | | |"
        )
    if any(oos_counts.values()):
        oos_desc = ", ".join(f"{v} {k}" for k, v in oos_counts.items() if v)
        lines.append(f"| _out-of-scope_ | | | | | | | | _{oos_desc} — excluded from headline_ | | | |")
    if supported_overheads:
        mean_ov = sum(supported_overheads) / len(supported_overheads)
        lines.append("")
        lines.append(
            f"**Mean overhead across {len(supported_overheads)} supported rows with ncu data:** "
            f"{mean_ov:.1f}%"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    n = len(out_rows)
    mape_s = f"{sum(supported_errs)/len(supported_errs):.2f}%" if supported_errs else "n/a"
    print(
        f"[{gpu}][microbench_ttft] supported MAPE {mape_s} "
        f"({len(supported_errs)}/{n} rows in scope) — wrote {report_path}"
    )


# ── serving_e2e (ISL/OSL → TTFT + TPOT + E2EL) validation ───────────────────

def validate_serving_e2e_gpu(gpu: str,
                              data_json_path: Path,
                              report_path: Path,
                              profile_name: str,
                              concurrency: int = 1) -> None:
    """Compare serving_e2e predictions vs real bench results for a given profile."""
    pred = PerKernelPredictor(gpu=gpu)
    if not pred.load():
        print(f"[{gpu}][serving_e2e] no pkls — skipping")
        return

    measured_rows = _load_measured_rows(
        data_json_path, gpu, concurrency=concurrency,
        profile_filter=profile_name,
    )
    if not measured_rows:
        print(f"[{gpu}][serving_e2e] no rows for profile={profile_name} conc={concurrency} — skipping")
        return

    out_rows: list[dict] = []
    for row in measured_rows:
        dir_name = _MODELSHORT_TO_DIR[row["modelShort"]]
        cfg = model_specs.get_model_config(
            dir_name, held_out=model_specs.is_held_out(dir_name, gpu))
        if cfg is None:
            continue
        isl = max(1, int(round(row["avg_isl"])))
        osl = max(0, int(round(row["avg_osl"])))
        bs = row["concurrency"]

        short = next((s for s, d in _SHORT_TO_DIR.items() if d == dir_name), row["modelShort"])
        arch = _arch_class(short, cfg)

        result = serving_e2e.predict_serving_e2e(pred, cfg, isl=isl, osl=osl, bs=bs)

        meas_ttft = row["measured_ttft_ms"]
        meas_tpot = row.get("median_tpot_ms")
        meas_e2el = row.get("median_e2el_ms")

        ttft_err = abs(result["ttft_ms"] - meas_ttft) / max(meas_ttft, 1e-9) * 100.0
        tpot_err = (abs(result["tpot_ms"] - meas_tpot) / max(meas_tpot, 1e-9) * 100.0
                    if meas_tpot is not None and meas_tpot > 0 else None)
        e2el_err = (abs(result["e2el_ms"] - meas_e2el) / max(meas_e2el, 1e-9) * 100.0
                    if meas_e2el is not None and meas_e2el > 0 else None)

        out_rows.append({
            "short": short, "arch": arch, "backend": row["backend"],
            "isl": isl, "osl": osl, "bs": bs,
            "pred_ttft": result["ttft_ms"],
            "pred_tpot": result["tpot_ms"],
            "pred_e2el": result["e2el_ms"],
            "pred_decode": result["decode_ms"],
            "meas_ttft": meas_ttft,
            "meas_tpot": meas_tpot,
            "meas_e2el": meas_e2el,
            "ttft_err": ttft_err,
            "tpot_err": tpot_err,
            "e2el_err": e2el_err,
            "held_out": model_specs.is_held_out(dir_name, gpu),
        })

    lines: list[str] = [
        f"# {gpu} — serving_e2e Validation: {profile_name}",
        "",
        f"- Predictor track: **serving_e2e** (ISL/OSL → TTFT + TPOT + E2EL)",
        f"- Predictor: {gpu} pkls ({sorted(pred.families_loaded())})",
        f"- Profile: `{profile_name}` (concurrency={concurrency})",
        f"- Ground truth: `summary.{{median_ttft_ms, median_tpot_ms, median_e2el_ms}}`",
        "- Headline MAPE = supported architectures only.",
        "",
        "## Per-row",
        "",
        "| Model | arch | backend | ISL | OSL | bs "
        "| pred TTFT | meas TTFT | TTFT err "
        "| pred TPOT | meas TPOT | TPOT err "
        "| pred E2EL | meas E2EL | E2EL err |",
        "|---|---|---|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:|",
    ]

    supported_ttft_errs: list[float] = []
    supported_tpot_errs: list[float] = []
    supported_e2el_errs: list[float] = []
    oos_counts = {"moe": 0, "hybrid_attn": 0}

    for r in out_rows:
        marker = " _(held-out)_" if r["held_out"] else ""
        tpot_pred = f"{r['pred_tpot']:.2f}" if r["osl"] > 0 else "\u2014"
        tpot_meas = f"{r['meas_tpot']:.2f}" if r["meas_tpot"] is not None else "\u2014"
        tpot_err_s = f"{r['tpot_err']:.1f}%" if r["tpot_err"] is not None else "\u2014"
        e2el_meas = f"{r['meas_e2el']:.2f}" if r["meas_e2el"] is not None else "\u2014"
        e2el_err_s = f"{r['e2el_err']:.1f}%" if r["e2el_err"] is not None else "\u2014"

        lines.append(
            f"| {r['short']}{marker} | {r['arch']} | {r['backend']} "
            f"| {r['isl']} | {r['osl']} | {r['bs']} "
            f"| {r['pred_ttft']:.2f} | {r['meas_ttft']:.2f} | {r['ttft_err']:.1f}% "
            f"| {tpot_pred} | {tpot_meas} | {tpot_err_s} "
            f"| {r['pred_e2el']:.2f} | {e2el_meas} | {e2el_err_s} |"
        )

        if r["arch"] == "supported":
            supported_ttft_errs.append(r["ttft_err"])
            if r["tpot_err"] is not None:
                supported_tpot_errs.append(r["tpot_err"])
            if r["e2el_err"] is not None:
                supported_e2el_errs.append(r["e2el_err"])
        else:
            oos_counts[r["arch"]] += 1

    lines.append("")
    lines.append("## Summary (supported architectures only)")
    lines.append("")

    def _mape(errs: list[float]) -> str:
        return f"{sum(errs)/len(errs):.2f}%" if errs else "n/a"

    lines.append(f"| Metric | MAPE | n rows |")
    lines.append(f"|---|---:|---:|")
    lines.append(f"| TTFT | {_mape(supported_ttft_errs)} | {len(supported_ttft_errs)} |")
    lines.append(f"| TPOT | {_mape(supported_tpot_errs)} | {len(supported_tpot_errs)} |")
    lines.append(f"| E2EL | {_mape(supported_e2el_errs)} | {len(supported_e2el_errs)} |")

    if any(oos_counts.values()):
        oos_desc = ", ".join(f"{v} {k}" for k, v in oos_counts.items() if v)
        lines.append(f"\n_Out-of-scope: {oos_desc} — excluded from headline._")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))

    ttft_s = _mape(supported_ttft_errs)
    tpot_s = _mape(supported_tpot_errs)
    e2el_s = _mape(supported_e2el_errs)
    print(
        f"[{gpu}][serving_e2e][{profile_name}] "
        f"TTFT MAPE={ttft_s}  TPOT MAPE={tpot_s}  E2EL MAPE={e2el_s} "
        f"({len(supported_ttft_errs)}/{len(out_rows)} in scope) — wrote {report_path}"
    )


# ── orchestration ────────────────────────────────────────────────────────────


# __ serving_e2e_conc (concurrency sweep) validation ______________________

DEFAULT_CONCURRENCIES = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]


def validate_serving_e2e_conc_gpu(gpu, data_json_path, report_path,
                                   profile_name, concurrencies=None):
    pred = PerKernelPredictor(gpu=gpu)
    if not pred.load():
        print("[%s][serving_e2e_conc] no pkls" % gpu)
        return

    concs = concurrencies or DEFAULT_CONCURRENCIES
    out_rows = []

    for conc in concs:
        measured_rows = _load_measured_rows(
            data_json_path, gpu, concurrency=conc,
            profile_filter=profile_name,
        )
        for row in measured_rows:
            dir_name = _MODELSHORT_TO_DIR.get(row["modelShort"])
            if not dir_name:
                continue
            cfg = model_specs.get_model_config(
                dir_name, held_out=model_specs.is_held_out(dir_name, gpu))
            if cfg is None:
                continue
            isl = max(1, int(round(row["avg_isl"])))
            osl = max(0, int(round(row["avg_osl"])))
            short = next((s for s, d in _SHORT_TO_DIR.items() if d == dir_name), row["modelShort"])
            arch = _arch_class(short, cfg)

            result = serving_e2e.predict_serving_e2e(
                pred, cfg, isl=isl, osl=osl, concurrency=conc)

            meas_ttft = row["measured_ttft_ms"]
            meas_tpot = row.get("median_tpot_ms")
            meas_e2el = row.get("median_e2el_ms")

            ttft_err = abs(result["ttft_ms"] - meas_ttft) / max(meas_ttft, 1e-9) * 100.0
            tpot_err = (abs(result["tpot_ms"] - meas_tpot) / max(meas_tpot, 1e-9) * 100.0
                        if meas_tpot and meas_tpot > 0 else None)
            e2el_err = (abs(result["e2el_ms"] - meas_e2el) / max(meas_e2el, 1e-9) * 100.0
                        if meas_e2el and meas_e2el > 0 else None)

            out_rows.append({
                "short": short, "arch": arch, "conc": conc,
                "isl": isl, "osl": osl, "bs_eff": result["bs_eff"],
                "saturated": result["saturated"],
                "pred_ttft": result["ttft_ms"], "pred_tpot": result["tpot_ms"],
                "pred_e2el": result["e2el_ms"],
                "meas_ttft": meas_ttft, "meas_tpot": meas_tpot, "meas_e2el": meas_e2el,
                "ttft_err": ttft_err, "tpot_err": tpot_err, "e2el_err": e2el_err,
            })

    if not out_rows:
        print("[%s][serving_e2e_conc][%s] no data" % (gpu, profile_name))
        return

    def _m(errs):
        return "%.1f%%" % (sum(errs)/len(errs)) if errs else "n/a"

    lines = [
        "# %s -- serving_e2e Concurrency Sweep: %s" % (gpu, profile_name),
        "",
        "- Predictor: steady-state batch size model + per-kernel XGBoost",
        "- Concurrencies: %s" % concs,
        "",
        "## Per-concurrency MAPE (supported architectures)",
        "",
        "| Conc | bs_eff | TTFT MAPE | TPOT MAPE | E2EL MAPE | n |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for conc in concs:
        cr = [r for r in out_rows if r["conc"] == conc and r["arch"] == "supported"]
        if not cr:
            continue
        te = [r["ttft_err"] for r in cr]
        pe = [r["tpot_err"] for r in cr if r["tpot_err"] is not None]
        ee = [r["e2el_err"] for r in cr if r["e2el_err"] is not None]
        bs = sum(r["bs_eff"] for r in cr) / len(cr)
        lines.append("| %d | %.1f | %s | %s | %s | %d |" % (conc, bs, _m(te), _m(pe), _m(ee), len(cr)))

    lines.append("")
    sup = [r for r in out_rows if r["arch"] == "supported"]
    sup_e2el = [r["e2el_err"] for r in sup if r["e2el_err"] is not None]
    sup_tpot = [r["tpot_err"] for r in sup if r["tpot_err"] is not None]
    lines.append("Overall supported MAPE: TPOT %s, E2EL %s" % (_m(sup_tpot), _m(sup_e2el)))
    lines.append("")

    lines.append("## Per-row detail")
    lines.append("")
    lines.append("| Model | arch | Conc | ISL | OSL | bs_eff "
                 "| pred TTFT | meas TTFT | TTFT err "
                 "| pred TPOT | meas TPOT | TPOT err "
                 "| pred E2EL | meas E2EL | E2EL err |")
    lines.append("|---|---|---:|---:|---:|---:"
                 "|---:|---:|---:"
                 "|---:|---:|---:"
                 "|---:|---:|---:|")

    for r in out_rows:
        tp = "%.1f" % r["pred_tpot"]
        tm = "%.1f" % r["meas_tpot"] if r["meas_tpot"] else "-"
        tes = "%.1f%%" % r["tpot_err"] if r["tpot_err"] is not None else "-"
        em = "%.1f" % r["meas_e2el"] if r["meas_e2el"] else "-"
        ees = "%.1f%%" % r["e2el_err"] if r["e2el_err"] is not None else "-"
        lines.append(
            "| %s | %s | %d | %d | %d | %.1f "
            "| %.1f | %.1f | %.1f%% "
            "| %s | %s | %s "
            "| %.1f | %s | %s |" % (
                r["short"], r["arch"], r["conc"], r["isl"], r["osl"], r["bs_eff"],
                r["pred_ttft"], r["meas_ttft"], r["ttft_err"],
                tp, tm, tes,
                r["pred_e2el"], em, ees))
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))

    print("[%s][serving_e2e_conc][%s] TPOT %s E2EL %s (%d/%d supported) -- wrote %s" % (
        gpu, profile_name, _m(sup_tpot), _m(sup_e2el),
        len(sup), len(out_rows), report_path))

def run(data_csv: Path, report_dir: Path, gpus: list[str], seq: int = 128,
        vs_measured: bool = False, data_json: Path | None = None,
        concurrency: int = 1,
        target_seq: int | None = None, seq_tolerance: int = 16,
        mode: str | None = None, profile: str | None = None) -> None:
    df = pd.read_csv(data_csv)

    for gpu in gpus:
        if mode == "serving_e2e_conc":
            profiles_list = [profile] if profile else [
                "chat-short", "chat-medium", "chat-long",
            ]
            for p in profiles_list:
                validate_serving_e2e_conc_gpu(
                    gpu, data_json,
                    report_dir / f"{gpu}_serving_e2e_conc_{p}.md",
                    profile_name=p,
                )
        elif mode == "serving_e2e":
            if data_json is None:
                print(f"[{gpu}][serving_e2e] --data-json required")
                continue
            profiles = [profile] if profile else [
                "chat-short", "chat-medium", "chat-long",
                "coding-agent", "prefill-heavy", "decode-heavy",
            ]
            for p in profiles:
                validate_serving_e2e_gpu(
                    gpu, data_json,
                    report_dir / f"{gpu}_serving_e2e_{p}.md",
                    profile_name=p,
                    concurrency=concurrency,
                )
        else:
            validate_gpu(df, gpu, report_dir / f"{gpu}_validation.md", seq=seq)
            if vs_measured and data_json is not None:
                suffix = f"_seq{target_seq}" if target_seq is not None else ""
                validate_vs_measured_gpu(
                    df, gpu, data_json,
                    report_dir / f"{gpu}_microbench_ttft_wallclock{suffix}.md",
                    concurrency=concurrency,
                    target_seq=target_seq,
                    seq_tolerance=seq_tolerance,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    ap.add_argument("--gpu", default=None, help="Shortcut for a single GPU")
    ap.add_argument("--seq", type=int, default=128)

    ap.add_argument("--mode", choices=["microbench_ttft", "serving_e2e", "serving_e2e_conc"], default=None,
                    help="Validation mode: microbench_ttft (prefill-only TTFT, default "
                         "when --vs-measured) or serving_e2e (ISL/OSL → TTFT+TPOT+E2EL).")
    ap.add_argument("--profile", default=None,
                    help="Workload profile for --mode serving_e2e (e.g. chat-short, "
                         "chat-medium, chat-long). If omitted, validates all standard profiles.")

    ap.add_argument("--vs-measured", action="store_true",
                    help="microbench_ttft mode: compare composer.predict_ttft_ms against "
                         "measured median_ttft_ms from data.json. "
                         "Equivalent to --mode microbench_ttft.")
    ap.add_argument("--data-json", default=None,
                    help="Path to inference-benchmark data.json.")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Concurrency filter (default 1).")
    ap.add_argument("--target-seq", type=int, default=None,
                    help="Restrict microbench_ttft rows to avg_seq within --seq-tolerance.")
    ap.add_argument("--seq-tolerance", type=int, default=16,
                    help="±tokens tolerance around --target-seq (default 16).")
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    data_csv = Path(args.data) if args.data else pkg_dir / "data" / "kernels_labeled.csv"
    data_csv = ensure_kernels_csv(data_csv)
    report_dir = Path(args.report_dir) if args.report_dir else pkg_dir / "reports"
    gpus = [args.gpu] if args.gpu else args.gpus

    mode = args.mode
    if mode is None and args.vs_measured:
        mode = "microbench_ttft"
    vs_measured = args.vs_measured or mode == "microbench_ttft"

    needs_data_json = vs_measured or mode in ("serving_e2e", "serving_e2e_conc")
    if needs_data_json:
        if args.data_json:
            data_json = Path(args.data_json)
        else:
            repo_root = pkg_dir.parent.parent.parent
            data_json = repo_root / "inference-benchmark" / "dashboard" / "public" / "data.json"
        if not data_json.exists():
            raise SystemExit(f"data.json not found: {data_json}")
    else:
        data_json = None

    run(data_csv, report_dir, gpus, seq=args.seq,
        vs_measured=vs_measured, data_json=data_json,
        concurrency=args.concurrency,
        target_seq=args.target_seq, seq_tolerance=args.seq_tolerance,
        mode=mode, profile=args.profile)


if __name__ == "__main__":
    main()
