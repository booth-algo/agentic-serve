"""Paper-facing validation — per-GPU MAPE + aggregate layer error.

Two comparison modes:

1. **ncu-self-consistency** (default):
   `composer.predict_ttft_ms(model, seq=128)` vs sum-of-kernel-times from
   `kernels_labeled.csv` for every `(gpu, model)`. Proves the kernel
   predictor reproduces the ncu numbers it was trained on.

2. **wall-clock** (`--vs-measured`):
   `composer.predict_ttft_ms(model, seq=avg_input_tokens, bs=concurrency)`
   vs `summary.median_ttft_ms` from the inference-benchmark `data.json`
   dump. Adds a third column that captures launch overhead + stream
   gaps + CPU dispatch — everything the ncu-self-consistency view
   misses.

The two-column "overhead %" on the wall-clock report is
`(wall_clock_ms - kernel_sum_ms) / wall_clock_ms * 100` and represents
the fraction of real TTFT we don't model with kernels.

CLI
---
    python -m llm_predict.training.per_kernel.validate --gpu A100
    python -m llm_predict.training.per_kernel.validate --vs-measured
    # Apples-to-apples against the ncu seq=128 sweep — requires benchmark
    # runs of the `fixed-seq128` workload profile so avg_seq ≈ 128:
    python -m llm_predict.training.per_kernel.validate --vs-measured \\
        --target-seq 128 --seq-tolerance 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor

from . import composer, model_specs


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
# `composer.predict_ttft_ms` has known gaps on two architecture classes:
#   - `moe`         (cfg.is_moe): expert dispatch not modelled.
#   - `hybrid_attn` (Qwen3.5 family): the composer assumes full attention at
#     every layer, but Qwen3.5 is hybrid (1/4 full + 3/4 linear) — it will
#     over-predict flash_attn 4×. See composer.predict_layer_ms and the
#     note in model_specs._FALLBACK for Qwen3.5-9B.
# Rows in these classes are reported separately so the headline MAPE
# reflects only architectures the composer fully supports.
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


# ── wall-clock (vs measured TTFT) validation ────────────────────────────────

def _load_measured_rows(data_json_path: Path,
                         gpu: str,
                         concurrency: int = 1,
                         target_seq: int | None = None,
                         seq_tolerance: int = 16) -> list[dict]:
    """Return measured-ttft rows from data.json matching a given GPU/concurrency.

    If `target_seq` is given, rows are restricted to profiles whose
    per-request `avg_seq` lies within ±`seq_tolerance` tokens of it.
    This is how the apples-to-apples comparison against the ncu
    prefill_seq128_bs1 sweep is built: run a `fixed-seq128` benchmark
    profile, then `--vs-measured --target-seq 128`.

    Each returned dict has:
        modelShort, backend, profile, concurrency,
        avg_seq, measured_ttft_ms, median_tpot_ms, successful_requests
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
        succ = int(summ.get("successful_requests", 0) or 0)
        if succ <= 0:
            continue
        total_in = float(summ.get("total_input_tokens", 0) or 0)
        median_ttft = summ.get("median_ttft_ms")
        if median_ttft is None or total_in <= 0:
            continue
        avg_seq = total_in / succ
        if target_seq is not None and abs(avg_seq - target_seq) > seq_tolerance:
            continue
        rows.append({
            "modelShort": ms,
            "backend": cfg.get("backend", ""),
            "profile": cfg.get("profile", ""),
            "concurrency": concurrency,
            "avg_seq": avg_seq,
            "measured_ttft_ms": float(median_ttft),
            "median_tpot_ms": (
                float(summ["median_tpot_ms"]) if summ.get("median_tpot_ms") is not None else None
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
        print(f"[{gpu}][vs-measured] no pkls — skipping")
        return

    measured_rows = _load_measured_rows(
        data_json_path, gpu, concurrency=concurrency,
        target_seq=target_seq, seq_tolerance=seq_tolerance,
    )
    if not measured_rows:
        print(f"[{gpu}][vs-measured] no matching rows in data.json — skipping")
        return

    # Per-(model) kernel-sum from ncu for the overhead column.
    df_gpu = df[df["gpu"] == gpu].copy()
    kernel_sum_per_model = df_gpu.groupby("model")["gpu_time_duration_ms"].sum().to_dict()

    out_rows: list[tuple] = []
    for row in measured_rows:
        dir_name = _MODELSHORT_TO_DIR[row["modelShort"]]
        cfg = model_specs.get_model_config(
            dir_name, held_out=model_specs.is_held_out(dir_name, gpu))
        if cfg is None:
            continue
        seq = max(1, int(round(row["avg_seq"])))
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
        f"# {gpu} — Wall-clock TTFT Validation (vs inference-benchmark data.json)",
        "",
        f"- Predictor: {gpu} pkls ({sorted(pred.families_loaded())})",
        f"- Ground truth: `summary.median_ttft_ms` per (model, backend, profile)",
        f"- Filter: concurrency={concurrency}, TP=1{seq_filter_note}",
        f"- Overhead % = `(measured - ncu_kernel_sum) / measured` — fraction of real TTFT not captured by kernels.",
        "- Headline MAPE = supported architectures only. MoE + hybrid-attn rows are known composer gaps; they are shown in the table but excluded from the aggregate.",
        "- `median tpot (ms)` is shown for reference; per-kernel composer currently only predicts TTFT, so no TPOT prediction column.",
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
        ncu_cell = f"{ncu_sum_ms:.2f}" if ncu_sum_ms is not None else "—"
        ov_cell = f"{overhead_pct:.1f}%" if overhead_pct is not None else "—"
        tpot_cell = f"{tpot_ms:.2f}" if tpot_ms is not None else "—"
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
        f"[{gpu}][vs-measured] supported MAPE {mape_s} "
        f"({len(supported_errs)}/{n} rows in scope) — wrote {report_path}"
    )


def run(data_csv: Path, report_dir: Path, gpus: list[str], seq: int = 128,
        vs_measured: bool = False, data_json: Path | None = None,
        concurrency: int = 1,
        target_seq: int | None = None, seq_tolerance: int = 16) -> None:
    df = pd.read_csv(data_csv)
    suffix = f"_seq{target_seq}" if target_seq is not None else ""
    for gpu in gpus:
        validate_gpu(df, gpu, report_dir / f"{gpu}_validation.md", seq=seq)
        if vs_measured and data_json is not None:
            validate_vs_measured_gpu(
                df, gpu, data_json,
                report_dir / f"{gpu}_wallclock_validation{suffix}.md",
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
    ap.add_argument("--vs-measured", action="store_true",
                    help="Also compare composer.predict_ttft_ms against measured "
                         "median_ttft_ms from inference-benchmark data.json.")
    ap.add_argument("--data-json", default=None,
                    help="Path to inference-benchmark data.json "
                         "(default: <repo>/inference-benchmark/dashboard/public/data.json)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Concurrency filter for --vs-measured (default 1).")
    ap.add_argument("--target-seq", type=int, default=None,
                    help="Restrict --vs-measured rows to those whose avg input "
                         "seq lies within --seq-tolerance of this target. Use "
                         "128 to match the ncu prefill_seq128_bs1 sweep.")
    ap.add_argument("--seq-tolerance", type=int, default=16,
                    help="±tokens tolerance around --target-seq (default 16).")
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    data_csv = Path(args.data) if args.data else pkg_dir / "data" / "kernels_labeled.csv"
    report_dir = Path(args.report_dir) if args.report_dir else pkg_dir / "reports"
    gpus = [args.gpu] if args.gpu else args.gpus

    if args.vs_measured:
        if args.data_json:
            data_json = Path(args.data_json)
        else:
            # <repo>/llm_predict/training/per_kernel/validate.py → <repo>
            repo_root = pkg_dir.parent.parent.parent
            data_json = repo_root / "inference-benchmark" / "dashboard" / "public" / "data.json"
        if not data_json.exists():
            raise SystemExit(f"[--vs-measured] data.json not found: {data_json}")
    else:
        data_json = None

    run(data_csv, report_dir, gpus, seq=args.seq,
        vs_measured=args.vs_measured, data_json=data_json,
        concurrency=args.concurrency,
        target_seq=args.target_seq, seq_tolerance=args.seq_tolerance)


if __name__ == "__main__":
    main()
