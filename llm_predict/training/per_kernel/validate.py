"""Paper-facing validation — per-GPU MAPE + aggregate layer error.

Compares composer.predict_ttft_ms(model, seq=128) against the ncu-measured
sum-of-kernel-times from `kernels_labeled.csv` for every (gpu, model). Writes
one report per GPU at `reports/{gpu}_validation.md`.

Ncu-self-consistency was chosen over external e2e (vLLM/sglang) measurements
because Hetzner doesn't have e2e data for RTX3090 / RTX2080Ti (see plan
decision #6). A100 cross-reference with runpod:measured/ is TODO.

CLI
---
    python -m llm_predict.training.per_kernel.validate --gpu A100
    python -m llm_predict.training.per_kernel.validate         # all GPUs
"""
from __future__ import annotations

import argparse
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


def validate_gpu(df: pd.DataFrame, gpu: str, report_path: Path, seq: int = 128) -> None:
    pred = PerKernelPredictor(gpu=gpu)
    if not pred.load():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"# {gpu} validation\n\nNo pkls loaded; run trainer first.\n")
        print(f"[{gpu}] no pkls — wrote placeholder report")
        return

    df_gpu = df[df["gpu"] == gpu].copy()
    measured_per_model = df_gpu.groupby("model")["gpu_time_duration_ms"].sum()

    rows: list[tuple[str, float, float, float, int]] = []
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
        rows.append((short, pred_ms, meas_ms, err_pct, n_ker))

    lines: list[str] = [
        f"# {gpu} — Per-Kernel Composition Validation",
        "",
        f"- Predictor: {gpu} pkls ({sorted(pred.families_loaded())})",
        f"- Ground truth: sum(gpu_time_duration_ms) per model from kernels_labeled.csv",
        f"- Input: bs=1, seq={seq}, tp=1",
        "",
        "## Per-model",
        "",
        "| Model | predicted TTFT (ms) | measured Σ (ms) | abs err % | n kernels |",
        "|---|---:|---:|---:|---:|",
    ]
    total_pred = 0.0
    total_meas = 0.0
    for short, pred_ms, meas_ms, err_pct, n in rows:
        marker = " _(held-out)_" if model_specs.is_held_out(_SHORT_TO_DIR[short], gpu) else ""
        lines.append(f"| {short}{marker} | {pred_ms:.2f} | {meas_ms:.2f} | {err_pct:.2f}% | {n} |")
        total_pred += pred_ms
        total_meas += meas_ms
    total_err = abs(total_pred - total_meas) / max(total_meas, 1e-9) * 100.0
    lines.append(f"| **TOTAL** | **{total_pred:.2f}** | **{total_meas:.2f}** | **{total_err:.2f}%** | |")
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
    print(f"[{gpu}] total err {total_err:.2f}% over {len(rows)} models — wrote {report_path}")


def run(data_csv: Path, report_dir: Path, gpus: list[str], seq: int = 128) -> None:
    df = pd.read_csv(data_csv)
    for gpu in gpus:
        validate_gpu(df, gpu, report_dir / f"{gpu}_validation.md", seq=seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    ap.add_argument("--gpu", default=None, help="Shortcut for a single GPU")
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    data_csv = Path(args.data) if args.data else pkg_dir / "data" / "kernels_labeled.csv"
    report_dir = Path(args.report_dir) if args.report_dir else pkg_dir / "reports"
    gpus = [args.gpu] if args.gpu else args.gpus

    run(data_csv, report_dir, gpus, seq=args.seq)


if __name__ == "__main__":
    main()
