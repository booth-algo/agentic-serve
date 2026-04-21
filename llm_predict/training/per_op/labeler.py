"""labeler.py — raw CUDA-event CSVs → per-op labeled training CSV.

Consumes torch.profiler-based phase1_bsseq_profiles.csv files (one per
model) and emits a single combined `data/per_op_labeled.csv` keyed on
(gpu, model, op, bs, seq). Replaces the ad-hoc per-path loading at
`experiment/train_perop_v4.py:load_phase1_data()`.

Raw input schema
----------------
Each input CSV is produced by torch.profiler CUDA-event capture
(e.g. `experiment/profile_phase1.py`). Expected columns (either spelling
is accepted):

    layer_name | op_name    — str, one of the module names OP_MAP expects
                              (self_attn / mlp / block_sparse_moe /
                               input_layernorm / post_attention_layernorm)
    bs                      — int, batch size
    seq                     — int, sequence length per batch
    n_tokens                — int, bs * seq during prefill; bs during decode
    kv_cache_len            — int, KV cache length (0 for prefill)
    latency_ns | duration_us — float, measured op latency

The per-GPU / per-model root is:
    {raw_root}/{GPU}/{model_dir}/tp1/phase1_bsseq_profiles.csv

where model_dir is one of the keys of `model_specs._DIR_TO_SHORT`.

Output schema
-------------
`data/per_op_labeled.csv` columns:

    gpu           str   — GPU tag ("A100" / "RTX3090" / "RTX2080Ti")
    model         str   — canonical short name (Llama-8B, Qwen-72B, ...)
    held_out      bool  — True if the model is in model_specs.HELD_OUT_BY_GPU[gpu]
    op            str   — one of (attn / ffn / norm_pre / norm_post)
    bs            int
    seq           int
    n_tokens      int
    kv_cache_len  int
    d             int   — hidden_size (from ModelConfig)
    h             int   — num_attention_heads
    kv            int   — num_key_value_heads
    ffn           int   — intermediate_size
    E             int   — num_local_experts (1 for dense)
    k             int   — num_experts_per_tok (1 for dense)
    duration_us   float — measured latency in microseconds (target)

CLI
---
    python -m llm_predict.training.per_op.labeler \\
        --raw-root /path/to/profiling_data \\
        --out      llm_predict/training/per_op/data/per_op_labeled.csv \\
        --gpus     A100 RTX3090 RTX2080Ti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import feature_spec
from llm_predict.training.per_kernel import model_specs


OUT_COLS: list[str] = [
    "gpu", "model", "held_out",
    "op", "bs", "seq", "n_tokens", "kv_cache_len",
    "d", "h", "kv", "ffn", "E", "k",
    "duration_us",
]


def _resolve_latency_us(row: pd.Series) -> float:
    """Accept either `duration_us` (new) or `latency_ns` (legacy v3/v4)."""
    if "duration_us" in row.index and pd.notna(row["duration_us"]):
        return float(row["duration_us"])
    if "latency_ns" in row.index and pd.notna(row["latency_ns"]):
        return float(row["latency_ns"]) / 1000.0
    raise KeyError("row has neither `duration_us` nor `latency_ns`")


def _resolve_op_name(row: pd.Series) -> str | None:
    """Accept either `op` (already mapped), `op_name`, or `layer_name`."""
    for col in ("op", "op_name", "layer_name"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "input" in df.columns and "n_tokens" not in df.columns:
        df = df.rename(columns={"input": "n_tokens"})
    if "latency(ns)" in df.columns and "latency_ns" not in df.columns:
        df = df.rename(columns={"latency(ns)": "latency_ns"})
    return df


def label_model_csv(csv_path: Path, gpu: str, model_dir: str) -> list[dict]:
    """Read one `phase1_bsseq_profiles.csv` and return labeled rows."""
    cfg = model_specs.get_model_config(
        model_dir, held_out=model_specs.is_held_out(model_dir, gpu)
    )
    if cfg is None:
        raise ValueError(f"unknown model dir: {model_dir!r}")
    short = model_specs._DIR_TO_SHORT[model_dir]

    df = _normalise_columns(pd.read_csv(csv_path))

    # Map raw module name → predictor op category via feature_spec.OP_MAP.
    raw_op_names = df.apply(_resolve_op_name, axis=1)
    df = df.assign(_op_raw=raw_op_names)
    df = df[df["_op_raw"].notna()].copy()
    df["op"] = df["_op_raw"].map(feature_spec.OP_MAP)
    df = df.dropna(subset=["op"])

    rows: list[dict] = []
    for _, r in df.iterrows():
        try:
            duration_us = _resolve_latency_us(r)
        except KeyError:
            continue
        bs = int(r.get("bs", 1))
        seq = int(r.get("seq", r.get("n_tokens", 0)))
        n_tokens = int(r.get("n_tokens", bs * seq))
        kv_cache_len = int(r.get("kv_cache_len", 0))
        rows.append({
            "gpu":          gpu,
            "model":        short,
            "held_out":     cfg.held_out,
            "op":           r["op"],
            "bs":           bs,
            "seq":          seq,
            "n_tokens":     n_tokens,
            "kv_cache_len": kv_cache_len,
            "d":            cfg.d,
            "h":            cfg.h,
            "kv":           cfg.kv,
            "ffn":          cfg.ffn,
            "E":            cfg.E,
            "k":            cfg.k,
            "duration_us":  duration_us,
        })
    return rows


def label_all(raw_root: Path, gpus: list[str],
              profile_filename: str = "phase1_bsseq_profiles.csv",
              tp_suffix: str = "tp1") -> pd.DataFrame:
    """Walk `{raw_root}/{gpu}/{model_dir}/{tp_suffix}/{profile_filename}` for
    every known model directory and concatenate the labeled rows."""
    all_rows: list[dict] = []
    for gpu in gpus:
        gpu_dir = raw_root / gpu
        if not gpu_dir.is_dir():
            print(f"[-] {gpu}: no raw-root dir at {gpu_dir}, skipping")
            continue
        for model_dir in model_specs._DIR_TO_SHORT:
            csv_path = gpu_dir / model_dir / tp_suffix / profile_filename
            if not csv_path.is_file():
                continue
            rows = label_model_csv(csv_path, gpu=gpu, model_dir=model_dir)
            print(f"[*] {gpu}/{model_dir}: {len(rows)} rows")
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows, columns=OUT_COLS)
    return df


def write_report(df: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Per-Op Labeling Report", ""]
    lines.append(f"- total rows: **{len(df)}**")
    if len(df) == 0:
        lines.append("- (no rows — raw data missing)")
        report_path.write_text("\n".join(lines))
        return
    lines.append(f"- GPUs: {sorted(df['gpu'].unique().tolist())}")
    lines.append(f"- models: {sorted(df['model'].unique().tolist())}")
    lines.append(f"- ops: {sorted(df['op'].unique().tolist())}")
    lines.append("")
    lines.append("## Rows per (gpu × model × op)")
    lines.append("")
    lines.append("| gpu | model | held_out | op | n_rows |")
    lines.append("|---|---|---|---|---:|")
    grp = df.groupby(["gpu", "model", "held_out", "op"]).size().reset_index(name="n")
    for _, r in grp.iterrows():
        lines.append(f"| {r['gpu']} | {r['model']} | {r['held_out']} | {r['op']} | {r['n']} |")
    report_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=False, default=None,
                    help="Root containing {gpu}/{model_dir}/tp1/phase1_bsseq_profiles.csv")
    ap.add_argument("--out", default=None,
                    help="Output labeled CSV (default: <pkg>/data/per_op_labeled.csv)")
    ap.add_argument("--report", default=None,
                    help="Output markdown report (default: <pkg>/reports/labeling_report.md)")
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    ap.add_argument("--profile-filename", default="phase1_bsseq_profiles.csv")
    ap.add_argument("--tp-suffix", default="tp1")
    args = ap.parse_args()

    pkg = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else pkg / "data" / "per_op_labeled.csv"
    report = Path(args.report) if args.report else pkg / "reports" / "labeling_report.md"

    if args.raw_root is None:
        raise SystemExit(
            "[!] --raw-root required. Phase 5 will supply the CUDA-event CSVs.")

    df = label_all(Path(args.raw_root), args.gpus,
                     profile_filename=args.profile_filename,
                     tp_suffix=args.tp_suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[*] wrote {len(df)} rows → {out}")
    write_report(df, report)
    print(f"[*] report: {report}")


if __name__ == "__main__":
    main()
