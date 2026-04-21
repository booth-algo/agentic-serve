"""Phase A1.5 splitter — combined labeled CSV → per-(GPU × family) CSVs.

Reads `data/kernels_labeled.csv` (produced by `labeler.py`) and writes
one CSV per (GPU, kernel-family) pair to
`data/per_family/{gpu}/{family}.csv` so the downstream trainer can
consume a lean, GPU-pinned, family-pure dataset without re-filtering
in memory each run.

Output schema is defined in `feature_spec.{GEMM,FLASH_ATTN,ELEMENTWISE,MISC}_RAW_COLS`
and includes ONLY raw shape columns + `target_us`. Derived features
(log_*, analytical_flops, one-hots) are NOT materialised here — they
are built on-the-fly by `feature_spec.build_*_features(df)` at train
time. This keeps the CSVs small and lets feature-engineering changes
ship without re-running the splitter.

Rules applied per `.claude/paper/predictor_notes.md`:
- Qwen3.5-{9B,27B} rows are excluded from `flash_attn.csv` only (hybrid
  attention — see docstring in `feature_spec.FAMILY_EXCLUDED_MODELS`).
  They remain admitted for gemm/elementwise/misc (shape-only).
- RTX2080Ti has zero flash_attn rows (sm_75 pre-FA2); `flash_attn.csv`
  is simply not written for that GPU.
- A (gpu, family) pair with fewer than `MIN_ROWS` usable rows is
  skipped with a logged warning rather than writing a truncated file.

CLI
---
    python -m llm_predict.training.per_kernel.split_by_family \\
        --data llm_predict/training/per_kernel/data/kernels_labeled.csv \\
        --out-dir llm_predict/training/per_kernel/data/per_family
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import feature_spec


MIN_ROWS = 5


def _coerce_target_us(df: pd.DataFrame) -> pd.DataFrame:
    """Add `target_us = gpu_time_duration_ms * 1000`. Drops rows where
    the target is NaN or ≤ 0 (cannot be used as a supervised sample)."""
    df = df.copy()
    t = pd.to_numeric(df["gpu_time_duration_ms"], errors="coerce")
    df["target_us"] = t * 1000.0
    return df[df["target_us"].notna() & (df["target_us"] > 0)].reset_index(drop=True)


def _select_for_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Return the subset of `df` belonging to the given training family.

    The combined CSV's `kernel_family` column uses the classifier's
    raw family names (gemm / flash_attn / elementwise / reduce /
    splitk_reduce / cast / copy / other). Training families remap
    those: gemm → {gemm}, flash_attn → {flash_attn}, elementwise →
    {elementwise}, misc → {reduce, splitk_reduce, cast, copy}.
    """
    cfg = feature_spec.FAMILY_CONFIG[family]
    mask = df["kernel_family"].isin(cfg["kernel_families"])
    return df[mask].copy()


def _apply_family_exclusions(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Drop rows for models that are locked out of this family pool."""
    excluded = feature_spec.FAMILY_EXCLUDED_MODELS.get(family, set())
    if not excluded:
        return df
    return df[~df["model"].isin(excluded)].reset_index(drop=True)


def _project_raw_cols(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Keep only the raw columns declared in `feature_spec.*_RAW_COLS`.
    Missing columns are added as NaN so callers get a stable schema."""
    cols = feature_spec.FAMILY_CONFIG[family]["raw_cols"]
    out = df.reindex(columns=cols).copy()
    return out


def split_one(df: pd.DataFrame, gpu: str, family: str) -> tuple[pd.DataFrame, dict]:
    gpu_df = df[df["gpu"] == gpu]
    fam_df = _select_for_family(gpu_df, family)
    n_pre = len(fam_df)
    fam_df = _apply_family_exclusions(fam_df, family)
    n_post = len(fam_df)
    fam_df = _project_raw_cols(fam_df, family)

    stats: dict = {
        "gpu": gpu,
        "family": family,
        "n_rows": len(fam_df),
        "n_excluded_by_model": n_pre - n_post,
        "excluded_models": sorted(feature_spec.FAMILY_EXCLUDED_MODELS.get(family, set())),
        "models_present": sorted(
            m for m in fam_df["model"].dropna().unique().tolist()
        ),
        "n_held_out": int(fam_df["held_out"].astype(bool).sum()) if len(fam_df) else 0,
    }
    return fam_df, stats


def run(data_csv: Path, out_dir: Path, report_path: Path,
        gpus: list[str], families: list[str]) -> list[dict]:
    if not data_csv.is_file():
        raise FileNotFoundError(f"combined labeled CSV not found: {data_csv}")

    print(f"[*] loading {data_csv}")
    raw = pd.read_csv(data_csv, low_memory=False)
    print(f"[*] {len(raw)} rows; gpus={sorted(raw['gpu'].unique())}")

    with_target = _coerce_target_us(raw)
    n_dropped_target = len(raw) - len(with_target)
    if n_dropped_target:
        print(f"[*] dropped {n_dropped_target} rows with null/≤0 gpu_time_duration_ms")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict] = []

    for gpu in gpus:
        sub = with_target[with_target["gpu"] == gpu]
        if len(sub) == 0:
            print(f"[-] {gpu}: no rows in combined CSV, skipping")
            continue
        gpu_dir = out_dir / gpu
        gpu_dir.mkdir(parents=True, exist_ok=True)

        for family in families:
            fam_df, stats = split_one(with_target, gpu, family)
            all_stats.append(stats)
            if stats["n_rows"] < MIN_ROWS:
                print(f"    [skip] {gpu}/{family}: only {stats['n_rows']} rows (<{MIN_ROWS})")
                continue
            out_path = gpu_dir / f"{family}.csv"
            fam_df.to_csv(out_path, index=False)
            excl = (
                f" (excluded {stats['n_excluded_by_model']} rows from "
                f"{stats['excluded_models']})"
                if stats["n_excluded_by_model"] else ""
            )
            print(f"    [ok]   {gpu}/{family}: {stats['n_rows']} rows, "
                  f"held_out={stats['n_held_out']}{excl}")

    _write_report(all_stats, with_target, report_path)
    return all_stats


def _write_report(stats: list[dict], df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Phase A1.5 Per-Family Split Report",
        "",
        f"- Source rows (valid target): {len(df)}",
        f"- Families: {list(feature_spec.FAMILY_CONFIG.keys())}",
        f"- GPUs: {sorted(df['gpu'].unique())}",
        f"- Excluded models per family: "
        f"{ {f: sorted(m) for f, m in feature_spec.FAMILY_EXCLUDED_MODELS.items()} }",
        "",
        "## Row counts per (GPU × family)",
        "",
        "| GPU | family | rows | held_out | excluded | models |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s in stats:
        mods = ", ".join(s["models_present"]) or "—"
        lines.append(
            f"| {s['gpu']} | {s['family']} | {s['n_rows']} | {s['n_held_out']} | "
            f"{s['n_excluded_by_model']} | {mods} |"
        )
    lines.append("")

    lines.append("## Raw column schemas")
    lines.append("")
    for fam, cfg in feature_spec.FAMILY_CONFIG.items():
        lines.append(f"- **{fam}.csv**: `{', '.join(cfg['raw_cols'])}`")
    lines.append("")
    lines.append(
        "Derived features are not materialised — the trainer rebuilds them "
        "at load time via `feature_spec.build_{gemm,flash_attn,elementwise,misc}_features`."
    )

    path.write_text("\n".join(lines))
    print(f"[*] report: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="Path to combined kernels_labeled.csv "
                         "(default: <pkg>/data/kernels_labeled.csv)")
    ap.add_argument("--out-dir", default=None,
                    help="Per-family output root "
                         "(default: <pkg>/data/per_family)")
    ap.add_argument("--report", default=None,
                    help="Markdown split report "
                         "(default: <pkg>/reports/per_family_split_report.md)")
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    ap.add_argument("--families", nargs="+",
                    default=list(feature_spec.FAMILY_CONFIG.keys()))
    args = ap.parse_args()

    pkg = Path(__file__).resolve().parent
    data = Path(args.data) if args.data else pkg / "data" / "kernels_labeled.csv"
    out_dir = Path(args.out_dir) if args.out_dir else pkg / "data" / "per_family"
    report = Path(args.report) if args.report else pkg / "reports" / "per_family_split_report.md"

    run(data, out_dir, report, args.gpus, args.families)


if __name__ == "__main__":
    main()
