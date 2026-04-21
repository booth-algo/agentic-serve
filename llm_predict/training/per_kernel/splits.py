"""splits.py — LOMO + held-out split iterators for per-kernel training.

Centralises the leave-one-model-out (LOMO) and canonical held-out split
logic so the next-stage trainer can consume `data/per_family/{gpu}/{family}.csv`
and iterate splits without re-implementing per-GPU model selection.
The held-out set per GPU is driven by `model_specs.HELD_OUT_BY_GPU`.

Two split types
---------------
1. **Held-out split** (`heldout_split`): canonical train/test split
   where `test` = rows flagged `held_out=True` by the labeler. Used to
   measure extrapolation to unseen models (e.g. Llama-3.3-70B on A100).
2. **Leave-one-model-out** (`lomo_splits`): iterate every training-pool
   model; each yielded split holds that model out as test while the
   others train. Used for within-pool cross-validation. Held-out-set
   models are **never** in either LOMO train or LOMO test (they're the
   permanent held-out set).

API
---
    from llm_predict.training.per_kernel import splits

    for family in ["gemm", "flash_attn", "elementwise", "misc"]:
        df = splits.load_per_family(per_family_dir, gpu="A100", family=family)
        train_df, test_df = splits.heldout_split(df, gpu="A100")
        for hold_model, train_df, test_df in splits.lomo_splits(df):
            ...

CLI
---
    python -m llm_predict.training.per_kernel.splits \\
        --per-family-dir llm_predict/training/per_kernel/data/per_family \\
        --report llm_predict/training/per_kernel/reports/splits_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import feature_spec
from . import model_specs


# Minimum rows for a LOMO fold to be considered trainable. Matches
# trainer.py's `MIN_TRAIN_ROWS`. A (gpu, family, lomo-fold) combination
# that falls below this is reported as "insufficient" and the downstream
# trainer will skip it.
MIN_TRAIN_ROWS = 5


# ───────── Helpers ─────────

def held_out_short_names(gpu: str) -> set[str]:
    """Translate `model_specs.HELD_OUT_BY_GPU` (ncu directory names) to
    the short names used in per-family CSVs' `model` column."""
    dir_names = model_specs.HELD_OUT_BY_GPU.get(gpu, set())
    out: set[str] = set()
    for d in dir_names:
        short = model_specs._DIR_TO_SHORT.get(d)
        if short is not None:
            out.add(short)
    return out


def load_per_family(per_family_dir: Path, gpu: str, family: str) -> pd.DataFrame | None:
    """Load `per_family/{gpu}/{family}.csv`. Returns None if the file
    doesn't exist (e.g. RTX2080Ti flash_attn, which the splitter skips)."""
    path = per_family_dir / gpu / f"{family}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, low_memory=False)


# ───────── Split iterators ─────────

def heldout_split(df: pd.DataFrame, gpu: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) where test = rows flagged `held_out=True`
    by the labeler. `gpu` is accepted for API symmetry with `lomo_splits`
    but is not used — the `held_out` flag is already GPU-aware (set by
    `model_specs.is_held_out(dir_name, gpu)` during labeling)."""
    del gpu  # unused, see docstring
    held = df["held_out"].astype(bool)
    return df[~held].reset_index(drop=True), df[held].reset_index(drop=True)


def lomo_models(df: pd.DataFrame) -> list[str]:
    """Models in the training pool (held_out=False) that participate in LOMO."""
    return sorted(df[~df["held_out"].astype(bool)]["model"].dropna().unique().tolist())


def lomo_splits(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (hold_model, train_df, test_df) for each training-pool model.

    - `train_df`: rows where `held_out=False` AND `model != hold_model`.
    - `test_df`:  rows where `held_out=False` AND `model == hold_model`.

    Held-out rows (`held_out=True`) are never in either side. Iteration
    order is alphabetical by model short-name (deterministic)."""
    pool = df[~df["held_out"].astype(bool)].copy()
    for m in lomo_models(df):
        test = pool[pool["model"] == m].reset_index(drop=True)
        train = pool[pool["model"] != m].reset_index(drop=True)
        yield m, train, test


# ───────── Summary + sanity ─────────

def summarise_one(df: pd.DataFrame, gpu: str, family: str) -> dict:
    train_df, test_df = heldout_split(df, gpu=gpu)
    pool = lomo_models(df)
    held_models = sorted(
        df[df["held_out"].astype(bool)]["model"].dropna().unique().tolist()
    )

    # LOMO fold sufficiency.
    lomo_info: dict[str, dict] = {}
    for hold_m, tr, te in lomo_splits(df):
        lomo_info[hold_m] = {
            "n_train": len(tr),
            "n_test":  len(te),
            "sufficient": (len(tr) >= MIN_TRAIN_ROWS and len(te) >= 1),
        }

    min_lomo_train = min((v["n_train"] for v in lomo_info.values()), default=0)
    min_lomo_test = min((v["n_test"] for v in lomo_info.values()), default=0)

    # Sanity check: every pool model must appear as a LOMO test split
    # exactly once (given the current iterator, this is tautological,
    # but we assert it to catch future refactors).
    seen_as_test = set(lomo_info.keys())
    assert seen_as_test == set(pool), (
        f"LOMO iterator drift at ({gpu}, {family}): "
        f"test-split models {sorted(seen_as_test)} != pool {pool}"
    )

    # Cross-check the held-out set against model_specs.
    expected_held = held_out_short_names(gpu)
    unexpected_held = set(held_models) - expected_held
    missing_held    = expected_held - set(held_models)

    return {
        "gpu": gpu, "family": family,
        "n_total": len(df),
        "n_train_heldout_split": len(train_df),
        "n_test_heldout_split":  len(test_df),
        "pool_models":  pool,
        "held_models":  held_models,
        "lomo":         lomo_info,
        "min_lomo_train_rows": int(min_lomo_train),
        "min_lomo_test_rows":  int(min_lomo_test),
        "lomo_all_sufficient": all(v["sufficient"] for v in lomo_info.values()),
        "unexpected_held": sorted(unexpected_held),
        "missing_held":    sorted(missing_held),
    }


def summarise_all(per_family_dir: Path, gpus: list[str], families: list[str]) -> list[dict]:
    rows: list[dict] = []
    for gpu in gpus:
        for family in families:
            df = load_per_family(per_family_dir, gpu, family)
            if df is None or len(df) == 0:
                rows.append({"gpu": gpu, "family": family, "n_total": 0, "missing": True})
                continue
            rows.append(summarise_one(df, gpu, family))
    return rows


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Per-Kernel Train/Test Splits Report",
        "",
        "Generated by `splits.py` — describes the canonical held-out split "
        "and leave-one-model-out (LOMO) folds available for each "
        "(GPU × family) pair in `data/per_family/`.",
        "",
        "## Held-out split (per `model_specs.HELD_OUT_BY_GPU`)",
        "",
        "| GPU | family | total | train | test | held-out models |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['gpu']} | {r['family']} | — | — | — | (missing file) |")
            continue
        held = ", ".join(r["held_models"]) or "—"
        lines.append(
            f"| {r['gpu']} | {r['family']} | {r['n_total']} | "
            f"{r['n_train_heldout_split']} | {r['n_test_heldout_split']} | {held} |"
        )
    lines.append("")

    lines.append("## LOMO folds — per-(GPU × family) summary")
    lines.append("")
    lines.append("| GPU | family | pool size | min train | min test | all sufficient |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in rows:
        if r.get("missing"):
            continue
        ok = "yes" if r["lomo_all_sufficient"] else "**NO**"
        lines.append(
            f"| {r['gpu']} | {r['family']} | {len(r['pool_models'])} | "
            f"{r['min_lomo_train_rows']} | {r['min_lomo_test_rows']} | {ok} |"
        )
    lines.append("")

    lines.append("## LOMO folds — per-model detail")
    lines.append("")
    for r in rows:
        if r.get("missing"):
            continue
        lines.append(f"### {r['gpu']} / {r['family']}")
        lines.append("")
        lines.append("| held-out model | n_train | n_test | sufficient |")
        lines.append("|---|---:|---:|---|")
        for m, info in r["lomo"].items():
            ok = "yes" if info["sufficient"] else "**NO**"
            lines.append(f"| {m} | {info['n_train']} | {info['n_test']} | {ok} |")
        lines.append("")

    # Sanity anomalies — flag any drift from model_specs.
    anomalies = [r for r in rows
                 if not r.get("missing")
                 and (r["unexpected_held"] or r["missing_held"])]
    if anomalies:
        lines.append("## ⚠️  Held-out set anomalies")
        lines.append("")
        lines.append("Entries where the `held_out=True` rows in the per-family "
                     "CSV don't match `model_specs.HELD_OUT_BY_GPU` — indicates "
                     "the labeler's held-out flag drifted from the spec.")
        lines.append("")
        for r in anomalies:
            if r["unexpected_held"]:
                lines.append(
                    f"- **{r['gpu']}/{r['family']}**: unexpected held models "
                    f"`{r['unexpected_held']}`"
                )
            if r["missing_held"]:
                lines.append(
                    f"- **{r['gpu']}/{r['family']}**: missing held models "
                    f"`{r['missing_held']}`"
                )
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"[*] report: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family-dir", default=None,
                    help="Per-family CSV root "
                         "(default: <pkg>/data/per_family)")
    ap.add_argument("--report", default=None,
                    help="Output markdown report "
                         "(default: <pkg>/reports/splits_report.md)")
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    ap.add_argument("--families", nargs="+",
                    default=list(feature_spec.FAMILY_CONFIG.keys()))
    args = ap.parse_args()

    pkg = Path(__file__).resolve().parent
    per_family_dir = (Path(args.per_family_dir) if args.per_family_dir
                      else pkg / "data" / "per_family")
    report = (Path(args.report) if args.report
              else pkg / "reports" / "splits_report.md")

    rows = summarise_all(per_family_dir, args.gpus, args.families)
    write_report(rows, report)


if __name__ == "__main__":
    main()
