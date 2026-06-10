#!/usr/bin/env python3
"""Merge dated raw decode-step run CSVs into one decode-grid CSV (newest run wins per cell).

The decode-step grids consumed by ``simulator/kernel_step_cost.load_grid`` (via each
deployment's ``data.decode_grid`` manifest entry) were so far single raw runs of
``profiling/gpu_profiling/vllm/cuda_events/decode_steps.py``. Raw run CSVs are APPEND-ONLY
session records (one file per run date, never edited); this builder is the deterministic
merge step that turns an ordered list of them into the grid artifact a deployment points at.

Merge rule (pre-registered in the 2026-06-10 L3 de-fit plan):
  * cells = union over all input runs, keyed by (batch_size, context_len);
  * where a cell exists in several runs, the LATEST input (list order = chronological)
    wins — re-measurements supersede older sessions (same script, same engine config);
    the superseded value and the drift vs it are reported on stdout and carried in the
    ``superseded_ms`` / ``drift_pct`` columns so the cross-check stays visible;
  * every output row records its ``source_file``. Extra columns are harmless to
    ``load_grid`` (DictReader; it reads batch_size / context_len / decode_step_ms /
    validation_status only).

H100x2 / Llama-3.1-8B defaults (the tp2 sub-linearity de-fit):
  inputs  profile_data/results/decode_profile_H100x2_2026-06-01.csv   (19 cells, sparse)
          profile_data/results/decode_profile_H100x2_2026-06-10_main.csv (54 cells, dense
          rectangle + T=24576 tail; re-measures all 19 old cells — session drift ex the
          known (1,512) warm-up outlier: median -0.85%, max |13.8|%)
  output  profile_data/results/decode_profile_H100x2_merged_2026-06-10.csv (54 cells)

The merged grid covers every (B, T) with B*(T+128) <= 998,656 KV tokens on the profiled
axes — i.e. up to the real H100x2 KV pool (62,416 blocks x 16), so serving states beyond
the measured hull are unreachable and the linear-in-b*ctx analytic fill no longer prices
any reachable tp2 decode step.

Deterministic (no RNG). Usage:
    python3 -m profiling.process.build_decode_grid                  # H100x2 defaults
    python3 -m profiling.process.build_decode_grid --inputs a.csv b.csv --out merged.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "profile_data" / "results"

DEFAULT_INPUTS = (
    RESULTS / "decode_profile_H100x2_2026-06-01.csv",
    RESULTS / "decode_profile_H100x2_2026-06-10_main.csv",
)
DEFAULT_OUT = RESULTS / "decode_profile_H100x2_merged_2026-06-10.csv"

# Raw decode_steps.py schema (kept first, in order) + merge-provenance columns.
RAW_FIELDS = [
    "gpu", "batch_size", "context_len", "observed_context_len", "total_kv_tokens",
    "decode_step_ms", "generated_tokens", "decode_intervals", "gpu_ms",
]
MERGE_FIELDS = RAW_FIELDS + ["source_file", "superseded_ms", "drift_pct"]


def merge(inputs: list[Path]) -> list[dict[str, object]]:
    """Union of cells over ``inputs`` (chronological order); the latest run wins per cell."""
    cells: dict[tuple[int, int], dict[str, object]] = {}
    for path in inputs:
        with path.open() as f:
            for r in csv.DictReader(f):
                key = (int(r["batch_size"]), int(r["context_len"]))
                row: dict[str, object] = {k: r.get(k, "") for k in RAW_FIELDS}
                row["source_file"] = path.name
                prev = cells.get(key)
                if prev is None:
                    row["superseded_ms"] = ""
                    row["drift_pct"] = ""
                else:
                    old_ms = float(str(prev["decode_step_ms"]))
                    new_ms = float(str(row["decode_step_ms"]))
                    row["superseded_ms"] = old_ms
                    row["drift_pct"] = round((new_ms - old_ms) / old_ms * 100.0, 2)
                cells[key] = row
    return [cells[k] for k in sorted(cells)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS),
                    metavar="CSV", help="raw run CSVs, OLDEST FIRST (later files win per cell)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="merged grid CSV")
    args = ap.parse_args(argv)

    for p in args.inputs:
        if not p.exists():
            raise SystemExit(f"missing raw run CSV: {p}")
    rows = merge(args.inputs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MERGE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    superseded = [r for r in rows if r["drift_pct"] != ""]
    drifts = sorted(float(str(r["drift_pct"])) for r in superseded)
    print(f"wrote {len(rows)} cells to {args.out} "
          f"({len(superseded)} re-measured; drift "
          f"{drifts[0]:+.2f}% .. {drifts[-1]:+.2f}%, median "
          f"{drifts[len(drifts) // 2]:+.2f}%)" if superseded else
          f"wrote {len(rows)} cells to {args.out} (no overlapping cells)")


if __name__ == "__main__":
    main()
