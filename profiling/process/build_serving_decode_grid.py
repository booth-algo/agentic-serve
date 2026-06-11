#!/usr/bin/env python3
"""Deterministic builder: raw serving-decode-grid JSONL.gz run(s) -> grid CSV.

The measurement client (``profiling/gpu_profiling/vllm/serving_decode_grid.py``) writes an
APPEND-ONLY raw per-request JSONL.gz per session (wall-anchored SSE event timestamps). This
builder re-derives every cell row from the raw events with the SAME pre-registered
post-processing (``summarize_cell``, imported from the client module — single source of truth):
steady window = [max first-token, min last-token]; per-request p50 mid-stream ITL inside the
window (>= 64 in-window deltas per request, else ``check``); cell decode_step_ms = median over
the per-request p50s; effective context_len = prompt + median in-window progress. Where a cell
appears in several inputs, the LATEST input (list order = chronological) wins — the
``build_decode_grid.py`` merge rule.

Output CSV columns (L11 pre-registration):
    batch_size, context_len, decode_step_ms, validation_status,
    nominal_T, prompt_tokens, osl, n_samples, steady_window_s, ...diagnostics
``simulator.kernel_step_cost.load_grid`` reads only the first four; extras are harmless
(DictReader). The grid consumer is agnostic to how cells were measured.

Deterministic (no RNG). Usage:
    python3 -m profiling.process.build_serving_decode_grid \
        --inputs profile_data/results/serving_decode_grid_H100x4_<date>.jsonl.gz \
        --out    profile_data/results/serving_decode_grid_H100x4_<date>.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_CLIENT = REPO_ROOT / "profiling" / "gpu_profiling" / "vllm" / "serving_decode_grid.py"

_spec = importlib.util.spec_from_file_location("serving_decode_grid_client", _CLIENT)
_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_client)  # aiohttp is lazily imported by the client; safe locally

summarize_cell = _client.summarize_cell
SUMMARY_FIELDS = _client.SUMMARY_FIELDS
OUT_FIELDS = SUMMARY_FIELDS + ["source_file"]


def read_run(path: Path) -> tuple[dict, dict[tuple[int, int], list[dict]]]:
    """One raw JSONL.gz -> (meta, {(B, nominal_T): [request records]})."""
    meta: dict = {}
    cells: dict[tuple[int, int], list[dict]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_meta"):
                meta = d
                continue
            key = (int(d["cell"][0]), int(d["cell"][1]))
            cells.setdefault(key, []).append(d)
    return meta, cells


def build(inputs: list[Path]) -> list[dict]:
    merged: dict[tuple[int, int], tuple[list[dict], str]] = {}
    for path in inputs:
        meta, cells = read_run(path)
        for key, recs in cells.items():
            if key in merged:
                print(f"cell {key}: superseded by {path.name}")
            merged[key] = (recs, path.name)
    rows = []
    for key in sorted(merged):
        recs, src = merged[key]
        ok = [r for r in recs if r.get("t_first_wall") is not None]
        dropped = len(recs) - len(ok)
        row = summarize_cell(ok)
        if dropped:
            row["validation_status"] = "check"
            print(f"cell {key}: {dropped} empty-stream request(s) -> check")
        row["source_file"] = src
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="raw JSONL(.gz) runs, chronological order (latest wins per cell)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = build([Path(p) for p in a.inputs])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
    n_ok = sum(1 for r in rows if r["validation_status"] == "ok")
    print(f"wrote {out} ({len(rows)} cells, {n_ok} ok / {len(rows) - n_ok} check)")


if __name__ == "__main__":
    main()
