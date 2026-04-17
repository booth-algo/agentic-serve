"""ncu long→wide CSV loader.

The ncu `--csv` export produces long-format output: one row per
(kernel_invocation × metric). This module pivots to wide form — one row per
kernel invocation, one column per metric — with units normalised to a common
base (ms for time, bytes for memory).

Schema differences between the 2026-04 collection and the earlier runpod
pipeline (`per-kernel-rebuild/label_all_v3.py`):

- Target metric is `gpu__time_duration.sum` (new) vs `.avg` (old). Same
  semantics for `--replay-mode application` collections (one sample per launch).
- Memory is combined `dram__bytes.sum` (new) vs split `dram__bytes_read.sum` +
  `dram__bytes_write.sum` (old). Gate-vs-Down GEMM disambiguation via write
  bytes is not possible with the new schema — callers must fall back to grid
  scoring.
- Launch columns (`block_size`, `grid_size`, `registers_per_thread`) are
  collected as metric rows, not as dedicated columns. The legacy `Block Size` /
  `Grid Size` columns in the CSV header carry tuple-formatted strings
  (e.g. `"(256, 1, 1)"`) and are ignored here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_METRIC = "gpu__time_duration.sum"

_TIME_UNIT_TO_MS = {
    "ns": 1e-6,
    "nsec": 1e-6,
    "us": 1e-3,
    "usec": 1e-3,
    "ms": 1.0,
    "msec": 1.0,
    "s": 1e3,
    "sec": 1e3,
}

_BYTE_UNIT_TO_BYTES = {
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
}

# Metric -> (output column name, value kind). Kind: "time_ms" | "bytes" | "int" | "float".
_METRIC_MAP = {
    "gpu__time_duration.sum":  ("gpu_time_duration_ms", "time_ms"),
    "gpu__time_duration.avg":  ("gpu_time_duration_ms", "time_ms"),
    "dram__bytes.sum":         ("dram_bytes_sum",       "bytes"),
    "dram__bytes_read.sum":    ("dram_bytes_read",      "bytes"),
    "dram__bytes_write.sum":   ("dram_bytes_write",     "bytes"),
    "launch__block_size":            ("launch_block_size",           "int"),
    "launch__grid_size":             ("launch_grid_size",            "int"),
    "launch__registers_per_thread":  ("launch_registers_per_thread", "int"),
    "launch__thread_count":          ("launch_thread_count",         "int"),
    "launch__shared_mem_per_block_dynamic": ("launch_shared_mem_dynamic", "int"),
    "launch__shared_mem_per_block_static":  ("launch_shared_mem_static",  "int"),
    "gpc__cycles_elapsed.max":                       ("gpc_cycles_elapsed_max", "int"),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_throughput_pct", "float"),
    "smsp__inst_executed.sum":                        ("smsp_inst_executed",   "int"),
}


def _to_number(s) -> float:
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    t = str(s).replace(",", "").strip()
    if not t:
        return np.nan
    try:
        return float(t)
    except ValueError:
        return np.nan


def _scale(value: float, unit: str, kind: str) -> float:
    if pd.isna(value):
        return np.nan
    if kind == "time_ms":
        mult = _TIME_UNIT_TO_MS.get(unit, 1.0)
    elif kind == "bytes":
        mult = _BYTE_UNIT_TO_BYTES.get(unit, 1.0)
    else:
        mult = 1.0
    return value * mult


def load(path: Path | str) -> pd.DataFrame:
    """Load a single ncu long-format CSV and return a wide DataFrame.

    Each row corresponds to one kernel launch, keyed by the `ID` column.
    Missing metrics become NaN columns. Kernel launch order in the returned
    DataFrame matches the order of first appearance in the source CSV —
    callers that rely on iteration order (layer-pattern walker) can treat
    it as monotonic.
    """
    path = Path(path)
    raw = pd.read_csv(path, low_memory=False, dtype=str)
    if len(raw) == 0:
        return pd.DataFrame()
    if "Metric Name" not in raw.columns or "Metric Value" not in raw.columns:
        raise ValueError(
            f"{path} does not look like an ncu long-format CSV "
            f"(missing Metric Name / Metric Value columns)"
        )

    # Preserve launch order by first-seen ID index.
    raw["_order"] = pd.to_numeric(raw["ID"], errors="coerce")
    first_seen = raw.drop_duplicates("ID", keep="first")[["ID", "Kernel Name", "_order"]]

    # Scale each metric row to canonical units.
    metric = raw["Metric Name"].astype(str)
    unit = raw["Metric Unit"].astype(str).fillna("")
    value = raw["Metric Value"].apply(_to_number)
    kinds = metric.map(lambda m: _METRIC_MAP.get(m, (None, None))[1])
    raw["_scaled"] = [_scale(v, u, k) for v, u, k in zip(value, unit, kinds)]

    # Pivot: one row per ID, one column per known metric.
    known = raw[raw["Metric Name"].isin(_METRIC_MAP)].copy()
    known["_col"] = known["Metric Name"].map(lambda m: _METRIC_MAP[m][0])
    wide = known.pivot_table(
        index="ID",
        columns="_col",
        values="_scaled",
        aggfunc="first",
    ).reset_index()

    # Attach kernel name + original order.
    wide = wide.merge(first_seen, on="ID", how="left")
    wide = wide.sort_values("_order", kind="mergesort").reset_index(drop=True)
    wide = wide.drop(columns="_order")

    return wide


def load_concat(paths: Iterable[Path | str], dedupe_keys: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Load and concatenate several ncu CSVs (e.g., a main + `_ext` pair).

    If `dedupe_keys` is provided, rows matching on those columns are deduped,
    keeping the first occurrence (prefer the main file if it is listed first).
    """
    frames = [load(p) for p in paths]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if dedupe_keys:
        out = out.drop_duplicates(list(dedupe_keys), keep="first").reset_index(drop=True)
    return out
