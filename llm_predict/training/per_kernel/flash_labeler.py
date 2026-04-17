"""Merge flash_attn sweep ncu output into the labeled training dataset.

Reads:
    {prefix}.csv           — ncu long-format CSV from collect_flash.sh
    {prefix}.manifest.json — sweep manifest from sweep_flash.py

Emits rows matching `labeler.py OUT_COLS` so they can be concatenated with
the model-prefill `kernels_labeled.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ncu_loader
from . import gpu_kernel_regex as gkr
from .labeler import OUT_COLS, DTYPE_DEFAULT


_DTYPE_NORMALIZE = {
    "bfloat16": "bf16", "torch.bfloat16": "bf16", "bf16": "bf16",
    "float16":  "fp16", "torch.float16":  "fp16", "fp16": "fp16",
}


def _empty_row(source: str, gpu: str, dtype: str) -> dict:
    row = {c: np.nan for c in OUT_COLS}
    row.update({
        "source": source, "gpu": gpu, "model": "flash_sweep",
        "kernel_family": "flash_attn", "kernel_name": "", "dtype": dtype,
        "held_out": False, "op_type": "flash_attn",
    })
    return row


def label_sweep(sweep_csv: Path, manifest_json: Path, gpu: str,
                  source: str = "flash_sweep") -> pd.DataFrame:
    with open(manifest_json) as f:
        manifest = json.load(f)
    invocations: list[dict] = manifest["invocations"]
    dtype = _DTYPE_NORMALIZE.get(manifest.get("dtype", "bf16"), DTYPE_DEFAULT)

    df = ncu_loader.load(sweep_csv)
    if len(df) == 0:
        print(f"[!] empty sweep CSV: {sweep_csv}")
        return pd.DataFrame(columns=OUT_COLS)

    pairs = df["Kernel Name"].fillna("").map(gkr.classify_kernel)
    df["kernel_family"] = pairs.map(lambda p: p[1])

    flash = df[df["kernel_family"] == "flash_attn"].reset_index(drop=True)

    n_flash, n_expected = len(flash), len(invocations)
    print(f"[*] sweep flash kernels = {n_flash}, expected invocations = {n_expected}")
    if n_flash == 0:
        return pd.DataFrame(columns=OUT_COLS)
    if n_flash != n_expected:
        ratio = n_flash / max(n_expected, 1)
        print(f"[!] mismatch; ratio flash/expected = {ratio:.3f} — may include fallback paths")

    n = min(n_flash, n_expected)
    rows: list[dict] = []
    for i in range(n):
        raw = flash.iloc[i]
        m = invocations[i]
        r = _empty_row(source, gpu, dtype)
        r["kernel_name"] = raw.get("Kernel Name", "")
        r["bs"] = int(m["bs"])
        r["seq"] = int(m["seq"])
        r["n_heads"] = int(m["n_heads"])
        r["head_dim"] = int(m["head_dim"])
        r["kv_heads"] = int(m["kv_heads"])
        r["gpu_time_duration_ms"] = raw.get("gpu_time_duration_ms", np.nan)
        r["launch_block_size"] = raw.get("launch_block_size", np.nan)
        r["launch_grid_size"] = raw.get("launch_grid_size", np.nan)
        r["launch_registers_per_thread"] = raw.get("launch_registers_per_thread", np.nan)
        r["dram_bytes_sum"] = raw.get("dram_bytes_sum", np.nan)
        rows.append(r)

    out = pd.DataFrame(rows, columns=OUT_COLS)
    print(f"[+] labeled flash rows: {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-prefix", required=True,
                    help="Path prefix for {prefix}.csv + {prefix}.manifest.json")
    ap.add_argument("--gpu", required=True, help="GPU short name (A100 / RTX3090)")
    ap.add_argument("--append-to", default=None,
                    help="Existing kernels_labeled.csv to append to. If omitted, writes "
                         "kernels_labeled_flash_{gpu}.csv alongside the sweep CSV.")
    args = ap.parse_args()

    prefix = Path(args.sweep_prefix)
    sweep_csv = Path(str(prefix) + ".csv") if not prefix.suffix else prefix.with_suffix(".csv")
    manifest  = Path(str(prefix) + ".manifest.json") if not prefix.suffix else prefix.with_suffix(".manifest.json")
    if not sweep_csv.is_file():
        alt = Path(str(prefix) + ".csv")
        if alt.is_file():
            sweep_csv = alt
    if not manifest.is_file():
        alt = Path(str(prefix) + ".manifest.json")
        if alt.is_file():
            manifest = alt
    if not sweep_csv.is_file():
        raise SystemExit(f"missing sweep CSV near {prefix}")
    if not manifest.is_file():
        raise SystemExit(f"missing manifest JSON near {prefix}")

    new_rows = label_sweep(sweep_csv, manifest, gpu=args.gpu)

    if args.append_to:
        main_csv = Path(args.append_to)
        if main_csv.is_file():
            existing = pd.read_csv(main_csv)
            mask = (existing["source"] == "flash_sweep") & (existing["gpu"] == args.gpu)
            existing = existing[~mask].copy()
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(main_csv, index=False)
        print(f"[+] appended {len(new_rows)} rows -> {main_csv} (total {len(combined)})")
    else:
        out_path = sweep_csv.with_name(f"kernels_labeled_flash_{args.gpu}.csv")
        new_rows.to_csv(out_path, index=False)
        print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()
