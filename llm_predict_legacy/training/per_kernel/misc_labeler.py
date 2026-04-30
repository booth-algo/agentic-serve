"""Merge synthetic misc-kernel sweep ncu output into the labeled training dataset.

Reads:
    {prefix}.csv           — ncu long-format CSV from profiling sweep_misc.py
    {prefix}.manifest.json — sweep manifest (op_type, M, d, numel per invocation)

Emits rows in `labeler.py OUT_COLS` schema. Labeling is iteration-order: each
synthetic op dispatches exactly one CUDA kernel at the shapes we sweep, so
the i-th classified misc-family kernel in CSV maps to the i-th manifest entry.

Subfamily mapping:
    op_type='reduce'  → kernel_family='reduce'
    op_type='softmax' → kernel_family='reduce'   (softmax classified as reduce by regex)
    op_type='cast'    → kernel_family='cast'
    op_type='copy'    → kernel_family='copy'
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

_OP_TO_FAMILY = {
    "reduce":  "reduce",
    "softmax": "reduce",
    "cast":    "cast",
    "copy":    "copy",
}


def _empty_row(source: str, gpu: str, family: str, dtype: str, op_type: str) -> dict:
    row = {c: np.nan for c in OUT_COLS}
    row.update({
        "source": source, "gpu": gpu, "model": "misc_sweep",
        "kernel_family": family, "kernel_name": "", "dtype": dtype,
        "held_out": False, "op_type": op_type,
    })
    return row


def label_sweep(sweep_csv: Path, manifest_json: Path, gpu: str,
                  source: str = "misc_sweep") -> pd.DataFrame:
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
    misc = df[df["kernel_family"].isin(["reduce", "cast", "copy", "elementwise"])].reset_index(drop=True)

    n_misc, n_expected = len(misc), len(invocations)
    print(f"[*] misc-family kernels = {n_misc}, expected invocations = {n_expected}")
    if n_misc == 0:
        return pd.DataFrame(columns=OUT_COLS)
    if n_misc != n_expected:
        ratio = n_misc / max(n_expected, 1)
        print(f"[!] mismatch; ratio misc/expected = {ratio:.3f}")

    n = min(n_misc, n_expected)
    rows: list[dict] = []
    for i in range(n):
        raw = misc.iloc[i]
        m = invocations[i]
        op = m["op_type"]
        family = _OP_TO_FAMILY.get(op, "other")
        r = _empty_row(source, gpu, family, dtype, op)
        r["kernel_name"] = raw.get("Kernel Name", "")
        r["M"] = int(m["M"])
        r["N"] = int(m["d"])
        r["numel"] = int(m["numel"])
        r["bs"], r["seq"] = 1, int(m["M"])
        r["gpu_time_duration_ms"] = raw.get("gpu_time_duration_ms", np.nan)
        r["launch_block_size"] = raw.get("launch_block_size", np.nan)
        r["launch_grid_size"] = raw.get("launch_grid_size", np.nan)
        r["launch_registers_per_thread"] = raw.get("launch_registers_per_thread", np.nan)
        r["dram_bytes_sum"] = raw.get("dram_bytes_sum", np.nan)
        rows.append(r)

    out = pd.DataFrame(rows, columns=OUT_COLS)
    by_fam = dict(pd.Series([r["kernel_family"] for r in rows]).value_counts())
    print(f"[+] labeled misc rows: {len(out)} by family: {by_fam}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-prefix", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--append-to", default=None)
    args = ap.parse_args()

    prefix = Path(args.sweep_prefix)
    sweep_csv = Path(str(prefix) + ".csv")
    manifest = Path(str(prefix) + ".manifest.json")
    if not sweep_csv.is_file():
        raise SystemExit(f"missing {sweep_csv}")
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}")

    new_rows = label_sweep(sweep_csv, manifest, gpu=args.gpu)

    if args.append_to:
        main_csv = Path(args.append_to)
        if main_csv.is_file():
            existing = pd.read_csv(main_csv)
            mask = (existing["source"] == "misc_sweep") & (existing["gpu"] == args.gpu)
            existing = existing[~mask].copy()
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(main_csv, index=False)
        print(f"[+] appended {len(new_rows)} rows -> {main_csv} (total {len(combined)})")
    else:
        out_path = sweep_csv.with_name(f"kernels_labeled_misc_{args.gpu}.csv")
        new_rows.to_csv(out_path, index=False)
        print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()
