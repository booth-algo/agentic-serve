"""Merge GEMM sweep ncu output into the labeled training dataset.

Reads:
    {prefix}.csv           — ncu long-format CSV from collect_gemm.sh
    {prefix}.manifest.json — sweep manifest from sweep_gemm.py

Emits rows matching the `labeler.py OUT_COLS` schema so they can be
concatenated with the model-prefill `kernels_labeled.csv`. Labeling is
iteration-order: the i-th primary GEMM kernel in the CSV maps to the i-th
matmul in the manifest. splitk_reduce rows are attached to the preceding
primary GEMM's shape; gemv rows (not expected in this sweep) are dropped.

CLI
---
    python -m llm_predict.training.per_kernel.roofline_labeler \\
        --sweep-prefix /tmp/ncu_gemm_sweep_A100 \\
        --gpu A100 \\
        --append-to   llm_predict/training/per_kernel/data/kernels_labeled.csv
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
    "float32":  "fp32", "torch.float32":  "fp32", "fp32": "fp32",
}


def _empty_row(source: str, gpu: str, family: str, dtype: str) -> dict:
    row = {c: np.nan for c in OUT_COLS}
    row.update({
        "source": source, "gpu": gpu, "model": "roofline",
        "kernel_family": family, "kernel_name": "", "dtype": dtype,
        "held_out": False, "op_type": "",
    })
    return row


def label_sweep(sweep_csv: Path, manifest_json: Path, gpu: str,
                  source: str = "roofline_sweep") -> pd.DataFrame:
    with open(manifest_json) as f:
        manifest = json.load(f)
    expected: list[dict] = manifest["matmuls"]
    dtype = _DTYPE_NORMALIZE.get(manifest.get("dtype", "bf16"), DTYPE_DEFAULT)

    df = ncu_loader.load(sweep_csv)
    if len(df) == 0:
        print(f"[!] empty sweep CSV: {sweep_csv}")
        return pd.DataFrame(columns=OUT_COLS)

    pairs = df["Kernel Name"].fillna("").map(gkr.classify_kernel)
    df["kernel_subtype"] = pairs.map(lambda p: p[0])
    df["kernel_family"]  = pairs.map(lambda p: p[1])

    gemms = df[df["kernel_family"] == "gemm"].copy()
    primary = gemms[~gemms["kernel_subtype"].isin(["gemv", "splitk_reduce"])].reset_index()
    # `primary.index` is 0..n; `primary["index"]` is the original df index.

    n_primary, n_expected = len(primary), len(expected)
    print(f"[*] sweep primary gemm kernels = {n_primary}, expected matmuls = {n_expected}")
    if n_primary != n_expected:
        ratio = n_primary / max(n_expected, 1)
        print(f"[!] mismatch; ratio primary/expected = {ratio:.3f} "
              f"(cuBLAS may have issued multi-kernel gemm for some shapes)")

    n = min(n_primary, n_expected)
    rows: list[dict] = []
    # i -> original df index for the i-th primary gemm row
    primary_original_idx: dict[int, int] = {}

    for i in range(n):
        raw = primary.iloc[i]
        m = expected[i]
        r = _empty_row(source, gpu, "gemm", dtype)
        r["kernel_name"] = raw.get("Kernel Name", "")
        r["M"], r["N"], r["K"] = int(m["M"]), int(m["N"]), int(m["K"])
        r["bs"], r["seq"] = 1, int(m["M"])
        r["op_type"] = m.get("tag", "synth")
        r["gpu_time_duration_ms"] = raw.get("gpu_time_duration_ms", np.nan)
        r["launch_block_size"] = raw.get("launch_block_size", np.nan)
        r["launch_grid_size"] = raw.get("launch_grid_size", np.nan)
        r["launch_registers_per_thread"] = raw.get("launch_registers_per_thread", np.nan)
        r["dram_bytes_sum"] = raw.get("dram_bytes_sum", np.nan)
        rows.append(r)
        primary_original_idx[i] = int(raw["index"])

    # Attach splitk_reduce rows to preceding primary gemm's shape.
    splitk = df[df["kernel_family"] == "splitk_reduce"]
    sorted_primary = sorted(primary_original_idx.items(), key=lambda kv: kv[1])
    for idx, raw in splitk.iterrows():
        preceding_i = None
        for i, pid in sorted_primary:
            if pid < idx:
                preceding_i = i
            else:
                break
        if preceding_i is None or preceding_i >= n:
            continue
        m = expected[preceding_i]
        r = _empty_row(source, gpu, "splitk_reduce", dtype)
        r["kernel_name"] = raw.get("Kernel Name", "")
        r["M"], r["N"], r["K"] = int(m["M"]), int(m["N"]), int(m["K"])
        r["bs"], r["seq"] = 1, int(m["M"])
        r["op_type"] = "splitk_reduce"
        r["gpu_time_duration_ms"] = raw.get("gpu_time_duration_ms", np.nan)
        r["launch_block_size"] = raw.get("launch_block_size", np.nan)
        r["launch_grid_size"] = raw.get("launch_grid_size", np.nan)
        r["launch_registers_per_thread"] = raw.get("launch_registers_per_thread", np.nan)
        r["dram_bytes_sum"] = raw.get("dram_bytes_sum", np.nan)
        rows.append(r)

    out = pd.DataFrame(rows, columns=OUT_COLS)
    print(f"[+] labeled rows: {len(out)}  "
          f"(gemm={int((out['kernel_family']=='gemm').sum())}, "
          f"splitk={int((out['kernel_family']=='splitk_reduce').sum())})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-prefix", required=True,
                    help="Path prefix for {prefix}.csv + {prefix}.manifest.json")
    ap.add_argument("--gpu", required=True,
                    help="GPU short name (A100 / RTX3090 / RTX2080Ti)")
    ap.add_argument("--append-to", default=None,
                    help="Existing kernels_labeled.csv to append to. If omitted, writes "
                         "kernels_labeled_roofline_{gpu}.csv alongside the sweep CSV.")
    args = ap.parse_args()

    prefix = Path(args.sweep_prefix)
    sweep_csv = Path(str(prefix) + ".csv") if not prefix.suffix else prefix.with_suffix(".csv")
    manifest  = Path(str(prefix) + ".manifest.json") if not prefix.suffix else prefix.with_suffix(".manifest.json")
    # Common invocation: prefix has no extension; handle that too
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
            # Only dedupe on the roofline-sweep subset to prevent re-append
            # of the same sweep; leave existing model-prefill rows untouched.
            mask = (existing["source"] == "roofline_sweep") & (existing["gpu"] == args.gpu)
            existing = existing[~mask].copy()
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_csv(main_csv, index=False)
        print(f"[+] appended {len(new_rows)} rows -> {main_csv} (total {len(combined)})")
    else:
        out_path = sweep_csv.with_name(f"kernels_labeled_roofline_{args.gpu}.csv")
        new_rows.to_csv(out_path, index=False)
        print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()
