"""Merge flash_attn sweep ncu output into the labeled training dataset.

Reads:
    {prefix}.csv           — ncu long-format CSV from collect_flash.sh
    {prefix}.manifest.json — sweep manifest from sweep_flash.py

v2 (2026-04-25): replaces broken positional 1:1 matching with SDPA-call
grouping. FA2 dispatches 1-N kernels per F.scaled_dot_product_attention
call (primary flash_fwd_kernel + optional splitkv + combine). This labeler
groups consecutive kernels into SDPA calls, pairs warmup/measured calls,
and sums gpu_time across all kernels in each call.

Emits rows matching `labeler.py OUT_COLS` so they can be concatenated with
the model-prefill `kernels_labeled.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labeler import OUT_COLS, DTYPE_DEFAULT


_DTYPE_NORMALIZE = {
    "bfloat16": "bf16", "torch.bfloat16": "bf16", "bf16": "bf16",
    "float16":  "fp16", "torch.float16":  "fp16", "fp16": "fp16",
}


def _ktype(name: str) -> str:
    if "splitkv_combine" in name:
        return "combine"
    if "splitkv_kernel" in name:
        return "splitkv"
    if "flash_fwd_kernel" in name:
        return "primary"
    return "other"


def _pivot_long_to_wide(df: pd.DataFrame) -> list[dict]:
    """Pivot ncu long-format (one row per metric) to one dict per kernel."""
    by_id: dict[str, dict] = {}
    id_order: list[str] = []
    for _, row in df.iterrows():
        kid = str(row["ID"])
        if kid not in by_id:
            id_order.append(kid)
            by_id[kid] = {"Kernel Name": row["Kernel Name"]}
        metric = row["Metric Name"]
        try:
            by_id[kid][metric] = float(str(row["Metric Value"]).replace(",", ""))
        except (ValueError, TypeError):
            by_id[kid][metric] = np.nan
    return [by_id[kid] for kid in id_order]


def _group_sdpa_calls(kernels: list[dict]) -> list[list[dict]]:
    """Group consecutive kernels into SDPA calls.
    A new call starts at each 'primary' kernel."""
    calls: list[list[dict]] = []
    current: list[dict] = []
    for k in kernels:
        t = _ktype(k["Kernel Name"])
        if t == "primary" and current:
            calls.append(current)
            current = []
        current.append(k)
    if current:
        calls.append(current)
    return calls


def _pair_warmup_measured(calls: list[list[dict]]) -> list[list[dict]]:
    """Extract measured SDPA calls by pairing consecutive same-signature calls.
    
    Each config runs warmup + measured (REPS=1). Consecutive calls with the
    same kernel-type signature are paired; the second (measured) is kept.
    Unpaired calls are kept as-is (single ncu capture)."""
    def sig(call):
        return tuple(_ktype(k["Kernel Name"]) for k in call)
    
    measured: list[list[dict]] = []
    i = 0
    while i < len(calls):
        if i + 1 < len(calls) and sig(calls[i]) == sig(calls[i + 1]):
            measured.append(calls[i + 1])
            i += 2
        else:
            measured.append(calls[i])
            i += 1
    return measured


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
    configs: list[dict] = manifest["configs"]
    dtype = _DTYPE_NORMALIZE.get(manifest.get("dtype", "bf16"), DTYPE_DEFAULT)

    df = pd.read_csv(sweep_csv)
    if len(df) == 0:
        print(f"[!] empty sweep CSV: {sweep_csv}")
        return pd.DataFrame(columns=OUT_COLS)

    kernels = _pivot_long_to_wide(df)
    flash_kernels = [k for k in kernels if _ktype(k["Kernel Name"]) != "other"]

    calls = _group_sdpa_calls(flash_kernels)
    measured = _pair_warmup_measured(calls)

    n_matched = min(len(measured), len(configs))
    print(f"[*] {gpu}: {len(flash_kernels)} flash kernels -> {len(calls)} SDPA calls "
          f"-> {len(measured)} measured -> {n_matched}/{len(configs)} configs matched")

    rows: list[dict] = []
    for i in range(n_matched):
        cfg = configs[i]
        call = measured[i]
        r = _empty_row(source, gpu, dtype)

        primary = [k for k in call if _ktype(k["Kernel Name"]) == "primary"]
        r["kernel_name"] = primary[0]["Kernel Name"] if primary else call[0]["Kernel Name"]
        r["bs"] = int(cfg["bs"])
        r["seq"] = int(cfg["seq"])
        r["n_heads"] = int(cfg["n_heads"])
        r["head_dim"] = int(cfg["head_dim"])
        r["kv_heads"] = int(cfg["kv_heads"])

        r["gpu_time_duration_ms"] = sum(
            k.get("gpu__time_duration.sum", 0) for k in call
        ) / 1000.0

        total_dram = sum(k.get("dram__bytes.sum", 0) for k in call)
        r["dram_bytes_sum"] = total_dram if total_dram > 0 else np.nan

        if primary:
            p = primary[0]
            r["launch_block_size"] = p.get("launch__block_size", np.nan)
            r["launch_grid_size"] = p.get("launch__grid_size", np.nan)
            r["launch_registers_per_thread"] = p.get("launch__registers_per_thread", np.nan)

        rows.append(r)

    out = pd.DataFrame(rows, columns=OUT_COLS)
    print(f"[+] labeled {len(out)} flash rows (summed across multi-kernel SDPA calls)")
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
    manifest = Path(str(prefix) + ".manifest.json") if not prefix.suffix else prefix.with_suffix(".manifest.json")
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
