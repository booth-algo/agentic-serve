"""Post-process flash attention NCU CSV into per-GPU latency tables.

Reads the NCU CSV export + manifest JSON, matches NVTX ranges to
(q_len, kv_len, n_heads, n_kv_heads, head_dim, batch) tuples,
aggregates median latency per shape, and writes a clean CSV for
the FlashAttnPredictor table.

Usage:
  python post_process_flash.py \\
      --ncu-csv /tmp/ncu_flash_serving_H100.csv \\
      --manifest /tmp/ncu_flash_serving_H100.manifest.json \\
      --out llm_predict/data/flash_attn/H100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

_GPU_KEY_PAT = re.compile(
    r"flash_q(\d+)_kv(\d+)_h(\d+)_kvh(\d+)_hd(\d+)_b(\d+)"
)

# NCU CSV column name variants
_DURATION_KEYS = [
    "gpu__time_duration.sum",
    "gpu__time_duration.sum [ns]",
    "GPU Time Duration (ns)",
    "gpu__time_duration.sum [nsecond]",
]

_NVTX_KEYS = [
    "NVTX Range",
    "NvtxRange",
    "nvtx_range",
    "NVTX Range:flash",
]


def _parse_duration_ns(row: dict) -> float | None:
    for key in _DURATION_KEYS:
        val = row.get(key, "")
        if not val:
            continue
        try:
            return float(str(val).replace(",", "").replace('"', ""))
        except ValueError:
            continue
    return None


def _find_nvtx(row: dict) -> str | None:
    for key in _NVTX_KEYS:
        val = row.get(key, "")
        if val and "flash" in str(val).lower():
            return str(val)
    # Also try partial key match
    for k, v in row.items():
        if "nvtx" in k.lower() and "flash" in str(v).lower():
            return str(v)
    return None


def _parse_shape(nvtx_str: str) -> tuple[int, int, int, int, int, int] | None:
    m = _GPU_KEY_PAT.search(nvtx_str)
    if m:
        return (
            int(m.group(1)), int(m.group(2)),
            int(m.group(3)), int(m.group(4)),
            int(m.group(5)), int(m.group(6)),
        )
    return None


def post_process(ncu_csv: Path, manifest_path: Path, out_path: Path) -> None:
    # Load manifest to know expected shapes and invocation order
    with open(manifest_path) as f:
        manifest = json.load(f)
    invocations = manifest.get("invocations", [])
    idx_to_shape: dict[int, tuple] = {}
    for inv in invocations:
        idx = inv["idx"]
        shape = (inv["q_len"], inv["kv_len"], inv["n_heads"],
                 inv["n_kv_heads"], inv["head_dim"], inv["batch"])
        idx_to_shape[idx] = shape
    expected_shapes = set(idx_to_shape.values())
    print(f"[*] {len(expected_shapes)} unique shapes in manifest, "
          f"{len(invocations)} invocations")

    # Parse NCU CSV
    with open(ncu_csv) as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    print(f"[*] {len(raw_rows)} CSV rows")

    # NCU CSV has two rows per kernel launch (one per metric).
    # Group by "ID" column and collect per-kernel duration values.
    kernel_durations: dict[int, list[float]] = defaultdict(list)
    for row in raw_rows:
        kernel_id_str = row.get("ID", "").strip('"')
        if not kernel_id_str:
            continue
        try:
            kernel_id = int(kernel_id_str)
        except ValueError:
            continue

        metric_name = row.get("Metric Name", "")
        metric_unit = row.get("Metric Unit", "")
        metric_value = row.get("Metric Value", "")

        if "gpu__time_duration" not in metric_name:
            continue

        try:
            val = float(metric_value.strip('"'))
        except ValueError:
            continue

        # Convert to microseconds if needed
        if "ns" in metric_unit:
            val = val / 1000.0
        elif "ms" in metric_unit:
            val = val * 1000.0
        # else assume microseconds

        if val > 0:
            kernel_durations[kernel_id].append(val)

    print(f"[*] {len(kernel_durations)} unique kernel launches with duration data")

    # Map kernel ID → shape via sequential ordering.
    # Each invocation in the manifest produces 1-3 kernels (fwd, splitkv, combine).
    # We bucket consecutive kernel IDs into their parent invocation.
    shape_kernel_count: dict[tuple, int] = defaultdict(int)
    shape_durations: dict[tuple, list[float]] = defaultdict(list)

    sorted_ids = sorted(kernel_durations)
    # Estimate kernels-per-invocation from the ratio
    kernels_per_inv = len(sorted_ids) / max(1, len(invocations))
    print(f"[*] ~{kernels_per_inv:.1f} kernels per invocation")

    # Simple sequential mapping: assume kernels_per_inv is an integer or close
    # Round to nearest integer, then map in blocks
    kpi = max(1, round(kernels_per_inv))
    inv_idx = 0
    for i, kid in enumerate(sorted_ids):
        if i > 0 and i % kpi == 0:
            inv_idx += 1
        if inv_idx >= len(invocations):
            inv_idx = len(invocations) - 1
        inv = invocations[min(inv_idx, len(invocations) - 1)]
        shape = (inv["q_len"], inv["kv_len"], inv["n_heads"],
                 inv["n_kv_heads"], inv["head_dim"], inv["batch"])
        shape_kernel_count[shape] += 1
        shape_durations[shape].extend(kernel_durations[kid])

    print(f"[*] {len(shape_durations)} shapes with duration data "
          f"({sum(len(v) for v in shape_durations.values())} total measurements)")

    if not shape_durations:
        print("[!] No shapes matched — check NVTX range naming")
        return

    # Write output table
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "q_len", "kv_len", "n_heads", "n_kv_heads",
            "head_dim", "batch", "latency_us",
        ])
        for shape in sorted(shape_durations):
            med = median(shape_durations[shape])
            writer.writerow([
                shape[0], shape[1], shape[2], shape[3],
                shape[4], shape[5], round(med, 2),
            ])

    # Coverage report
    missing = expected_shapes - set(shape_durations)
    print(f"[+] {len(shape_durations)} shapes written to {out_path}")
    if missing:
        print(f"[!] {len(missing)} shapes in manifest but not in NCU output "
              f"(sampled {len(expected_shapes) - len(missing)}/{len(expected_shapes)})")
    else:
        print(f"[+] All {len(expected_shapes)} shapes covered")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-process flash attention NCU CSV"
    )
    ap.add_argument("--ncu-csv", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    post_process(args.ncu_csv, args.manifest, args.out)


if __name__ == "__main__":
    main()
