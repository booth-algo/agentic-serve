"""Build per-layer kernel profiling parquet from R2 analytical + NCU data."""

import argparse
import csv
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://get-data.kevinlauofficial01.workers.dev/"

ANALYTICAL_FILES = [
    "perkernel/main/analytical/llama8b_analytical_prefill_B1.json",
    "perkernel/main/analytical/llama8b_analytical_prefill_B80.json",
]

NCU_CSV_PATH = "perkernel/main/ncu/bs1_8layers/Llama-3.1-8B_prefill_bs1_8layers.csv"

MODEL = "Llama-3.1-8B"
HARDWARE = "H100"
PHASE = "prefill"

NCU_METRIC_MAP = {
    ("GPU Speed Of Light Throughput", "Duration"): "duration_us",
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "compute_sm_throughput_pct",
    ("GPU Speed Of Light Throughput", "DRAM Throughput"): "dram_throughput_pct",
    ("GPU Speed Of Light Throughput", "Memory Throughput"): "memory_throughput_pct",
    ("GPU Speed Of Light Throughput", "L1/TEX Cache Throughput"): "l1_tex_cache_throughput_pct",
    ("GPU Speed Of Light Throughput", "L2 Cache Throughput"): "l2_cache_throughput_pct",
    ("GPU Speed Of Light Throughput", "SM Frequency"): "sm_frequency_ghz",
    ("GPU Speed Of Light Throughput", "DRAM Frequency"): "dram_frequency_ghz",
}


def download(url: str, dest: Path) -> Path:
    print(f"  Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "AgentPerfBench/1.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)
    except urllib.error.URLError as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        sys.exit(1)
    return dest


def parse_analytical(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    batch_size = data["B"]
    seq_len = data["S"]

    rows = []
    rows.append({
        "record_type": "analytical_total",
        "model": MODEL,
        "hardware": HARDWARE,
        "phase": PHASE,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "component_name": "total",
        "bound": data["overall_bound"],
        "flops": data["total_flops"],
        "bytes_accessed": data["total_bytes"],
        "operational_intensity": data["oi"],
        "ridge_point": data["ridge_point"],
    })

    for comp in data["components"]:
        rows.append({
            "record_type": "analytical_component",
            "model": MODEL,
            "hardware": HARDWARE,
            "phase": PHASE,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "component_name": comp["name"],
            "bound": comp["bound"],
            "flops": comp["flops"],
            "bytes_accessed": comp["bytes"],
            "operational_intensity": comp["oi"],
            "ridge_point": None,
        })

    return rows


def parse_ncu_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    kernels: dict[str, dict] = {}
    for row in raw_rows:
        kid = row["ID"]
        section = row["Section Name"]
        metric = row["Metric Name"]
        value = row["Metric Value"]

        if kid not in kernels:
            kernels[kid] = {
                "kernel_id": int(kid),
                "kernel_name": row["Kernel Name"],
                "block_size": None,
                "grid_size": None,
            }

        key = (section, metric)
        if key in NCU_METRIC_MAP:
            col = NCU_METRIC_MAP[key]
            try:
                kernels[kid][col] = float(value.replace(",", ""))
            except (ValueError, TypeError):
                kernels[kid][col] = None

        if section == "Launch Statistics" and metric == "Block Size":
            kernels[kid]["block_size"] = value
        elif section == "Launch Statistics" and metric == "Grid Size":
            kernels[kid]["grid_size"] = value

    rows = []
    for kid in sorted(kernels, key=lambda x: int(x)):
        k = kernels[kid]
        rows.append({
            "record_type": "ncu_kernel",
            "model": MODEL,
            "hardware": HARDWARE,
            "phase": PHASE,
            "batch_size": 1,
            "sequence_length": 512,
            "kernel_id": k["kernel_id"],
            "kernel_name": k["kernel_name"],
            "block_size": k.get("block_size"),
            "grid_size": k.get("grid_size"),
            "duration_us": k.get("duration_us"),
            "compute_sm_throughput_pct": k.get("compute_sm_throughput_pct"),
            "dram_throughput_pct": k.get("dram_throughput_pct"),
            "memory_throughput_pct": k.get("memory_throughput_pct"),
            "l1_tex_cache_throughput_pct": k.get("l1_tex_cache_throughput_pct"),
            "l2_cache_throughput_pct": k.get("l2_cache_throughput_pct"),
            "sm_frequency_ghz": k.get("sm_frequency_ghz"),
            "dram_frequency_ghz": k.get("dram_frequency_ghz"),
        })

    return rows


COLUMNS = [
    "record_type", "model", "hardware", "phase", "batch_size", "sequence_length",
    "component_name", "bound", "flops", "bytes_accessed", "operational_intensity", "ridge_point",
    "kernel_id", "kernel_name", "block_size", "grid_size",
    "duration_us", "compute_sm_throughput_pct", "dram_throughput_pct",
    "memory_throughput_pct", "l1_tex_cache_throughput_pct", "l2_cache_throughput_pct",
    "sm_frequency_ghz", "dram_frequency_ghz",
]


def main():
    parser = argparse.ArgumentParser(description="Build per-layer kernel parquet from R2 data")
    parser.add_argument("--r2-base", type=str, default=R2_BASE, help="R2 base URL")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "hf_dataset",
                        help="Output directory for parquets")
    parser.add_argument("--analytical-dir", type=Path, default=None,
                        help="Local directory with analytical JSONs (skip R2 download)")
    parser.add_argument("--ncu-csv", type=Path, default=None,
                        help="Local path to NCU CSV (skip R2 download)")
    args = parser.parse_args()

    r2_base = args.r2_base.rstrip("/")
    all_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        print("Building per_layer_kernel config...")

        print("  Parsing analytical JSONs...")
        for rel_path in ANALYTICAL_FILES:
            if args.analytical_dir:
                local = args.analytical_dir / Path(rel_path).name
            else:
                local = download(f"{r2_base}/{rel_path}", tmp / Path(rel_path).name)
            all_rows.extend(parse_analytical(local))

        print("  Parsing NCU CSV...")
        if args.ncu_csv:
            ncu_path = args.ncu_csv
        else:
            ncu_path = download(f"{r2_base}/{NCU_CSV_PATH}", tmp / "ncu.csv")
        all_rows.extend(parse_ncu_csv(ncu_path))

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["batch_size"] = df["batch_size"].astype("int64")
    df["sequence_length"] = df["sequence_length"].astype("int64")

    output_path = args.output_dir / "per_layer_kernel" / "summary.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"  Written: {output_path}")
    print(f"  Rows: {len(df)} ({len([r for r in all_rows if r['record_type'].startswith('analytical')])} analytical + {len([r for r in all_rows if r['record_type'] == 'ncu_kernel'])} ncu)")
    print(f"  Columns: {len(df.columns)}")

    analytical = df[df["record_type"].str.startswith("analytical")]
    ncu = df[df["record_type"] == "ncu_kernel"]
    print(f"  Analytical batch sizes: {sorted(analytical['batch_size'].unique())}")
    print(f"  NCU kernels: {len(ncu)}")


if __name__ == "__main__":
    main()
