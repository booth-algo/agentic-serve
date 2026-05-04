"""Build kernel profiling parquets from R2 data."""

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://get-data.kevinlauofficial01.workers.dev/"

KERNELS_SCHEMA = {
    "source": "string",
    "gpu": "string",
    "model": "string",
    "kernel_family": "string",
    "kernel_name": "string",
    "dtype": "string",
    "held_out": "bool",
    "M": "float64",
    "N": "float64",
    "K": "float64",
    "bs": "float64",
    "seq": "float64",
    "n_heads": "float64",
    "head_dim": "float64",
    "kv_heads": "float64",
    "numel": "float64",
    "op_type": "string",
    "gpu_time_duration_ms": "float64",
    "launch_block_size": "float64",
    "launch_grid_size": "float64",
    "dram_bytes_sum": "float64",
    "launch_registers_per_thread": "float64",
}

ROOFLINE_SCHEMA = {
    "model": "string",
    "profile": "string",
    "concurrency": "int64",
    "engine": "string",
    "hardware": "string",
    "oi": "float64",
    "cf_gb": "float64",
    "output_tput": "float64",
    "tpot_ms": "float64",
    "ttft_ms": "float64",
}


def download(url: str, dest: Path) -> Path:
    print(f"Downloading {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgentPerfBench/1.0"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)
    except urllib.error.URLError as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        sys.exit(1)
    return dest


def build_kernels_labeled(csv_path: Path, output_path: Path) -> pd.DataFrame:
    # kernel_name values contain commas and angle brackets that break the C parser
    df = pd.read_csv(csv_path, engine="python")
    df = df.astype({col: dtype for col, dtype in KERNELS_SCHEMA.items() if dtype != "bool"})
    df["held_out"] = df["held_out"].astype("bool")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Written: {output_path}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    return df


def build_roofline_quadrant(json_path: Path, output_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["points"])
    df = df[list(ROOFLINE_SCHEMA.keys())]
    df = df.astype(ROOFLINE_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Written: {output_path}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build kernel profiling parquets from R2 data")
    parser.add_argument("--kernels-csv", type=Path, default=None, help="Path to kernels_labeled.csv")
    parser.add_argument("--roofline-json", type=Path, default=None, help="Path to roofline-quadrant.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "hf_dataset",
                        help="Output directory for parquets")
    parser.add_argument("--r2-base", type=str, default=R2_BASE, help="R2 base URL")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        kernels_csv = args.kernels_csv
        if kernels_csv is None:
            kernels_csv = download(
                args.r2_base.rstrip("/") + "/profiling-data/kernels_labeled.csv",
                tmp / "kernels_labeled.csv",
            )

        roofline_json = args.roofline_json
        if roofline_json is None:
            roofline_json = download(
                args.r2_base.rstrip("/") + "/roofline-quadrant.json",
                tmp / "roofline-quadrant.json",
            )

        build_kernels_labeled(kernels_csv, args.output_dir / "kernel_profiles" / "kernels_labeled.parquet")
        build_roofline_quadrant(roofline_json, args.output_dir / "kernel_profiles" / "roofline_quadrant.parquet")


if __name__ == "__main__":
    main()
