"""Build kernel_profiles.parquet from parsed roofline JSON files."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def profile_id(model: str, phase: str, batch_size: int, seq_len: int) -> str:
    key = f"{model}|{phase}|{batch_size}|{seq_len}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def load_parsed_jsons(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)

        pid = profile_id(
            data["model"], data["phase"], data["batch_size"], data["seq_len"]
        )

        for k in data["kernels"]:
            rows.append({
                "profile_id": pid,
                "model": data["model"],
                "phase": data["phase"],
                "batch_size": data["batch_size"],
                "seq_len": data["seq_len"],
                "num_layers_profiled": data["num_layers_profiled"],
                "profiler": data["profiler"],
                "kernel_name": k["name"],
                "kernel_category": k["category"],
                "cuda_time_us": k["cuda_time_us"],
                "calls": k["calls"],
                "flops": k["flops"],
                "theoretical_bytes": k["theoretical_bytes"],
                "arithmetic_intensity": k["arithmetic_intensity"],
                "achieved_tflops": k["achieved_tflops"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/roofline/parsed"))
    parser.add_argument("--output", type=Path, default=Path("hf_dataset/roofline/kernel_profiles.parquet"))
    args = parser.parse_args()

    rows = load_parsed_jsons(args.input_dir)
    df = pd.DataFrame(rows)

    df = df.astype({
        "profile_id": "str",
        "model": "str",
        "phase": "str",
        "batch_size": "int64",
        "seq_len": "int64",
        "num_layers_profiled": "int64",
        "profiler": "str",
        "kernel_name": "str",
        "kernel_category": "str",
        "cuda_time_us": "float64",
        "calls": "int64",
        "flops": "int64",
        "theoretical_bytes": "int64",
        "arithmetic_intensity": "float64",
        "achieved_tflops": "float64",
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"Written: {args.output}")
    print(f"  Profiles: {df['profile_id'].nunique()}")
    print(f"  Models:   {df['model'].nunique()} ({', '.join(sorted(df['model'].unique()))})")
    print(f"  Rows:     {len(df)}")
    print(f"  Columns:  {list(df.columns)}")


if __name__ == "__main__":
    main()
