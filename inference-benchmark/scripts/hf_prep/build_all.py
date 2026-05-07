"""Orchestrate the full HuggingFace dataset build pipeline."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://get-data.kevinlauofficial01.workers.dev/"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent.parent / "hf_dataset"
DEFAULT_REPO = "agent-perf-bench/AgentPerfBench"

EXPECTED_PARQUETS = [
    "trace_replay/summary.parquet",
    "synthetic_distributional/summary.parquet",
    "kernel_profiles/kernels_labeled.parquet",
    "per_layer_kernel/summary.parquet",
]


def run_script(name: str, args: list[str]):
    script = SCRIPT_DIR / name
    cmd = [sys.executable, str(script)] + args
    print(f"\n>>> {name} {' '.join(args)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"Error: {name} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def validate(output_dir: Path) -> list[tuple[str, int]]:
    results = []
    missing = []
    for rel in EXPECTED_PARQUETS:
        path = output_dir / rel
        if not path.exists():
            missing.append(rel)
            continue
        df = pd.read_parquet(path)
        if len(df) == 0:
            print(f"Warning: {rel} has 0 rows", file=sys.stderr)
        results.append((rel, len(df)))
    if missing:
        print(f"Error: missing parquets: {missing}", file=sys.stderr)
        sys.exit(1)
    return results


PARQUET_TO_CROISSANT_ID = {
    "trace_replay/summary.parquet": "trace-replay-summary-parquet",
    "synthetic_distributional/summary.parquet": "synthetic-distributional-summary-parquet",
    "kernel_profiles/kernels_labeled.parquet": "kernels-labeled-parquet",
    "mse_validation/summary.parquet": "mse-validation-summary-parquet",
    "per_layer_kernel/summary.parquet": "per-layer-kernel-summary-parquet",
}


def rehash_croissant(output_dir: Path):
    croissant_path = output_dir / "croissant.json"
    if not croissant_path.exists():
        print("  No croissant.json found, skipping")
        return
    with open(croissant_path) as f:
        croissant = json.load(f)

    id_to_hash = {}
    for rel_path, croissant_id in PARQUET_TO_CROISSANT_ID.items():
        path = output_dir / rel_path
        if path.exists():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            id_to_hash[croissant_id] = h.hexdigest()

    updated = 0
    for dist in croissant.get("distribution", []):
        file_id = dist.get("@id", "")
        if file_id in id_to_hash:
            dist["sha256"] = id_to_hash[file_id]
            updated += 1

    with open(croissant_path, "w") as f:
        json.dump(croissant, f, indent=2)
        f.write("\n")
    print(f"  Updated {updated} SHA256 hashes in croissant.json")


def main():
    parser = argparse.ArgumentParser(description="Build all HuggingFace dataset parquets")
    parser.add_argument("--r2-base", type=str, default=R2_BASE, help="R2 base URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after build")
    parser.add_argument("--token", type=str, default=None, help="HF token for upload")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="HF repo id")
    args = parser.parse_args()

    r2_base = args.r2_base.rstrip("/")
    output_dir = args.output_dir.resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_json = Path(tmpdir) / "data.json"
        print(f"Downloading data.json from {r2_base}")
        req = urllib.request.Request(f"{r2_base}/data.json", headers={"User-Agent": "AgentPerfBench/1.0"})
        with urllib.request.urlopen(req) as resp, open(data_json, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)
        print(f"  Saved to {data_json}")

        run_script("build_dataset.py", [
            "--data-json", str(data_json),
            "--output-dir", str(output_dir),
        ])

    run_script("build_kernels.py", [
        "--r2-base", r2_base,
        "--output-dir", str(output_dir),
    ])

    run_script("build_perkernel.py", [
        "--r2-base", r2_base,
        "--output-dir", str(output_dir),
    ])

    print("\nValidating parquets...")
    results = validate(output_dir)

    print("\nRecomputing croissant hashes for all parquets...")
    rehash_croissant(output_dir)

    if args.upload:
        upload_args = ["--repo", args.repo, "--dataset-dir", str(output_dir)]
        if args.token:
            upload_args += ["--token", args.token]
        run_script("upload.py", upload_args)

    print("\nBuild summary:")
    for rel, rows in results:
        print(f"  {rel}: {rows} rows")
    print(f"  Total: {sum(r for _, r in results)} rows across {len(results)} files")


if __name__ == "__main__":
    main()
