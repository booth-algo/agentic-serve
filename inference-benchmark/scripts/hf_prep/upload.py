"""Upload hf_dataset/ directory to HuggingFace Hub."""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument("--repo", type=str, default="agent-perf-bench/AgentPerfBench")
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "hf_dataset",
                        help="Path to hf_dataset/ directory")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: No token provided. Use --token or set HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    if not args.dataset_dir.exists():
        print(f"Error: {args.dataset_dir} does not exist. Run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    expected_files = [
        "trace_replay/summary.parquet",
        "distributional/summary.parquet",
        "kernel_profiles/kernels_labeled.parquet",
    ]
    missing = [f for f in expected_files if not (args.dataset_dir / f).exists()]
    if missing:
        print(f"Warning: missing expected files: {missing}", file=sys.stderr)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    print(f"Uploading {args.dataset_dir} to {args.repo}")

    api.upload_folder(
        folder_path=str(args.dataset_dir),
        repo_id=args.repo,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{args.repo}"
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
