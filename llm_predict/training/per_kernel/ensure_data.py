"""Auto-download kernels_labeled.csv from R2 when missing."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_R2_ENDPOINT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
_R2_BUCKET = "agent-bench"
_R2_KEY = "profiling-data/kernels_labeled.csv"
_AWS_PROFILE = "r2"


def _try_aws(dest: Path) -> bool:
    if not shutil.which("aws"):
        return False
    try:
        subprocess.run(
            [
                "aws", "--profile", _AWS_PROFILE,
                "--endpoint-url", _R2_ENDPOINT,
                "s3", "cp", f"s3://{_R2_BUCKET}/{_R2_KEY}", str(dest),
            ],
            check=True, capture_output=True, text=True,
        )
        return dest.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_kernels_csv(data_csv: Path) -> Path:
    """Return *data_csv* if it exists, else download from R2.

    Tries ``aws s3 cp`` first. If that fails, exits with a clear
    error message showing the exact commands to run.
    """
    if data_csv.exists():
        return data_csv

    print(f"[!] {data_csv} not found — attempting download from R2 ...")
    data_csv.parent.mkdir(parents=True, exist_ok=True)

    if _try_aws(data_csv):
        print(f"[+] Downloaded {data_csv} ({data_csv.stat().st_size / 1e6:.1f} MB)")
        return data_csv

    msg = f"""\
[ERROR] kernels_labeled.csv is missing and auto-download failed.

The file is stored on Cloudflare R2 (gitignored, ~46 MB).
Download it using ONE of these methods:

  # Method 1: pull_pkls.sh (from a host with aws CLI + R2 profile)
  bash llm_predict/training/per_kernel/pull_pkls.sh

  # Method 2: aws CLI directly
  aws --profile {_AWS_PROFILE} --endpoint-url {_R2_ENDPOINT} \\
      s3 cp s3://{_R2_BUCKET}/{_R2_KEY} {data_csv}

  # Method 3: rsync from another host that has it
  rsync -az <host>:~/agentic-serve/llm_predict/training/per_kernel/data/kernels_labeled.csv \\
      {data_csv}
"""
    print(msg, file=sys.stderr)
    raise SystemExit(1)
