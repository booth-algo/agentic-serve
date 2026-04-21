#!/usr/bin/env python3
"""Build profiling-state.json from profiling_manifest.yaml and write to dashboard/public/.

Reads llm_predict/training/per_kernel/profiling_manifest.yaml (produced by Phase 1).
If the manifest does not exist, prints a warning and leaves the existing stub JSON in place.

Run (no upload):
    python scripts/publish_profiling_state.py --no-upload

Run (upload to R2):
    python scripts/publish_profiling_state.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "llm_predict" / "training" / "per_kernel" / "profiling_manifest.yaml"
OUTPUT_FILE = HERE.parent / "inference-benchmark" / "dashboard" / "public" / "profiling-state.json"

# Resolve OUTPUT_FILE relative to repo root (scripts/ is inside inference-benchmark/)
OUTPUT_FILE = HERE.parent / "dashboard" / "public" / "profiling-state.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "profiling-state.json"


def build_state(manifest: dict) -> dict:
    gpus: list[str] = manifest.get("gpus", [])
    models: list[str] = manifest.get("models", [])
    cells_raw: list[dict] = manifest.get("cells", [])

    cells = []
    for entry in cells_raw:
        def parse_status(raw: object) -> dict:
            if isinstance(raw, str):
                return {"status": raw}
            if isinstance(raw, dict):
                return raw
            return {"status": "missing"}

        cells.append({
            "gpu": str(entry["gpu"]),
            "model": str(entry["model"]),
            "per_kernel_prefill":  parse_status(entry.get("per_kernel_prefill",  "missing")),
            "per_kernel_roofline": parse_status(entry.get("per_kernel_roofline", "missing")),
            "per_op_cuda_events":  parse_status(entry.get("per_op_cuda_events",  "missing")),
            "per_op_trained_pkl":  parse_status(entry.get("per_op_trained_pkl",  "missing")),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "models": models,
        "cells": cells,
    }


def upload_r2(path: Path, endpoint: str, bucket: str, profile: str) -> None:
    cmd = [
        "aws", "--profile", profile, "s3", "cp",
        str(path), f"s3://{bucket}/{R2_KEY}",
        "--endpoint-url", endpoint,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-upload", action="store_true", help="skip R2 upload")
    ap.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = ap.parse_args()

    if not args.manifest.exists():
        print(
            f"WARNING: manifest not found at {args.manifest} — "
            "leaving existing profiling-state.json stub in place.",
            file=sys.stderr,
        )
        return 0

    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        return 1

    manifest = yaml.safe_load(args.manifest.read_text())
    state = build_state(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    print(f"wrote {args.out} ({len(state[cells])} cells)", file=sys.stderr)

    if not args.no_upload:
        try:
            upload_r2(args.out, args.endpoint, args.bucket, args.profile)
            print(f"uploaded to s3://{args.bucket}/{R2_KEY}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"R2 upload failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
