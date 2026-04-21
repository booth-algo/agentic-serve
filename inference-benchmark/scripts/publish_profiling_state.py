#!/usr/bin/env python3
"""Build profiling-state.json from profiling_manifest.yaml and write to dashboard/public/.

Reads llm_predict/training/per_kernel/profiling_manifest.yaml (produced by Phase 1).
If the manifest does not exist, prints a warning and leaves the existing stub JSON in place.

The manifest is nested (gpus -> model -> per_kernel|per_op -> status_entry); the dashboard
consumes a flat cells list, so build_state() walks the nested dict and emits one cell per
(gpu, model) pair with four status columns.

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
# scripts/ is inside inference-benchmark/ ; so HERE.parent is inference-benchmark/.
MANIFEST = HERE.parent.parent / "llm_predict" / "training" / "per_kernel" / "profiling_manifest.yaml"
OUTPUT_FILE = HERE.parent / "dashboard" / "public" / "profiling-state.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "profiling-state.json"


def _normalise(raw: object, keep: tuple[str, ...]) -> dict:
    """Reduce a manifest status entry to the dashboard's minimal schema:
    {status, reason?, rows?, version?}. Accepts either a plain string status
    or a dict with extra fields."""
    if isinstance(raw, str):
        return {"status": raw}
    if isinstance(raw, dict):
        out: dict = {"status": raw.get("status", "missing")}
        for k in keep:
            v = raw.get(k)
            if v is not None:
                out[k] = v
        return out
    return {"status": "missing"}


def build_state(manifest: dict) -> dict:
    """Walk manifest['gpus'][gpu][model] and emit a flat cells list."""
    gpus_data = manifest.get("gpus", {})
    if not isinstance(gpus_data, dict):
        raise ValueError(
            "manifest['gpus'] must be a nested dict (gpu -> model -> ...); "
            f"got {type(gpus_data).__name__}"
        )

    gpus = sorted(gpus_data.keys())
    all_models: set[str] = set()
    for models_data in gpus_data.values():
        if isinstance(models_data, dict):
            all_models.update(models_data.keys())
    models = sorted(all_models)

    cells: list[dict] = []
    for gpu in gpus:
        models_data = gpus_data.get(gpu, {})
        if not isinstance(models_data, dict):
            continue
        for model in sorted(models_data.keys()):
            entry = models_data[model] if isinstance(models_data[model], dict) else {}
            per_kernel = entry.get("per_kernel", {}) if isinstance(entry, dict) else {}
            per_op = entry.get("per_op", {}) if isinstance(entry, dict) else {}
            cells.append({
                "gpu": gpu,
                "model": model,
                "per_kernel_prefill":  _normalise(per_kernel.get("prefill_seq128_bs1"), ("reason", "rows")),
                "per_kernel_roofline": _normalise(per_kernel.get("roofline_sweep"), ("reason", "rows")),
                "per_op_cuda_events":  _normalise(per_op.get("cuda_events"), ("reason",)),
                "per_op_trained_pkl":  _normalise(per_op.get("trained_pkl"), ("reason", "version")),
            })

    return {
        "generated_at": manifest.get("generated_at") or datetime.now(timezone.utc).isoformat(),
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
            f"WARNING: manifest not found at {args.manifest} -- "
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
    print(f"wrote {args.out} ({len(state['cells'])} cells)", file=sys.stderr)

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
