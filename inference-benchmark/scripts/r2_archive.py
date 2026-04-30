#!/usr/bin/env python3
"""Archive generated R2 JSON artifacts without deleting live objects.

This helper is intentionally conservative:

- default mode is dry-run
- it only copies generated dashboard/state JSON artifacts
- it never deletes source objects

Raw benchmark result directories are much larger and should be archived with a
separate, explicit command after an inventory/verification pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"


ROOT_JSON_KEYS = [
    "data.json",
    "gemm-extrapolation.json",
    "predictor-coverage.json",
    "profiling-state.json",
    "roofline-quadrant.json",
    "sweep-state.json",
]


REMOTE_OPTIONAL_JSON_KEYS = [
    ("predictor/gemm-eval.json", "gemm-eval.json"),
]


LOCAL_OPTIONAL_JSONS = [
    "inference-benchmark/dashboard/public/gemm-eval.json",
    "inference-benchmark/dashboard/public/serving-predictions.json",
]


def _run(cmd: list[str], execute: bool) -> None:
    print(" ".join(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def copy_remote_key(
    *,
    bucket: str,
    endpoint: str,
    profile: str,
    src_key: str,
    dst_key: str,
    execute: bool,
) -> None:
    cache_dir = Path("/tmp") / "r2-json-archive-cache"
    cache_path = cache_dir / src_key.replace("/", "__")
    download_cmd = [
        "aws",
        "--profile",
        profile,
        "--endpoint-url",
        endpoint,
        "s3",
        "cp",
        _s3_uri(bucket, src_key),
        str(cache_path),
    ]
    upload_cmd = [
        "aws",
        "--profile",
        profile,
        "--endpoint-url",
        endpoint,
        "s3",
        "cp",
        str(cache_path),
        _s3_uri(bucket, dst_key),
    ]
    if execute:
        cache_dir.mkdir(parents=True, exist_ok=True)
    _run(download_cmd, execute)
    _run(upload_cmd, execute)


def copy_local_file(
    *,
    bucket: str,
    endpoint: str,
    profile: str,
    src_path: Path,
    dst_key: str,
    execute: bool,
) -> None:
    cmd = [
        "aws",
        "--profile",
        profile,
        "--endpoint-url",
        endpoint,
        "s3",
        "cp",
        str(src_path),
        _s3_uri(bucket, dst_key),
    ]
    _run(cmd, execute)


def write_manifest(
    *,
    bucket: str,
    endpoint: str,
    profile: str,
    archive_name: str,
    copied_keys: list[dict],
    execute: bool,
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_name": archive_name,
        "description": "Generated dashboard JSON archive before R2 layout cleanup.",
        "copied_keys": copied_keys,
        "notes": [
            "This archive helper copies JSON artifacts only.",
            "No source objects are deleted.",
            "Raw benchmark results require a separate explicit archive pass.",
        ],
    }
    tmp_path = Path("/tmp") / f"r2-json-archive-manifest-{archive_name}.json"
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    copy_local_file(
        bucket=bucket,
        endpoint=endpoint,
        profile=profile,
        src_path=tmp_path,
        dst_key=f"json/archive/{archive_name}/manifest.json",
        execute=execute,
    )


def archive_jsons(args: argparse.Namespace) -> int:
    copied: list[dict] = []
    archive_prefix = f"json/archive/{args.archive_name}"

    for key in ROOT_JSON_KEYS:
        dst_name = key.split("/")[-1]
        archive_key = f"{archive_prefix}/{dst_name}"
        current_key = f"json/current/{dst_name}"
        copied.append({"source": key, "archive": archive_key, "current": current_key})
        copy_remote_key(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_key=key,
            dst_key=archive_key,
            execute=args.execute,
        )
        copy_remote_key(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_key=key,
            dst_key=current_key,
            execute=args.execute,
        )

    local_optional = [Path(p) for p in LOCAL_OPTIONAL_JSONS]
    local_names = {p.name for p in local_optional if p.is_file()}
    for key, dst_name in REMOTE_OPTIONAL_JSON_KEYS:
        if dst_name in local_names:
            print(f"# skip remote {key}: local {dst_name} will be archived", file=sys.stderr)
            continue
        archive_key = f"{archive_prefix}/{dst_name}"
        current_key = f"json/current/{dst_name}"
        copied.append({"source": key, "archive": archive_key, "current": current_key})
        copy_remote_key(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_key=key,
            dst_key=archive_key,
            execute=args.execute,
        )
        copy_remote_key(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_key=key,
            dst_key=current_key,
            execute=args.execute,
        )

    for src_path in local_optional:
        if not src_path.is_file():
            print(f"# skip missing local optional artifact: {src_path}", file=sys.stderr)
            continue
        dst_name = src_path.name
        archive_key = f"{archive_prefix}/{dst_name}"
        current_key = f"json/current/{dst_name}"
        copied.append({"source": str(src_path), "archive": archive_key, "current": current_key})
        copy_local_file(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_path=src_path,
            dst_key=archive_key,
            execute=args.execute,
        )
        copy_local_file(
            bucket=args.bucket,
            endpoint=args.endpoint,
            profile=args.profile,
            src_path=src_path,
            dst_key=current_key,
            execute=args.execute,
        )

    write_manifest(
        bucket=args.bucket,
        endpoint=args.endpoint,
        profile=args.profile,
        archive_name=args.archive_name,
        copied_keys=copied,
        execute=args.execute,
    )

    if not args.execute:
        print("\n# dry-run only; add --execute to copy these objects", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-name",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d-pre-distributional"),
        help="archive directory name under json/archive/",
    )
    parser.add_argument("--execute", action="store_true", help="perform copies; default is dry-run")
    parser.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    parser.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = parser.parse_args()
    return archive_jsons(args)


if __name__ == "__main__":
    raise SystemExit(main())
