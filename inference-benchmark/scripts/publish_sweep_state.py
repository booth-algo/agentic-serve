#!/usr/bin/env python3
"""Build sweep-state.json from sweep.yaml + orchestrator state files, upload to R2.

For each cell in sweep.yaml (including `known_oom` entries), produce a status
record by reading /tmp/bench_jobs/state/<job_id>.status. Cells without a state
file get status "pending". known_oom entries override runtime status.

Output is written to dashboard/public/sweep-state.json and optionally uploaded
to R2 at s3://agent-bench/sweep-state.json so the live dashboard can fetch it
from https://pub-38e30ed030784867856634f1625c7130.r2.dev/sweep-state.json.

Run locally (no upload):
    python scripts/publish_sweep_state.py --no-upload

Run from orchestrator tick (default — uploads):
    python scripts/publish_sweep_state.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SWEEP_YAML = HERE / "sweep.yaml"
STATE_DIR = Path("/tmp/bench_jobs/state")
OUTPUT_FILE = HERE.parent / "dashboard" / "public" / "sweep-state.json"

R2_ENDPOINT_DEFAULT = "https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
R2_BUCKET_DEFAULT = "agent-bench"
R2_KEY = "sweep-state.json"


def hw_label(host_cfg: dict, tp: int) -> str:
    base = str(host_cfg["hardware_label"])
    return base if tp == 1 else f"{base}x{tp}"


def job_id(host: str, model: str, tp: int, mode: str) -> str:
    return f"{host}_{model}_tp{tp}_{mode}"


def read_state(jid: str) -> dict:
    out: dict = {"status": "pending", "attempt": 0, "max_len_override": None, "updated_at": None}
    p = STATE_DIR / f"{jid}.status"
    if p.exists():
        out["status"] = p.read_text().strip() or "pending"
        out["updated_at"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    att = STATE_DIR / f"{jid}.attempt"
    if att.exists():
        try:
            out["attempt"] = int(att.read_text().strip())
        except ValueError:
            pass
    ov = STATE_DIR / f"{jid}.max_len_override"
    if ov.exists():
        try:
            out["max_len_override"] = int(ov.read_text().strip())
        except ValueError:
            pass
    return out


def build_state(manifest: dict) -> dict:
    # Normalize host keys to strings — YAML parses unquoted `3090:` as int
    # but cells reference hosts by string name.
    hosts = {str(k): v for k, v in manifest["hosts"].items()}
    cells = []

    # One record per sweep cell, augmented with runtime state.
    for cell in manifest["cells"]:
        host_name = str(cell["host"])
        host_cfg = hosts[host_name]
        tp = int(cell["tp"])
        mode = str(cell["mode"])
        model = str(cell["model"])
        jid = job_id(host_name, model, tp, mode)
        rt = read_state(jid)

        cells.append({
            "host": host_name,
            "hw_label": hw_label(host_cfg, tp),
            "model": model,
            "tp": tp,
            "mode": mode,
            "status": rt["status"],
            "attempt": rt["attempt"],
            "max_len_override": rt["max_len_override"],
            "reason": None,
            "updated_at": rt["updated_at"],
        })

    # Known-OOM entries override any runtime state for matching cells, and are
    # appended as new (host, model, tp) entries if no sweep cell exists yet.
    for entry in manifest.get("known_oom", []):
        host_name = str(entry["host"])
        host_cfg = hosts[host_name]
        tp = int(entry["tp"])
        label = hw_label(host_cfg, tp)
        model = str(entry["model"])
        matched = False
        for c in cells:
            if c["host"] == host_name and c["model"] == model and c["tp"] == tp:
                c["status"] = "known_oom"
                c["reason"] = entry["reason"]
                matched = True
        if not matched:
            cells.append({
                "host": host_name,
                "hw_label": label,
                "model": model,
                "tp": tp,
                "mode": "single",
                "status": "known_oom",
                "attempt": 0,
                "max_len_override": None,
                "reason": entry["reason"],
                "updated_at": None,
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feasibility_ratio": float(manifest["feasibility_ratio"]),
        "hosts": {
            h: {
                "hardware_label": str(cfg["hardware_label"]),
                "vram_gb_per_gpu": int(cfg["vram_gb_per_gpu"]),
                "total_gpus": int(cfg["total_gpus"]),
            }
            for h, cfg in hosts.items()
        },
        "models": {
            m: {"weights_gb": int(cfg["weights_gb"])}
            for m, cfg in manifest["models"].items()
        },
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=SWEEP_YAML)
    ap.add_argument("--out", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-upload", action="store_true", help="skip R2 upload")
    ap.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT", R2_ENDPOINT_DEFAULT))
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", R2_BUCKET_DEFAULT))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "r2"))
    args = ap.parse_args()

    manifest = yaml.safe_load(args.yaml.read_text())
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
