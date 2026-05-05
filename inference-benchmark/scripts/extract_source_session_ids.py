#!/usr/bin/env python3
"""Extract unique source session IDs from a benchmark result JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _request_source_id(row: dict) -> str | None:
    metadata = row.get("request_metadata") or {}
    source_id = metadata.get("source_session_id")
    if source_id is None:
        source_id = metadata.get("sampled_source_session_id")
    return str(source_id) if source_id else None


def extract_source_session_ids(result_path: Path) -> list[str]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload.get("per_request")
    if not isinstance(rows, list):
        raise ValueError(f"{result_path} does not contain a per_request list")

    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = _request_source_id(row)
        if not source_id or source_id in seen:
            continue
        ids.append(source_id)
        seen.add(source_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract newline-delimited source_session_id values from a benchmark "
            "result JSON, preserving first-seen session order."
        )
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    try:
        ids = extract_source_session_ids(args.result_json)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not ids:
        print(f"error: no source session IDs found in {args.result_json}", file=sys.stderr)
        return 1

    text = "\n".join(ids) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {len(ids)} IDs to {args.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
