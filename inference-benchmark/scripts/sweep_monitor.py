#!/usr/bin/env python3
"""Deterministic benchmark sweep monitor.

Compares three layers:

1. Local desired/runtime state generated from sweep.yaml + /tmp state files.
2. Local dashboard artifacts in dashboard/public.
3. Published dashboard artifacts in R2 json/current, which the website reads.

The script is read-only. It does not dispatch jobs, publish JSON, or edit state.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import publish_sweep_state


HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
REPO_ROOT = BENCH_ROOT.parent

DEFAULT_PUBLIC_BASE = "https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current"
DEFAULT_LOCAL_DATA = BENCH_ROOT / "dashboard" / "public" / "data.json"
DEFAULT_LOCAL_STATE_FILE = BENCH_ROOT / "dashboard" / "public" / "sweep-state.json"
DEFAULT_SWEEP_YAML = HERE / "sweep.yaml"
DEFAULT_STATE_DIR = Path("/tmp/bench_jobs/state")

CellKey = tuple[str, str, int, str, str]  # host, model, tp, mode, backend
PointKey = tuple[str, str, str, str, str, int]  # hw, model, backend, mode, profile, conc
ProfileKey = tuple[str, str, int, str, str, str]  # host, model, tp, mode, backend, profile


@dataclass(frozen=True)
class JsonSource:
    label: str
    ref: str
    ok: bool
    error: str | None = None
    bytes_read: int = 0


def is_url(ref: str) -> bool:
    return ref.startswith("http://") or ref.startswith("https://")


def load_json_ref(ref: str, label: str, timeout: float) -> tuple[Any | None, JsonSource]:
    try:
        if is_url(ref):
            req = urllib.request.Request(ref, headers={"User-Agent": "agentic-serve-sweep-monitor/1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        else:
            raw = Path(ref).read_bytes()
        return json.loads(raw), JsonSource(label=label, ref=ref, ok=True, bytes_read=len(raw))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, JsonSource(label=label, ref=ref, ok=False, error=str(exc))


def load_local_generated_state(sweep_yaml: Path, state_dir: Path) -> dict[str, Any]:
    publish_sweep_state.STATE_DIR = state_dir
    manifest = yaml.safe_load(sweep_yaml.read_text())
    return publish_sweep_state.build_state(manifest)


def cell_key(cell: dict[str, Any]) -> CellKey:
    return (
        str(cell.get("host", "")),
        str(cell.get("model", "")),
        int(cell.get("tp", 0) or 0),
        str(cell.get("mode", "")),
        str(cell.get("backend", "vllm")),
    )


def profile_key(item: dict[str, Any]) -> ProfileKey:
    return (
        str(item.get("host", "")),
        str(item.get("model", "")),
        int(item.get("tp", 0) or 0),
        str(item.get("mode", "")),
        str(item.get("backend", "vllm")),
        str(item.get("profile", "")),
    )


def index_cells(state: dict[str, Any]) -> tuple[dict[CellKey, dict[str, Any]], dict[CellKey, int]]:
    grouped: dict[CellKey, list[dict[str, Any]]] = defaultdict(list)
    for cell in state.get("cells", []):
        grouped[cell_key(cell)].append(cell)
    indexed = {key: items[-1] for key, items in grouped.items()}
    duplicates = {key: len(items) for key, items in grouped.items() if len(items) > 1}
    return indexed, duplicates


def mode_to_data_mode(mode: str) -> str:
    if mode == "single":
        return "single-turn"
    if mode == "multi":
        return "multi-turn"
    return mode


def expected_points_from_state(state: dict[str, Any]) -> set[PointKey]:
    blocked_profiles = {profile_key(item) for item in state.get("profile_infeasible", [])}
    points: set[PointKey] = set()
    for cell in state.get("cells", []):
        if cell.get("status") == "known_oom":
            continue
        host = str(cell.get("host", ""))
        model = str(cell.get("model", ""))
        tp = int(cell.get("tp", 0) or 0)
        mode = str(cell.get("mode", ""))
        backend = str(cell.get("backend", "vllm"))
        hw = str(cell.get("hw_label", ""))
        data_mode = mode_to_data_mode(mode)
        profiles = [str(p) for p in cell.get("profiles") or []]
        concurrencies = [int(c) for c in cell.get("concurrencies") or []]
        for profile in profiles:
            if (host, model, tp, mode, backend, profile) in blocked_profiles:
                continue
            for conc in concurrencies:
                points.add((hw, model, backend, data_mode, profile, conc))
    return points


def data_points(data: list[dict[str, Any]], scope: str) -> set[PointKey]:
    points: set[PointKey] = set()
    for row in data:
        if scope != "all" and row.get("dataScope") != scope:
            continue
        cfg = row.get("config") or {}
        hardware = row.get("hardware")
        model = row.get("modelShort") or row.get("model_short")
        backend = cfg.get("backend") or "vllm"
        mode = cfg.get("mode")
        profile = cfg.get("profile")
        conc = cfg.get("concurrency")
        if not hardware or not model or not mode or not profile or conc is None:
            continue
        try:
            conc_i = int(conc)
        except (TypeError, ValueError):
            continue
        points.add((str(hardware), str(model), str(backend), str(mode), str(profile), conc_i))
    return points


def data_scope_counts(data: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("dataScope") or "unknown") for row in data)


def state_status_counts(state: dict[str, Any]) -> Counter[str]:
    return Counter(str(cell.get("status") or "unknown") for cell in state.get("cells", []))


def format_cell_key(key: CellKey) -> str:
    host, model, tp, mode, backend = key
    return f"{host}/{model}/tp{tp}/{mode}/{backend}"


def format_point_group(key: tuple[str, str, str, str, str], concs: list[int]) -> str:
    hw, model, backend, mode, profile = key
    shown = ",".join(str(c) for c in concs[:12])
    if len(concs) > 12:
        shown += ",..."
    return f"{hw}/{model}/{backend}/{mode}/{profile}: C={shown}"


def group_points(points: set[PointKey]) -> list[tuple[tuple[str, str, str, str, str], list[int]]]:
    grouped: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for hw, model, backend, mode, profile, conc in points:
        grouped[(hw, model, backend, mode, profile)].append(conc)
    return sorted(
        ((key, sorted(concs)) for key, concs in grouped.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )


def compare_state(local: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    local_cells, local_dups = index_cells(local)
    other_cells, other_dups = index_cells(other)
    local_keys = set(local_cells)
    other_keys = set(other_cells)
    common = local_keys & other_keys
    field_mismatches: list[tuple[CellKey, list[str]]] = []
    for key in sorted(common):
        left = local_cells[key]
        right = other_cells[key]
        changed: list[str] = []
        for field in ("status", "max_len", "gpu_mem"):
            if left.get(field) != right.get(field):
                changed.append(f"{field}: local={left.get(field)!r} other={right.get(field)!r}")
        if sorted(left.get("profiles") or []) != sorted(right.get("profiles") or []):
            changed.append("profiles")
        if sorted(left.get("concurrencies") or []) != sorted(right.get("concurrencies") or []):
            changed.append("concurrencies")
        if changed:
            field_mismatches.append((key, changed))

    local_profiles = {profile_key(item) for item in local.get("profile_infeasible", [])}
    other_profiles = {profile_key(item) for item in other.get("profile_infeasible", [])}
    return {
        "only_local": sorted(local_keys - other_keys),
        "only_other": sorted(other_keys - local_keys),
        "field_mismatches": field_mismatches,
        "local_duplicates": local_dups,
        "other_duplicates": other_dups,
        "profile_only_local": sorted(local_profiles - other_profiles),
        "profile_only_other": sorted(other_profiles - local_profiles),
    }


def generated_at(state: dict[str, Any] | None) -> str:
    if not state:
        return "unavailable"
    return str(state.get("generated_at") or "missing")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def print_list(title: str, items: list[str], limit: int) -> None:
    print(f"- {title}: {len(items)}")
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  - ... {len(items) - limit} more")


def state_drift_score(drift: dict[str, Any]) -> int:
    return (
        len(drift["only_local"])
        + len(drift["only_other"])
        + len(drift["field_mismatches"])
        + len(drift["profile_only_local"])
        + len(drift["profile_only_other"])
    )


def build_report(args: argparse.Namespace) -> int:
    local_state = load_local_generated_state(args.sweep_yaml, args.state_dir)
    local_state_file, local_state_source = load_json_ref(str(args.local_state), "local dashboard sweep-state", args.timeout)
    published_state, published_state_source = load_json_ref(args.published_state, "published R2 sweep-state", args.timeout)
    local_data, local_data_source = load_json_ref(str(args.local_data), "local dashboard data", args.timeout)
    published_data, published_data_source = load_json_ref(args.published_data, "published R2 data", args.timeout)

    if local_state_file is None:
        local_state_file = {"cells": [], "profile_infeasible": []}
    if published_state is None:
        published_state = {"cells": [], "profile_infeasible": []}
    if not isinstance(local_data, list):
        local_data = []
    if not isinstance(published_data, list):
        published_data = []

    local_vs_published = compare_state(local_state, published_state)
    generated_vs_file = compare_state(local_state, local_state_file)

    expected = expected_points_from_state(local_state)
    local_points = data_points(local_data, args.scope)
    published_points = data_points(published_data, args.scope)

    missing_published = expected - published_points
    missing_local = expected - local_points
    local_not_published = local_points - published_points
    published_not_local = published_points - local_points
    published_extra_vs_yaml = published_points - expected

    fetch_errors = [
        source for source in (
            local_state_source,
            published_state_source,
            local_data_source,
            published_data_source,
        )
        if not source.ok
    ]
    drift = state_drift_score(local_vs_published)
    data_drift = len(local_not_published) + len(published_not_local)
    if fetch_errors:
        health = "error"
    elif drift or data_drift:
        health = "degraded"
    elif missing_published:
        health = "incomplete"
    else:
        health = "healthy"

    print("# Sweep Monitor Report")
    print()
    print(f"- generated_at: {now_iso()}")
    print(f"- health: {health}")
    print(f"- scope: {args.scope}")
    print(f"- local sweep yaml: {args.sweep_yaml}")
    print(f"- local state dir: {args.state_dir}")
    print()

    print("## Inputs")
    print(f"- local generated state: {len(local_state.get('cells', []))} cells, generated_at={generated_at(local_state)}")
    print(
        f"- local dashboard sweep-state: {len(local_state_file.get('cells', []))} cells, "
        f"generated_at={generated_at(local_state_file)}"
    )
    print(
        f"- published R2 sweep-state: {len(published_state.get('cells', []))} cells, "
        f"generated_at={generated_at(published_state)}"
    )
    print(f"- local data rows: {len(local_data)} total, scopes={dict(data_scope_counts(local_data))}")
    print(f"- published data rows: {len(published_data)} total, scopes={dict(data_scope_counts(published_data))}")
    for source in (local_state_source, published_state_source, local_data_source, published_data_source):
        suffix = f"{source.bytes_read} bytes" if source.ok else f"ERROR: {source.error}"
        print(f"- {source.label}: {source.ref} ({suffix})")
    print()

    print("## State Drift")
    print(f"- local generated vs published drift items: {drift}")
    print(f"- local generated vs local dashboard file drift items: {state_drift_score(generated_vs_file)}")
    print(f"- local status counts: {dict(state_status_counts(local_state))}")
    print(f"- published status counts: {dict(state_status_counts(published_state))}")
    print_list("cells in local YAML/state but missing from published sweep-state", [format_cell_key(k) for k in local_vs_published["only_local"]], args.limit)
    print_list("cells in published sweep-state but absent from local YAML/state", [format_cell_key(k) for k in local_vs_published["only_other"]], args.limit)
    mismatch_lines = [
        f"{format_cell_key(key)} ({'; '.join(changes)})"
        for key, changes in local_vs_published["field_mismatches"]
    ]
    print_list("cell field mismatches", mismatch_lines, args.limit)
    print_list(
        "profile-infeasible records missing from published sweep-state",
        ["/".join(map(str, k)) for k in local_vs_published["profile_only_local"]],
        args.limit,
    )
    print_list(
        "profile-infeasible records only in published sweep-state",
        ["/".join(map(str, k)) for k in local_vs_published["profile_only_other"]],
        args.limit,
    )
    if local_vs_published["local_duplicates"] or local_vs_published["other_duplicates"]:
        local_dup_lines = [
            f"{format_cell_key(key)} x{count}"
            for key, count in sorted(local_vs_published["local_duplicates"].items())
        ]
        other_dup_lines = [
            f"{format_cell_key(key)} x{count}"
            for key, count in sorted(local_vs_published["other_duplicates"].items())
        ]
        print_list("duplicate local cells", local_dup_lines, args.limit)
        print_list("duplicate published cells", other_dup_lines, args.limit)
    print()

    print("## Coverage Against Local YAML")
    print(f"- expected profile-concurrency points from local YAML/state: {len(expected)}")
    print(f"- present in local data.json: {len(expected & local_points)} / {len(expected)}")
    print(f"- present in published R2 data.json: {len(expected & published_points)} / {len(expected)}")
    print(f"- missing from local data.json: {len(missing_local)}")
    print(f"- missing from published R2 data.json: {len(missing_published)}")
    print(f"- published current points not expected by local YAML/state: {len(published_extra_vs_yaml)}")
    print()

    print("## Published Missing Points")
    for key, concs in group_points(missing_published)[:args.limit]:
        print(f"- {format_point_group(key, concs)}")
    if len(group_points(missing_published)) > args.limit:
        print(f"- ... {len(group_points(missing_published)) - args.limit} more profile groups")
    print()

    print("## Local Vs Published Data")
    print(f"- local current points not published: {len(local_not_published)}")
    print(f"- published current points not local: {len(published_not_local)}")
    for key, concs in group_points(local_not_published)[:args.limit]:
        print(f"- local-only: {format_point_group(key, concs)}")
    for key, concs in group_points(published_not_local)[:args.limit]:
        print(f"- published-only: {format_point_group(key, concs)}")
    print()

    print("## Stop Condition")
    print("- Healthy means: published sweep-state matches local generated state, published data matches local data, and no expected current points are missing.")
    print("- Incomplete means: state is in sync, but the local YAML still expects benchmark rows that are absent from published data.")
    print("- Degraded means: the website/R2 artifacts are stale or divergent from local state/data.")

    if args.fail_on_drift and (fetch_errors or drift or data_drift):
        return 2
    if args.fail_on_missing and missing_published:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-yaml", type=Path, default=DEFAULT_SWEEP_YAML)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--local-state", type=Path, default=DEFAULT_LOCAL_STATE_FILE)
    parser.add_argument("--local-data", type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument("--published-base", default=DEFAULT_PUBLIC_BASE)
    parser.add_argument("--published-state", default=None)
    parser.add_argument("--published-data", default=None)
    parser.add_argument("--scope", choices=("current", "archive", "all"), default="current")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    base = str(args.published_base).rstrip("/")
    if args.published_state is None:
        args.published_state = f"{base}/sweep-state.json"
    if args.published_data is None:
        args.published_data = f"{base}/data.json"

    return build_report(args)


if __name__ == "__main__":
    sys.exit(main())
