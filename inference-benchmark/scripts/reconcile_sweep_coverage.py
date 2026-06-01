#!/usr/bin/env python3
"""Reconcile sweep job state against actual coverage points.

The orchestrator tracks coarse jobs: host + model + tp + mode + backend. The
dashboard coverage is finer-grained: profile + concurrency rows inside each
coarse job. A job can therefore be marked `done` even when only a subset of its
expected profile/concurrency JSON files reached the dashboard data.

This script uses sweep.yaml as the desired matrix, data.json as the observed
coverage, and /tmp/bench_jobs/state as the dispatch state. It reports missing
coverage per job and can reset stale terminal jobs back to `pending`.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import compile_sweep
import publish_sweep_state


HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent

DEFAULT_PUBLIC_BASE = "https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current"
DEFAULT_DATA = f"{DEFAULT_PUBLIC_BASE}/data.json"
DEFAULT_SWEEP_YAML = HERE / "sweep.yaml"
DEFAULT_BENCH_JOBS = HERE / "bench_jobs.txt"
DEFAULT_STATE_DIR = Path("/tmp/bench_jobs/state")
DEFAULT_SWEEP_STATE = BENCH_ROOT / "dashboard" / "public" / "sweep-state.json"
DEFAULT_REPORT = Path("/tmp/sweep-coverage-reconcile.md")
DEFAULT_MISSING_JOBS = Path("/tmp/bench_jobs/missing_synthetic_distributional_bench_jobs.txt")

BLOCKING_STATUSES = {"done", "skipped", "failed", "known_oom"}
DEFAULT_RESET_STATUSES = {"done"}
COVERAGE_REQUEUE_COUNT_SUFFIX = "coverage_requeue_count"
COVERAGE_BLOCKER_SUFFIX = "coverage_blocker.json"
N_A_FAILURE_CATEGORIES = {
    "oom_or_kv_cache",
    "success_rate_below_min",
    "zero_results",
    "incomplete_outputs",
}

PointKey = tuple[str, str, str, str, str, int]  # hw, model, backend, data_mode, profile, conc
ProfileKey = tuple[str, str, str, int, str, str, str]  # scope, host, model, tp, mode, backend, profile


@dataclass(frozen=True)
class JsonSource:
    label: str
    ref: str
    ok: bool
    error: str | None = None
    bytes_read: int = 0


@dataclass
class JobCoverage:
    job_id: str
    data_scope: str
    host: str
    hw_label: str
    model: str
    tp: int
    mode: str
    backend: str
    status: str
    reason: str | None
    attempt: int | None = None
    failure_metadata: dict[str, Any] | None = None
    expected: set[PointKey] = field(default_factory=set)
    present: set[PointKey] = field(default_factory=set)

    @property
    def missing(self) -> set[PointKey]:
        return self.expected - self.present

    @property
    def is_stale_terminal(self) -> bool:
        return bool(self.missing) and self.status in BLOCKING_STATUSES


@dataclass
class ResetOutcome:
    reset: list[JobCoverage] = field(default_factory=list)
    exhausted: list[JobCoverage] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_url(ref: str) -> bool:
    return ref.startswith("http://") or ref.startswith("https://")


def load_json_ref(ref: str, label: str, timeout: float) -> tuple[Any | None, JsonSource]:
    try:
        if is_url(ref):
            req = urllib.request.Request(ref, headers={"User-Agent": "agentic-serve-sweep-reconcile/1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        else:
            raw = Path(ref).read_bytes()
        return json.loads(raw), JsonSource(label=label, ref=ref, ok=True, bytes_read=len(raw))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, JsonSource(label=label, ref=ref, ok=False, error=str(exc))


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    return manifest


def load_generated_state(sweep_yaml: Path, state_dir: Path) -> dict[str, Any]:
    publish_sweep_state.STATE_DIR = state_dir
    manifest = load_manifest(sweep_yaml)
    return publish_sweep_state.build_state(manifest)


def mode_to_data_mode(mode: str) -> str:
    if mode == "single":
        return "single-turn"
    if mode == "multi":
        return "multi-turn"
    return mode


def scope_matches(scope_value: str, scope: str) -> bool:
    return scope == "all" or compile_sweep.dashboard_scope_for(scope_value) == compile_sweep.dashboard_scope_for(scope)


def profile_key(item: dict[str, Any]) -> ProfileKey:
    return (
        compile_sweep.dashboard_scope_for(str(item.get("data_scope") or "archived")),
        str(item.get("host", "")),
        str(item.get("model", "")),
        int(item.get("tp", 0) or 0),
        str(item.get("mode", "")),
        str(item.get("backend", "vllm")),
        str(item.get("profile", "")),
    )


def point_from_row(row: dict[str, Any]) -> PointKey | None:
    cfg = row.get("config") or {}
    hardware = row.get("hardware")
    model = row.get("modelShort") or row.get("model_short")
    backend = cfg.get("backend") or "vllm"
    mode = cfg.get("mode")
    profile = cfg.get("profile")
    conc = cfg.get("concurrency")
    if not hardware or not model or not mode or not profile or conc is None:
        return None
    try:
        conc_i = int(conc)
    except (TypeError, ValueError):
        return None
    return (str(hardware), str(model), str(backend), str(mode), str(profile), conc_i)


def data_points(data: list[dict[str, Any]], scope: str) -> set[PointKey]:
    points: set[PointKey] = set()
    for row in data:
        row_scope = compile_sweep.dashboard_scope_for(str(row.get("dataScope") or "trace_replay"))
        if scope != "all" and row_scope != compile_sweep.dashboard_scope_for(scope):
            continue
        point = point_from_row(row)
        if point is not None:
            points.add(point)
    return points


def data_scope_counts(data: list[dict[str, Any]]) -> Counter[str]:
    return Counter(compile_sweep.dashboard_scope_for(str(row.get("dataScope") or "unknown")) for row in data)


def expected_by_job(
    state: dict[str, Any],
    scope: str,
    runnable_job_ids: set[str] | None,
) -> dict[str, JobCoverage]:
    blocked_profiles = {profile_key(item) for item in state.get("profile_infeasible", [])}
    jobs: dict[str, JobCoverage] = {}
    for cell in state.get("cells", []):
        data_scope = compile_sweep.dashboard_scope_for(str(cell.get("data_scope") or "archived"))
        if not scope_matches(data_scope, scope):
            continue
        output_scope = compile_sweep.dashboard_scope_for(scope) if scope != "all" else data_scope

        host = str(cell.get("host", ""))
        model = str(cell.get("model", ""))
        tp = int(cell.get("tp", 0) or 0)
        mode = str(cell.get("mode", ""))
        backend = str(cell.get("backend", "vllm"))
        hw_label = str(cell.get("hw_label", ""))
        status = str(cell.get("status") or "pending")
        reason = cell.get("reason")
        jid = publish_sweep_state.job_id(host, model, tp, mode, backend)
        if runnable_job_ids is not None and jid not in runnable_job_ids:
            continue

        cov = jobs.get(jid)
        if cov is None:
            cov = JobCoverage(
                job_id=jid,
                data_scope=output_scope,
                host=host,
                hw_label=hw_label,
                model=model,
                tp=tp,
                mode=mode,
                backend=backend,
                status=status,
                reason=str(reason) if reason else None,
                attempt=int(cell.get("attempt", 0) or 0),
                failure_metadata=cell.get("failure_metadata") if isinstance(cell.get("failure_metadata"), dict) else None,
            )
            jobs[jid] = cov

        data_mode = mode_to_data_mode(mode)
        profiles = [str(p) for p in cell.get("profiles") or []]
        concurrencies = [int(c) for c in cell.get("concurrencies") or []]
        for profile in profiles:
            if (data_scope, host, model, tp, mode, backend, profile) in blocked_profiles:
                continue
            for conc in concurrencies:
                cov.expected.add((hw_label, model, backend, data_mode, profile, conc))

    return jobs


def apply_present_points(jobs: dict[str, JobCoverage], present_points: set[PointKey]) -> None:
    for cov in jobs.values():
        cov.present = cov.expected & present_points


def parse_reset_statuses(raw: str) -> set[str]:
    statuses = {part.strip() for part in raw.split(",") if part.strip()}
    invalid = statuses - BLOCKING_STATUSES
    if invalid:
        allowed = ",".join(sorted(BLOCKING_STATUSES))
        raise argparse.ArgumentTypeError(f"invalid statuses {sorted(invalid)}; allowed: {allowed}")
    if not statuses:
        raise argparse.ArgumentTypeError("at least one status is required")
    return statuses


def parse_bench_jobs(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        host, _model_path, tp, short, mode, backend = parts[:6]
        backend = backend or "vllm"
        try:
            tp_i = int(tp)
        except ValueError:
            continue
        jid = publish_sweep_state.job_id(host.strip(), short.strip(), tp_i, mode.strip(), backend.strip())
        rows[jid] = raw
    return rows


def compiled_bench_jobs(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, str]], str, int]:
    compile_sweep.validate(manifest)
    emitted, skipped = compile_sweep.compile_jobs(manifest)
    rows: dict[str, str] = {}
    rows_by_scope: dict[str, dict[str, str]] = defaultdict(dict)
    for cell, row in emitted:
        backend = str(cell.get("backend", "vllm"))
        jid = publish_sweep_state.job_id(
            str(cell["host"]),
            str(cell["model"]),
            int(cell["tp"]),
            str(cell["mode"]),
            backend,
        )
        rows[jid] = row
        rows_by_scope[compile_sweep.dashboard_scope_for(publish_sweep_state.cell_data_scope(cell))][jid] = row
    synthetic_emitted, _synthetic_skipped = compile_sweep.compile_jobs(manifest, "synthetic_distributional")
    for cell, row in synthetic_emitted:
        backend = str(cell.get("backend", "vllm"))
        jid = publish_sweep_state.job_id(
            str(cell["host"]),
            str(cell["model"]),
            int(cell["tp"]),
            str(cell["mode"]),
            backend,
        )
        rows_by_scope["synthetic_distributional"][jid] = row
    return rows, rows_by_scope, compile_sweep.render_file(emitted), len(skipped)


def write_local_sweep_state(sweep_yaml: Path, state_dir: Path, out: Path) -> None:
    state = load_generated_state(sweep_yaml, state_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2) + "\n")


def target_state_dir_for_cov(state_dir: Path, cov: JobCoverage) -> Path:
    return state_dir if state_dir.name == cov.data_scope else state_dir / cov.data_scope


def read_int_file(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip() or default)
    except (OSError, ValueError):
        return default


def write_coverage_blocker(
    target_state_dir: Path,
    cov: JobCoverage,
    *,
    scope: str,
    status: str,
    timestamp: str,
    requeue_count: int,
    max_requeues: int,
    reason: str,
) -> None:
    payload = {
        "generated_at": timestamp,
        "job_id": cov.job_id,
        "scope": scope,
        "status": status,
        "host": cov.host,
        "hardware": cov.hw_label,
        "model": cov.model,
        "tp": cov.tp,
        "mode": cov.mode,
        "backend": cov.backend,
        "present": len(cov.present),
        "expected": len(cov.expected),
        "missing": group_missing_for_job(cov.missing),
        "expected_points": points_payload(cov.expected),
        "present_points": points_payload(cov.present),
        "missing_points": points_payload(cov.missing),
        "missing_count": len(cov.missing),
        "attempt": cov.attempt,
        "failure": failure_payload(cov),
        "requeue_count": requeue_count,
        "max_requeues": max_requeues,
        "reason": reason,
    }
    (target_state_dir / f"{cov.job_id}.{COVERAGE_BLOCKER_SUFFIX}").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def reset_stale_jobs(
    jobs: list[JobCoverage],
    state_dir: Path,
    reset_statuses: set[str],
    scope: str,
    write_reason: bool,
    max_requeues: int,
) -> ResetOutcome:
    targets = [cov for cov in jobs if cov.status in reset_statuses and cov.missing]
    outcome = ResetOutcome()
    timestamp = now_iso()
    for cov in targets:
        target_state_dir = target_state_dir_for_cov(state_dir, cov)
        target_state_dir.mkdir(parents=True, exist_ok=True)
        count_path = target_state_dir / f"{cov.job_id}.{COVERAGE_REQUEUE_COUNT_SUFFIX}"
        current_count = read_int_file(count_path)
        if max_requeues >= 0 and current_count >= max_requeues:
            root_cause = failure_summary(cov)
            reason = (
                f"coverage incomplete for scope={scope}: "
                f"missing {len(cov.missing)}/{len(cov.expected)} points; "
                f"coverage requeue limit reached {current_count}/{max_requeues}"
            )
            if root_cause:
                reason = f"{reason}; last failure: {root_cause}"
            if write_reason:
                (target_state_dir / f"{cov.job_id}.reason").write_text(reason + "\n")
            write_coverage_blocker(
                target_state_dir,
                cov,
                scope=scope,
                status="requeue_exhausted",
                timestamp=timestamp,
                requeue_count=current_count,
                max_requeues=max_requeues,
                reason=reason,
            )
            outcome.exhausted.append(cov)
            continue

        next_count = current_count + 1
        (target_state_dir / f"{cov.job_id}.status").write_text("pending\n")
        count_path.write_text(f"{next_count}\n")
        if write_reason:
            reason = (
                f"coverage incomplete for scope={scope}: "
                f"missing {len(cov.missing)}/{len(cov.expected)} points; "
                f"coverage requeue {next_count}/{max_requeues if max_requeues >= 0 else 'unlimited'} "
                f"by reconcile_sweep_coverage.py at {timestamp}"
            )
            (target_state_dir / f"{cov.job_id}.reason").write_text(reason + "\n")
        write_coverage_blocker(
            target_state_dir,
            cov,
            scope=scope,
            status="requeued",
            timestamp=timestamp,
            requeue_count=next_count,
            max_requeues=max_requeues,
            reason=reason if write_reason else "coverage incomplete; reset to pending",
        )
        outcome.reset.append(cov)
    return outcome


def group_missing_for_job(points: set[PointKey]) -> str:
    grouped: dict[str, list[int]] = defaultdict(list)
    for _hw, _model, _backend, _mode, profile, conc in points:
        grouped[profile].append(conc)
    parts = []
    for profile, concs in sorted(grouped.items()):
        shown = ",".join(str(c) for c in sorted(concs))
        parts.append(f"{profile}: C={shown}")
    return "; ".join(parts)


def point_payload(points: set[PointKey]) -> list[dict[str, Any]]:
    rows = []
    for hw, model, backend, mode, profile, conc in sorted(points):
        rows.append({
            "hardware": hw,
            "model": model,
            "backend": backend,
            "mode": mode,
            "profile": profile,
            "concurrency": conc,
        })
    return rows


def points_payload(points: set[PointKey]) -> list[dict[str, Any]]:
    return point_payload(points)


def optional_present_points(data: list[dict[str, Any]], scope: str, expected_points: set[PointKey]) -> set[PointKey]:
    """Return accepted synthetic points that are outside the required grid.

    Some synthetic runs intentionally exceed the current required sweep grid
    because a previously-waived high-concurrency or workaround cell completed.
    Keep those visible as optional coverage without letting them expand the
    required denominator or missing-work set.
    """
    if compile_sweep.dashboard_scope_for(scope) != "synthetic_distributional":
        return set()
    return {
        point
        for point in data_points(data, scope) - expected_points
        if point[4].endswith("-synth")
    }


def failure_category(reason: str | None) -> str:
    text = (reason or "").lower()
    if any(token in text for token in (
        "xid",
        "nvml",
        "driver",
        "gpu has fallen off",
        "cuda error",
        "uncorrectable",
        "nvidia-smi",
        "cuda initialization",
    )):
        return "driver_failure"
    if any(token in text for token in (
        "out of memory",
        "cuda out of memory",
        "kv-cache",
        "kv cache",
        "cache blocks",
    )):
        return "oom_or_kv_cache"
    if "success rate" in text and "below minimum" in text:
        return "success_rate_below_min"
    if "[warn]" in text and "failed" in text:
        return "benchmark_failed"
    if "zero results" in text or "zero expected outputs" in text:
        return "zero_results"
    if "incomplete" in text or "expected outputs missing" in text:
        return "incomplete_outputs"
    return "unknown"


def failure_category_label(category: str) -> str:
    labels = {
        "driver_failure": "driver failure",
        "oom_or_kv_cache": "OOM / KV-cache limit",
        "success_rate_below_min": "success rate below threshold",
        "benchmark_failed": "benchmark command failed",
        "zero_results": "zero results",
        "incomplete_outputs": "incomplete outputs",
        "unknown": "unknown failure",
    }
    return labels.get(category, category.replace("_", " "))


def failure_payload(cov: JobCoverage) -> dict[str, Any] | None:
    metadata = cov.failure_metadata or {}
    reason = str(metadata.get("reason") or cov.reason or "")
    if not metadata and not reason:
        return None
    category = failure_category(reason)
    attempt = metadata.get("attempt", cov.attempt)
    return {
        "category": category,
        "label": failure_category_label(category),
        "kind": metadata.get("kind"),
        "status": metadata.get("status", cov.status),
        "reason": reason or None,
        "attempt": attempt,
        "max_attempts": metadata.get("max_attempts"),
        "expected_outputs_present": metadata.get("expected_outputs_present"),
        "expected_outputs_total": metadata.get("expected_outputs_total"),
        "missing_outputs": metadata.get("missing_outputs") if isinstance(metadata.get("missing_outputs"), list) else [],
        "remote_log": metadata.get("remote_log"),
        "mirror_status": metadata.get("mirror_status"),
        "updated_at": metadata.get("updated_at"),
    }


def failure_summary(cov: JobCoverage) -> str | None:
    failure = failure_payload(cov)
    if not failure:
        return None
    attempt = failure.get("attempt")
    max_attempts = failure.get("max_attempts")
    attempts = ""
    if attempt is not None and max_attempts is not None:
        attempts = f" after {attempt}/{max_attempts} attempts"
    elif attempt is not None:
        attempts = f" after {attempt} attempts"
    reason = str(failure.get("reason") or "").strip()
    if len(reason) > 240:
        reason = reason[:237] + "..."
    return f"{failure.get('label')}{attempts}{(': ' + reason) if reason else ''}"


def coverage_disposition(cov: JobCoverage) -> str | None:
    """Classify terminal missing coverage as either fillable failure or N/A.

    Red failures are cases where another run could plausibly fix repo/infra
    behavior. Blue N/A means the job reached a terminal, explainable limit
    after the configured retry path, so the missing points should not keep
    counting against fillable synthetic coverage.
    """
    if not cov.missing or cov.status not in BLOCKING_STATUSES:
        return None
    if cov.status == "known_oom":
        return "na"
    failure = failure_payload(cov)
    category = str((failure or {}).get("category") or "unknown")
    if category in N_A_FAILURE_CATEGORIES:
        return "na"
    return "failed"


def coverage_explanation(cov: JobCoverage, disposition: str | None) -> str | None:
    summary = failure_summary(cov) or cov.reason
    if not summary:
        return None
    if disposition == "na":
        return f"N/A after retry exhaustion: {summary}"
    if disposition == "failed":
        return f"failed after retry; needs inspection: {summary}"
    return summary


def format_job(cov: JobCoverage) -> str:
    return f"{cov.host}/{cov.hw_label}/{cov.model}/tp{cov.tp}/{cov.backend}/{cov.mode}"


def build_report(
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    current_rows: dict[str, str],
    compiled_rows: dict[str, str],
    compiled_scope_rows: dict[str, str],
    compiled_skipped: int,
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
    limit: int,
    wrote_bench_jobs: bool,
    wrote_missing_jobs: Path | None,
    wrote_sweep_state: Path | None,
    wrote_blockers_json: Path | None,
) -> str:
    all_jobs = sorted(jobs.values(), key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode))
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]
    reset_candidates = [cov for cov in stale_jobs if cov.status in reset_statuses]
    expected_points = {point for cov in all_jobs for point in cov.expected}
    present_points = {point for cov in all_jobs for point in cov.present}
    optional_points = optional_present_points(data, scope, expected_points)
    missing_points = expected_points - present_points
    coverage_na_points = {
        point
        for cov in stale_jobs
        if coverage_disposition(cov) == "na"
        for point in cov.missing
    }
    coverage_failed_points = {
        point
        for cov in stale_jobs
        if coverage_disposition(cov) == "failed"
        for point in cov.missing
    }
    required_points = expected_points - coverage_na_points
    status_counts = Counter(cov.status for cov in all_jobs)
    missing_by_status = Counter(cov.status for cov in missing_jobs)

    drift_compiled_rows = compiled_scope_rows if scope != "all" else compiled_rows
    only_current = sorted(set(current_rows) - set(drift_compiled_rows))
    only_compiled = sorted(set(drift_compiled_rows) - set(current_rows))

    lines = [
        "# Sweep Coverage Reconcile",
        "",
        f"- generated_at: {now_iso()}",
        f"- scope: {scope}",
        f"- data source: {data_source.ref}",
        f"- data source status: {'ok' if data_source.ok else 'error'}"
        + (f" ({data_source.bytes_read} bytes)" if data_source.ok else f" ({data_source.error})"),
        f"- data rows: {len(data)} total, scopes={dict(data_scope_counts(data))}",
        f"- expected coverage points: {len(expected_points)}",
        f"- present expected points: {len(present_points)} / {len(expected_points)}",
        f"- optional present synthetic points outside required grid: {len(optional_points)}",
        f"- observed synthetic points: {len(present_points | optional_points)} / {len(required_points)} fillable ({len(expected_points)} grid points)",
        f"- missing expected points: {len(missing_points)}",
        f"- N/A attempted points: {len(coverage_na_points)}",
        f"- failed points needing inspection: {len(coverage_failed_points)}",
        f"- expected jobs: {len(all_jobs)}",
        f"- compiled runnable jobs for scope: {len(compiled_scope_rows)}",
        f"- jobs with missing coverage: {len(missing_jobs)}",
        f"- stale terminal/blocking jobs with missing coverage: {len(stale_jobs)}",
        f"- reset candidates for statuses {sorted(reset_statuses)}: {len(reset_candidates)}",
        f"- reset performed: {len(reset_outcome.reset)} jobs",
        f"- reset exhausted by coverage requeue limit {max_requeues if max_requeues >= 0 else 'unlimited'}: {len(reset_outcome.exhausted)} jobs",
        f"- job status counts: {dict(status_counts)}",
        f"- missing jobs by status: {dict(missing_by_status)}",
        "",
        "## Bench Jobs Drift",
        f"- current bench_jobs rows: {len(current_rows)}",
        f"- compiled runnable rows from sweep.yaml: {len(drift_compiled_rows)}",
        f"- compiled skipped rows: {compiled_skipped}",
        f"- rows in current bench_jobs only: {len(only_current)}",
        f"- rows in compiled sweep only: {len(only_compiled)}",
        f"- rewrote bench_jobs.txt: {wrote_bench_jobs}",
    ]

    if only_current[:limit]:
        lines.append("- current-only job ids:")
        lines.extend(f"  - {jid}" for jid in only_current[:limit])
    if len(only_current) > limit:
        lines.append(f"  - ... {len(only_current) - limit} more")
    if only_compiled[:limit]:
        lines.append("- compiled-only job ids:")
        lines.extend(f"  - {jid}" for jid in only_compiled[:limit])
    if len(only_compiled) > limit:
        lines.append(f"  - ... {len(only_compiled) - limit} more")

    lines.extend([
        "",
        "## Stale Terminal Or Blocking Jobs",
        "| job_id | status | job | present/expected | missing |",
        "|---|---:|---|---:|---|",
    ])
    for cov in stale_jobs[:limit]:
        lines.append(
            f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
            f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
        )
    if len(stale_jobs) > limit:
        lines.append(f"| ... | ... | {len(stale_jobs) - limit} more | ... | ... |")

    if reset_outcome.exhausted:
        lines.extend([
            "",
            "## Requeue Exhausted",
            "| job_id | status | job | present/expected | missing |",
            "|---|---:|---|---:|---|",
        ])
        for cov in reset_outcome.exhausted[:limit]:
            lines.append(
                f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
                f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
            )
        if len(reset_outcome.exhausted) > limit:
            lines.append(f"| ... | ... | {len(reset_outcome.exhausted) - limit} more | ... | ... |")

    lines.extend([
        "",
        "## Non-terminal Missing Jobs",
        "| job_id | status | job | present/expected | missing |",
        "|---|---:|---|---:|---|",
    ])
    non_terminal_missing = [cov for cov in missing_jobs if cov.status not in BLOCKING_STATUSES]
    for cov in non_terminal_missing[:limit]:
        lines.append(
            f"| {cov.job_id} | {cov.status} | {format_job(cov)} | "
            f"{len(cov.present)}/{len(cov.expected)} | {group_missing_for_job(cov.missing)} |"
        )
    if len(non_terminal_missing) > limit:
        lines.append(f"| ... | ... | {len(non_terminal_missing) - limit} more | ... | ... |")

    lines.extend([
        "",
        "## Outputs",
        f"- missing jobs file: {wrote_missing_jobs if wrote_missing_jobs else 'not written'}",
        f"- local sweep-state.json: {wrote_sweep_state if wrote_sweep_state else 'not written'}",
        f"- blockers json: {wrote_blockers_json if wrote_blockers_json else 'not written'}",
        "",
        "## Stop Condition",
        f"- Complete for scope={scope} means every expected point from sweep.yaml exists in data.json.",
        "- If stale terminal/blocking jobs are nonzero, the orchestrator can believe there is no work left while coverage is still missing.",
    ])
    return "\n".join(lines) + "\n"


def coverage_payload(
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
) -> dict[str, Any]:
    all_jobs = sorted(jobs.values(), key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode))
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]
    expected_points = {point for cov in all_jobs for point in cov.expected}
    present_points = {point for cov in all_jobs for point in cov.present}
    optional_points = optional_present_points(data, scope, expected_points)
    coverage_na_points = {
        point
        for cov in stale_jobs
        if coverage_disposition(cov) == "na"
        for point in cov.missing
    }
    coverage_failed_points = {
        point
        for cov in stale_jobs
        if coverage_disposition(cov) == "failed"
        for point in cov.missing
    }
    coverage_required_points = expected_points - coverage_na_points
    status_counts = Counter(cov.status for cov in all_jobs)
    missing_by_status = Counter(cov.status for cov in missing_jobs)

    def cov_payload(cov: JobCoverage, *, include_missing_points: bool) -> dict[str, Any]:
        failure = failure_payload(cov)
        disposition = coverage_disposition(cov)
        explanation = coverage_explanation(cov, disposition)
        payload = {
            "job_id": cov.job_id,
            "status": cov.status,
            "scope": cov.data_scope,
            "host": cov.host,
            "hardware": cov.hw_label,
            "model": cov.model,
            "tp": cov.tp,
            "mode": cov.mode,
            "backend": cov.backend,
            "present": len(cov.present),
            "expected": len(cov.expected),
            "missing_count": len(cov.missing),
            "missing": group_missing_for_job(cov.missing),
            "present_points": points_payload(cov.present),
            "attempt": cov.attempt,
            "failure": failure,
            "reason": cov.reason,
            "coverage_disposition": disposition,
            "coverage_explanation": explanation,
        }
        if include_missing_points:
            payload["missing_points"] = points_payload(cov.missing)
        return payload

    failure_category_counts = Counter(
        failure.get("category", "unknown")
        for cov in stale_jobs
        if (failure := failure_payload(cov))
    )
    disposition_counts = Counter(
        disposition
        for cov in stale_jobs
        if (disposition := coverage_disposition(cov))
    )
    disposition_point_counts = Counter()
    for cov in stale_jobs:
        disposition = coverage_disposition(cov)
        if disposition:
            disposition_point_counts[disposition] += len(cov.missing)

    return {
        "generated_at": now_iso(),
        "scope": scope,
        "data_source": {
            "ref": data_source.ref,
            "ok": data_source.ok,
            "error": data_source.error,
            "bytes_read": data_source.bytes_read,
        },
        "data_rows": len(data),
        "data_scopes": dict(data_scope_counts(data)),
        "expected_points": len(expected_points),
        "coverage_required_points": len(coverage_required_points),
        "coverage_missing_required_points": len((expected_points - present_points) - coverage_na_points),
        "coverage_na_points": len(coverage_na_points),
        "coverage_failed_points": len(coverage_failed_points),
        "present_points": len(present_points),
        "observed_present_points": len(present_points | optional_points),
        "optional_present_points_count": len(optional_points),
        "optional_present_points": points_payload(optional_points),
        "missing_points": len(expected_points - present_points),
        "jobs_total": len(all_jobs),
        "jobs_with_missing_coverage": len(missing_jobs),
        "stale_terminal_jobs": len(stale_jobs),
        "job_status_counts": dict(status_counts),
        "missing_jobs_by_status": dict(missing_by_status),
        "reset_statuses": sorted(reset_statuses),
        "max_requeues": max_requeues,
        "reset_performed": [cov.job_id for cov in reset_outcome.reset],
        "reset_exhausted": [cov.job_id for cov in reset_outcome.exhausted],
        "failure_category_counts": dict(failure_category_counts),
        "failure_disposition_counts": dict(disposition_counts),
        "failure_disposition_point_counts": dict(disposition_point_counts),
        "jobs": [cov_payload(cov, include_missing_points=False) for cov in all_jobs],
        "blockers": [cov_payload(cov, include_missing_points=True) for cov in stale_jobs],
    }


def write_blockers_json(
    path: Path,
    *,
    scope: str,
    data_source: JsonSource,
    data: list[dict[str, Any]],
    jobs: dict[str, JobCoverage],
    reset_statuses: set[str],
    reset_outcome: ResetOutcome,
    max_requeues: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = coverage_payload(
        scope=scope,
        data_source=data_source,
        data=data,
        jobs=jobs,
        reset_statuses=reset_statuses,
        reset_outcome=reset_outcome,
        max_requeues=max_requeues,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_missing_bench_jobs(path: Path, missing_jobs: list[JobCoverage], compiled_rows: dict[str, str], scope: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark job subset with missing coverage.",
        "# GENERATED by scripts/reconcile_sweep_coverage.py.",
        f"# SCOPE: {scope}",
        "# Format matches scripts/bench_jobs.txt.",
        "",
    ]
    for cov in sorted(missing_jobs, key=lambda c: (c.host, c.hw_label, c.model, c.tp, c.backend, c.mode)):
        row = compiled_rows.get(cov.job_id)
        if row:
            lines.append(row)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(
            "trace_replay",
            "synthetic_distributional",
            "synthetic-distributional",
            "archived",
            "synthetic",
            "latest",
            "current",
            "fixed",
            "mse",
            "archive",
            "all",
        ),
        default="synthetic_distributional",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="data.json file path or URL; defaults to published R2 current data")
    parser.add_argument("--sweep-yaml", type=Path, default=DEFAULT_SWEEP_YAML)
    parser.add_argument("--bench-jobs", type=Path, default=DEFAULT_BENCH_JOBS)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--sweep-state-out", type=Path, default=DEFAULT_SWEEP_STATE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="write markdown report to this path")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--write-bench-jobs", action="store_true", help="rewrite bench_jobs.txt from sweep.yaml")
    parser.add_argument(
        "--write-missing-jobs",
        nargs="?",
        const=str(DEFAULT_MISSING_JOBS),
        default=None,
        help="write a bench_jobs-format subset containing jobs with missing coverage",
    )
    parser.add_argument("--write-sweep-state", action="store_true", help="refresh local dashboard/public/sweep-state.json")
    parser.add_argument("--reset-stale", action="store_true", help="reset stale terminal jobs with missing coverage to pending")
    parser.add_argument(
        "--reset-statuses",
        type=parse_reset_statuses,
        default=DEFAULT_RESET_STATUSES,
        help="comma-separated terminal statuses to reset; default: done",
    )
    parser.add_argument(
        "--max-coverage-requeues",
        type=int,
        default=1,
        help="maximum automatic coverage requeues per job; use -1 for unlimited",
    )
    parser.add_argument(
        "--write-blockers-json",
        type=Path,
        default=None,
        help="write machine-readable coverage blocker/requeue summary JSON",
    )
    parser.add_argument("--no-reset-reason", action="store_true", help="do not write/reset .reason files")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--fail-on-stale", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.sweep_yaml)
    compiled_rows, compiled_rows_by_scope, compiled_text, compiled_skipped = compiled_bench_jobs(manifest)
    if args.scope == "all":
        compiled_scope_rows = compiled_rows
    else:
        compiled_scope_rows = compiled_rows_by_scope.get(compile_sweep.dashboard_scope_for(args.scope), {})
    current_rows = parse_bench_jobs(args.bench_jobs)

    data_raw, data_source = load_json_ref(args.data, "coverage data", args.timeout)
    if not data_source.ok:
        print(f"failed to read {args.data}: {data_source.error}", file=sys.stderr)
        return 2
    if not isinstance(data_raw, list):
        print(f"{args.data} did not contain a JSON array", file=sys.stderr)
        return 2
    data: list[dict[str, Any]] = [row for row in data_raw if isinstance(row, dict)]

    state = load_generated_state(args.sweep_yaml, args.state_dir)
    jobs = expected_by_job(state, args.scope, set(compiled_scope_rows))
    apply_present_points(jobs, data_points(data, args.scope))

    all_jobs = list(jobs.values())
    missing_jobs = [cov for cov in all_jobs if cov.missing]
    stale_jobs = [cov for cov in missing_jobs if cov.status in BLOCKING_STATUSES]

    reset_outcome = ResetOutcome()
    if args.reset_stale:
        reset_outcome = reset_stale_jobs(
            stale_jobs,
            args.state_dir,
            args.reset_statuses,
            args.scope,
            write_reason=not args.no_reset_reason,
            max_requeues=args.max_coverage_requeues,
        )

    wrote_bench_jobs = False
    if args.write_bench_jobs:
        args.bench_jobs.write_text(compiled_text)
        wrote_bench_jobs = True

    wrote_missing_jobs: Path | None = None
    if args.write_missing_jobs:
        wrote_missing_jobs = Path(args.write_missing_jobs)
        write_missing_bench_jobs(wrote_missing_jobs, missing_jobs, compiled_scope_rows, args.scope)

    wrote_sweep_state: Path | None = None
    if args.write_sweep_state:
        write_local_sweep_state(args.sweep_yaml, args.state_dir, args.sweep_state_out)
        wrote_sweep_state = args.sweep_state_out

    wrote_blockers_json: Path | None = None
    if args.write_blockers_json:
        write_blockers_json(
            args.write_blockers_json,
            scope=args.scope,
            data_source=data_source,
            data=data,
            jobs=jobs,
            reset_statuses=args.reset_statuses,
            reset_outcome=reset_outcome,
            max_requeues=args.max_coverage_requeues,
        )
        wrote_blockers_json = args.write_blockers_json

    report = build_report(
        scope=args.scope,
        data_source=data_source,
        data=data,
        jobs=jobs,
        current_rows=current_rows,
        compiled_rows=compiled_rows,
        compiled_scope_rows=compiled_scope_rows,
        compiled_skipped=compiled_skipped,
        reset_statuses=args.reset_statuses,
        reset_outcome=reset_outcome,
        max_requeues=args.max_coverage_requeues,
        limit=args.limit,
        wrote_bench_jobs=wrote_bench_jobs,
        wrote_missing_jobs=wrote_missing_jobs,
        wrote_sweep_state=wrote_sweep_state,
        wrote_blockers_json=wrote_blockers_json,
    )

    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    sys.stdout.write(report)

    if args.fail_on_stale and stale_jobs:
        return 4
    if args.fail_on_missing and any(cov.missing for cov in all_jobs):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
