#!/usr/bin/env python3
"""Read-only sweep progress and GPU occupancy reporter.

This script complements bench_orchestrator.sh. It does not dispatch jobs,
publish data, or edit sweep state. It reads local /tmp/bench_jobs state,
polls each GPU host with nvidia-smi over SSH, and emits a compact Markdown
snapshot suitable for a long-running monitor loop.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable


HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
DEFAULT_JOBS_FILE = HERE / "bench_jobs.txt"
DEFAULT_STATE_DIR = Path("/tmp/bench_jobs/state")
DEFAULT_HOSTS = ("gpu-4", "3090", "2080ti", "h100")
PORTS = tuple(range(8089, 8097))
LEGACY_STATE_FALLBACK = os.environ.get("BENCH_STATE_LEGACY_FALLBACK") == "1"

REMOTE_SNAPSHOT_SCRIPT = r"""
set -uo pipefail
echo "__WHOAMI__"
whoami 2>/dev/null || true
echo "__GPU__"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
echo "__PROC__"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo "__PS__"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -nu | tr '\n' ',')
if [ -n "$pids" ]; then
  ps -o pid=,user=,ppid=,etimes=,cmd= -p "${pids%,}" 2>/dev/null || true
fi
echo "__PORTS__"
for p in 8089 8090 8091 8092 8093 8094 8095 8096; do
  line=$(ss -ltnp 2>/dev/null | awk -v p=":$p" '$4 ~ p"$" {print; exit}')
  if [ -n "$line" ]; then printf "%s %s\n" "$p" "$line"; fi
done
"""


@dataclass(frozen=True)
class Job:
    host: str
    model_path: str
    tp: int
    short: str
    mode: str
    backend: str
    max_len: str
    gpu_mem: str
    concs: str
    profiles: str
    extra_env: str
    scope: str
    line_no: int

    @property
    def job_id(self) -> str:
        suffix = "" if self.backend == "vllm" else f"_{self.backend}"
        return f"{self.host}_{self.short}_tp{self.tp}_{self.mode}{suffix}"


@dataclass
class JobState:
    job: Job
    status: str
    gpus: str
    port: str
    attempt: str
    age_seconds: int | None
    max_len_override: str


@dataclass
class GpuInfo:
    index: str
    uuid: str
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    util_pct: int | None


@dataclass
class GpuProcess:
    gpu_index: str
    gpu_uuid: str
    pid: str
    process_name: str
    used_memory_mib: int | None
    user: str
    ppid: str
    age_seconds: int | None
    cmd: str
    kind: str = "unknown"


@dataclass
class HostSnapshot:
    host: str
    ok: bool
    remote_user: str = ""
    gpus: list[GpuInfo] | None = None
    processes: list[GpuProcess] | None = None
    ports: list[str] | None = None
    error: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_csv(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def split_gpu_list(raw: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", raw.strip()) if part]


def compact_cmd(cmd: str, max_len: int = 96) -> str:
    clean = " ".join(cmd.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def human_age(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h{minutes % 60:02d}m"


def human_mem(used: int | None, total: int | None = None) -> str:
    if used is None:
        return "-"
    if total is None:
        return f"{used}MiB"
    return f"{used}/{total}MiB"


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def state_path(state_dir: Path, job: Job, suffix: str) -> Path:
    # Job IDs contain model-version dots, so Path.with_suffix() would corrupt
    # names like Llama-3.1-8B_tp4_multi.
    if job.scope and job.scope != "all":
        if state_dir.name == job.scope:
            return state_dir / f"{job.job_id}.{suffix}"
        scoped = state_dir / job.scope / f"{job.job_id}.{suffix}"
        if scoped.exists() or not LEGACY_STATE_FALLBACK:
            return scoped
    return state_dir / f"{job.job_id}.{suffix}"


def parse_jobs(jobs_file: Path) -> list[Job]:
    jobs: list[Job] = []
    scope = "all"
    for line_no, raw in enumerate(jobs_file.read_text().splitlines(), start=1):
        line = raw.strip()
        if line.startswith("# SCOPE:"):
            scope = line.split(":", 1)[1].strip() or "all"
            continue
        if not line or line.startswith("#"):
            continue
        parts = raw.rstrip("\n").split("|")
        while len(parts) < 11:
            parts.append("")
        host, model_path, tp, short, mode, backend, max_len, gpu_mem, concs, profiles, extra_env = parts[:11]
        host = host.strip()
        backend = (backend.strip() or "vllm")
        jobs.append(
            Job(
                host=host,
                model_path=model_path.strip(),
                tp=int(tp.strip()),
                short=short.strip(),
                mode=mode.strip(),
                backend=backend,
                max_len=max_len.strip(),
                gpu_mem=gpu_mem.strip(),
                concs=concs.strip(),
                profiles=profiles.strip(),
                extra_env=extra_env.strip(),
                scope=scope,
                line_no=line_no,
            )
        )
    return jobs


def load_job_states(jobs: Iterable[Job], state_dir: Path) -> list[JobState]:
    states: list[JobState] = []
    now = time.time()
    for job in jobs:
        status_file = state_path(state_dir, job, "status")
        status = read_text(status_file, "pending") or "pending"
        age_seconds: int | None = None
        try:
            age_seconds = max(0, int(now - status_file.stat().st_mtime))
        except OSError:
            pass
        states.append(
            JobState(
                job=job,
                status=status,
                gpus=read_text(state_path(state_dir, job, "gpus")),
                port=read_text(state_path(state_dir, job, "port")),
                attempt=read_text(state_path(state_dir, job, "attempt"), "0") or "0",
                age_seconds=age_seconds,
                max_len_override=read_text(state_path(state_dir, job, "max_len_override")),
            )
        )
    return states


def ssh_snapshot(host: str, timeout: int) -> HostSnapshot:
    command = f"bash -lc {shlex.quote(REMOTE_SNAPSHOT_SCRIPT)}"
    try:
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, command],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return HostSnapshot(host=host, ok=False, error=str(exc))
    if proc.returncode != 0 and not proc.stdout:
        return HostSnapshot(host=host, ok=False, error=(proc.stderr or f"ssh exit {proc.returncode}").strip())
    return parse_host_snapshot(host, proc.stdout, proc.stderr)


def parse_host_snapshot(host: str, stdout: str, stderr: str) -> HostSnapshot:
    sections: dict[str, list[str]] = defaultdict(list)
    section = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("__") and stripped.endswith("__"):
            section = stripped.strip("_").lower()
            continue
        if section:
            sections[section].append(line.rstrip())

    remote_user = next((line.strip() for line in sections.get("whoami", []) if line.strip()), "")

    gpus: list[GpuInfo] = []
    uuid_to_index: dict[str, str] = {}
    for line in sections.get("gpu", []):
        if not line.strip() or "failed" in line.lower():
            continue
        parts = parse_csv(line)
        if len(parts) < 6:
            continue
        index, uuid, name, mem_used, mem_total, util = [part.strip() for part in parts[:6]]
        uuid_to_index[uuid] = index
        gpus.append(
            GpuInfo(
                index=index,
                uuid=uuid,
                name=name,
                memory_used_mib=parse_int(mem_used),
                memory_total_mib=parse_int(mem_total),
                util_pct=parse_int(util),
            )
        )

    ps_by_pid: dict[str, tuple[str, str, int | None, str]] = {}
    for line in sections.get("ps", []):
        match = re.match(r"\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if not match:
            continue
        pid, user, ppid, etimes, cmd = match.groups()
        ps_by_pid[pid] = (user, ppid, parse_int(etimes), cmd)

    processes: list[GpuProcess] = []
    for line in sections.get("proc", []):
        if not line.strip() or "No running processes" in line:
            continue
        parts = parse_csv(line)
        if len(parts) < 4:
            continue
        gpu_uuid, pid, process_name, used_memory = [part.strip() for part in parts[:4]]
        user, ppid, age_seconds, cmd = ps_by_pid.get(pid, ("?", "?", None, process_name))
        processes.append(
            GpuProcess(
                gpu_index=uuid_to_index.get(gpu_uuid, "?"),
                gpu_uuid=gpu_uuid,
                pid=pid,
                process_name=process_name,
                used_memory_mib=parse_int(used_memory),
                user=user,
                ppid=ppid,
                age_seconds=age_seconds,
                cmd=cmd,
            )
        )

    ports = [line.strip() for line in sections.get("ports", []) if line.strip()]
    error = stderr.strip()
    return HostSnapshot(host=host, ok=True, remote_user=remote_user, gpus=gpus, processes=processes, ports=ports, error=error)


def running_jobs_by_host_gpu(states: list[JobState]) -> dict[tuple[str, str], list[JobState]]:
    out: dict[tuple[str, str], list[JobState]] = defaultdict(list)
    for state in states:
        if state.status != "running":
            continue
        for gpu in split_gpu_list(state.gpus):
            out[(state.job.host, gpu)].append(state)
    return out


def classify_process(proc: GpuProcess, snapshot: HostSnapshot, sweep_jobs: list[JobState]) -> str:
    cmd = proc.cmd.lower()
    markers = (
        "/tmp/inference-benchmark",
        "sweep_all_profiles",
        "sweep_multiturn_profiles",
        "src.benchmark.runner",
        "/tmp/results/synthetic",
        "/tmp/results/latest",
        "/tmp/results/current",
        "/tmp/results/fixed",
        "/tmp/results/mse",
    )
    if any(marker in cmd for marker in markers):
        return "sweep"
    if sweep_jobs:
        return "sweep-slot"
    if snapshot.remote_user and proc.user not in ("?", snapshot.remote_user):
        return "other-user"
    return "same-user-nonsweep"


def job_label(state: JobState) -> str:
    port = f":{state.port}" if state.port else ""
    age = human_age(state.age_seconds)
    return f"{state.job.job_id}{port} age={age}"


def build_report(args: argparse.Namespace) -> str:
    jobs = parse_jobs(args.jobs_file)
    states = load_job_states(jobs, args.state_dir)
    states_by_host: dict[str, list[JobState]] = defaultdict(list)
    for state in states:
        states_by_host[state.job.host].append(state)

    running_by_gpu = running_jobs_by_host_gpu(states)
    snapshots = [ssh_snapshot(host, args.ssh_timeout) for host in args.hosts]
    lines: list[str] = []

    total_counts = Counter(state.status for state in states)
    lines.append("# Sweep Progress Snapshot")
    lines.append("")
    lines.append(f"- generated_at: {now_iso()}")
    lines.append(f"- jobs_file: {args.jobs_file}")
    lines.append(f"- state_dir: {args.state_dir}")
    lines.append(f"- total_jobs: {len(states)} ({format_counts(total_counts)})")
    lines.append("")

    lines.append("## Host Progress")
    lines.append("")
    lines.append("| Host | Jobs | Running sweep jobs | Listening ports |")
    lines.append("| --- | --- | --- | --- |")
    for snapshot in snapshots:
        host_states = states_by_host.get(snapshot.host, [])
        counts = Counter(state.status for state in host_states)
        running = [state for state in host_states if state.status == "running"]
        running_text = "<br>".join(job_label(state) for state in running) or "-"
        ports_text = "<br>".join(snapshot.ports or []) if snapshot.ok and snapshot.ports else "-"
        if not snapshot.ok:
            ports_text = f"SSH ERROR: {snapshot.error}"
        lines.append(
            f"| {snapshot.host} | {len(host_states)} ({format_counts(counts)}) | "
            f"{running_text} | {ports_text} |"
        )
    lines.append("")

    lines.append("## GPU Occupancy")
    for snapshot in snapshots:
        lines.append("")
        lines.append(f"### {snapshot.host}")
        if not snapshot.ok:
            lines.append(f"- SSH ERROR: {snapshot.error}")
            continue
        lines.append(f"- ssh_user: {snapshot.remote_user or '?'}")
        lines.append("")
        lines.append("| GPU | Mem | Util | Sweep assignment | GPU processes |")
        lines.append("| --- | --- | --- | --- | --- |")
        processes_by_gpu: dict[str, list[GpuProcess]] = defaultdict(list)
        for proc in snapshot.processes or []:
            jobs_for_gpu = running_by_gpu.get((snapshot.host, proc.gpu_index), [])
            proc.kind = classify_process(proc, snapshot, jobs_for_gpu)
            processes_by_gpu[proc.gpu_index].append(proc)
        for gpu in snapshot.gpus or []:
            jobs_for_gpu = running_by_gpu.get((snapshot.host, gpu.index), [])
            assignment = "<br>".join(job_label(state) for state in jobs_for_gpu) or "-"
            procs = processes_by_gpu.get(gpu.index, [])
            proc_text = "<br>".join(format_process(proc) for proc in procs) or "-"
            lines.append(
                f"| {gpu.index} | {human_mem(gpu.memory_used_mib, gpu.memory_total_mib)} | "
                f"{gpu.util_pct if gpu.util_pct is not None else '-'}% | {assignment} | {proc_text} |"
            )

        non_sweep = [
            proc
            for proc in (snapshot.processes or [])
            if proc.kind not in ("sweep", "sweep-slot")
        ]
        if non_sweep:
            lines.append("")
            lines.append("Non-sweep GPU processes:")
            for proc in sorted(non_sweep, key=lambda item: (item.gpu_index, item.user, item.pid)):
                lines.append(f"- gpu{proc.gpu_index}: {format_process(proc)}")
    lines.append("")

    lines.append("## Stop / Tail")
    lines.append("- Reporter loop lock: `/tmp/sweep-progress-reporter.lock`")
    lines.append("- Latest report: `/tmp/sweep-progress-latest.md`")
    lines.append("- History: `/tmp/sweep-progress-history.md`")
    lines.append("- Stop command: `pkill -f sweep-progress-reporter.lock`")
    lines.append("")
    return "\n".join(lines)


def format_counts(counts: Counter[str]) -> str:
    order = ("done", "running", "pending", "skipped", "failed", "known_oom")
    parts = [f"{key}={counts[key]}" for key in order if counts.get(key)]
    for key in sorted(counts):
        if key not in order:
            parts.append(f"{key}={counts[key]}")
    return ", ".join(parts) or "none"


def format_process(proc: GpuProcess) -> str:
    return (
        f"{proc.kind} pid={proc.pid} user={proc.user} mem={human_mem(proc.used_memory_mib)} "
        f"age={human_age(proc.age_seconds)} cmd=`{compact_cmd(proc.cmd)}`"
    )


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=f".{path.name}.") as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def append_history(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip())
        handle.write("\n\n---\n\n")


def run_once(args: argparse.Namespace) -> str:
    try:
        report = build_report(args)
    except Exception as exc:  # noqa: BLE001 - monitor must keep running.
        report = f"# Sweep Progress Snapshot\n\n- generated_at: {now_iso()}\n- health: reporter-error\n- error: `{exc}`\n"
    atomic_write(args.out, report)
    if args.history:
        append_history(args.history, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, default=DEFAULT_JOBS_FILE)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    parser.add_argument("--ssh-timeout", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sweep-progress-latest.md"))
    parser.add_argument("--history", type=Path, default=Path("/tmp/sweep-progress-history.md"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        report = run_once(args)
        first_counts = next((line for line in report.splitlines() if line.startswith("- total_jobs:")), "- total_jobs: unknown")
        print(f"{now_iso()} wrote {args.out} ({first_counts.removeprefix('- ')})", flush=True)
        if args.once:
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    sys.exit(main())
