"""Aggregate a per-step vLLM engine-trace JSONL into a turn-level summary CSV.

The summary CSV matches the schema of existing
``profile_data/results/vllm_engine_step_trace_*_summary.csv`` files so the
``compare_sim_to_engine_trace`` comparator can consume it without changes.

Each input JSONL record is one scheduler step.  Records are grouped by
``(profile, concurrency, turn_index)`` and rolled up to one summary row.
Profile aliases let us reconcile capture-time names with predictor-time
names (e.g. ``swebench-multiturn`` from the live server captures with
``swebench-multiturn-synth`` used by the benchmark CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SUMMARY_FIELDS = (
    "profile",
    "concurrency",
    "turn_index",
    "primary_eval",
    "scheduled_request_count",
    "batch_size",
    "context_len",
    "output_tokens",
    "new_prefill_tokens",
    "cache_hit_rate",
    "steps",
    "decode_only_steps",
    "prefill_only_steps",
    "mixed_decode_prefill_steps",
    "empty_or_unknown_steps",
    "total_decode_slots",
    "total_prefill_tokens",
    "total_scheduled_tokens",
    "max_decode_batch",
    "mean_decode_batch",
    "mean_decode_batch_ratio",
    "max_waiting_queue",
    "max_running_queue",
    "min_free_kv_blocks",
    "total_preemptions",
    "total_swaps",
    "total_recomputes",
    "prefill_intrusion_candidate",
    "effective_microbatching_candidate",
    "decode_wave_candidate",
    "kv_pressure_candidate",
)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_nonempty(records: list[dict], key: str) -> Any:
    for r in records:
        v = r.get(key)
        if v not in (None, ""):
            return v
    return ""


def _parse_alias_arg(alias_args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in alias_args:
        if "=" not in entry:
            raise SystemExit(f"--profile-alias expects from=to, got {entry!r}")
        src, dst = entry.split("=", 1)
        out[src.strip()] = dst.strip()
    return out


def _resolve_turn_index(record: Mapping[str, Any]) -> str:
    raw = record.get("turn_index")
    if raw in (None, ""):
        raw = record.get("target_turn_index")
    return "" if raw in (None, "") else str(raw)


def aggregate(records: list[dict], profile_alias: Mapping[str, str]) -> list[dict]:
    """Return one summary row per (profile, concurrency, turn_index) group."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        profile = str(rec.get("profile", ""))
        profile = profile_alias.get(profile, profile)
        concurrency = str(rec.get("concurrency", ""))
        turn = _resolve_turn_index(rec)
        if not profile or not concurrency or not turn:
            continue
        groups[(profile, concurrency, turn)].append(rec)

    out: list[dict] = []
    for (profile, concurrency, turn), step_records in sorted(groups.items()):
        steps = len(step_records)
        decode_batches = [_to_int(r.get("decode_batch")) for r in step_records]
        prefill_tokens = [_to_int(r.get("prefill_tokens")) for r in step_records]
        total_scheduled = [_to_int(r.get("total_scheduled_tokens")) for r in step_records]
        free_kv = [_to_int(r.get("free_kv_blocks")) for r in step_records if r.get("free_kv_blocks") not in (None, "")]
        waiting = [_to_int(r.get("waiting_queue")) for r in step_records]
        running = [_to_int(r.get("running_queue")) for r in step_records]
        preemptions = [_to_int(r.get("preemptions")) for r in step_records]
        swaps = [_to_int(r.get("swaps")) for r in step_records]
        recomputes = [_to_int(r.get("recomputes")) for r in step_records]

        decode_only = sum(1 for d, p in zip(decode_batches, prefill_tokens) if d > 0 and p == 0)
        prefill_only = sum(1 for d, p in zip(decode_batches, prefill_tokens) if d == 0 and p > 0)
        mixed = sum(1 for d, p in zip(decode_batches, prefill_tokens) if d > 0 and p > 0)
        empty = sum(1 for d, p in zip(decode_batches, prefill_tokens) if d == 0 and p == 0)

        max_decode = max(decode_batches, default=0)
        mean_decode = statistics.fmean(decode_batches) if decode_batches else 0.0
        scheduled = _to_int(_first_nonempty(step_records, "scheduled_request_count"))
        mean_decode_ratio = (mean_decode / scheduled) if scheduled > 0 else 0.0

        intrusion = mixed > 0
        microbatch = scheduled > 0 and 0 < max_decode < scheduled
        decode_wave = scheduled > 0 and 0.0 < mean_decode_ratio < 0.75
        kv_pressure = (free_kv and min(free_kv) <= 0) or any(w > 0 for w in waiting)

        out.append({
            "profile": profile,
            "concurrency": concurrency,
            "turn_index": turn,
            "primary_eval": _first_nonempty(step_records, "primary_eval"),
            "scheduled_request_count": scheduled,
            "batch_size": _to_int(_first_nonempty(step_records, "batch_size"), default=scheduled),
            "context_len": _first_nonempty(step_records, "context_len"),
            "output_tokens": _first_nonempty(step_records, "output_tokens"),
            "new_prefill_tokens": _first_nonempty(step_records, "new_prefill_tokens"),
            "cache_hit_rate": _first_nonempty(step_records, "cache_hit_rate"),
            "steps": steps,
            "decode_only_steps": decode_only,
            "prefill_only_steps": prefill_only,
            "mixed_decode_prefill_steps": mixed,
            "empty_or_unknown_steps": empty,
            "total_decode_slots": sum(decode_batches),
            "total_prefill_tokens": sum(prefill_tokens),
            "total_scheduled_tokens": sum(total_scheduled),
            "max_decode_batch": max_decode,
            "mean_decode_batch": f"{mean_decode:.4f}",
            "mean_decode_batch_ratio": f"{mean_decode_ratio:.4f}",
            "max_waiting_queue": max(waiting, default=0),
            "max_running_queue": max(running, default=0),
            "min_free_kv_blocks": min(free_kv) if free_kv else 0,
            "total_preemptions": sum(preemptions),
            "total_swaps": sum(swaps),
            "total_recomputes": sum(recomputes),
            "prefill_intrusion_candidate": "true" if intrusion else "false",
            "effective_microbatching_candidate": "true" if microbatch else "false",
            "decode_wave_candidate": "true" if decode_wave else "false",
            "kv_pressure_candidate": "true" if kv_pressure else "false",
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Per-step JSONL trace.")
    p.add_argument("--output", type=Path, required=True, help="Destination summary CSV.")
    p.add_argument(
        "--profile-alias",
        action="append",
        default=[],
        help="Map source profile name to canonical (e.g. swebench-multiturn=swebench-multiturn-synth).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aliases = _parse_alias_arg(args.profile_alias)
    records: list[dict] = []
    with args.input.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise SystemExit(f"no records parsed from {args.input}")
    rows = aggregate(records, aliases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    print(f"Wrote {args.output} ({len(rows)} turns)")


if __name__ == "__main__":
    main()
