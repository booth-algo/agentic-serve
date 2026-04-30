#!/usr/bin/env python3
"""Build compact synthetic-workload distributions from existing traces.

This is intentionally a no-op with respect to benchmark behavior: it only reads
existing trace/result files and writes JSON summaries under data/distributions/.
The runner/profile/dashboard wiring should consume these artifacts in a later
phase after the distributions have been inspected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = DATA_DIR / "distributions"

TRAJECTORY_SOURCES = {
    "swebench_multiturn": DATA_DIR / "swebench_trajectories.jsonl",
    "terminalbench_multiturn": DATA_DIR / "terminalbench_trajectories.jsonl",
    "osworld_multiturn": DATA_DIR / "osworld_trajectories.jsonl",
}

CHAT_RESULT_PROFILES = {
    "chat-multiturn-short",
    "chat-multiturn-medium",
    "chat-multiturn-long",
}

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class TurnSample:
    turn_index: int
    total_context_tokens: int
    new_prefill_tokens: int
    output_tokens: int
    cache_hit_rate: float


@dataclass
class SessionSample:
    session_id: str
    turn_count: int
    turns: list[TurnSample]
    context_decrease_turns: int = 0
    context_non_growth_turns: int = 0
    estimated_context_turns: int = 0


def estimate_tokens(value: Any) -> int:
    """Estimate tokens using the same coarse word ratio used by datasets.py."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.35))


def message_tokens(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def stats(values: Iterable[float], *, round_digits: int = 3) -> dict[str, float | int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    out: dict[str, float | int] = {
        "count": len(vals),
        "min": round(min(vals), round_digits),
        "mean": round(statistics.fmean(vals), round_digits),
        "max": round(max(vals), round_digits),
    }
    for q in QUANTILES:
        key = f"p{int(q * 100):02d}"
        out[key] = round(percentile(vals, q), round_digits)
    return out


def int_histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def round_turn_sample(sample: TurnSample) -> dict[str, int | float]:
    return {
        "turn_index": sample.turn_index,
        "total_context_tokens": sample.total_context_tokens,
        "new_prefill_tokens": sample.new_prefill_tokens,
        "output_tokens": sample.output_tokens,
        "cache_hit_rate": round(sample.cache_hit_rate, 4),
    }


def build_trajectory_distribution(name: str, path: Path) -> dict[str, Any]:
    sessions: list[SessionSample] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            raw_turns = row.get("turns") or []
            turns: list[TurnSample] = []
            previous_context = 0
            context_decrease_turns = 0
            context_non_growth_turns = 0
            estimated_context_turns = 0
            for idx, turn in enumerate(raw_turns):
                messages = turn.get("messages") or []
                total_context_raw = turn.get("input_tokens")
                if total_context_raw is None:
                    estimated_context_turns += 1
                total_context = int(total_context_raw or message_tokens(messages))
                if total_context <= 0:
                    continue
                output_tokens = int(
                    turn.get("output_tokens")
                    or turn.get("osl_tokens")
                    or turn.get("max_tokens")
                    or 1
                )
                if turns and total_context < previous_context:
                    context_decrease_turns += 1
                if turns and total_context <= previous_context:
                    context_non_growth_turns += 1
                new_prefill = max(1, total_context - previous_context)
                cache_hit_rate = max(0.0, min(1.0, 1.0 - new_prefill / total_context))
                turns.append(
                    TurnSample(
                        turn_index=int(turn.get("turn_idx", idx)),
                        total_context_tokens=total_context,
                        new_prefill_tokens=new_prefill,
                        output_tokens=max(1, output_tokens),
                        cache_hit_rate=cache_hit_rate,
                    )
                )
                previous_context = total_context

            if not turns:
                skipped += 1
                continue
            sessions.append(
                SessionSample(
                    session_id=str(row.get("session_id", len(sessions))),
                    turn_count=len(turns),
                    turns=turns,
                    context_decrease_turns=context_decrease_turns,
                    context_non_growth_turns=context_non_growth_turns,
                    estimated_context_turns=estimated_context_turns,
                )
            )

    return build_distribution_json(
        name=name,
        source_kind="trajectory_jsonl",
        source_path=path,
        sessions=sessions,
        skipped_sessions=skipped,
    )


def build_chat_distribution_from_results(name: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    sessions: list[SessionSample] = []
    skipped = 0
    for idx, row in enumerate(rows):
        config = row.get("config") or {}
        profile = str(config.get("profile") or "")
        if profile not in CHAT_RESULT_PROFILES:
            continue
        per_turn = row.get("perTurn")
        if not per_turn:
            skipped += 1
            continue

        turns: list[TurnSample] = []
        previous_context = 0
        for turn in per_turn:
            total_context = int(round(
                turn.get("median_input_tokens")
                or turn.get("avg_input_tokens")
                or 0
            ))
            if total_context <= 0:
                continue
            output_tokens = int(round(
                turn.get("median_output_tokens")
                or turn.get("avg_output_tokens")
                or 1
            ))
            new_prefill = int(round(
                turn.get("median_new_prefill_tokens")
                or max(1, total_context - previous_context)
            ))
            cache_hit_rate = float(
                turn.get("median_cache_hit_rate")
                if turn.get("median_cache_hit_rate") is not None
                else max(0.0, min(1.0, 1.0 - new_prefill / total_context))
            )
            turns.append(
                TurnSample(
                    turn_index=int(turn.get("turn_index", len(turns))),
                    total_context_tokens=total_context,
                    new_prefill_tokens=max(1, new_prefill),
                    output_tokens=max(1, output_tokens),
                    cache_hit_rate=max(0.0, min(1.0, cache_hit_rate)),
                )
            )
            previous_context = total_context

        if not turns:
            skipped += 1
            continue
        session_id = f"{profile}:{config.get('backend', 'unknown')}:{config.get('concurrency', 'unknown')}:{idx}"
        sessions.append(
            SessionSample(
                session_id=session_id,
                turn_count=len(turns),
                turns=turns,
            )
        )

    if not sessions:
        return None
    return build_distribution_json(
        name=name,
        source_kind="dashboard_per_turn_summary",
        source_path=path,
        sessions=sessions,
        skipped_sessions=skipped,
        note=(
            "ShareGPT raw multi-turn source is not stored locally, so this "
            "artifact is derived from existing per-turn benchmark summaries."
        ),
    )


def build_distribution_json(
    *,
    name: str,
    source_kind: str,
    source_path: Path,
    sessions: list[SessionSample],
    skipped_sessions: int,
    note: str | None = None,
) -> dict[str, Any]:
    turns = [turn for session in sessions for turn in session.turns]
    by_turn: dict[int, list[TurnSample]] = defaultdict(list)
    for turn in turns:
        by_turn[turn.turn_index].append(turn)
    context_decrease_turns = sum(s.context_decrease_turns for s in sessions)
    context_non_growth_turns = sum(s.context_non_growth_turns for s in sessions)
    estimated_context_turns = sum(s.estimated_context_turns for s in sessions)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": source_kind,
            "path": str(source_path.relative_to(ROOT)),
            "sessions": len(sessions),
            "turns": len(turns),
            "skipped_sessions": skipped_sessions,
        },
        "token_estimator": "estimated_tokens = int(word_count * 1.35) when raw input_tokens are absent",
        "summary": {
            "turn_count": stats([s.turn_count for s in sessions], round_digits=2),
            "total_context_tokens": stats([t.total_context_tokens for t in turns], round_digits=2),
            "new_prefill_tokens": stats([t.new_prefill_tokens for t in turns], round_digits=2),
            "output_tokens": stats([t.output_tokens for t in turns], round_digits=2),
            "cache_hit_rate": stats([t.cache_hit_rate for t in turns], round_digits=4),
        },
        "diagnostics": {
            "context_decrease_turns": context_decrease_turns,
            "context_non_growth_turns": context_non_growth_turns,
            "estimated_context_turns": estimated_context_turns,
            "estimated_context_turn_fraction": (
                round(estimated_context_turns / len(turns), 4) if turns else 0.0
            ),
            "note": (
                "Context deltas are token-count estimates. Non-growth or decreases "
                "mean the source rows may not be literal prefix-growing prompts; "
                "future sampling should treat those deltas as approximate."
            ),
        },
        "histograms": {
            "turn_count": int_histogram(s.turn_count for s in sessions),
        },
        "samples": {
            "turn_count": [s.turn_count for s in sessions],
            "turns": [round_turn_sample(t) for t in turns],
        },
        "by_turn_index": [
            {
                "turn_index": idx,
                "num_samples": len(samples),
                "total_context_tokens": stats([t.total_context_tokens for t in samples], round_digits=2),
                "new_prefill_tokens": stats([t.new_prefill_tokens for t in samples], round_digits=2),
                "output_tokens": stats([t.output_tokens for t in samples], round_digits=2),
                "cache_hit_rate": stats([t.cache_hit_rate for t in samples], round_digits=4),
            }
            for idx, samples in sorted(by_turn.items())
        ],
    }
    if note:
        payload["note"] = note
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for distribution JSON files.",
    )
    parser.add_argument(
        "--dashboard-data",
        type=Path,
        default=ROOT / "dashboard" / "public" / "data.json",
        help="Dashboard data.json used only to derive chat_multiturn when perTurn summaries exist.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Do not derive chat_multiturn from dashboard per-turn summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrote: list[Path] = []

    for name, path in TRAJECTORY_SOURCES.items():
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue
        payload = build_trajectory_distribution(name, path)
        out_path = args.out_dir / f"{name}.json"
        write_json(out_path, payload)
        wrote.append(out_path)
        print(
            f"[write] {out_path} "
            f"({payload['source']['sessions']} sessions, {payload['source']['turns']} turns)"
        )

    if not args.skip_chat:
        payload = build_chat_distribution_from_results("chat_multiturn", args.dashboard_data)
        if payload is None:
            print(f"[skip] chat_multiturn: no usable perTurn summaries in {args.dashboard_data}")
        else:
            out_path = args.out_dir / "chat_multiturn.json"
            write_json(out_path, payload)
            wrote.append(out_path)
            print(
                f"[write] {out_path} "
                f"({payload['source']['sessions']} summary rows, {payload['source']['turns']} turns)"
            )

    print(f"[done] wrote {len(wrote)} distribution files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
