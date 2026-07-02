# simulator_v2/getters/workload.py

""" Getter for workload files """

import json
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from simulator_v2.core.mode import Mode, mode
from simulator_v2.core.types import Cell, Turn, TurnPrediction

_CONC_RE = re.compile(r"_conc(\d+)\.json$")


@mode(Mode.SHARED)
def trapezoid_mean(quantiles: Sequence[float]) -> float:
    """Mean of a distribution given evenly-spaced quantiles (p0..p100), via the
    trapezoid rule over the inverse CDF: E[X] = ∫₀¹ Q(u) du ≈
    (0.5·q₀ + q₁ + … + qₙ₋₂ + 0.5·qₙ₋₁) / (n − 1). Pure math, no constant."""
    n = len(quantiles)
    if n == 0:
        return 1.0
    if n == 1:
        return float(quantiles[0])
    return (0.5 * quantiles[0] + sum(quantiles[1:-1]) + 0.5 * quantiles[-1]) / (n - 1)


@mode(Mode.SHARED)
def cohort_context_spread(scale_quantiles: Sequence[float]) -> float:
    """The cohort's context-size spread: the mean per-session
    `total_context / median_context`, from its quantiles. The amplifier uses
    `z = pressure · context_spread`. Returns 1.0 when no spread data is available
    (median-session pressure, onset at pool-full -- the no-spread fallback).

    SHARED + data-source-agnostic: the caller supplies the quantiles, so backtest
    feeds the measured realized distribution and forward feeds the input ISL
    distribution -- identical math either way."""
    return trapezoid_mean(scale_quantiles) if scale_quantiles else 1.0


@mode(Mode.BACKTEST)
def _req_tokens(req: dict) -> tuple[float, float, float]:
    """(total_context, cached, new_prefill) tokens for one request, tolerant of
    schema differences: multi-turn rows carry total_context_tokens /
    cached_context_tokens / new_prefill_tokens; single-turn rows carry only
    input_tokens (the whole prompt, all new). Missing new_prefill falls back to
    total - cached."""
    cached = float(req.get("cached_context_tokens") or 0.0)
    total = req.get("total_context_tokens")
    if isinstance(total, (int, float)):
        total = float(total)
    else:
        inp = req.get("input_tokens")
        total = float(inp) if isinstance(inp, (int, float)) else 0.0
    new = req.get("new_prefill_tokens")
    new = float(new) if isinstance(new, (int, float)) else max(0.0, total - cached)
    return total, cached, new


@mode(Mode.BACKTEST)
def _shared_prefix_tokens(reqs: list[dict]) -> float:
    """Median `request_metadata.shared_prefix_actual_tokens` over a turn's requests --
    the profile-constant cross-session APC prefix vLLM dedups (one session prefills it,
    the rest hit). 0.0 when absent (non-prefix-aware workloads)."""
    vals = [
        float((r.get("request_metadata") or {}).get("shared_prefix_actual_tokens"))
        for r in reqs
        if isinstance((r.get("request_metadata") or {}).get("shared_prefix_actual_tokens"), (int, float))
    ]
    vals = [v for v in vals if v > 0.0]
    return statistics.median(vals) if vals else 0.0


@mode(Mode.BACKTEST)
def _parse_cell_meta(path: Path) -> tuple[str, int]:
    match = _CONC_RE.search(path.name)
    if not match:
        raise ValueError(f"benchmark filename must match *_conc<N>.json: {path.name}")
    profile = path.name[: match.start()]
    return profile, int(match.group(1))


@mode(Mode.BACKTEST)
def load_benchmark(path: Path) -> Cell:
    """Load one benchmark JSON into a Cell with per-turn medians + ground truth."""
    profile, concurrency = _parse_cell_meta(path)
    data = json.loads(path.read_text())
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for req in data.get("per_request") or []:
        if not req.get("success"):
            continue
        by_turn[int(req.get("turn_index") or 0)].append(req)

    turns: list[Turn] = []
    ground_truth: list[TurnPrediction] = []
    for turn_index in sorted(by_turn):
        reqs = by_turn[turn_index]

        def med(key: str) -> float:
            vals = [float(r[key]) for r in reqs if isinstance(r.get(key), (int, float))]
            return statistics.median(vals) if vals else 0.0

        def med_of(values: list[float]) -> float:
            return statistics.median(values) if values else 0.0

        toks = [_req_tokens(r) for r in reqs]
        turns.append(
            Turn(
                isl_tokens=med_of([t[0] for t in toks]),
                osl_tokens=max(1.0, med("output_tokens")),
                cache_hit_tokens=med_of([t[1] for t in toks]),
                new_prefill_tokens=med_of([t[2] for t in toks]),
                scheduled_requests=float(len(reqs)),
            )
        )
        ground_truth.append(
            TurnPrediction(
                ttft_ms=round(med("ttft_ms"), 4),
                tpot_ms=round(med("tpot_ms"), 4),
                e2el_ms=round(med("e2el_ms"), 4),
            )
        )

    # Per-session trajectories: each session's successful turns in order. Real
    # session diversity (varied sizes / turn counts) is what de-synchronizes the
    # cohort the queue sim replays.
    by_session: dict = defaultdict(list)
    for i, req in enumerate(data.get("per_request") or []):
        if not req.get("success"):
            continue
        # Single-turn profiles have no session_id; key each unsessioned request to
        # its own single-turn session instead of collapsing them all into one giant
        # trajectory (which the cohort would then replay concurrency-many times).
        sid = req.get("session_id")
        by_session[sid if sid is not None else ("_unsessioned", i)].append(req)

    trajectories: list[list[Turn]] = []
    for reqs in by_session.values():
        reqs.sort(key=lambda r: int(r.get("turn_index") or 0))
        traj = []
        for r in reqs:
            total, cached, new = _req_tokens(r)
            traj.append(
                Turn(
                    isl_tokens=total,
                    osl_tokens=max(1.0, float(r.get("output_tokens", 1.0))),
                    cache_hit_tokens=cached,
                    new_prefill_tokens=new,
                    scheduled_requests=0.0,
                )
            )
        if traj:
            trajectories.append(traj)

    return Cell(
        turns=turns,
        concurrency=concurrency,
        profile=profile,
        ground_truth=ground_truth,
        shared_prefix_tokens=_shared_prefix_tokens(by_turn.get(0, [])),
        trajectories=trajectories or None,
    )


@mode(Mode.FORWARD)
def load_distribution(path: Path) -> Cell:
    """Load an ISL:OSL distribution file into a Cell (no ground truth).

    Expects concurrency (and profile/samples) to be defined in the file itself.
    """
    raise NotImplementedError("DistributionWorkload not yet implemented")
