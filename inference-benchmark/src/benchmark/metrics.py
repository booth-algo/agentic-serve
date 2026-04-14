"""
Metrics aggregation for benchmark results.

Computes p50/p90/p99 for TTFT, TPOT, ITL, E2EL.
Tracks successful vs failed requests separately.
Reports input tok/s and output tok/s separately (not just total).
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics
import json
import time


@dataclass
class RequestResult:
    """Per-request benchmark result. Shared across all backends."""
    success: bool
    ttft: Optional[float] = None          # seconds to first token
    itl: list = field(default_factory=list)  # inter-token latencies (seconds)
    e2el: Optional[float] = None          # end-to-end latency (seconds)
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    turn_index: Optional[int] = None      # multi-turn: which turn (0-indexed)

    @property
    def tpot(self) -> Optional[float]:
        """Time per output token (mean ITL), excluding first token.

        Falls back to (e2el - ttft) / output_tokens when ITL data is
        missing (e.g. models that don't stream token-by-token).
        """
        if self.itl:
            return sum(self.itl) / len(self.itl)
        # Fallback: compute from e2el and ttft
        if self.e2el is not None and self.ttft is not None and self.output_tokens > 1:
            decode_time = self.e2el - self.ttft
            return decode_time / (self.output_tokens - 1)
        return None


@dataclass
class BenchmarkSummary:
    """Aggregated metrics for a benchmark run."""

    # Run config
    model: str = ""
    profile: str = ""
    concurrency: int = 0
    num_requests: int = 0
    duration_s: float = 0.0

    # Request counts
    successful_requests: int = 0
    failed_requests: int = 0

    # Throughput
    request_throughput: float = 0.0     # req/s
    input_token_throughput: float = 0.0  # input tok/s
    output_token_throughput: float = 0.0  # output tok/s
    total_token_throughput: float = 0.0   # (input + output) tok/s

    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # TTFT (ms)
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    # TPOT / mean ITL (ms) — time per output token excluding first
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    # ITL (ms) — individual inter-token latencies (all tokens pooled)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # E2EL (ms)
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "errors"}
        d["errors"] = self.errors[:10]  # cap error list in JSON
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) of a sorted or unsorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def aggregate(results, duration_s: float, model: str = "", profile: str = "", concurrency: int = 0) -> BenchmarkSummary:
    """
    Aggregate a list of RequestResult into a BenchmarkSummary.

    Args:
        results: list of RequestResult from client.py
        duration_s: total wall-clock time for the benchmark run
        model: model name for labeling
        profile: workload profile name for labeling
        concurrency: concurrency level used
    """
    summary = BenchmarkSummary(
        model=model,
        profile=profile,
        concurrency=concurrency,
        num_requests=len(results),
        duration_s=duration_s,
    )

    ttfts = []
    tpots = []
    itls = []   # all individual inter-token latencies pooled across requests
    e2els = []

    for r in results:
        if r.success:
            summary.successful_requests += 1
            summary.total_input_tokens += r.input_tokens
            summary.total_output_tokens += r.output_tokens

            if r.ttft is not None:
                ttfts.append(r.ttft * 1000)  # convert to ms
            if r.tpot is not None:
                tpots.append(r.tpot * 1000)
            if r.itl:
                itls.extend(t * 1000 for t in r.itl)  # convert to ms
            if r.e2el is not None:
                e2els.append(r.e2el * 1000)
        else:
            summary.failed_requests += 1
            if r.error:
                summary.errors.append(r.error)

    if duration_s > 0:
        summary.request_throughput = summary.successful_requests / duration_s
        summary.input_token_throughput = summary.total_input_tokens / duration_s
        summary.output_token_throughput = summary.total_output_tokens / duration_s
        summary.total_token_throughput = (
            summary.total_input_tokens + summary.total_output_tokens
        ) / duration_s

    if ttfts:
        summary.mean_ttft_ms = statistics.mean(ttfts)
        summary.median_ttft_ms = statistics.median(ttfts)
        summary.p90_ttft_ms = _percentile(ttfts, 90)
        summary.p99_ttft_ms = _percentile(ttfts, 99)

    if tpots:
        summary.mean_tpot_ms = statistics.mean(tpots)
        summary.median_tpot_ms = statistics.median(tpots)
        summary.p90_tpot_ms = _percentile(tpots, 90)
        summary.p99_tpot_ms = _percentile(tpots, 99)

    if itls:
        summary.mean_itl_ms = statistics.mean(itls)
        summary.median_itl_ms = statistics.median(itls)
        summary.p90_itl_ms = _percentile(itls, 90)
        summary.p99_itl_ms = _percentile(itls, 99)

    if e2els:
        summary.mean_e2el_ms = statistics.mean(e2els)
        summary.median_e2el_ms = statistics.median(e2els)
        summary.p90_e2el_ms = _percentile(e2els, 90)
        summary.p99_e2el_ms = _percentile(e2els, 99)

    return summary


@dataclass
class TurnSummary:
    """Per-turn metrics for multi-turn benchmarks."""
    turn_index: int
    num_requests: int = 0
    successful: int = 0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def aggregate_per_turn(results_by_turn: dict[int, list]) -> list[TurnSummary]:
    """Aggregate metrics per turn for multi-turn benchmarks."""
    summaries = []
    for turn_idx in sorted(results_by_turn.keys()):
        results = results_by_turn[turn_idx]
        if not results:
            continue

        ts = TurnSummary(turn_index=turn_idx)
        ts.num_requests = len(results)

        ttfts = []
        tpots = []
        e2els = []
        input_toks = []
        output_toks = []

        for r in results:
            if r.success:
                ts.successful += 1
                if r.ttft is not None:
                    ttfts.append(r.ttft * 1000)
                if r.tpot is not None:
                    tpots.append(r.tpot * 1000)
                if r.e2el is not None:
                    e2els.append(r.e2el * 1000)
                input_toks.append(r.input_tokens)
                output_toks.append(r.output_tokens)

        if ttfts:
            ts.mean_ttft_ms = statistics.mean(ttfts)
            ts.median_ttft_ms = statistics.median(ttfts)
            ts.p90_ttft_ms = _percentile(ttfts, 90)
            ts.p99_ttft_ms = _percentile(ttfts, 99)
        if tpots:
            ts.mean_tpot_ms = statistics.mean(tpots)
            ts.median_tpot_ms = statistics.median(tpots)
        if e2els:
            ts.mean_e2el_ms = statistics.mean(e2els)
            ts.median_e2el_ms = statistics.median(e2els)
        if input_toks:
            ts.avg_input_tokens = statistics.mean(input_toks)
        if output_toks:
            ts.avg_output_tokens = statistics.mean(output_toks)

        summaries.append(ts)
    return summaries


def print_multi_turn_summary(turn_summaries: list, overall: BenchmarkSummary) -> None:
    """Print per-turn metrics table for multi-turn benchmarks."""
    print_summary(overall)
    print(f"{'=' * 72}")
    print(f" Per-Turn Breakdown (prefix cache effect visible in TTFT trend)")
    print(f"{'=' * 72}")
    print(f" {'Turn':>4}  {'Reqs':>5}  {'Avg ISL':>8}  {'TTFT p50':>9}  {'TTFT p90':>9}  {'TPOT p50':>9}  {'E2EL p50':>9}")
    print(f" {'─' * 4}  {'─' * 5}  {'─' * 8}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}")
    for ts in turn_summaries:
        print(f" {ts.turn_index + 1:>4}  {ts.successful:>5}  {ts.avg_input_tokens:>8.0f}  "
              f"{ts.median_ttft_ms:>8.1f}ms  {ts.p90_ttft_ms:>8.1f}ms  "
              f"{ts.median_tpot_ms:>8.1f}ms  {ts.median_e2el_ms:>8.1f}ms")
    print(f"{'=' * 72}\n")


def print_summary(s: BenchmarkSummary) -> None:
    """Print a formatted benchmark summary to stdout."""
    print(f"\n{'=' * 52}")
    print(f" Benchmark Results: {s.profile} | concurrency={s.concurrency}")
    print(f"{'=' * 52}")
    print(f" Model:                    {s.model}")
    print(f" Duration:                 {s.duration_s:.2f}s")
    print(f" Requests:                 {s.successful_requests} ok / {s.failed_requests} failed")
    print(f" Request throughput:       {s.request_throughput:.2f} req/s")
    print(f" Input token throughput:   {s.input_token_throughput:.0f} tok/s")
    print(f" Output token throughput:  {s.output_token_throughput:.0f} tok/s")
    print(f" Total token throughput:   {s.total_token_throughput:.0f} tok/s")
    print(f"{'─' * 52}")
    print(f" TTFT  mean/p50/p90/p99:   {s.mean_ttft_ms:.1f} / {s.median_ttft_ms:.1f} / {s.p90_ttft_ms:.1f} / {s.p99_ttft_ms:.1f} ms")
    print(f" TPOT  mean/p50/p90/p99:   {s.mean_tpot_ms:.1f} / {s.median_tpot_ms:.1f} / {s.p90_tpot_ms:.1f} / {s.p99_tpot_ms:.1f} ms")
    print(f" ITL   mean/p50/p90/p99:   {s.mean_itl_ms:.1f} / {s.median_itl_ms:.1f} / {s.p90_itl_ms:.1f} / {s.p99_itl_ms:.1f} ms")
    print(f" E2EL  mean/p50/p90/p99:   {s.mean_e2el_ms:.1f} / {s.median_e2el_ms:.1f} / {s.p90_e2el_ms:.1f} / {s.p99_e2el_ms:.1f} ms")
    print(f"{'=' * 52}\n")
    if s.errors:
        print(f" Errors ({len(s.errors)} total, first {min(3,len(s.errors))}):")
        for e in s.errors[:3]:
            print(f"   {e}")
        print()
