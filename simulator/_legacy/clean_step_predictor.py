"""Step-type-dispatching pricer for vLLM scheduler steps.

Replaces the kernel-composition pricer in ``vllm_engine_step_cost.py`` with
direct lookups into the three validated step-level predictors documented in
``profiling/docs/prediction_pipeline.yaml``:

- ``full_prefill_ms(N)``  — 1D log-linear over ``prefill_profile_H100_dense.csv``
- ``cached_prefill_ms(U, P)`` — 2D bilinear-in-log over ``cached_prefill_v3_H100.csv``
- ``decode_ms(B, T)``     — 2D bilinear-in-log over ``decode_profile_H100_large_2026-05-17.csv``

The dispatcher ``step_ms`` classifies a scheduler step:

- ``decode_only`` (B > 0, prefill = 0)   → ``decode_ms(B, T_avg)``
- ``prefill_only`` (B = 0, prefill > 0)  → sum of ``cached_prefill_ms`` per chunk
- ``mixed``                              → ``max(prefill_total, decode_total)``

The ``max`` semantic captures the GPU overlap: in real vLLM, decode and prefill
compute share kernels (one GEMM on combined tokens) and the wall is dominated
by whichever side has more work.  See
``profiling/docs/scheduler-shape-correction-2026-05-25.md`` for the trace
evidence.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from simulator._legacy.vllm_scheduler_shape import VllmPrefillChunk, VllmStepShape


def _log2(value: float) -> float:
    return math.log2(max(1.0, float(value)))


def _log_linear_1d(samples: Sequence[tuple[int, float]], x: int) -> float:
    """Piecewise log-linear interpolation over (x, y).

    Clamps at endpoints.  Operates in log2 space on x.
    """
    if not samples:
        return 0.0
    points = sorted(samples)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        if points[i][0] <= x <= points[i + 1][0]:
            lo, hi = points[i], points[i + 1]
            break
    log_x_lo, log_x_hi, log_x = _log2(lo[0]), _log2(hi[0]), _log2(x)
    if log_x_hi == log_x_lo:
        return lo[1]
    frac = (log_x - log_x_lo) / (log_x_hi - log_x_lo)
    return lo[1] + frac * (hi[1] - lo[1])


def _bilinear_log2d(
    lattice: Mapping[tuple[int, int], float], x: int, y: int
) -> float:
    """2D bilinear interpolation in log2(x) × log2(y) space.

    Falls back to nearest neighbor when corner vertices are missing.
    Clamps at lattice bounds.
    """
    if not lattice:
        return 0.0
    xs = sorted({k[0] for k in lattice})
    ys = sorted({k[1] for k in lattice})
    x = max(xs[0], min(xs[-1], x))
    y = max(ys[0], min(ys[-1], y))

    def _bracket(values: Sequence[int], target: int) -> tuple[int, int]:
        lo = values[0]
        hi = values[-1]
        for v in values:
            if v <= target:
                lo = v
            if v >= target:
                hi = v
                break
        return lo, hi

    x_lo, x_hi = _bracket(xs, x)
    y_lo, y_hi = _bracket(ys, y)
    corners = [
        (x_lo, y_lo),
        (x_hi, y_lo),
        (x_lo, y_hi),
        (x_hi, y_hi),
    ]
    if not all(c in lattice for c in corners):
        # Nearest available corner.
        best = min(
            lattice,
            key=lambda k: (abs(_log2(k[0]) - _log2(x)) + abs(_log2(k[1]) - _log2(y))),
        )
        return lattice[best]

    log_x_lo = _log2(x_lo)
    log_x_hi = _log2(x_hi)
    log_y_lo = _log2(y_lo)
    log_y_hi = _log2(y_hi)
    log_x = _log2(x)
    log_y = _log2(y)
    fx = 0.0 if log_x_hi == log_x_lo else (log_x - log_x_lo) / (log_x_hi - log_x_lo)
    fy = 0.0 if log_y_hi == log_y_lo else (log_y - log_y_lo) / (log_y_hi - log_y_lo)
    return (
        (1 - fx) * (1 - fy) * lattice[(x_lo, y_lo)]
        + fx * (1 - fy) * lattice[(x_hi, y_lo)]
        + (1 - fx) * fy * lattice[(x_lo, y_hi)]
        + fx * fy * lattice[(x_hi, y_hi)]
    )


@dataclass(frozen=True)
class CleanStepCost:
    """Decomposed cost so callers can still see the dispatch result."""

    decode_ms: float
    prefill_ms: float
    total_ms: float
    classification: str  # "decode_only", "prefill_only", "mixed", "empty"


class CleanStepPredictor:
    """Wraps the 3 validated step-level CSVs.

    Use ``from_csvs`` to construct from default paths, then call
    ``step_ms(step, context_lens)`` to price one scheduler step.
    """

    def __init__(
        self,
        full_prefill_samples: Sequence[tuple[int, float]],
        cached_prefill_lattice: Mapping[tuple[int, int], float],
        decode_lattice: Mapping[tuple[int, int], float],
    ) -> None:
        self._full_prefill = tuple(sorted(full_prefill_samples))
        self._cached_prefill = dict(cached_prefill_lattice)
        self._decode = dict(decode_lattice)

    @classmethod
    def from_csvs(
        cls,
        *,
        full_prefill_path: Path,
        cached_prefill_path: Path,
        decode_path: Path,
    ) -> "CleanStepPredictor":
        full_prefill: list[tuple[int, float]] = []
        with full_prefill_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    full_prefill.append(
                        (int(row["prefill_tokens"]), float(row["prefill_ms"]))
                    )
                except (KeyError, ValueError):
                    continue
        cached: dict[tuple[int, int], float] = {}
        with cached_prefill_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    cached[(int(row["U"]), int(row["P"]))] = float(row["prefill_ms"])
                except (KeyError, ValueError):
                    continue
        decode: dict[tuple[int, int], float] = {}
        with decode_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    decode[(int(row["batch_size"]), int(row["context_len"]))] = float(
                        row["decode_step_ms"]
                    )
                except (KeyError, ValueError):
                    continue
        return cls(full_prefill, cached, decode)

    def full_prefill_ms(self, N: int) -> float:
        return max(0.0, _log_linear_1d(self._full_prefill, int(N)))

    def cached_prefill_ms(self, U: int, P: int) -> float:
        if P <= 0:
            return self.full_prefill_ms(U)
        return max(0.0, _bilinear_log2d(self._cached_prefill, int(U), int(P)))

    def decode_ms(self, B: int, T: int) -> float:
        return max(0.0, _bilinear_log2d(self._decode, int(B), int(T)))

    def prefill_chunks_ms(self, chunks: Sequence[VllmPrefillChunk]) -> float:
        total = 0.0
        for chunk in chunks:
            U = max(1, int(chunk.scheduled_tokens))
            P = max(0, int(chunk.prefix_tokens))
            total += self.cached_prefill_ms(U, P)
        return total

    def step_ms(
        self, step: VllmStepShape, context_lens: Mapping[int, int]
    ) -> CleanStepCost:
        """Price one scheduler step via type dispatch.

        ``context_lens`` maps request_id → current resident context length
        for decoded requests; used to compute the average T for decode.
        """
        prefill_total = self.prefill_chunks_ms(step.prefill_chunks)
        decode_total = 0.0
        if step.decode_batch > 0 and step.decoded_request_ids:
            ts = [
                int(context_lens.get(rid, 0))
                for rid in step.decoded_request_ids
                if context_lens.get(rid) is not None
            ]
            ts = [t for t in ts if t > 0]
            t_avg = int(sum(ts) / len(ts)) if ts else 1
            decode_total = self.decode_ms(step.decode_batch, t_avg)

        if step.prefill_tokens > 0 and step.decode_batch > 0:
            return CleanStepCost(
                decode_ms=decode_total,
                prefill_ms=prefill_total,
                total_ms=max(prefill_total, decode_total),
                classification="mixed",
            )
        if step.prefill_tokens > 0:
            return CleanStepCost(
                decode_ms=0.0,
                prefill_ms=prefill_total,
                total_ms=prefill_total,
                classification="prefill_only",
            )
        if step.decode_batch > 0:
            return CleanStepCost(
                decode_ms=decode_total,
                prefill_ms=0.0,
                total_ms=decode_total,
                classification="decode_only",
            )
        return CleanStepCost(
            decode_ms=0.0,
            prefill_ms=0.0,
            total_ms=0.0,
            classification="empty",
        )
