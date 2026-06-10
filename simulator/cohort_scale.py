"""Cohort context-size spread: trapezoid mean of the measured scale quantiles.

The 2026-06-10 ramp restructure (``kernel_tpot._overflow_weight``) needs one number
per (profile, concurrency, gpu) cell: ``qbar`` — the mean of the cohort's MEASURED
per-session context-size scale distribution, so that
``z = pressure · qbar`` is the distribution-integrated KV demand / pool whose
crossing of 1 (pool overflow) starts eviction.

``qbar`` is the trapezoid mean of ``context_scale_quantiles`` (p0..p100, the
per-session ``median(total_context / per-(conc,turn)-median)`` distribution,
success-filtered) from ``inference-benchmark/data/distributions/*_realized*.json``,
resolved through the EXISTING fallback chain in
``simulator.ramp_tpot.context_scale_quantiles`` (per-conc block → in-file pooled →
legacy pooled; per-GPU file when present). Pure measured workload artifact — no
tuned constant. Profiles without a spec return 1.0 (median-session pressure,
onset at pool-full — the pre-restructure behavior).

Measured values (pooled artifacts, 2026-06-10): swebench 1.1269, terminalbench
1.3463, osworld 0.9834, chat 1.0003 → computed onsets in median-pressure units
(1/qbar): 0.887 / 0.743 / 1.017 / 1.000.

Lives in its own module (not ``kernel_tpot``) because ``ramp_tpot`` imports
``kernel_tpot`` — importing the resolver from ``kernel_tpot`` would be circular.
"""

from __future__ import annotations

from simulator.ramp_tpot import context_scale_quantiles


def trapezoid_mean(quantiles: list[float]) -> float:
    """Mean of a distribution given evenly-spaced quantiles (p0..p100).

    Trapezoid rule over the inverse CDF: E[X] = ∫₀¹ Q(u) du ≈
    (0.5·q₀ + q₁ + … + qₙ₋₂ + 0.5·qₙ₋₁) / (n − 1). Derived math, no constant.
    """
    n = len(quantiles)
    if n == 0:
        return 1.0
    if n == 1:
        return float(quantiles[0])
    return (0.5 * quantiles[0] + sum(quantiles[1:-1]) + 0.5 * quantiles[-1]) / (n - 1)


def cohort_scale_mean(
    profile: str, concurrency: float | None = None, gpu_key: str | None = None
) -> float:
    """``qbar`` for a (profile, concurrency, gpu) cell, or 1.0 when no measured
    quantile artifact exists (→ ``kernel_tpot`` falls back to median-session
    pressure with saturation onset at pool-full)."""
    q = context_scale_quantiles(profile, concurrency, gpu_key)
    if not q:
        return 1.0
    return trapezoid_mean(q)


__all__ = ["cohort_scale_mean", "trapezoid_mean"]
