"""Forward (workload-only) TTFT predictor for vLLM / H100 / Llama-3.1-8B.

TTFT is **not** a quasi-static per-step quantity like TPOT — it is the wall-clock
**queue wait** a request sees before its first token (client_queue_wait≈0, so all
server-side queueing is folded into the measured ttft_ms). Within a (profile,
concurrency) cell the per-turn TTFT is a saturate-ramp-recover curve driven by the
cohort: backlog builds while the cohort is oversubscribed and drains as sessions
finish. See profiling/docs + the `ttft-e2el-scoping` memory.

Model (strictly physical — no fitted constants without basis):

    TTFT[t] = baseline_prefill[t]                               # queue-free floor
            + RESIDUAL · min(1, pressure) · sched · decode_step  # sub-saturation wait
            + max(0, pressure − 1) · output · tpot               # oversubscription backlog

  * baseline_prefill — queue-free single-session prefill: the measured cached-prefill
    step grid for new_prefill ≤ 1024, else the prefill-compute roofline
    (2·N·U / (peak_flops·util)).
  * sub-saturation term — a uniformly-arriving request waits, on average, the mean
    *residual service life* of the work already in flight. For (roughly)
    deterministic service the renewal/Pollaczek–Khinchine residual is S/2, hence
    RESIDUAL = 0.5. Scaled by utilization min(1, pressure) so it vanishes when the
    server is idle, and by the in-flight count `sched` × the per-step decode wall
    (`decode_step_ms`) — the work the arrival queues behind. Output-independent.
  * oversubscription term — once KV demand exceeds capacity (pressure > 1), the
    excess is a backlog measured in *turns*: each unit of oversubscription costs one
    turn-decode (output · tpot) of waiting. This is the regime that dominates TTFT
    magnitude and collapses tightly onto pressure−1 in the measurements.

`pressure`, `sched_hat` (forward cohort drain from the profile survival curve) and
the measured kernel grids are all reused from the TPOT stack — see [[ramp_tpot]],
[[kernel_step_cost]], [[kernel_tpot]]. `oracle=True` swaps the forward cohort for the
measured `scheduled_requests` to isolate amplifier-shape error from drain-forecast
error (forward ≈ oracle here, so the residual error is the model, not the drain).

Honest accuracy (forward, measured 2026-05-29): overall ~61% MAPE — the high-pressure
oversubscription regime is well-captured, but the sub-saturation contention wait does
not scale cleanly with workload aggregates (it is an arrival-rate effect), so a static
per-turn model has a floor here. The faithful path to TPOT-level accuracy is a
wall-clock multi-turn queue simulation.
"""

from __future__ import annotations

import math
from typing import Any

from simulator.cached_prefill_lookup import cached_prefill_step_ms
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator.ramp_tpot import PROFILE_DIST, sched_hat

__all__ = ["PROFILE_DIST", "predict_turn_ttft", "predict_cell_ttft"]

# Renewal / Pollaczek–Khinchine mean-residual-life factor for ~deterministic
# service: a random arrival waits on average S/2 of the in-flight work. Physical,
# not fitted.
RESIDUAL = 0.5

# Largest new-prefill the measured cached-prefill grid covers; above it the grid
# clamps and under-counts, so fall back to the prefill-compute roofline.
_GRID_U_MAX = 1024.0


def _prefill_per_token_ms(params: RooflineParams) -> float:
    """Chunk-saturated prefill compute per token (FLOPs roofline)."""
    return (2.0 * params.n_params) / (params.peak_flops_per_s * params.util_flops) * 1e3


def _baseline_prefill_ms(
    new_prefill: float, cached: float, params: RooflineParams
) -> float:
    """Queue-free single-session prefill floor (no concurrency term)."""
    u = max(1.0, float(new_prefill))
    p = max(1.0, float(cached))
    if u <= _GRID_U_MAX:
        return cached_prefill_step_ms(u, p)
    return u * _prefill_per_token_ms(params) + params.scheduler_overhead_ms_per_step


def predict_turn_ttft(
    cached: float,
    new_prefill: float,
    output: float,
    sched: float,
    tpot: float,
    params: RooflineParams | None = None,
) -> float:
    """TTFT (ms) for one turn from its workload + forecast cohort `sched`.

    `tpot` is the predicted per-output-token latency for the turn (the kernel TPOT
    headline); it sets the turn-decode time the oversubscription backlog is measured
    in.
    """
    p = params or RooflineParams()
    kv = float(p.available_kv_blocks)
    ctx_mid = max(1.0, float(cached) + float(new_prefill) + 0.5 * float(output))
    blocks = math.ceil(ctx_mid / p.cache_block_size)
    capacity_batch = kv / blocks
    pressure = sched * blocks / kv

    baseline = _baseline_prefill_ms(new_prefill, cached, p)
    step = decode_step_ms(min(sched, capacity_batch), float(cached) + float(new_prefill), p)
    sub = RESIDUAL * min(1.0, pressure) * sched * step
    over = max(0.0, pressure - 1.0) * float(output) * float(tpot)
    return baseline + sub + over


def _scheduled(turn: dict[str, Any], fallback: float) -> float:
    v = turn.get("scheduled_requests")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return fallback


def predict_cell_ttft(
    turns: list[dict[str, Any]],
    profile: str,
    concurrency: float,
    params: RooflineParams | None = None,
    *,
    oracle: bool = False,
    tpot_preds: list[float] | None = None,
) -> list[float]:
    """Per-turn TTFT (ms) for a (profile, concurrency) cell.

    Forward by default — the cohort `sched_hat[t]` is forecast from the profile's
    session-length survival curve, reading nothing from the measured trajectory.
    `oracle=True` consumes the measured `scheduled_requests` instead (drain
    isolation). `tpot_preds` overrides the internally-computed kernel TPOT (e.g. to
    reuse `tpot_pred_kernel` already in the dashboard JSON).
    """
    if not turns:
        return []
    p = params or RooflineParams()

    # Resolve the cohort ONCE — forward (sched_hat) or oracle (measured) — and use
    # it for both the queue terms and the internal TPOT, so the forward path never
    # reads measured scheduler state.
    if oracle:
        scheds = [_scheduled(t, float(concurrency)) for t in turns]
    else:
        scheds = [
            sched_hat(profile, float(concurrency), int(t.get("turn_index", 0)))
            for t in turns
        ]

    if tpot_preds is None:
        inputs = [
            KernelTurnInput(
                cached_context_tokens=float(t.get("cached_context_tokens") or 0.0),
                new_prefill_tokens=float(t.get("new_prefill_tokens") or 0.0),
                output_tokens=float(t.get("output_tokens") or 0.0),
                scheduled_requests=sched,
            )
            for t, sched in zip(turns, scheds)
        ]
        tpot_preds = predict_cell_tpot(inputs, p)

    out: list[float] = []
    for t, tpot, sched in zip(turns, tpot_preds, scheds):
        out.append(
            predict_turn_ttft(
                float(t.get("cached_context_tokens") or 0.0),
                float(t.get("new_prefill_tokens") or 0.0),
                float(t.get("output_tokens") or 0.0),
                sched,
                float(tpot),
                p,
            )
        )
    return out
