# simulator_v2/engine/serving_frontend.py

"""vLLM API-server FRONTEND serialization (HTTP/parse/chat-template/tokenize/IPC/stream),
the serving-harness cost in front of the engine -- kept out of `queue_sim.py` (scheduler
only). A barrier herd is serviced ~serially before it reaches the scheduler, staggering
first tokens; this drives the sub-saturation TTFT climb the engine sim can't see.

Measured (H100/Llama-3.1-8B, vLLM 0.19.0) via `serving_herd_scaling.py`, server-side:
f(new,cached) = FLOOR + RATE*(new+cached); serialization ~ (N+1)/2 * f, sub-linear by N=20.

NOT wired into predict(): a faithful term needs load-dependent frontend parallelism and
engine pipelining -- a fixed lane count fits one band and breaks the other. See docs/ttft.md.
"""

from __future__ import annotations

from simulator_v2.core.mode import Mode, mode

# Probe-measured (not fit): c1 intercept + per-total-token slope (~ the host re-tokenize rate).
FRONTEND_FLOOR_MS = 6.5
FRONTEND_MS_PER_TOKEN = 0.0046


@mode(Mode.SHARED)
def frontend_service_ms(new_tokens: float, cached_tokens: float) -> float:
    """Single-request frontend service time (ms) over the full re-sent prompt (new+cached)."""
    total = max(0.0, float(new_tokens)) + max(0.0, float(cached_tokens))
    return FRONTEND_FLOOR_MS + FRONTEND_MS_PER_TOKEN * total


@mode(Mode.SHARED)
def barrier_stagger_epochs(
    services_ms: list[float], release_epoch: float, lanes: int = 1
) -> list[float]:
    """[deferred, not wired] Engine-arrival epoch per request when a barrier herd is served
    by `lanes` serial frontend servers (FIFO). TTFT is still clocked from `release_epoch`."""
    n = max(1, int(lanes))
    lane_free = [float(release_epoch)] * n
    out: list[float] = []
    for f in services_ms:
        i = min(range(n), key=lambda k: lane_free[k])
        lane_free[i] += max(0.0, float(f))
        out.append(lane_free[i])
    return out
