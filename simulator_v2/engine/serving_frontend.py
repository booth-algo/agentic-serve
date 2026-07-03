# simulator_v2/engine/serving_frontend.py

"""API-server frontend (HTTP/parse/tokenize/IPC) in front of the engine. A herd's
host work drains near-serially before requests reach the scheduler; `queue_sim`
delays each ARRIVAL to its drain epoch while TTFT stays clocked from the release,
so the wait shows only where the engine doesn't hide it. Constants come from the
GPU YAML `frontend:` section (measured, client-referenced — provenance in
docs/knobs.md and docs/ttft.md)."""

import math

from simulator_v2.core.mode import Mode, mode
from simulator_v2.core.types import FrontendParams, Turn

@mode(Mode.SHARED)
def _load_mult(fp: FrontendParams, herd_n: int) -> float:
    """Streaming-load multiplier on f, evaluated at mid-drain load (herd/2);
    clamped at the curve ends. Flat `load_mult` when no curve."""
    if herd_n <= 1:
        return 1.0
    curve = fp.mult_curve
    if not curve:
        return fp.load_mult
    d = herd_n / 2.0
    if d <= curve[0][0]:
        return curve[0][1]
    for (d0, m0), (d1, m1) in zip(curve, curve[1:]):
        if d <= d1:
            return m0 + (m1 - m0) * (d - d0) / (d1 - d0)
    return curve[-1][1]


@mode(Mode.SHARED)
def service_ms(fp: FrontendParams, new_tokens: float, cached_tokens: float,
               herd_n: int) -> float:
    """Per-request frontend service (ms) over the full re-sent prompt."""
    f = (fp.floor_ms
         + fp.new_ms_per_token * max(0.0, float(new_tokens))
         + fp.cached_ms_per_token * max(0.0, float(cached_tokens)))
    return f * _load_mult(fp, herd_n)


@mode(Mode.SHARED)
def effective_lanes(fp: FrontendParams, herd_n: int) -> float:
    """Frontend parallelism for a herd (piecewise-linear in log2(conc), linear
    extrapolation past the last point)."""
    curve = fp.lanes_curve
    if not curve:
        return 1.0
    n = max(1, int(herd_n))
    if n <= curve[0][0]:
        return curve[0][1]
    for (c0, l0), (c1, l1) in zip(curve, curve[1:]):
        if n <= c1:
            t = (math.log2(n) - math.log2(c0)) / (math.log2(c1) - math.log2(c0))
            return l0 + t * (l1 - l0)
    (c0, l0), (c1, l1) = curve[-2], curve[-1]
    slope = (l1 - l0) / (math.log2(c1) - math.log2(c0))
    return l1 + slope * (math.log2(n) - math.log2(c1))


@mode(Mode.SHARED)
def herd_arrival_epochs(
    fp: FrontendParams, turns: list[Turn], release_epoch: float
) -> list[float]:
    """Engine-arrival epoch per request for a herd released at `release_epoch`:
    FIFO drain, arrival_i = release + cumsum(f_1..f_i)/lanes. Disabled params ->
    all epochs = release."""
    if not fp.enabled or not turns:
        return [float(release_epoch)] * len(turns)
    n = len(turns)
    lanes = max(1.0, effective_lanes(fp, n))
    out, acc = [], 0.0
    for t in turns:
        acc += service_ms(fp, t.new_prefill_tokens, t.cache_hit_tokens, n)
        out.append(float(release_epoch) + acc / lanes)
    return out
