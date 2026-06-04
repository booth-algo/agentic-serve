"""Forward 3D-roofline eviction-deficit RAMP predictor for per-turn TPOT (ITL).

Companion to ``simulator/kernel_tpot.py``. Where the production kernel predictor is
a pressure-driven amplifier whose rise can LAG the measured saturation step by a few
turns, this predictor models the ramp from first principles and forecasts its timing
**forward from the workload spec** — concurrency + the profile's session-length
distribution + per-turn ISL:OSL — with NO measured cohort trajectory.

Three roofline bounds (compute / memory-bandwidth / KV-capacity-eviction):

    ITL[t] = T_bw[t] + frac[t] · (T_ceiling[t] − T_bw[t])

* ``T_bw`` (memory-bandwidth floor) = ``decode_step_ms(b_eff, ctx_mid)`` — measured
  decode kernel grid (``kernel_step_cost``, 7.4% MAPE). ``b_eff`` is the KV-throttled
  running batch ``min(sched_hat, 27250/blk)``.
* ``T_ceiling`` (compute-bound recompute spike) = the output-keyed saturated ceiling
  ``saturated_ceiling_ms``, reached immediately for short-output families and only as
  the eviction deficit develops for long-output ones (drain-aware ``lift``).
* ``frac`` = ``smoothstep(defcap; DEF_LO, DEF_HI)`` — the KV-capacity transition. The
  ramp emerges from the turn-by-turn growth of the eviction **deficit**
  ``defcap = pressure − 1`` crossing the eviction watermark (``DEF_LO``, pool ~88%
  committed) toward full recompute (``DEF_HI``). Gated by the output-sustain factor.

**Forward cohort drain (the key forward piece).** Instead of the measured per-turn
``scheduled_requests``, the resident cohort is forecast as ``sched_hat[t] = round(C ·
S(t))`` where ``S(t)`` is the survival function of the profile's session-length
distribution (``histograms['turn_count']`` in the workload spec) — fully forward,
pure workload, no telemetry. The forecast preserves the deficit trajectory that
drives the ramp onset to within ~0.01 (defcap) for swe/terminal/chat; osworld's
steep, concurrency-sensitive drain is the dominant residual (see module notes).

Validated (workflows wf_e342dabb + wf_a9364128, fully forward, vs production
``kernel`` 16.48%): this is a **targeted ramp-tracking win** — terminalbench plateau
18.3→15.3, swebench plateau 9.2→8.7, terminalbench c80 64.8→51, swebench c40 61→33,
terminalbench c120 (the turn-11 lag) 23.5→17.5, jump-turn MAE 1.7, chat byte-flat.

The saturated plateau is preemption/recompute/queueing latency (NOT an amortizable
roofline term — both a two-roofline amortized ceiling and a KV-read roofline ceiling
were measured to collapse to the bandwidth floor and were rejected), so the fitted
output-keyed ``saturated_ceiling_ms`` magnitude is kept. osworld's regression — the
ceiling over-predicting its early/partial/RECOVERING plateau — is fixed fit-free by a
FORWARD WATERMARK-RECOVERY CAP in ``predict_cell_tpot_ramp`` (the forward analog of
``kernel_tpot_hint``'s ``min(ramp_target, max(pressure_path[t:]))``, built from the
forecast cohort): osworld plateau 25.9→20.2 with the recovery tracked, overall forward
19.1→18.3, swe/terminal/chat unchanged (the cap is a no-op on flat-survival cells).
The earlier ``frac·=survival^0.5`` idea was rejected — its exponent is a fitted knob
masking this ceiling/recovery effect (see [[tpot-amplifier-pressure-law]]).
Wired as a side-by-side comparison column (``tpot_pred_ramp``), NOT the headline.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import decode_step_ms
from simulator.kernel_tpot import (
    OUT_KNEE_HI,
    OUT_KNEE_LO,
    SAT_SUSTAIN_HI,
    SAT_SUSTAIN_LO,
    _smoothstep,
    saturated_ceiling_ms,
)

# --- eviction-deficit ramp band (defcap = pressure − 1) ----------------------
# The saturation jump is the KV-pool eviction WATERMARK crossing, not the pool-full
# crossing. Measured across all real-jump cells the jump fires at pressure ≈ 0.88–1.22
# (pool ~88–92% committed). So the ramp band straddles defcap = 0:
DEF_LO = -0.12   # eviction-watermark onset: pool ~88% committed (pressure 0.88). Read
                 # off the measured jump-pressure cluster floor (term c80 @ -0.12),
                 # NOT a MAPE fit. Sibling of kernel P_LO=0.8 amplifier onset.
DEF_HI = 0.22    # second-wave-recompute knee: pressure ~1.22, ~22% oversubscription
                 # forces the full prefill recompute. Cluster max (swe c200 @ +0.22).
DEF_SAT = 1.0    # eviction fully developed at 2× pool commit (pressure 2.0) — the
                 # two-roofline wave-factor knee. Controls the long-output ceiling lift.

# Output-sustain gate + short/long-output split: reused verbatim from kernel_tpot
# (SAT_SUSTAIN_LO/HI = 10/24 tok; OUT_KNEE_LO/HI = 40/80 tok).

_PARAMS = RooflineParams()
KV_BLOCKS = float(_PARAMS.available_kv_blocks)   # 27250
BLK_SIZE = int(_PARAMS.cache_block_size)         # 16

_REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = _REPO_ROOT / "inference-benchmark" / "data" / "distributions"

# profile -> session-length distribution spec (the *-multiturn-synth mappings in
# inference-benchmark/src/workloads/profiles.py). survival(t) is read from each
# file's histograms['turn_count']; this is the forward cohort-drain source.
PROFILE_DIST = {
    "swebench-multiturn-synth": "swebench_multiturn_short_tracereplay_filtered-mse_realized.json",
    "terminalbench-multiturn-synth": "terminalbench_multiturn_short_tracereplay_filtered-mse_realized.json",
    "osworld-multiturn-synth": "osworld_multiturn_realized.json",
    "chat-multiturn-synth": "chat_multiturn_realized.json",
}

_SURVIVAL_CACHE: dict[tuple, list[float]] = {}
_REALIZED_CACHE: dict[tuple[str, str], dict | None] = {}


def _gpu_slug(gpu_key: str | None) -> str:
    """Filesystem-safe slug for a GPU key, SHARED by the generator and the resolver so
    writer and reader agree (e.g. 'A100'->'a100', 'A100 (sglang)'->'a100sglang'). '' -> ''."""
    return re.sub(r"[^a-z0-9]", "", str(gpu_key).lower()) if gpu_key else ""


def _resolve_dist_path(profile: str, gpu_key: str | None = None) -> Path | None:
    """Realized-dist path: the per-GPU file (``<spec>_realized_<slug>.json``) if it exists,
    else the pooled ``PROFILE_DIST`` file. None when the profile has no spec."""
    if profile not in PROFILE_DIST:
        return None
    pooled = PROFILE_DIST[profile]
    slug = _gpu_slug(gpu_key)
    if slug:
        cand = DIST_DIR / pooled.replace(".json", f"_{slug}.json")
        if cand.exists():
            return cand
    return DIST_DIR / pooled


def _load_realized(profile: str, gpu_key: str | None = None) -> dict | None:
    """Parsed realized dist JSON (per-GPU if present, else pooled), cached by (profile, slug)."""
    if profile not in PROFILE_DIST:
        return None
    key = (profile, _gpu_slug(gpu_key))
    if key not in _REALIZED_CACHE:
        path = _resolve_dist_path(profile, gpu_key)
        try:
            _REALIZED_CACHE[key] = json.loads(path.read_text()) if path else None
        except Exception:
            _REALIZED_CACHE[key] = None
    return _REALIZED_CACHE[key]


def _select_conc(realized: dict | None, concurrency: float | None) -> str | None:
    """Nearest measured-concurrency key in ``by_concurrency`` (exact preferred; ties -> smaller
    conc, deterministic), or None to fall back to the pooled curve."""
    if not realized or concurrency is None:
        return None
    bc = realized.get("by_concurrency")
    if not bc:
        return None
    keys = sorted(int(k) for k in bc)
    if not keys:
        return None
    target = int(round(float(concurrency)))
    if str(target) in bc:
        return str(target)
    return str(min(keys, key=lambda k: (abs(k - target), k)))


def _survival_from_hist(hist: dict[int, int]) -> list[float]:
    """``S(t)`` = fraction of sessions with ``turn_count > t`` from a turn_count histogram.
    ``S(0) == 1`` by construction; pure workload distribution."""
    n = sum(hist.values())
    if n <= 0:
        return [1.0]
    tmax = max(hist)
    return [sum(v for k, v in hist.items() if k > t) / n for t in range(tmax)]


def forward_survival(dist_file: str) -> list[float]:
    """``S(t)`` read forward from a dist file's POOLED ``histograms.turn_count`` (public API,
    unchanged signature — the pooled-curve path). The per-(conc,gpu) path goes via ``_survival``."""
    d = json.loads((DIST_DIR / dist_file).read_text())
    hist = {int(k): int(v) for k, v in d["histograms"]["turn_count"].items()}
    return _survival_from_hist(hist)


def _survival(
    profile: str, concurrency: float | None = None, gpu_key: str | None = None
) -> list[float] | None:
    """Survival curve for a (profile, concurrency, gpu). Fallback chain: per-conc block (nearest
    measured conc) -> in-file pooled -> legacy ``PROFILE_DIST`` pooled. ``concurrency=None,
    gpu_key=None`` reproduces the legacy pooled curve byte-identically. None if no spec."""
    if profile not in PROFILE_DIST:
        return None
    realized = _load_realized(profile, gpu_key)
    conc_key = _select_conc(realized, concurrency)
    cache_key = (profile, _gpu_slug(gpu_key), conc_key)
    if cache_key not in _SURVIVAL_CACHE:
        hist: dict[int, int] | None = None
        if realized:
            if conc_key is not None:
                block = (realized.get("by_concurrency") or {}).get(conc_key) or {}
                if block.get("turn_count"):
                    hist = {int(k): int(v) for k, v in block["turn_count"].items()}
            if hist is None:  # in-file pooled fallback
                pooled = (realized.get("histograms") or {}).get("turn_count")
                if pooled:
                    hist = {int(k): int(v) for k, v in pooled.items()}
        _SURVIVAL_CACHE[cache_key] = (
            _survival_from_hist(hist) if hist is not None
            else forward_survival(PROFILE_DIST[profile])  # last resort: legacy pooled file
        )
    return _SURVIVAL_CACHE[cache_key]


def survival_for(
    profile: str, concurrency: float | None = None, gpu_key: str | None = None
) -> list[float] | None:
    """Public per-(profile, concurrency, gpu) survival curve. Default args == pooled curve."""
    return _survival(profile, concurrency, gpu_key)


def trajectory_pool(
    profile: str, concurrency: float | None = None, gpu_key: str | None = None
) -> list | None:
    """Real per-session trajectory POOL for concurrency-MATCHED cohort REPLAY (per-GPU), or None. Each
    session is a list of ``[cached, new, output]`` per turn. This is the JOINT cohort input (survival +
    context-scale + their correlation) that reaches the oracle floor. Selects the NEAREST measured
    concurrency's pool from ``by_concurrency`` — osworld's trajectory shapes are concurrency-dependent,
    so conc-matching beats a single pooled-over-conc pool (tournament 2026-06-04). Falls back to a
    top-level pooled ``trajectory_pool`` if present, else None -> caller uses the survival/scale
    marginals (pooled when ``gpu_key=None`` → byte-identical)."""
    realized = _load_realized(profile, gpu_key)
    if not realized:
        return None
    conc_key = _select_conc(realized, concurrency)
    if conc_key is not None:
        blk = (realized.get("by_concurrency") or {}).get(conc_key) or {}
        if blk.get("trajectory_pool"):
            return blk["trajectory_pool"]
    return realized.get("trajectory_pool") or None


_SCALE_CACHE: dict[tuple, list[float]] = {}


def context_scale_quantiles(
    profile: str, concurrency: float | None = None, gpu_key: str | None = None
) -> list[float] | None:
    """Per-session context-size SCALE quantiles (p0..p100) for a (profile, concurrency, gpu),
    or None. Each session runs systematically larger/smaller contexts than the per-turn median
    (a measured workload property); the cohort applies a session's quantile scale to the median
    trajectory so the KV working set has the measured SPREAD (small sessions stay resident=hits,
    the large minority is evicted -> the osworld saturate-RECOVER). Fallback: per-conc block
    (nearest measured conc) -> in-file pooled. ``concurrency=None, gpu_key=None`` == pooled curve
    (byte-identical to legacy). Pure workload distribution — no TTFT fit."""
    if profile not in PROFILE_DIST:
        return None
    realized = _load_realized(profile, gpu_key)
    conc_key = _select_conc(realized, concurrency)
    cache_key = (profile, _gpu_slug(gpu_key), conc_key)
    if cache_key not in _SCALE_CACHE:
        q = None
        if realized:
            if conc_key is not None:
                block = (realized.get("by_concurrency") or {}).get(conc_key) or {}
                q = block.get("context_scale_quantiles")
            if not q:  # in-file pooled fallback
                q = realized.get("context_scale_quantiles")
        _SCALE_CACHE[cache_key] = [float(x) for x in q] if q else []
    return _SCALE_CACHE[cache_key] or None


def sched_hat(
    profile: str, concurrency: float, turn_index: int, gpu_key: str | None = None
) -> float:
    """Forward cohort estimate ``round(C · S(t))`` (replaces measured scheduled_requests).
    ``gpu_key=None`` (default, used by the ramp column + ttft_predict fallback) -> the pooled
    survival, byte-identical to legacy (the pooled ``*_realized.json`` carry no per-conc block)."""
    s = _survival(profile, concurrency, gpu_key)
    if not s:
        return max(1.0, float(concurrency))
    frac = s[turn_index] if turn_index < len(s) else s[-1]
    return max(1.0, round(float(concurrency) * frac))


def _drain_aware_ceiling_ms(out: float, defcap: float, kstep: float, t_outkey: float) -> float:
    """Saturated ceiling for the turn. Short-output turns get the full output-keyed
    ceiling immediately; long-output turns (osworld/chat) only approach it as the
    eviction deficit develops toward 2× pool commit — so a draining/recovering cohort
    is not pinned to the static 1/output ceiling."""
    longw = _smoothstep(out, OUT_KNEE_LO, OUT_KNEE_HI)
    lift = (1.0 - longw) + longw * _smoothstep(defcap, 0.0, DEF_SAT)
    return kstep + lift * (t_outkey - kstep)


def _ramp_pieces(
    cached: float,
    new_prefill: float,
    output: float,
    sched: float,
    ceiling_output: float,
    params: RooflineParams | None = None,
) -> dict:
    """One turn's ramp decomposition: the ITL ``s_hat`` plus the pieces the
    cell-level forward recovery cap reuses (``kstep``, ``t_ceil``, ``blk``)."""
    p = params or _PARAMS
    out = max(1.0, float(output))
    sch = max(1.0, float(sched))
    ctx_mid = float(cached) + float(new_prefill) + 0.5 * out
    blk = max(1, math.ceil(ctx_mid / max(1, p.cache_block_size)))
    pressure = sch * blk / p.available_kv_blocks
    defcap = pressure - 1.0
    b_eff = max(1.0, min(sch, p.available_kv_blocks / blk))
    kstep = decode_step_ms(b_eff, ctx_mid, p)
    t_outkey = saturated_ceiling_ms(ceiling_output)
    t_ceil = max(kstep, _drain_aware_ceiling_ms(out, defcap, kstep, t_outkey))
    frac = _smoothstep(defcap, DEF_LO, DEF_HI) * _smoothstep(out, SAT_SUSTAIN_LO, SAT_SUSTAIN_HI)
    s_hat = kstep + frac * (t_ceil - kstep)
    return {"s_hat": s_hat, "kstep": kstep, "t_ceil": t_ceil, "blk": blk}


def predict_turn_ramp(
    cached: float,
    new_prefill: float,
    output: float,
    sched: float,
    ceiling_output: float,
    params: RooflineParams | None = None,
) -> float:
    """One turn's ITL (ms) from the eviction-deficit roofline composition (the
    *uncapped* ramp; the cell path adds the forward recovery cap)."""
    return _ramp_pieces(cached, new_prefill, output, sched, ceiling_output, params)["s_hat"]


def predict_cell_tpot_ramp(
    turns: list[dict],
    profile: str,
    concurrency: float,
    params: RooflineParams | None = None,
    *,
    oracle: bool = False,
) -> list[float]:
    """Per-turn ITL for a (profile, concurrency) cell, fully forward.

    ``turns`` are per-turn dicts carrying ``cached_context_tokens``,
    ``new_prefill_tokens``, ``output_tokens``, ``turn_index`` (and, only for the
    ``oracle`` baseline, ``scheduled_requests``).

    The forward path forecasts the resident cohort via ``sched_hat`` and NEVER reads
    ``scheduled_requests``. ``oracle=True`` swaps in the measured cohort — used only to
    separate forward-drain error from ramp-model error (per the standing constraint),
    never for production.

    A FORWARD WATERMARK-RECOVERY CAP (the forward analog of kernel_tpot_hint's
    measured-cohort ``min(ramp_target, max(pressure_path[t:]))``) lets a draining
    cohort RECOVER below the saturated ceiling. Build a forecast watermark path
    ``W[t] = kstep[t] + rel[t]·(t_ceil[t] − kstep[t])`` where ``rel[t] =
    load[t] / max(load[0..t])`` is the forecast KV-block demand ``load = sched_hat·blk``
    relative to its causal running peak (a dimensionless residency ratio in [0,1]),
    then cap ``ITL[t] = min(s_hat[t], max(W[t..end]))``. As the forecast cohort drains,
    ``rel`` falls, ``W`` descends, and its forward-max relaxes the lift — tracking
    osworld's recovery. For flat-survival workloads (swe/terminal/chat) ``rel`` stays
    ~1, so ``W`` ≈ ``s_hat`` and the cap is a no-op. Fit-free: only structural
    min / forward-max over already-physical quantities, no new constant. One-sided
    (only lowers), so no turn can regress above its uncapped ramp value.
    """
    p = params or _PARAMS
    if not turns:
        return []
    outs = [max(1.0, float(t.get("output_tokens", 1.0) or 1.0)) for t in turns]
    ceiling_output = statistics.median(outs) if outs else 1.0

    s_hat: list[float] = []
    kstep: list[float] = []
    t_ceil: list[float] = []
    blk: list[int] = []
    sched_used: list[float] = []
    for t in turns:
        ti = int(t.get("turn_index", 0))
        if oracle:
            sched = max(1.0, float(t.get("scheduled_requests", concurrency) or concurrency))
        else:
            sched = sched_hat(profile, concurrency, ti)
        sched_used.append(sched)
        pc = _ramp_pieces(
            float(t.get("cached_context_tokens", 0.0) or 0.0),
            float(t.get("new_prefill_tokens", 0.0) or 0.0),
            float(t.get("output_tokens", 1.0) or 1.0),
            sched,
            ceiling_output,
            p,
        )
        s_hat.append(pc["s_hat"])
        kstep.append(pc["kstep"])
        t_ceil.append(pc["t_ceil"])
        blk.append(pc["blk"])

    # Forward watermark-recovery cap (fit-free: structural min / forward-max + rel in [0,1]).
    n = len(turns)
    load = [sched_used[i] * blk[i] for i in range(n)]  # forecast KV-block demand
    rel: list[float] = []
    run_peak = 0.0
    for i in range(n):
        run_peak = max(run_peak, load[i])
        rel.append(load[i] / run_peak if run_peak > 0 else 1.0)
    watermark = [kstep[i] + rel[i] * (t_ceil[i] - kstep[i]) for i in range(n)]
    fwd_max = [0.0] * n
    running = float("-inf")
    for i in range(n - 1, -1, -1):
        running = max(running, watermark[i])
        fwd_max[i] = running
    return [min(s_hat[i], fwd_max[i]) for i in range(n)]


__all__ = [
    "predict_cell_tpot_ramp",
    "predict_turn_ramp",
    "forward_survival",
    "survival_for",
    "trajectory_pool",
    "context_scale_quantiles",
    "sched_hat",
    "PROFILE_DIST",
    "DEF_LO",
    "DEF_HI",
]
