#!/usr/bin/env python3
"""Measure the cached-prefill HOST shared/per-request split from the live B-sweep.

``simulator/ttft_queue_sim.py`` prices the cached host serving-stack cost (HTTP body parse +
chat-template + tokenize + ZMQ IPC; SUM = 6.103e-3 ms/cached-token, benchmark-fitted and
live-validated at 5.89 — see profiling/docs/prefill_stage_split_results.md) as two terms:
``PREFILL_HOST_SHARED_MS_PER_TOKEN`` (amortized once per engine step) and
``PREFILL_HOST_PERREQ_MS_PER_TOKEN`` (summed over concurrent prefills). The SUM is measured;
the 50/50 PARTITION shipped in commit 41e35f5 was chosen to maximize the gate within a
measured 40-54% band — i.e. a fit (spec .omc/specs/deep-dive-whether-there-are-fitted.md
rows 2-3). This script builds the missing reproducible measurement: the point estimate of
the B-slope-vs-c1-rate ratio that bounds that band, from the two live-probe CSVs already on
disk (no GPU, no server).

Data (both produced 2026-06-03 by live probes against the real vLLM OpenAI server on h100
GPU 7 over loopback HTTP — commit 9dce1dc; instruments ``live_ttft_probe.py`` /
``live_split_probe.py``):
  * ``profile_data/results/prefill_live_ttft_H100.csv`` — c1 grid (new x cached). 2-var
    lstsq ``ttft = floor + new_rate*new + c1_rate*cached`` gives the c1 cached host+GPU
    rate ``c1_rate`` (= 5.887e-3 ms/tok; the band's documented denominator).
  * ``profile_data/results/prefill_live_split_H100.csv`` — THE B-sweep: B in {1,2,4,8,16}
    concurrent cache-hit requests sharing one primed prefix of P in {2000,8000,16000}
    tokens (fresh 8-token tail each), median per-request TTFT.

Model (the simulator's own pricing): ``TTFT(B,P) = intercept(P) + perreq*(B*P)``, so the
per-added-request slope is ``perreq*P`` and ``shared_frac = 1 - perreq/c1_rate``.

Estimator (PRE-REGISTERED — the spec's "point estimate, not the gate-max"): pooled OLS of
the common ``B*P`` slope with per-P intercepts over the two band-defining planes
P in {8000, 16000}. P=2000 is EXCLUDED exactly as the documented band excluded it: at short
prefixes a fixed ~8.3 ms per-request serving cost (4-param refit, reported in the artifact)
is misattributed as per-token, driving its plane's shared_frac to ~-0.02 (non-physical).
Band endpoints for context: P=8000 -> 0.402, P=16000 -> 0.546 (the in-code "~40-54%").

Caveats recorded in the artifact:
  * Do NOT use ``live_split_probe.py``'s built-in ``fit()`` (global 3-param lstsq over all
    15 rows): it returns shared = -1.303 ms/1k (negative) because the model omits the fixed
    per-request cost; the band never came from it.
  * The pooled per-P intercepts come out INVERTED (84.3 ms at P=8000 vs 52.4 ms at
    P=16000) — same omitted-fixed-cost symptom. A 4-param refit (fixed-per-request term)
    gives shared_frac ~0.63, OUTSIDE the band; adopting it would add a new pricing term to
    the simulator, beyond this partition-only de-fit.
  * Denominator: the same-day live c1 rate 5.887 (internally consistent with the band);
    using the shipped sum 6.103 instead would give 0.540 (reported as sensitivity).

Deterministic (no RNG). Usage:
    python3 -m profiling.process.build_host_split [--out <json>]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "profile_data" / "results"
C1_CSV = RESULTS / "prefill_live_ttft_H100.csv"
BSWEEP_CSV = RESULTS / "prefill_live_split_H100.csv"
OUT_JSON = REPO_ROOT / "profile_data" / "kernels" / "prefill_host_split_H100.json"

# The benchmark-fitted cached host SUM the split partitions (ms/cached-token). MEASURED:
# corroborated by the live c1 probe (5.887, below) and the 2026-06-05 c1 stage-split
# (frontend.cached 5.174 + prefill_span.cached 0.815 = 5.989; serving_stage_split_H100.csv).
HOST_SUM_MS_PER_TOKEN = 6.103e-3

# Pre-registered estimator scope: the band-defining planes. P=2000 excluded (see docstring).
BAND_PLANES = (8000, 16000)


def fit_c1_rate() -> dict:
    """2-var lstsq of the c1 live grid: ttft = floor + new_rate*new + c1_rate*cached."""
    rows = list(csv.DictReader(C1_CSV.open()))
    A = np.array([[1.0, float(r["new"]), float(r["cached"])] for r in rows])
    y = np.array([float(r["ttft_ms"]) for r in rows])
    floor, new_rate, c1_rate = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"floor_ms": floor, "new_ms_per_tok": new_rate, "c1_cached_ms_per_tok": c1_rate,
            "n_rows": len(rows)}


def load_bsweep() -> dict[int, dict[int, float]]:
    data: dict[int, dict[int, float]] = {}
    for r in csv.DictReader(BSWEEP_CSV.open()):
        data.setdefault(int(r["P"]), {})[int(r["B"])] = float(r["ttft_ms"])
    return data


def endpoint_slopes(data: dict[int, dict[int, float]], c1_rate: float) -> dict:
    """Per-plane B=1->16 endpoint slope -> perreq -> shared_frac (the band's arithmetic)."""
    out = {}
    for P, d in sorted(data.items()):
        b_lo, b_hi = min(d), max(d)
        slope = (d[b_hi] - d[b_lo]) / (b_hi - b_lo)          # ms per added request
        perreq = slope / P                                    # ms per token
        out[str(P)] = {"b_slope_ms_per_req": slope, "perreq_ms_per_tok": perreq,
                       "shared_frac": 1.0 - perreq / c1_rate}
    return out


def pooled_perreq(data: dict[int, dict[int, float]]) -> dict:
    """PRE-REGISTERED point estimate: pooled OLS, per-P intercepts + common B*P slope,
    over BAND_PLANES only."""
    pts = [(B, P, t) for P in BAND_PLANES for B, t in sorted(data[P].items())]
    X = np.array([[float(P == p) for p in BAND_PLANES] + [B * P] for B, P, _ in pts])
    y = np.array([t for _, _, t in pts])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"intercept_ms_per_plane": {str(p): coef[i] for i, p in enumerate(BAND_PLANES)},
            "perreq_ms_per_tok": coef[-1], "n_rows": len(pts), "planes": list(BAND_PLANES)}


def fixed_cost_refit(data: dict[int, dict[int, float]]) -> dict:
    """Diagnostic 4-param refit over BAND_PLANES: per-P intercepts + fixed-per-request B
    term + per-token B*P term. NOT the adopted estimator (changes the pricing structure)."""
    pts = [(B, P, t) for P in BAND_PLANES for B, t in sorted(data[P].items())]
    X = np.array([[float(P == p) for p in BAND_PLANES] + [B, B * P] for B, P, _ in pts])
    y = np.array([t for _, _, t in pts])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"fixed_ms_per_req": coef[-2], "perreq_ms_per_tok": coef[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    c1 = fit_c1_rate()
    c1_rate = c1["c1_cached_ms_per_tok"]
    data = load_bsweep()
    per_plane = endpoint_slopes(data, c1_rate)
    pooled = pooled_perreq(data)
    shared_frac = 1.0 - pooled["perreq_ms_per_tok"] / c1_rate

    band = sorted(per_plane[str(p)]["shared_frac"] for p in BAND_PLANES)
    if not band[0] <= shared_frac <= band[1]:
        raise SystemExit(f"pooled shared_frac {shared_frac:.4f} outside the measured band "
                         f"[{band[0]:.4f}, {band[1]:.4f}] — data or estimator changed")

    artifact = {
        "schema": "prefill_host_split.v1",
        "inputs": {"c1_csv": str(C1_CSV.relative_to(REPO_ROOT)),
                   "bsweep_csv": str(BSWEEP_CSV.relative_to(REPO_ROOT)),
                   "host_sum_ms_per_tok": HOST_SUM_MS_PER_TOKEN},
        "c1_fit": c1,
        "per_plane_endpoint": per_plane,           # includes excluded P=2000 for the record
        "pooled_fit": pooled,
        "shared_frac": shared_frac,
        "band": {"lo": band[0], "hi": band[1], "planes": list(BAND_PLANES)},
        "constants": {
            "PREFILL_HOST_SHARED_MS_PER_TOKEN": shared_frac * HOST_SUM_MS_PER_TOKEN,
            "PREFILL_HOST_PERREQ_MS_PER_TOKEN": (1.0 - shared_frac) * HOST_SUM_MS_PER_TOKEN,
        },
        "alternatives": {
            "endpoint_mean": float(np.mean(band)),
            "denominator_host_sum": 1.0 - pooled["perreq_ms_per_tok"] / HOST_SUM_MS_PER_TOKEN,
        },
        "diagnostic_fixed_cost_refit": fixed_cost_refit(data),
        "excluded": {"P=2000": "fixed per-request cost misattributed as per-token "
                               "(shared_frac ~ -0.02); excluded by the documented band"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"c1 rate: {c1_rate*1e3:.4f} ms/1k (floor {c1['floor_ms']:.2f}, "
          f"new {c1['new_ms_per_tok']*1e3:.3f})")
    for P in sorted(data):
        pp = per_plane[str(P)]
        tag = "" if P in BAND_PLANES else "  [EXCLUDED]"
        print(f"P={P}: B-slope {pp['b_slope_ms_per_req']:.3f} ms/req -> perreq "
              f"{pp['perreq_ms_per_tok']*1e3:.4f} ms/1k -> shared_frac "
              f"{pp['shared_frac']:.4f}{tag}")
    print(f"pooled perreq {pooled['perreq_ms_per_tok']*1e3:.4f} ms/1k -> "
          f"shared_frac = {shared_frac:.4f}  (band [{band[0]:.3f}, {band[1]:.3f}])")
    print(f"SHARED = {artifact['constants']['PREFILL_HOST_SHARED_MS_PER_TOKEN']:.6e}  "
          f"PERREQ = {artifact['constants']['PREFILL_HOST_PERREQ_MS_PER_TOKEN']:.6e}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
