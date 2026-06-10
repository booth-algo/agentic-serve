#!/usr/bin/env python3
"""Measure the TPOT amplifier ramp knees (P_LO / P_HI_SHORT / P_HI_LONG) from benchmark GT.

``kernel_tpot.predict_turn_tpot`` ramps the measured decode-kernel step toward the measured
saturated ceiling with ``weight = smoothstep(pressure; P_LO, p_hi)``. The three knees were
hand-tuned (see profiling/docs/fitted_constants_audit.md) — this script builds the missing
reproducible measurement: invert the per-cell measured weight-vs-pressure curve back to its
smoothstep band and aggregate, writing a ``profile_data/kernels/ramp_knees_*.json`` artifact
(the same measured-anchor pattern as ``build_saturated_ceiling.py``).

Per (profile, concurrency) cell with ground truth:
  * per turn: ``pressure = scheduled·ceil(ctx/block)/available_kv_blocks`` and the **implied
    ramp weight** ``w = clip((tpot_meas − kernel_step)/(t_upper − kernel_step), 0, 1)`` —
    the same kernel_step / ceiling-at-cell-median-output the predictor uses.
  * detect where the pressure-ordered rolling-median of ``w`` sustainably crosses W_LOW and
    W_HIGH; only cells with BOTH crossings interior to their pressure range are usable
    (one crossing cannot separate the band's onset from its width).
  * invert the smoothstep through the two crossings: with u_k = smoothstep⁻¹(k),
    ``hi − lo = (p_high − p_low)/(u_high − u_low)`` and ``lo = p_low − u_low·(hi − lo)``.

Aggregate (PRE-REGISTERED — fixed before looking at any MAPE outcome; see RULE below):
P_LO = median per-cell ``lo`` over all usable cells; P_HI_SHORT / P_HI_LONG = median per-cell
``hi`` over short-output (cell median ≤ OUT_KNEE_LO) / long-output (≥ OUT_KNEE_HI) cells —
mid-output cells are excluded from the knee estimates because their per-turn p_hi is a blend.
A knee is ADOPTABLE only if ≥ MIN_CELLS usable cells from ≥ 2 profiles support it and the
leave-one-profile-out / leave-one-concurrency-out jackknife stays within
max(0.05, 0.5·IQR) of the full-sample value (a real physical band is resampling-stable; a
fit chases its cells). Knees that fail keep their production value and are relabeled
honestly as tuned-knob (spec .omc/specs/deep-dive-whether-there-are-fitted.md, fallback).

v2 (ramp_knee_adoption_plan.md Phases 0+1) ADDITIONALLY measures the pressure-independent
decode floor-excess ``D`` per deployment (pre-registered: turns with pressure <
FLOOR_PRESSURE_MAX, output >= SAT_SUSTAIN_HI, tpot_meas > 0; D_raw = median(tpot_meas −
kernel_step); D = max(0, D_raw)) and re-derives the band against the corrected floor
``kernel_step + D`` with the IDENTICAL detection rule. The artifact keeps the ORIGINAL
uncorrected ``knees`` block byte-compatible (pinned by
test_kernel_tpot.test_ramp_knees_tuned_values_and_measured_band_both_pinned) and adds
``floor_excess_ms`` + ``knees_corrected``. ``--exclude-profile`` (LOCO) drops one profile
from the BAND cells only — D stays full-sample per the plan; LOCO runs must write to a
non-canonical ``--out-dir``.

Deterministic (no RNG). Usage:
    python3 -m profiling.process.build_ramp_knees [--exclude-profile <name>] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.process.build_simulator_rows import (  # noqa: E402
    BENCH_BASE, CONCURRENCIES, PROFILES, build_turns,
)
from configs.loader import all_deployments  # noqa: E402
from simulator import kernel_step_cost, kernel_tpot  # noqa: E402
from simulator.kernel_step_cost import load_grid  # noqa: E402
from simulator.kernel_tpot import (  # noqa: E402
    OUT_KNEE_HI, OUT_KNEE_LO, SAT_SUSTAIN_HI,
    KernelTurnInput, _kernel_step_ms, saturated_ceiling_ms,
)

# --- the pre-registered rule (every number fixed up front; no MAPE feedback) ---
W_LOW = 0.1            # lower weight crossing the detector latches onto
W_HIGH = 0.9           # upper weight crossing
ROLL_WINDOW = 5        # rolling-median window over pressure-ordered turns
SUSTAIN_STEPS = 2      # crossing must hold for this many consecutive windows
MIN_TURNS_PER_CELL = 8
MIN_COND_GAP_MS = 10.0     # t_upper − kernel_step must exceed this …
MIN_COND_REL = 0.5         # … and 0.5·kernel_step (implied-weight conditioning)
MIN_CELLS_PER_KNEE = 3
MIN_PROFILES_PER_KNEE = 2  # single-profile support can't be jackknifed -> not adoptable
STABILITY_ABS = 0.05       # jackknife max-deviation tolerance: max(this, 0.5·IQR)
# Phase-0 floor-excess D (ramp_knee_adoption_plan.md, pre-registered): saturation is
# impossible below this pressure, so any (tpot_meas − kernel_step) excess there is
# pressure-INDEPENDENT serving overhead, not ramp.
FLOOR_PRESSURE_MAX = 0.30

CANONICAL_OUT_DIR = Path("profile_data/kernels")


def _smoothstep_inv(y: float) -> float:
    """u such that u²(3−2u) = y, u in [0,1] (bisection; avoids a magic literal)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if mid * mid * (3.0 - 2.0 * mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


U_LOW = _smoothstep_inv(W_LOW)
U_HIGH = _smoothstep_inv(W_HIGH)


@dataclass(frozen=True)
class CellEstimate:
    profile: str
    conc: int
    cluster: str          # short | mid | long (by cell median output)
    n_turns: int
    p_low: float          # pressure at the sustained W_LOW crossing
    p_high: float         # pressure at the sustained W_HIGH crossing
    lo: float             # inverted smoothstep onset
    hi: float             # inverted smoothstep full-saturation knee


def _rolling_median(xs: list[float], window: int) -> list[float]:
    half = window // 2
    return [st.median(xs[max(0, i - half): i + half + 1]) for i in range(len(xs))]


def _sustained_crossing(r: list[float], thr: float) -> int | None:
    """First index where the rolling median reaches ``thr`` and holds SUSTAIN_STEPS windows."""
    for i in range(len(r) - SUSTAIN_STEPS + 1):
        if all(r[i + k] >= thr for k in range(SUSTAIN_STEPS)):
            return i
    return None


def _cell_estimate(profile: str, conc: int, turns: list[dict], params,
                   floor_excess: float = 0.0) -> tuple[CellEstimate | None, str]:
    """One cell's inverted (lo, hi) band, or (None, censoring verdict).

    ``floor_excess`` is the Phase-1 corrected-floor mode: ``floor = kernel_step + D`` is
    substituted for ``kernel_step`` EVERYWHERE the detection rule uses the floor (implied
    weight numerator/denominator, conditioning check, t_upper). With 0.0 the math is
    byte-identical to the original (v1, uncorrected) rule.
    """
    if not turns:
        return None, "no-turns"
    median_out = st.median(max(1.0, float(t["output_tokens"])) for t in turns)
    t_ceiling = saturated_ceiling_ms(median_out)  # the predictor's de-swung cell ceiling

    pts: list[tuple[float, float]] = []  # (pressure, implied weight)
    for t in turns:
        out = max(1.0, float(t["output_tokens"]))
        meas = float(t["tpot_meas"])
        if out < SAT_SUSTAIN_HI or meas <= 0:   # sustain-clean only: gate ≈ 1
            continue
        kin = KernelTurnInput(t["cached_context_tokens"], t["new_prefill_tokens"],
                              out, t["scheduled_requests"])
        floor = _kernel_step_ms(kin, params) + floor_excess
        t_upper = max(floor, t_ceiling)
        if t_upper - floor < max(MIN_COND_GAP_MS, MIN_COND_REL * floor):
            continue                             # ill-conditioned denominator
        ctx = (float(t["cached_context_tokens"]) + float(t["new_prefill_tokens"]) + 0.5 * out)
        psb = max(1, math.ceil(ctx / max(1, params.cache_block_size)))
        pressure = max(1.0, float(t["scheduled_requests"])) * psb / params.available_kv_blocks
        w = min(1.0, max(0.0, (meas - floor) / (t_upper - floor)))
        pts.append((pressure, w))

    if len(pts) < MIN_TURNS_PER_CELL:
        return None, "too-few-turns"
    pts.sort()
    pressures = [p for p, _ in pts]
    r = _rolling_median([w for _, w in pts], ROLL_WINDOW)

    i_low = _sustained_crossing(r, W_LOW)
    if i_low is None:
        return None, "never-saturates"
    if r[0] >= W_LOW:
        return None, "left-censored"
    i_high = _sustained_crossing(r, W_HIGH)
    if i_high is None:
        return None, "knee-censored"
    p_low, p_high = pressures[i_low], pressures[i_high]
    if p_high <= p_low:
        return None, "non-monotone"

    width = (p_high - p_low) / (U_HIGH - U_LOW)
    lo = p_low - U_LOW * width
    cluster = ("short" if median_out <= OUT_KNEE_LO
               else "long" if median_out >= OUT_KNEE_HI else "mid")
    return CellEstimate(profile, conc, cluster, len(pts),
                        round(p_low, 4), round(p_high, 4),
                        round(lo, 4), round(lo + width, 4)), "ok"


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties; scipy-free, deterministic)."""
    n = len(xs)
    if n < 2:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0          # average rank over the tie group, 1-based
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0.0 or syy == 0.0:
        return None                            # a constant has no rank correlation
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def _measure_floor_excess(cell_turns: dict[tuple[str, int], list[dict]], params) -> dict:
    """Phase-0 pre-registered floor-excess D (one per deployment, FULL-sample).

    Over turns with pressure < FLOOR_PRESSURE_MAX (saturation impossible), output >=
    SAT_SUSTAIN_HI (sustain-clean), tpot_meas > 0: ``D_raw = median(tpot_meas −
    kernel_step)``; ``D = max(0, D_raw)`` (a serving overhead cannot be negative).
    Also reports n, IQR and Spearman(excess, scheduled_requests) — a strong batch trend
    is documented but D stays a constant this round (no new fitted shape).
    """
    excess: list[float] = []
    scheds: list[float] = []
    for (_profile, _conc), turns in sorted(cell_turns.items()):
        for t in turns:
            out = max(1.0, float(t["output_tokens"]))
            meas = float(t["tpot_meas"])
            if out < SAT_SUSTAIN_HI or meas <= 0:
                continue
            sched = max(1.0, float(t["scheduled_requests"]))
            ctx = (float(t["cached_context_tokens"]) + float(t["new_prefill_tokens"]) + 0.5 * out)
            psb = max(1, math.ceil(ctx / max(1, params.cache_block_size)))
            if sched * psb / params.available_kv_blocks >= FLOOR_PRESSURE_MAX:
                continue
            kin = KernelTurnInput(t["cached_context_tokens"], t["new_prefill_tokens"],
                                  out, t["scheduled_requests"])
            excess.append(meas - _kernel_step_ms(kin, params))
            scheds.append(sched)
    if not excess:
        return {"value": 0.0, "raw_median": None, "iqr": None, "n": 0,
                "spearman_vs_sched": None}
    raw = st.median(excess)
    vals = sorted(excess)
    iqr = (vals[min(len(vals) - 1, int(0.75 * len(vals)))]
           - vals[int(0.25 * len(vals))])     # same quartile convention as _knee
    rho = _spearman(excess, scheds)
    return {"value": round(max(0.0, raw), 4), "raw_median": round(raw, 4),
            "iqr": round(iqr, 4), "n": len(excess),
            "spearman_vs_sched": None if rho is None else round(rho, 4)}


def _knee(cells: list[CellEstimate], pick: str, clusters: set[str]) -> dict:
    vals = sorted(getattr(c, pick) for c in cells if c.cluster in clusters)
    if not vals:
        return {"value": None, "n": 0, "iqr": None, "profiles": []}
    iqr = (vals[min(len(vals) - 1, int(0.75 * len(vals)))]
           - vals[int(0.25 * len(vals))])
    return {"value": round(st.median(vals), 4), "n": len(vals), "iqr": round(iqr, 4),
            "profiles": sorted({c.profile for c in cells if c.cluster in clusters})}


def _aggregate(cells: list[CellEstimate]) -> dict[str, dict]:
    return {
        "P_LO": _knee(cells, "lo", {"short", "mid", "long"}),
        "P_HI_SHORT": _knee(cells, "hi", {"short"}),
        "P_HI_LONG": _knee(cells, "hi", {"long"}),
    }


def _jackknife(cells: list[CellEstimate], full: dict[str, dict]) -> dict:
    """Leave-one-profile-out + leave-one-conc-out max deviation per knee."""
    out: dict[str, dict] = {k: {"max_dev": 0.0, "replicates": 0, "emptied": 0} for k in full}
    axes = ([("profile", p) for p in sorted({c.profile for c in cells})]
            + [("conc", c) for c in sorted({c.conc for c in cells})])
    for attr, held in axes:
        sub = [c for c in cells if getattr(c, attr) != held]
        agg = _aggregate(sub)
        for k in full:
            if full[k]["value"] is None:
                continue
            if agg[k]["value"] is None:
                out[k]["emptied"] += 1     # dropping one axis value emptied the estimate
                continue
            out[k]["replicates"] += 1
            out[k]["max_dev"] = round(max(out[k]["max_dev"],
                                          abs(agg[k]["value"] - full[k]["value"])), 4)
    return out


def _knee_block(cells: list[CellEstimate]) -> dict:
    """Aggregate + jackknife + adoptability verdict (the v1 logic, factored so the
    corrected-floor band reuses it unchanged)."""
    full = _aggregate(cells)
    jk = _jackknife(cells, full)
    knees = {}
    for k, agg in full.items():
        stable = adoptable = False
        if agg["value"] is not None:
            tol = max(STABILITY_ABS, 0.5 * (agg["iqr"] or 0.0))
            stable = jk[k]["max_dev"] <= tol and jk[k]["emptied"] == 0
            adoptable = (stable and agg["n"] >= MIN_CELLS_PER_KNEE
                         and len(agg["profiles"]) >= MIN_PROFILES_PER_KNEE)
        knees[k] = {**agg, "jackknife_max_dev": jk[k]["max_dev"],
                    "jackknife_emptied": jk[k]["emptied"], "stable": stable,
                    "adoptable": adoptable}
    return knees


def build(dep, exclude_profile: str | None = None) -> dict | None:
    bench_root = BENCH_BASE / dep.bench_dir
    if not bench_root.exists():
        print(f"SKIP {dep.gpu_key}: bench root missing ({bench_root})")
        return None
    params = dep.roofline
    cell_turns: dict[tuple[str, int], list[dict]] = {}
    for profile in PROFILES:
        for conc in CONCURRENCIES:
            f = bench_root / f"{profile}_conc{conc}.json"
            if f.exists():
                cell_turns[(profile, conc)] = build_turns(f)[0]

    # Phase 0: pressure-independent floor excess D — FULL sample (LOCO never drops it).
    floor_excess = _measure_floor_excess(cell_turns, params)
    d_ms = floor_excess["value"]

    def _band(fe: float) -> tuple[list[CellEstimate], dict[str, int]]:
        cells: list[CellEstimate] = []
        verdicts: dict[str, int] = {}
        for (profile, conc), turns in cell_turns.items():
            if exclude_profile is not None and profile == exclude_profile:
                continue   # LOCO drops BAND cells only
            est, verdict = _cell_estimate(profile, conc, turns, params, fe)
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if est:
                cells.append(est)
        return cells, verdicts

    cells, verdicts = _band(0.0)        # v1 uncorrected band — byte-compatible
    knees = _knee_block(cells)
    cells_c, verdicts_c = _band(d_ms)   # Phase-1 band against the corrected floor
    knees_corrected = _knee_block(cells_c)

    return {
        "gpu": dep.gpu_key,
        "model": dep.model,
        "tensor_parallel": dep.tp,
        "criterion": ("per-cell inversion of the measured implied-weight-vs-pressure curve "
                      f"through its sustained w={W_LOW}/w={W_HIGH} rolling-median crossings; "
                      "knees = median of the per-cell inverted band edges (P_LO over all usable "
                      "cells; P_HI_SHORT/P_HI_LONG over short-/long-output cells only)"),
        "rule": {
            "w_low": W_LOW, "w_high": W_HIGH, "u_low": round(U_LOW, 6), "u_high": round(U_HIGH, 6),
            "rolling_window": ROLL_WINDOW, "sustain_steps": SUSTAIN_STEPS,
            "sustain_min_output": SAT_SUSTAIN_HI, "min_turns_per_cell": MIN_TURNS_PER_CELL,
            "conditioning": f"t_upper - kernel_step >= max({MIN_COND_GAP_MS} ms, {MIN_COND_REL}*kernel_step)",
            "short_cell_max_median_output": OUT_KNEE_LO, "long_cell_min_median_output": OUT_KNEE_HI,
            "min_cells_per_knee": MIN_CELLS_PER_KNEE, "min_profiles_per_knee": MIN_PROFILES_PER_KNEE,
            "stability": f"jackknife max-dev <= max({STABILITY_ABS}, 0.5*IQR), no emptied replicate",
        },
        "knees": knees,
        "floor_excess_ms": floor_excess,
        "knees_corrected": knees_corrected,
        "corrected_rule": {
            "floor": "kernel_step + floor_excess_ms.value (D) substituted for kernel_step "
                     "everywhere (implied-weight numerator/denominator, conditioning, t_upper)",
            "floor_excess": (f"D_raw = median(tpot_meas - kernel_step) over turns with "
                             f"pressure < {FLOOR_PRESSURE_MAX}, output >= {SAT_SUSTAIN_HI}, "
                             f"tpot_meas > 0; D = max(0, D_raw); FULL-sample even under "
                             f"--exclude-profile (LOCO drops band cells only)"),
            "band": "identical pre-registered detection rule as 'knees' (see 'rule')",
        },
        "defcap_units": {k: (None if v["value"] is None else round(v["value"] - 1.0, 4))
                         for k, v in knees.items()},  # ramp_tpot DEF_* cross-reference
        # What production runs at build time. Since the 2026-06-10 ramp restructure
        # kernel_tpot has NO tuned knee literals (saturation onset/width are computed by
        # _overflow_weight) -> getattr returns None; the committed artifacts keep the
        # historical {0.88, 1.22, 2.0} record (pinned in test_kernel_tpot).
        "current_literals": {"P_LO": getattr(kernel_tpot, "P_LO", None),
                             "P_HI_SHORT": getattr(kernel_tpot, "P_HI_SHORT", None),
                             "P_HI_LONG": getattr(kernel_tpot, "P_HI_LONG", None)},
        "verdict_counts": dict(sorted(verdicts.items())),
        "n_usable_cells": len(cells),
        "cells": [vars(c) for c in sorted(cells, key=lambda c: (c.profile, c.conc))],
        "verdict_counts_corrected": dict(sorted(verdicts_c.items())),
        "n_usable_cells_corrected": len(cells_c),
        "cells_corrected": [vars(c) for c in sorted(cells_c, key=lambda c: (c.profile, c.conc))],
        "excluded_profile": exclude_profile,
        "source": str(bench_root),
        "_notes": ("Measured ramp-knee band for the kernel_tpot amplifier. Regenerate: "
                   "python3 -m profiling.process.build_ramp_knees. 'adoptable' is the "
                   "MEASUREMENT-QUALITY verdict only (enough cells/profiles + jackknife-stable); "
                   "production adoption is additionally gate-conditional (no TPOT/TTFT/E2EL "
                   "regression). OUTCOME 2026-06-09: the measured band DISAGREES with the tuned "
                   "production knees and failed the gates (H100 TPOT 15.4->23.3%; per-knee "
                   "isolation also failed) -> production keeps the tuned values with honest "
                   "tuned-knob labels; this artifact documents the disagreement. The knees are "
                   "compensating fits for the ramp shape itself (the implied weight attributes "
                   "any excess over kernel_step to KV pressure). See the De-fit log in "
                   "profiling/docs/prediction_construction.md."),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exclude-profile", default=None, choices=PROFILES, metavar="PROFILE",
                    help="LOCO: exclude this bench profile from the BAND cells only "
                         "(D stays full-sample). Requires a non-canonical --out-dir.")
    ap.add_argument("--out-dir", default=str(CANONICAL_OUT_DIR), metavar="DIR",
                    help=f"artifact output directory (default: {CANONICAL_OUT_DIR})")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.exclude_profile and out_dir.resolve() == CANONICAL_OUT_DIR.resolve():
        raise SystemExit("--exclude-profile (LOCO) must NOT overwrite the canonical "
                         f"artifacts: pass --out-dir != {CANONICAL_OUT_DIR}")

    orig_grid = kernel_step_cost._default_grid
    orig_ceiling = kernel_tpot._active_ceiling_json
    try:
        for dep in all_deployments():
            # Needs ground truth + an OWNED measured ceiling + a MEASURED decode grid — an
            # analytic-roofline kernel_step would taint the implied weight. (dep.engine is the
            # serving engine; dep.backend is the calibration label, e.g. "kernel-headline".)
            if not (dep.model == "Llama-3.1-8B" and dep.engine == "vllm" and dep.ground_truth
                    and dep.saturated_ceiling is not None
                    and dep.decode_grid is not None and dep.decode_grid.exists()):
                continue
            grid = load_grid(dep.decode_grid)
            kernel_step_cost._default_grid = lambda grid=grid: grid
            kernel_tpot._active_ceiling_json = dep.saturated_ceiling

            payload = build(dep, exclude_profile=args.exclude_profile)
            if payload is None:
                continue
            slug = dep.gpu_key.lower().replace(" ", "_").replace("(", "").replace(")", "")
            out = out_dir / f"ramp_knees_{slug}_llama31_8b.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"\n{dep.gpu_key}: wrote {out}")
            print(f"  verdicts: {payload['verdict_counts']}")
            for k, v in payload["knees"].items():
                cur = payload["current_literals"][k]
                print(f"  {k:<11} measured={v['value']} (n={v['n']}, iqr={v['iqr']}, "
                      f"jk_dev={v['jackknife_max_dev']}, profiles={len(v['profiles'])}) "
                      f"current={cur}  -> {'ADOPTABLE' if v['adoptable'] else 'RELABEL (keep current)'}")
            fe = payload["floor_excess_ms"]
            print(f"  floor_excess D={fe['value']} ms (raw_median={fe['raw_median']}, "
                  f"iqr={fe['iqr']}, n={fe['n']}, spearman_vs_sched={fe['spearman_vs_sched']})")
            for k, v in payload["knees_corrected"].items():
                print(f"  corrected {k:<11} measured={v['value']} (n={v['n']}, iqr={v['iqr']}, "
                      f"jk_dev={v['jackknife_max_dev']}, profiles={len(v['profiles'])}) "
                      f"-> {'ADOPTABLE' if v['adoptable'] else 'not adoptable'}")
    finally:
        kernel_step_cost._default_grid = orig_grid
        kernel_tpot._active_ceiling_json = orig_ceiling


if __name__ == "__main__":
    main()
