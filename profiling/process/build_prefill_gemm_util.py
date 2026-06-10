#!/usr/bin/env python3
"""Build the measured prefill-GEMM util(m) artifact from the per-step CUDA-event sweep.

Phase C of `ttft_pricing_defit_plan.md` Item 2 (audit-v2 R1/S6): converts
``profile_data/results/prefill_util_sweep_H100.csv`` (written on the H100 by
``profiling/gpu_profiling/vllm/cuda_events/prefill_util_sweep.py`` — per-step device-ms at
exact full-budget chunk sizes m) into a small measured-anchor artifact, the same pattern as
the decode grid / saturated ceiling: one (m, util_sim) anchor per measured budget, linear
interpolation between them, clamp outside the measured range.

``util_sim`` is the sim's pricing convention — ``roofline_ms(m) = 2·N_PARAMS_SIM·m/PEAK``
over the measured median device-ms — i.e. exactly the util that
``ttft_queue_sim._prefill_gemm_per_tok_loaded`` divides by. The artifact REPLACES the
``util_flops → PREFILL_GEMM_UTIL_SAT=1.0`` linear ramp (a validation-anchored cap with no
measurement behind it; the debunked "15.5 ms/1k GT cohort" anchor is recorded in the De-fit
log) with measured anchors. ``util_gemm`` (executed fused-linear FLOPs, N=6.979e9) is
reported per anchor for comparison with the offline microbench — NOT for pricing.

Deterministic (closed-form OLS per budget; no RNG). Usage:
    python3 -m profiling.process.build_prefill_gemm_util
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_PARAMS_SIM = 8.03e9    # the sim's pricing convention (matches _prefill_gemm_per_tok)
N_GEMM = 6.979e9         # executed fused-linear FLOPs (microbench-comparison only)
PEAK_FLOPS = 989e12      # H100 bf16 dense

CSV_IN = Path("profile_data/results/prefill_util_sweep_H100.csv")
OUT_JSON = Path("profile_data/kernels/prefill_gemm_util_H100.json")
MIN_STEPS_PER_ANCHOR = 6   # full-budget steps required to trust an anchor's median


def main() -> None:
    if not CSV_IN.exists():
        raise SystemExit(f"missing {CSV_IN} — pull it from the H100 run "
                         f"(prefill_util_sweep.py, see its RUN block)")
    by_budget: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with CSV_IN.open() as f:
        for row in csv.DictReader(f):
            assert int(row["tokens"]) == int(row["budget"]), "non-full step leaked into the CSV"
            by_budget[int(row["budget"])].append((int(row["step"]), float(row["device_ms"])))

    anchors = []
    for m in sorted(by_budget):
        pts = by_budget[m]
        if len(pts) < MIN_STEPS_PER_ANCHOR:
            print(f"SKIP m={m}: only {len(pts)} full steps (<{MIN_STEPS_PER_ANCHOR})")
            continue
        # Per-step device time = GEMM(m) + attention over the GROWING resident prefix. The sim
        # prices the attention part SEPARATELY (PREFILL_FA3_MS_PER_TOKEN2·M·(R+0.5·M)), so the
        # util anchor must be the ZERO-PREFIX intercept: OLS of device_ms against the sim's own
        # FA3 regressor x = (step+0.5)·m² (M=m, R=step·m). Slope = an independent re-measurement
        # of the FA3 coefficient (cross-check, reported per anchor).
        xs = [(i + 0.5) * m * m for i, _ in pts]
        ys = [y for _, y in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        a = my - slope * mx
        ss_res = sum((y - (a + slope * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        anchors.append({
            "m_tokens": m,
            "gemm_intercept_ms": round(a, 4),
            "n_steps": n,
            "fa3_slope_ms_per_tok2": float(f"{slope:.4e}"),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
            "util_sim": round(2.0 * N_PARAMS_SIM * m / PEAK_FLOPS * 1e3 / a, 4),
            "util_gemm": round(2.0 * N_GEMM * m / PEAK_FLOPS * 1e3 / a, 4),
        })
    if len(anchors) < 3:
        raise SystemExit(f"only {len(anchors)} usable anchors — sweep incomplete")

    payload = {
        "schema": "prefill_gemm_util.v1",
        "gpu": "H100",
        "model": "Llama-3.1-8B",
        "source_csv": str(CSV_IN),
        "method": ("per-step CUDA events around GPUModelRunner.execute_model (in-process mp=0, "
                   "tp1, prefix caching OFF, chunked prefill ON, max_num_batched_tokens=m); "
                   "exact full-budget steps only, token counts read from SchedulerOutput"),
        "conventions": {"util_sim_n_params": N_PARAMS_SIM, "util_gemm_n_params": N_GEMM,
                        "peak_flops_per_s": PEAK_FLOPS},
        "anchors": anchors,
        "lookup": ("linear interpolation of util_sim in m between anchors; clamp to the nearest "
                   "anchor outside the measured range"),
        "_notes": ("Measured replacement for the retired util_flops->PREFILL_GEMM_UTIL_SAT=1.0 "
                   "linear ramp (audit-v2 R1/S6: the cap's claimed GT anchor was a shared-prefix "
                   "double-count; see prediction_construction.md De-fit log 2026-06-10). "
                   "Regenerate: python3 -m profiling.process.build_prefill_gemm_util."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_JSON}")
    for a in anchors:
        print(f"  m={a['m_tokens']:>5}  gemm-intercept {a['gemm_intercept_ms']:8.3f} ms "
              f"(n={a['n_steps']}, r2={a['r2']}, fa3 {a['fa3_slope_ms_per_tok2']}) -> "
              f"util_sim {a['util_sim']:.4f} | util_gemm {a['util_gemm']:.4f}")


if __name__ == "__main__":
    main()
