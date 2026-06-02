#!/usr/bin/env python3
"""Derive the measured saturated-ITL ceiling anchors for the kernel TPOT amplifier.

Replaces the retired least-squares ceiling ``118.7 + 3263/output`` (a 2-coefficient
regression to measured plateau ITL) with a small set of **measured anchors**: the
median measured ``tpot_ms`` over turns in the saturated regime (KV pressure >= 2.5,
i.e. the "C=300+" asymptote), grouped by output-length cluster. ``saturated_ceiling_ms``
then linearly interpolates between these measured points (the same fit-free
measured-anchor + interpolation pattern as the decode kernel grid).

Pressure is the same workload quantity the predictor uses:
``pressure = scheduled_requests * per_session_blocks / available_kv_blocks``.

Saturated turns fall into disjoint output clusters (short-output agentic coding vs
long-output osworld); the cluster split sits in the empty output gap, so the anchors
are invariant to its exact value. One anchor per populated cluster.

Usage:
    python3 -m profiling.process.build_saturated_ceiling
"""
from __future__ import annotations

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

PRESSURE_THRESHOLD = 2.5      # saturated regime (the C=300+ asymptote the ceiling models)
CLUSTER_SPLIT_OUTPUT = 50.0   # sits in the empty output gap [35,75]; anchors invariant to it
CACHE_BLOCK_SIZE = 16


@dataclass(frozen=True)
class CeilingConfig:
    gpu: str
    tp: int
    bench_dir: str
    available_kv_blocks: int
    out_json: Path


# Generate a ceiling for every deployment that OWNS one (manifest data.saturated_ceiling status
# measured/derived, with a path). Configs that INHERIT the H100 ceiling (e.g. H100x2) are skipped.
# Driven by configs/deployments/*.json — to add one, set that deployment's saturated_ceiling status.
CONFIGS = [
    CeilingConfig(d.gpu_key, d.tp, d.bench_dir, d.available_kv_blocks,
                  Path((d.data.get("saturated_ceiling") or {})["path"]))
    for d in all_deployments()
    if (d.data.get("saturated_ceiling") or {}).get("status") in ("measured", "derived")
    and (d.data.get("saturated_ceiling") or {}).get("path")
]


def _saturated_turns(cfg: CeilingConfig) -> list[tuple[float, float]]:
    """(output_tokens, tpot_meas) for turns at pressure >= PRESSURE_THRESHOLD."""
    root = BENCH_BASE / cfg.bench_dir
    out: list[tuple[float, float]] = []
    for prof in PROFILES:
        for c in CONCURRENCIES:
            f = root / f"{prof}_conc{c}.json"
            if not f.exists():
                continue
            for tn in build_turns(f):
                output = max(1.0, float(tn["output_tokens"]))
                sched = max(1.0, float(tn["scheduled_requests"]))
                ctx = (tn["cached_context_tokens"] + tn["new_prefill_tokens"]
                       + 0.5 * output)
                psb = max(1, math.ceil(ctx / CACHE_BLOCK_SIZE))
                pressure = sched * psb / cfg.available_kv_blocks
                meas = float(tn["tpot_meas"])
                if pressure >= PRESSURE_THRESHOLD and meas > 0:
                    out.append((output, meas))
    return out


def build(cfg: CeilingConfig) -> dict:
    sat = _saturated_turns(cfg)
    if not sat:
        raise SystemExit(f"{cfg.gpu}: no saturated turns (pressure>={PRESSURE_THRESHOLD})")
    short = [(o, m) for o, m in sat if o < CLUSTER_SPLIT_OUTPUT]
    long = [(o, m) for o, m in sat if o >= CLUSTER_SPLIT_OUTPUT]
    anchors = []
    for cluster in (short, long):
        if not cluster:
            continue
        outs = [o for o, _ in cluster]
        ms = [m for _, m in cluster]
        anchors.append({
            "output_tokens": round(st.median(outs)),
            "plateau_ms": round(st.median(ms), 1),
            "n": len(cluster),
        })
    anchors.sort(key=lambda a: a["output_tokens"])
    all_ms = sorted(m for _, m in sat)
    return {
        "gpu": cfg.gpu,
        "model": "Llama-3.1-8B",
        "tensor_parallel": cfg.tp,
        "criterion": (f"median measured tpot_ms over turns at KV pressure >= "
                      f"{PRESSURE_THRESHOLD} (the saturated 'C=300+' asymptote)"),
        "pressure_threshold": PRESSURE_THRESHOLD,
        "cluster_split_output": CLUSTER_SPLIT_OUTPUT,
        "source": str(BENCH_BASE / cfg.bench_dir),
        "n_saturated_turns": len(sat),
        "max_measured_plateau_ms": round(all_ms[-1], 1),
        "p90_measured_plateau_ms": round(all_ms[min(len(all_ms) - 1, int(len(all_ms) * 0.9))], 1),
        "anchors": anchors,
        "lookup": ("linear interpolation in output between anchors; clamp to the "
                   "nearest anchor outside the range (monotone non-increasing: "
                   "short output saturates higher)"),
        "_notes": ("Measured-anchor replacement for the retired least-squares ceiling "
                   "118.7 + 3263/output (no fit; measured plateau medians + interpolation, "
                   "same pattern as the decode kernel grid). Regenerate: "
                   "python3 -m profiling.process.build_saturated_ceiling. "
                   "See profiling/docs/fitted_constants_audit.md."),
    }


def main() -> None:
    for cfg in CONFIGS:
        if not (BENCH_BASE / cfg.bench_dir).exists():
            print(f"SKIP {cfg.gpu}: bench root missing ({BENCH_BASE / cfg.bench_dir})")
            continue
        payload = build(cfg)
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        anchors = ", ".join(f"out={a['output_tokens']}->{a['plateau_ms']}ms(n={a['n']})"
                            for a in payload["anchors"])
        print(f"{cfg.gpu}: {payload['n_saturated_turns']} saturated turns -> anchors [{anchors}] "
              f"-> {cfg.out_json}")


if __name__ == "__main__":
    main()
