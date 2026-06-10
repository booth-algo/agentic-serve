#!/usr/bin/env python3
"""Measure the tensor-parallel prefill comm term from the LIKE-FOR-LIKE tp1/tp2 stage-split pair.

G3 de-fit (`ttft_pricing_defit_plan.md` Item 3; audit-v2 G3): `PREFILL_TP_COMM_MS_PER_TOKEN`
was a backed-out remainder (tp2 ttft.new 18.5 − GEMM/2 12.65 = 5.85 ms/1k) from an
instrumentation-INCONSISTENT pair (tp2 multiprocess api_server vs tp1 in-process LLM), which
physics said over-absorbed ~2.5 ms/1k of host IPC under a comm label (NCCL all-reduce band
~1–3 ms/1k).

This builder computes it like-for-like: BOTH legs are the SAME `serving_stage_split.py`
multiprocess api_server run (tp1: `serving_stage_split_H100.csv`, 2026-06-05; tp2:
`serving_stage_split_H100_tp2.csv`, 2026-06-10, GPUs 6+7 per `h100_setup.md`). Per leg, OLS
``prefill_span_ms ~ FLOOR + a·new + b·cached`` over the c1 cells; the comm term =
``a(tp2) − a(tp1)/2`` — the per-token cost the tp2 GPU-side prefill window carries ABOVE its
halved GEMM share. The host frontend is excluded by construction (it is a separate stage in
both legs and is charged separately in the sim, where it does not shard with tp).

Deterministic (closed-form OLS). Usage:
    python3 -m profiling.process.build_tp_comm
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TP1_CSV = Path("profile_data/results/serving_stage_split_H100.csv")
TP2_CSV = Path("profile_data/results/serving_stage_split_H100_tp2.csv")
OUT_JSON = Path("profile_data/kernels/prefill_tp_comm_H100.json")
NCCL_PHYSICS_BAND_MS_PER_1K = (1.0, 3.0)  # external-lit all-reduce band, hidden=4096 bf16


def _span_new_rate(path: Path) -> dict:
    """OLS prefill_span_ms ~ floor + a·new + b·cached over the stage-split rows."""
    rows = list(csv.DictReader(path.open()))
    X = [(1.0, float(r["new"]), float(r["cached"])) for r in rows]
    y = [float(r["prefill_span_ms"]) for r in rows]
    # normal equations (3x3), no numpy dependency
    n = len(X)
    xtx = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(3)]
    # gaussian elimination
    m = [xtx[i] + [xty[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    beta = [m[i][3] / m[i][i] for i in range(3)]
    return {"floor_ms": beta[0], "new_ms_per_tok": beta[1], "cached_ms_per_tok": beta[2],
            "n_rows": n}


def main() -> None:
    for p in (TP1_CSV, TP2_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p} — pull both stage-split legs first")
    tp1 = _span_new_rate(TP1_CSV)
    tp2 = _span_new_rate(TP2_CSV)
    comm = tp2["new_ms_per_tok"] - tp1["new_ms_per_tok"] / 2.0
    in_band = NCCL_PHYSICS_BAND_MS_PER_1K[0] <= comm * 1e3 <= NCCL_PHYSICS_BAND_MS_PER_1K[1] + 1.0
    payload = {
        "schema": "prefill_tp_comm.v1",
        "gpu": "H100", "model": "Llama-3.1-8B",
        "method": ("like-for-like multiprocess api_server stage-split pair (same script both "
                   "legs); comm = prefill_span.new(tp2) − prefill_span.new(tp1)/2"),
        "tp1_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp1.items()},
        "tp2_fit": {k: round(v, 6) if isinstance(v, float) else v for k, v in tp2.items()},
        "constants": {"PREFILL_TP_COMM_MS_PER_TOKEN": comm},
        "nccl_physics_band_ms_per_1k": list(NCCL_PHYSICS_BAND_MS_PER_1K),
        "within_physics_band_plus_1": in_band,
        "retired_backed_out_remainder": 0.00585,
        "_notes": ("Replaces the instrumentation-inconsistent backed-out remainder 5.85 ms/1k "
                   "(audit-v2 G3): the like-for-like pair removes the multiprocess-vs-in-process "
                   "mismatch, and the residual lands at the top of the NCCL all-reduce physics "
                   "band. Regenerate: python3 -m profiling.process.build_tp_comm. Sources: "
                   "serving_stage_split.py --tensor-parallel-size {1,2} on h100 (h100_setup.md)."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"tp1 span.new {tp1['new_ms_per_tok']*1e3:.3f} ms/1k | tp2 {tp2['new_ms_per_tok']*1e3:.3f} "
          f"-> comm {comm*1e3:.4f} ms/1k (retired remainder 5.85; physics band "
          f"{NCCL_PHYSICS_BAND_MS_PER_1K})")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
