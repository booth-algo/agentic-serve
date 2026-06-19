"""Forward-MAPE companion to build_simulator_rows.

The backtester (build_simulator_rows) reads each cell's realized trajectory_pool and scores
prediction-vs-measured. This runs the FORWARD predictor (simulator.forward.predict_forward) over the
SAME cells, fed the per-session multi-turn trajectories extracted from the GT — i.e. the workload AS A
CLIENT WOULD PROVIDE IT — and computes the forward path's MAPE against the same measured GT. Writes
forward-predictions.json; the dashboard joins it to the backtester rows on (gpu_key, model, profile,
concurrency) to show forward-pred + forward-MAPE alongside the backtester's.

For Llama-8B (has a trajectory_pool) forward ≈ backtester (validation). For the 70B / MoE cells the
backtester uses MARGINAL cohorts (no Llama pool) while forward uses the JOINT trajectories, so the
columns diverge — that delta is the point.

Run: PYTHONPATH=/root/agentic-serve python -m profiling.process.build_forward_rows [model-filter]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import profiling.process.build_simulator_rows as B  # noqa: E402
from configs.loader import all_deployments  # noqa: E402
from simulator.forward import predict_forward  # noqa: E402

FORWARD_JSON = Path("inference-benchmark/dashboard/public/forward-predictions.json")


def _trajectories(bench: Path) -> list[list[tuple[float, float, float]]]:
    """Per-session multi-turn [(cached, new, output), ...] from a benchmark run's per_request array."""
    data = json.loads(bench.read_text())
    by_sid: dict = defaultdict(dict)
    for r in data.get("per_request") or []:
        if not r.get("success"):
            continue
        by_sid[r["session_id"]][int(r.get("turn_index") or 0)] = (
            float(r.get("cached_context_tokens") or 0.0),
            float(r.get("new_prefill_tokens") or 0.0),
            max(1.0, float(r.get("output_tokens") or 1.0)),
        )
    return [[turns[ti] for ti in sorted(turns)] for turns in by_sid.values() if turns]


def _cell_mape(fwd: list[float], turns: list[dict], meas_key: str) -> float | None:
    n = min(len(fwd), len(turns))
    aps = [abs(fwd[i] - turns[i][meas_key]) / turns[i][meas_key] * 100.0
           for i in range(n) if isinstance(turns[i].get(meas_key), (int, float)) and turns[i][meas_key]]
    return round(st.mean(aps), 4) if aps else None


def build_forward_row(cfg, profile: str, conc: int, bench: Path) -> dict | None:
    trajs = _trajectories(bench)
    if not trajs:
        return None
    turns, sp = B.build_turns(bench)
    if not turns:
        return None
    res = predict_forward(gpu=cfg.gpu, model=cfg.model, tp=cfg.tp, engine=cfg.engine,
                          concurrency=conc, trajectories=trajs, shared_prefix_tokens=sp)
    fwd = res.per_turn
    gt = cfg.ground_truth
    return {
        "model": cfg.model, "backend": cfg.backend, "profile": profile, "concurrency": conc,
        "tensor_parallel_size": cfg.tp,
        "fwd_tpot_pred": round(res.tpot_ms, 4),
        "fwd_ttft_pred": round(res.ttft_ms, 4),
        "fwd_e2el_pred": round(res.e2el_ms, 4),
        "fwd_tpot_err": _cell_mape(fwd["tpot"], turns, "tpot_meas") if gt else None,
        "fwd_ttft_err": _cell_mape(fwd["ttft"], turns, "ttft_meas") if gt else None,
        "fwd_e2el_err": _cell_mape(fwd["e2el"], turns, "e2el_meas") if gt else None,
        "fwd_calibration": res.calibration_status,
    }


def main(model_filter: str | None = None) -> None:
    payload: dict[str, list[dict]] = {}
    for cfg in all_deployments():
        if model_filter and cfg.model != model_filter:
            continue
        root = B.BENCH_BASE / cfg.bench_dir
        if not root.exists():
            continue
        rows: list[dict] = []
        for profile in B.PROFILES:
            for conc in B.CONCURRENCIES:
                bench = root / f"{profile}_conc{conc}.json"
                if not bench.exists():
                    continue
                try:
                    r = build_forward_row(cfg, profile, conc, bench)
                except Exception as e:  # one bad cell shouldn't sink the run
                    print(f"  ERR {cfg.gpu_key} {profile} c{conc}: {e}")
                    r = None
                if r:
                    rows.append(r)
        if rows:
            payload.setdefault(cfg.gpu_key, []).extend(rows)
            print(f"{cfg.gpu_key} [{cfg.model}] += {len(rows)} forward rows")
    FORWARD_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {sum(len(v) for v in payload.values())} forward rows across {len(payload)} configs "
          f"-> {FORWARD_JSON}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
