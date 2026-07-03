#!/usr/bin/env python3
"""Build simulator-v2sim-predictions.json for the dashboard "Simulator v2" tab.

Deployment-driven like v1's emitter: iterates simulator_v2/configs/deployments/*.yaml
and emits `{gpu_key: ServingRow[]}` for every deployment whose bench_dir has ground
truth. The calibrated config (H100 / Llama-3.1-8B / tp1 / vllm) runs the kernel
composition; everything else runs the analytic Roofline as a LABELED first-cut, with
the saturated ceiling scaled from the measured H100-8B plateau (v1's anchoring).

Usage:  python3 inference-benchmark/scripts/build_simulator_v2_predictions.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # make simulator_v2 (at repo root) importable
os.chdir(REPO)  # simulator_v2 resolves kernel artifacts via a CWD-relative path

from simulator_v2.engine.predict import predict  # noqa: E402
from simulator_v2.getters.hardware import (  # noqa: E402
    Roofline,
    load_kernel_composition_hardware,
)
from simulator_v2.configs.config_loader import load_gpu_config, load_model_config  # noqa: E402
from simulator_v2.core.types import SchedulerSettings  # noqa: E402
from simulator_v2.getters.workload import load_benchmark  # noqa: E402
from simulator_v2.kv_wall.saturated_ceiling import load_saturated_ceiling  # noqa: E402

STORE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
DEPLOY_DIR = REPO / "simulator_v2/configs/deployments"
CFG = REPO / "simulator_v2/configs"
OUT = REPO / "inference-benchmark/dashboard/public/simulator-v2sim-predictions.json"
H100_CEILING_JSON = REPO / "profile_data/kernels/saturated_ceiling/H100.json"

GPU_YAMLS = {"H100": "h100.yaml", "A100": "a100.yaml",
             "RTX3090": "rtx3090.yaml", "RTX2080Ti": "rtx2080ti.yaml"}
MODEL_YAMLS = {"Llama-3.1-8B": "llama3.1-8b.yaml", "Llama-3.1-70B": "llama-3.1-70b.yaml",
               "Llama-3.3-70B": "llama-3.3-70b.yaml", "Mixtral-8x7B": "mixtral-8x7b.yaml",
               "Qwen2.5-72B": "qwen2.5-72b.yaml", "Qwen3.5-27B": "qwen3.5-27b.yaml",
               "Qwen3.5-9B": "qwen3.5-9b.yaml", "gpt-oss-20b": "gpt-oss-20b.yaml",
               "gpt-oss-120b": "gpt-oss-120b.yaml"}

PROFILES = frozenset({
    "chat-multiturn-synth", "osworld-multiturn-synth",
    "swebench-multiturn-synth", "terminalbench-multiturn-synth",
})

# Uncalibrated (roofline) configs skip cells beyond this KV pressure: a consumer
# GPU's pool at c320 is 50x oversubscribed -- the event sim grinds toward its
# event guard and the prediction is meaningless anyway. The calibrated page maxes
# out near pressure ~10 (swebench c320), so 12 keeps everything comparable.
PRESSURE_CAP = 12.0

# Per-deployment result cache: a worker crash / restart skips finished configs.
CACHE_DIR = Path("/tmp/wf-hint/v2rows")


def _analytic_sat_bytes(gpu, model, tp: int, pool_blocks: int) -> float:
    """Per-GPU bytes moved in one saturated decode step: sharded weights + this
    shard of a full-pool KV read."""
    kv_shard = max(1, min(tp, model.kv_heads))
    pool_tokens = pool_blocks * model.cache_block_size
    return (model.n_params * model.bytes_per_param / tp
            + pool_tokens * model.kv_bytes_per_token / kv_shard)


class RooflineFirstCut(Roofline):
    """Roofline with a first-cut saturated ceiling: analytic full-pool bandwidth
    step, anchored to the measured H100/Llama-3.1-8B plateau (v1's scaling)."""
    _ceiling_ms: float = 200.0

    def saturated_ceiling_ms(self, output: float) -> float:
        return self._ceiling_ms


def _h100_8b_anchor() -> tuple[float, float]:
    """(measured H100-8B plateau ms, analytic H100-8B saturated-step ms)."""
    gpu = load_gpu_config(CFG / "gpu_configs/h100.yaml")
    model = load_model_config(CFG / "model_configs/llama3.1-8b.yaml")
    measured = load_saturated_ceiling(H100_CEILING_JSON).ceiling_ms(1.0)
    analytic = _analytic_sat_bytes(gpu, model, 1, 27250) / (
        gpu.peak_bw_bytes_per_s * gpu.util_bw) * 1e3
    return measured, analytic


def _clean_profile(raw: str, prefix: str) -> str:
    return raw[len(prefix):] if raw.startswith(prefix) else raw


def _ape(pred: float, meas: float) -> float | None:
    return round(abs(pred - meas) / meas * 100.0, 2) if meas and meas > 0 else None


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.mean(xs), 4) if xs else 0.0


def _cell_mape(per_turn: list[dict], err_key: str) -> float | None:
    apes = [pt[err_key] for pt in per_turn if isinstance(pt.get(err_key), (int, float))]
    return round(st.mean(apes), 2) if apes else None


def _row(cell, preds, dep: dict, profile: str) -> dict:
    turns, gt = cell.turns, cell.ground_truth or []
    per_turn = []
    for ti, (t, g, p) in enumerate(zip(turns, gt, preds)):
        total = float(t.isl_tokens)
        per_turn.append({
            "turn_index": ti,
            "successful": int(t.scheduled_requests) or 1,
            "total_context_tokens": total,
            "new_prefill_tokens": float(t.new_prefill_tokens),
            "cached_context_tokens": float(t.cache_hit_tokens),
            "cache_hit_rate": (t.cache_hit_tokens / total) if total > 0 else 0.0,
            "output_tokens": float(t.osl_tokens),
            "scheduled_requests": float(t.scheduled_requests),
            "ttft_meas": g.ttft_ms, "ttft_pred": round(p.ttft_ms, 4), "ttft_err": _ape(p.ttft_ms, g.ttft_ms),
            "tpot_meas": g.tpot_ms, "tpot_pred": round(p.tpot_ms, 4), "tpot_err": _ape(p.tpot_ms, g.tpot_ms),
            "e2el_meas": g.e2el_ms, "e2el_pred": round(p.e2el_ms, 4), "e2el_err": _ape(p.e2el_ms, g.e2el_ms),
        })
    return {
        "model": dep["model"],
        "backend": dep.get("backend", "v2"),
        "profile": profile,
        "data_scope": "synthetic_distributional",
        "mode": "single-turn" if len(turns) <= 1 else "multi-turn",
        "concurrency": cell.concurrency,
        "isl": float(turns[0].isl_tokens) if turns else 0.0,
        "osl": float(turns[0].osl_tokens) if turns else 0.0,
        "tensor_parallel_size": int(dep.get("tp", 1)),
        "predicted_turn_count": len(turns),
        "multiturn_prediction_mode": dep["_pred_mode"],
        "calibration_status": dep["_calibration"],
        "multiturn_turn_predictions": per_turn,
        "tpot_pred": _mean([x["tpot_pred"] for x in per_turn]),
        "tpot_meas": _mean([x["tpot_meas"] for x in per_turn]),
        "tpot_err": _cell_mape(per_turn, "tpot_err"),
        "ttft_pred": _mean([x["ttft_pred"] for x in per_turn]),
        "ttft_meas": _mean([x["ttft_meas"] for x in per_turn]),
        "ttft_err": _cell_mape(per_turn, "ttft_err"),
        "e2el_pred": _mean([x["e2el_pred"] for x in per_turn]),
        "e2el_meas": _mean([x["e2el_meas"] for x in per_turn]),
        "e2el_err": _cell_mape(per_turn, "e2el_err"),
    }


def _is_calibrated(dep: dict) -> bool:
    return (dep["gpu"] == "H100" and dep["model"] == "Llama-3.1-8B"
            and int(dep.get("tp", 1)) == 1 and dep.get("engine") == "vllm")


def _hardware(dep: dict, anchor: tuple[float, float]):
    if _is_calibrated(dep):
        return load_kernel_composition_hardware(
            CFG / "gpu_configs/h100.yaml", CFG / "model_configs/llama3.1-8b.yaml", tp=1)
    gpu = load_gpu_config(CFG / "gpu_configs" / GPU_YAMLS[dep["gpu"]])
    model = load_model_config(CFG / "model_configs" / MODEL_YAMLS[dep["model"]])
    tp = int(dep.get("tp", 1))
    pool = int(dep.get("available_kv_blocks") or 0)
    sched = SchedulerSettings(
        max_num_batched_tokens=int(dep.get("max_num_batched_tokens") or 8192),
        long_prefill_token_threshold=int(gpu.max_model_len * 0.04),
    )
    hw = RooflineFirstCut(gpu=gpu, model=model, tp=tp,
                          kv_pool_blocks=pool, sched=sched)
    measured, analytic = anchor
    own = _analytic_sat_bytes(gpu, model, tp, hw.kv_pool_blocks) / (
        gpu.peak_bw_bytes_per_s * gpu.util_bw) * 1e3
    hw._ceiling_ms = own * (measured / analytic)
    return hw


def _build_deployment(args_tuple):
    dep_path, anchor = args_tuple
    cache = CACHE_DIR / (Path(dep_path).stem + ".json")
    if cache.exists():
        data = json.loads(cache.read_text())
        return (data["gpu_key"], data["rows"],
                None if data["rows"] else f"{Path(dep_path).name}: cached, 0 rows")
    dep = yaml.safe_load(Path(dep_path).read_text())
    bench = STORE / dep.get("bench_dir", "")
    if not dep.get("bench_dir") or not bench.is_dir():
        return dep.get("gpu_key", "?"), [], f"{Path(dep_path).name}: no bench_dir"
    calibrated = _is_calibrated(dep)
    dep["_pred_mode"] = ("kernel_composition_v2" if calibrated
                         else "v2_analytic_roofline")
    dep["_calibration"] = ("simulator_v2_kernel_composition" if calibrated
                           else "v2_roofline_firstcut")
    try:
        hw = _hardware(dep, anchor)
    except (KeyError, FileNotFoundError) as e:
        return dep.get("gpu_key", "?"), [], f"{Path(dep_path).name}: {e}"
    prefix = dep["bench_dir"].split("_", 1)[1] + "_"
    rows = []
    capped = 0
    pool_tokens = max(1, hw.kv_pool_blocks * hw.cache_block_size)
    files = sorted(f for f in glob.glob(str(bench / "*_conc*.json"))
                   if "_per_turn" not in f)
    for f in files:
        cell = load_benchmark(Path(f))
        profile = _clean_profile(cell.profile, prefix)
        if profile not in PROFILES or not cell.turns or not cell.ground_truth:
            continue
        if not calibrated:
            pressure = cell.concurrency * max(
                t.isl_tokens + t.osl_tokens for t in cell.turns) / pool_tokens
            if pressure > PRESSURE_CAP:
                capped += 1
                continue
        preds = predict(hw, cell.turns, cell.concurrency,
                        trajectories=cell.trajectories,
                        shared_prefix_tokens=cell.shared_prefix_tokens)
        rows.append(_row(cell, preds, dep, profile))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"gpu_key": dep["gpu_key"], "rows": rows}))
    note = f" ({capped} cells over pressure cap)" if capped else ""
    return dep["gpu_key"], rows, None if rows else f"{Path(dep_path).name}: 0 rows{note}"


def main() -> None:
    ap = argparse.ArgumentParser()
    # 7GB box: workers json.load multi-hundred-MB bench files (~1-2GB in-process);
    # more than ~3 concurrent parsers thrashes/OOMs the pool.
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()

    anchor = _h100_8b_anchor()
    deps = sorted(glob.glob(str(DEPLOY_DIR / "*.yaml")))
    out: dict[str, list] = {}
    skipped = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for gpu_key, rows, err in pool.map(
                _build_deployment, [(d, anchor) for d in deps]):
            if err:
                skipped.append(err)
                continue
            if not rows:
                skipped.append(f"{gpu_key}: bench_dir has no in-scope cells")
                continue
            out.setdefault(gpu_key, []).extend(rows)
            print(f"  {gpu_key}: +{len(rows)} rows", flush=True)

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    total = sum(len(v) for v in out.values())
    print(f"wrote {total} rows across {len(out)} gpu_keys -> {OUT}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped[:15]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
