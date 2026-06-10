#!/usr/bin/env python3
"""Config loader — composes model + GPU + deployment JSONs from ``configs/`` into resolved deployments.

```
configs/
  models/<model>.json      model constants (n_params, kv_bytes_per_token, kv_heads, bytes_per_param, cache_block_size)
  gpus/<gpu>.json           hardware (peak_flops_per_s, peak_bw_bytes_per_s, util_flops, util_bw, scheduler_overhead_ms_per_step)
  deployments/<name>.json   gpu_key, gpu, model, tp, available_kv_blocks, bench_dir, ground_truth, + a per-input `data` manifest
```

A :class:`Deployment` composes its gpu + model into a :class:`RooflineParams` and resolves which measured
artifacts this config OWNS (manifest status ``measured``/``derived`` → use the path) vs INHERITS (→ ``None`` =
the in-code H100 module default). Consumed by ``build_simulator_rows`` / ``build_saturated_ceiling`` (replacing
their hardcoded CONFIGS) and ``configs/coverage_report.py``. Read-only; no fitted constants live here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
# Dashboard display order (mirrors ServingPredictionsPage SERVING_GPU_ORDER); gpu_keys not listed
# (e.g. the " (sglang)"-suffixed ones) sort after, in JSON insertion order.
_GPU_ORDER = ["H100", "H100x2", "H100x4", "A100", "A100x2", "A100x4", "A100x8",
              "RTX3090", "RTX3090x2", "RTX3090x4", "RTX3090x8",
              "RTX2080Ti", "RTX2080Tix2", "RTX2080Tix4"]
# Manifest statuses where THIS config provides the artifact (else it inherits the H100 module default).
_OWNED = {"measured", "derived"}


@dataclass(frozen=True)
class Deployment:
    gpu_key: str
    gpu: str
    model: str
    tp: int
    engine: str
    available_kv_blocks: int
    bench_dir: str
    backend: str
    calibration_status: str
    ground_truth: bool
    decode_grid: Path | None        # None -> in-code default (H100) decode grid
    saturated_ceiling: Path | None  # None -> in-code default (H100) saturated-ITL ceiling
    roofline: RooflineParams        # composed from gpu + model + (available_kv_blocks, tp)
    data: dict                      # per-input provenance / coverage manifest


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _owned_path(data: dict, key: str) -> Path | None:
    """The artifact path iff THIS config owns it (measured/derived); else None (inherit default)."""
    entry = data.get(key) or {}
    if entry.get("status") in _OWNED and entry.get("path"):
        return Path(entry["path"])
    return None


def load_deployment(dep_path: Path) -> Deployment:
    d = _read(dep_path)
    gpu = _read(CONFIGS_DIR / "gpus" / f"{d['gpu']}.json")
    model = _read(CONFIGS_DIR / "models" / f"{d['model']}.json")
    merged = {**gpu, **model,
              "available_kv_blocks": d["available_kv_blocks"], "tensor_parallel": d["tp"]}
    # Optional ENGINE-CONFIG key (not a fit knob): the vLLM per-step token budget the
    # deployment actually ran with. vLLM resolves the OPENAI_API_SERVER default by device
    # name (vllm/engine/arg_utils.py _set_default_args): 8192 for >=70GiB non-A100 GPUs
    # (= the RooflineParams default), 2048 for A100 — a100 deployments pin 2048 here.
    # Any explicitly-launched value belongs in the deployment JSON, same key.
    if "max_num_batched_tokens" in d:
        merged["max_num_batched_tokens"] = int(d["max_num_batched_tokens"])
    fields = RooflineParams.__dataclass_fields__
    roofline = RooflineParams(**{k: v for k, v in merged.items() if k in fields})
    data = d.get("data", {})
    return Deployment(
        gpu_key=d["gpu_key"], gpu=d["gpu"], model=d["model"], tp=int(d["tp"]),
        engine=d.get("engine", "vllm"), available_kv_blocks=int(d["available_kv_blocks"]),
        bench_dir=d["bench_dir"], backend=d["backend"], calibration_status=d["calibration_status"],
        ground_truth=bool(d.get("ground_truth", True)),
        decode_grid=_owned_path(data, "decode_grid"),
        saturated_ceiling=_owned_path(data, "saturated_ceiling"),
        roofline=roofline, data=data,
    )


def all_deployments() -> list[Deployment]:
    """All configs/deployments/*.json, sorted by dashboard GPU order (stable JSON key order)."""
    deps = [load_deployment(p) for p in (CONFIGS_DIR / "deployments").glob("*.json")]

    def order_key(dep: Deployment) -> tuple[int, str]:
        try:
            return (_GPU_ORDER.index(dep.gpu_key), dep.gpu_key)
        except ValueError:
            return (len(_GPU_ORDER), dep.gpu_key)

    return sorted(deps, key=order_key)
