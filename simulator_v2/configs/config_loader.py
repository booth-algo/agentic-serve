# simulator_v2/configs/config_loader.py

import yaml
from pathlib import Path

from simulator_v2.core.types import FrontendParams, GpuConfig, ModelConfig
from simulator_v2.core.mode import Mode, mode


@mode(Mode.SHARED)
def load_gpu_config(path: Path) -> GpuConfig:
    data = yaml.safe_load(path.read_text())
    compute = data["compute"]
    memory = data["memory"]
    scheduler = data["scheduler"]
    host = data.get("prefill_host") or {}
    fe = data.get("frontend") or {}
    frontend = FrontendParams(
        floor_ms=float(fe.get("floor_ms", 0.0)),
        new_ms_per_token=float(fe.get("new_ms_per_token", 0.0)),
        cached_ms_per_token=float(fe.get("cached_ms_per_token", 0.0)),
        load_mult=float(fe.get("load_mult", 1.0)),
        mult_curve=tuple((int(c), float(m)) for c, m in (fe.get("mult_curve") or [])),
        lanes_curve=tuple((int(c), float(l)) for c, l in (fe.get("lanes_curve") or [])),
    )
    return GpuConfig(
        name=data["name"],
        peak_flops_per_s=float(compute["peak_flops_per_s"]),
        util_flops=float(compute["util_flops"]),
        peak_bw_bytes_per_s=float(memory["peak_bw_bytes_per_s"]),
        util_bw=float(memory["util_bw"]),
        total_bytes=int(memory["total_bytes"]),
        gpu_mem_util=float(memory["gpu_mem_util"]),
        scheduler_overhead_ms_per_step=float(scheduler["overhead_ms_per_step"]),
        max_model_len=int(scheduler.get("max_model_len", 32768)),
        request_overhead_ms=float(scheduler.get("request_overhead_ms", 25.0)),
        kv_pool_blocks=int(memory.get("kv_pool_blocks", 0)),
        prefill_host_shared_ms_per_token=float(host.get("shared_ms_per_token", 0.0)),
        prefill_host_perreq_ms_per_token=float(host.get("perreq_ms_per_token", 0.0)),
        prefill_host_new_ms_per_token=float(host.get("new_ms_per_token", 0.0)),
        cross_attn_ms_per_token_pair=float(compute.get("cross_attn_ms_per_token_pair", 0.0)),
        frontend=frontend,
    )


@mode(Mode.SHARED)
def load_model_config(path: Path) -> ModelConfig:
    data = yaml.safe_load(path.read_text())

    def opt_int(key: str) -> int | None:
        val = data.get(key)
        return int(val) if val is not None else None

    return ModelConfig(
        name=data["name"],
        n_params=int(data["n_params"]),
        kv_bytes_per_token=float(data["kv_bytes_per_token"]),
        kv_heads=int(data["kv_heads"]),
        bytes_per_param=float(data["bytes_per_param"]),
        cache_block_size=int(data["cache_block_size"]),
        n_layers=opt_int("n_layers"),
        hidden_dim=opt_int("hidden_dim"),
        intermediate_size=opt_int("intermediate_size"),
        n_heads=opt_int("n_heads"),
        head_dim=opt_int("head_dim"),
        vocab_size=opt_int("vocab_size"),
    )
