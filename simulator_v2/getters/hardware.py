# simulator_v2/getters/hardware.py

""" Getter for hardware configs """

from dataclasses import dataclass
from pathlib import Path

from configs.kv_pool import available_kv_blocks
from simulator_v2.configs.config_loader import load_gpu_config, load_model_config
from simulator_v2.core.mode import Mode, mode
from simulator_v2.core.types import GpuConfig, ModelConfig, SchedulerSettings
from simulator_v2.kernel_floor.sum_kernels import KernelFloor, load_kernel_floor
from simulator_v2.kv_wall.saturated_ceiling import (
    SaturatedCeiling,
    load_saturated_ceiling,
)

@mode(Mode.SHARED)
def _analytic_kv_pool(gpu: GpuConfig, model: ModelConfig, tp: int) -> int:
    """Analytic vLLM KV-pool size (blocks) -- the fallback when no measured pool is
    pinned in the GPU config (`gpu.kv_pool_blocks`)."""
    return available_kv_blocks(
        total_memory_bytes=gpu.total_bytes,
        gpu_mem_util=gpu.gpu_mem_util,
        weight_bytes=model.n_params * model.bytes_per_param,
        tp=tp,
        kv_bytes_per_token=model.kv_bytes_per_token,
        kv_heads=model.kv_heads,
        block_size=model.cache_block_size,
    )


# I don't really get this class right now
@mode(Mode.FORWARD)
@dataclass
class Roofline:
    gpu: GpuConfig
    model: ModelConfig
    tp: int = 1
    kv_pool_blocks: int = 0
    saturated_step_ms: float = 200.0  #! placeholder — saturated ITL ceiling, not yet measured/config-derived
    sched: SchedulerSettings | None = None

    def __post_init__(self) -> None:
        if self.kv_pool_blocks <= 0:
            self.kv_pool_blocks = self.gpu.kv_pool_blocks or _analytic_kv_pool(self.gpu, self.model, self.tp)
        if self.sched is None:
            self.sched = SchedulerSettings(
                max_num_batched_tokens=8192,  #! vLLM default — should come from deployment config
                long_prefill_token_threshold=int(self.gpu.max_model_len * 0.04),  # vLLM chunked-prefill per-request cap
            )

    @property
    def cache_block_size(self) -> int:
        return self.model.cache_block_size

    @property
    def request_overhead_ms(self) -> float:
        return self.gpu.request_overhead_ms

    @property
    def prefill_host_rates(self) -> tuple[float, float, float]:
        return (
            self.gpu.prefill_host_shared_ms_per_token,
            self.gpu.prefill_host_perreq_ms_per_token,
            self.gpu.prefill_host_new_ms_per_token,
        )

    @property
    def cross_attn_ms_per_token_pair(self) -> float:
        return self.gpu.cross_attn_ms_per_token_pair

    @classmethod
    def from_config(
        cls,
        gpu: GpuConfig,
        model: ModelConfig,
        *,
        tp: int = 1,
        saturated_step_ms: float = 200.0,
    ) -> "Roofline":
        return cls(
            gpu=gpu,
            model=model,
            tp=tp,
            saturated_step_ms=saturated_step_ms,
        )

    def decode_step_ms(self, batch: int, ctx_tokens: float) -> float:
        batch = max(0, int(batch))
        if batch == 0:
            return 0.0

        compute_ms = (
            2.0 * self.model.n_params * batch  #! 2.0 = FLOPs per param (multiply-add)
            / (self.gpu.peak_flops_per_s * self.gpu.util_flops)
            * 1e3  #! unit: seconds -> milliseconds
        )
        weight_bytes = self.model.n_params * self.model.bytes_per_param
        kv_bpt = self.model.kv_bytes_per_token
        ctx = max(0.0, float(ctx_tokens))
        bytes_total = (
            weight_bytes
            + batch * ctx * kv_bpt
            + batch * kv_bpt
        )
        bandwidth_ms = bytes_total / (self.gpu.peak_bw_bytes_per_s * self.gpu.util_bw) * 1e3  #! unit: seconds -> milliseconds
        return max(compute_ms, bandwidth_ms) + self.gpu.scheduler_overhead_ms_per_step

    def prefill_ms(self, new: float, cached: float, batch: int) -> float:
        del cached  #! c=1 simplification — ignores cached prefix (queue-free path)
        tokens = max(0.0, float(new)) * max(1, int(batch))
        if tokens == 0.0:
            return 0.0

        compute_ms = (
            2.0 * self.model.n_params * tokens  #! 2.0 = FLOPs per param (multiply-add)
            / (self.gpu.peak_flops_per_s * self.gpu.util_flops)
            * 1e3  #! unit: seconds -> milliseconds
        )
        weight_bytes = self.model.n_params * self.model.bytes_per_param
        kv_bpt = self.model.kv_bytes_per_token
        bytes_total = weight_bytes + tokens * kv_bpt + tokens * kv_bpt
        bandwidth_ms = bytes_total / (self.gpu.peak_bw_bytes_per_s * self.gpu.util_bw) * 1e3  #! unit: seconds -> milliseconds
        return max(compute_ms, bandwidth_ms)

    def fused_step_ms(self, prefill_tokens: int, decode_batch: int, decode_ctx: float) -> float:
        #! Analytic co-scheduled step: the decode batch and the prefill chunk share the
        #! forward pass, so the step costs the larger of the two (the piggyback).
        decode = self.decode_step_ms(decode_batch, decode_ctx) if decode_batch > 0 else 0.0
        prefill = self.prefill_ms(prefill_tokens, 0.0, 1) if prefill_tokens > 0 else 0.0
        return max(decode, prefill)

    def saturated_ceiling_ms(self, output: float) -> float:
        #! Forward-mode ceiling is unsolved: there's no measured plateau for an
        #! uncalibrated config, and the analytic bandwidth bound was ~10x low (dropped).
        #! The right fallback is to inherit measured anchors from the nearest config.
        raise NotImplementedError(
            "forward-mode saturated ceiling is not implemented; supply measured "
            "anchors (kv_wall.load_saturated_ceiling) or use KernelComposition"
        )

# ===== Kernel composition (backtest mode) =====
#
# Backtest hardware is the measured leaf-kernel composition (kernel_floor.KernelFloor:
# decode floor validated 9.7%, prefill 3.4%) bound to a model/GPU. KernelComposition is
# a thin Hardware-protocol adapter over it. The saturation ceiling (kv_wall/) and the
# eviction amplifier (engine/predict.py) consume these floors -- they don't live here.


@mode(Mode.BACKTEST)
@dataclass
class KernelComposition:
    """Backtest hardware: per-step floors from the measured leaf-kernel composition,
    plus the measured saturation ceiling the TPOT amplifier ramps toward."""
    gpu: GpuConfig
    model: ModelConfig
    floor: KernelFloor
    ceiling: SaturatedCeiling
    tp: int = 1
    kv_pool_blocks: int = 0
    saturated_step_ms: float = 0.0  # protocol scalar; set from the ceiling in __post_init__
    sched: SchedulerSettings | None = None

    def __post_init__(self) -> None:
        if self.kv_pool_blocks <= 0:
            # Measured pool (gpu.kv_pool_blocks) preferred; analytic estimate otherwise.
            self.kv_pool_blocks = self.gpu.kv_pool_blocks or _analytic_kv_pool(self.gpu, self.model, self.tp)
        if self.saturated_step_ms <= 0.0:
            self.saturated_step_ms = self.ceiling.ceiling_ms(1.0)  # highest (short-output) plateau
        if self.sched is None:
            self.sched = SchedulerSettings(
                max_num_batched_tokens=8192,  #! should come from deployment config
                long_prefill_token_threshold=int(self.gpu.max_model_len * 0.04),  # vLLM chunked-prefill per-request cap
            )

    @property
    def cache_block_size(self) -> int:
        return self.model.cache_block_size

    @property
    def request_overhead_ms(self) -> float:
        return self.gpu.request_overhead_ms

    @property
    def prefill_host_rates(self) -> tuple[float, float, float]:
        return (
            self.gpu.prefill_host_shared_ms_per_token,
            self.gpu.prefill_host_perreq_ms_per_token,
            self.gpu.prefill_host_new_ms_per_token,
        )

    @property
    def cross_attn_ms_per_token_pair(self) -> float:
        return self.gpu.cross_attn_ms_per_token_pair

    def decode_step_ms(self, batch: int, ctx_tokens: float) -> float:
        """Per-step decode floor -> KernelFloor.decode_step_ms."""
        return self.floor.decode_step_ms(batch, ctx_tokens)

    def prefill_ms(self, new: float, cached: float, batch: int) -> float:
        """Queue-free prefill-chunk floor -> KernelFloor.prefill_step_ms."""
        del cached, batch  #! c=1 simplification — one sequence's chunk, no cached prefix
        return self.floor.prefill_step_ms(int(new))

    def fused_step_ms(self, prefill_tokens: int, decode_batch: int, decode_ctx: float) -> float:
        """Co-scheduled prefill+decode step cost -> KernelFloor.fused_step_ms."""
        return self.floor.fused_step_ms(prefill_tokens, decode_batch, decode_ctx)

    def saturated_ceiling_ms(self, output: float) -> float:
        """Measured saturated-ITL plateau at `output` tokens -> SaturatedCeiling."""
        return self.ceiling.ceiling_ms(output)


# ----- composite getters (paths -> hardware object) -----
#
# Forward mode uses the analytic roofline; backtest mode uses the measured kernels.

_KERNEL_DIR = Path("profile_data/kernels")


@mode(Mode.BACKTEST)
def _kernel_artifact_paths(gpu_name: str) -> dict[str, Path]:
    """Measured leaf-kernel artifact paths for a GPU, keyed by GpuConfig.name
    (e.g. 'H100' -> forward_pass/gemm/H100.csv, flash_attn/H100.csv, ...)."""
    return {
        "gemm_path": _KERNEL_DIR / "forward_pass" / "gemm" / f"{gpu_name}.csv",
        "flash_attn_path": _KERNEL_DIR / "flash_attn" / f"{gpu_name}.csv",
        "prefill_attn_path": _KERNEL_DIR / f"fa3_prefill_{gpu_name}.csv",
        "elementwise_path": _KERNEL_DIR / "elementwise" / f"{gpu_name}.json",
        "ceiling_path": _KERNEL_DIR / "saturated_ceiling" / f"{gpu_name}.json",
    }


@mode(Mode.FORWARD)
def load_roofline_hardware(
    gpu_config: str | Path, model_config: str | Path, *, tp: int = 1
) -> Roofline:
    gpu = load_gpu_config(Path(gpu_config))
    model = load_model_config(Path(model_config))
    return Roofline.from_config(gpu, model, tp=tp)


@mode(Mode.BACKTEST)
def load_kernel_composition_hardware(
    gpu_config: str | Path, model_config: str | Path, *, tp: int = 1
) -> KernelComposition:
    """Load this GPU's measured leaf tables (resolved from GpuConfig.name), compose
    them into a KernelFloor, and bind it as backtest hardware."""
    gpu = load_gpu_config(Path(gpu_config))
    model = load_model_config(Path(model_config))
    paths = _kernel_artifact_paths(gpu.name)
    floor = load_kernel_floor(
        model, gpu,
        paths["gemm_path"], paths["flash_attn_path"],
        paths["prefill_attn_path"], paths["elementwise_path"],
    )
    ceiling = load_saturated_ceiling(paths["ceiling_path"])
    return KernelComposition(gpu=gpu, model=model, floor=floor, ceiling=ceiling, tp=tp)
