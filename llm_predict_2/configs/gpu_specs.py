"""Calibrated GPU specifications for roofline modeling.

peak_flops_tflops: sustained GEMM throughput (not theoretical peak).
hbm_bw_gb_s: sustained HBM bandwidth from STREAM-like workloads.
kernel_floor_us: minimum kernel launch overhead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_flops_tflops: float  # bf16/fp16 sustained
    hbm_bw_gb_s: float        # sustained, not theoretical
    kernel_floor_us: float     # minimum launch overhead


GPU_SPECS: dict[str, GpuSpec] = {
    "H100": GpuSpec(
        name="H100-SXM5-80GB",
        peak_flops_tflops=989.0,   # bf16 tensor core sustained ~989 TFLOPS
        hbm_bw_gb_s=2600.0,        # HBM3 ~3.35 TB/s theoretical, ~2.6 sustained
        kernel_floor_us=2.0,
    ),
    "A100": GpuSpec(
        name="A100-SXM4-40GB",
        peak_flops_tflops=312.0,   # bf16 tensor core sustained
        hbm_bw_gb_s=1555.0,        # HBM2e ~2 TB/s theoretical, ~1.55 sustained
        kernel_floor_us=3.0,
    ),
    "RTX3090": GpuSpec(
        name="RTX3090-24GB",
        peak_flops_tflops=142.0,   # bf16 tensor core sustained
        hbm_bw_gb_s=760.0,         # GDDR6X ~936 GB/s theoretical, ~760 sustained
        kernel_floor_us=4.0,
    ),
    "RTX2080Ti": GpuSpec(
        name="RTX2080Ti-11GB",
        peak_flops_tflops=53.8,    # fp16 tensor core (no bf16 native on Turing)
        hbm_bw_gb_s=520.0,         # GDDR6 ~616 GB/s theoretical, ~520 sustained
        kernel_floor_us=5.0,
    ),
}


def get_gpu(name: str) -> GpuSpec:
    if name not in GPU_SPECS:
        raise ValueError(f"Unknown GPU: {name}. Available: {list(GPU_SPECS)}")
    return GPU_SPECS[name]
