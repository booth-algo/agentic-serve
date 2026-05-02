"""Calibrated GPU specifications for roofline and communication modeling.

peak_flops_tflops: sustained GEMM throughput (not theoretical peak).
hbm_bw_gb_s: sustained HBM bandwidth from STREAM-like workloads.
kernel_floor_us: minimum kernel launch overhead.
tp_comm_latency_us: all-reduce barrier latency per reduction for tensor parallelism.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_flops_tflops: float  # bf16/fp16 sustained
    hbm_bw_gb_s: float        # sustained, not theoretical
    kernel_floor_us: float     # minimum launch overhead
    tp_comm_latency_us: float  # all-reduce latency (NVSwitch ~5us, NVLink ~10us, PCIe ~40us)


GPU_SPECS: dict[str, GpuSpec] = {
    "H100": GpuSpec(
        name="H100-SXM5-80GB",
        peak_flops_tflops=989.0,
        hbm_bw_gb_s=2600.0,
        kernel_floor_us=2.0,
        tp_comm_latency_us=5.0,    # NVSwitch, ~2-5us per all-reduce
    ),
    "A100": GpuSpec(
        name="A100-SXM4-40GB",
        peak_flops_tflops=312.0,
        hbm_bw_gb_s=1555.0,
        kernel_floor_us=3.0,
        tp_comm_latency_us=10.0,   # NVLink, ~5-10us per all-reduce
    ),
    "RTX3090": GpuSpec(
        name="RTX3090-24GB",
        peak_flops_tflops=142.0,
        hbm_bw_gb_s=760.0,
        kernel_floor_us=4.0,
        tp_comm_latency_us=40.0,   # PCIe 3.0, no NVLink, ~20-50us per all-reduce
    ),
    "RTX2080Ti": GpuSpec(
        name="RTX2080Ti-11GB",
        peak_flops_tflops=53.8,
        hbm_bw_gb_s=520.0,
        kernel_floor_us=5.0,
        tp_comm_latency_us=50.0,   # PCIe 3.0, Turing, ~30-60us per all-reduce
    ),
}


def get_gpu(name: str) -> GpuSpec:
    if name not in GPU_SPECS:
        raise ValueError(f"Unknown GPU: {name}. Available: {list(GPU_SPECS)}")
    return GPU_SPECS[name]
