"""Analytical roofline baseline for any kernel.

Provides the physics-grounded prediction that XGBoost learns residuals on top of.
"""

from ..configs.gpu_specs import GpuSpec


def compute_bound_us(flops: float, gpu: GpuSpec) -> float:
    return flops / (gpu.peak_flops_tflops * 1e6)


def memory_bound_us(bytes_moved: float, gpu: GpuSpec) -> float:
    return bytes_moved / (gpu.hbm_bw_gb_s * 1e3)


def roofline_us(flops: float, bytes_moved: float, gpu: GpuSpec) -> float:
    return max(
        compute_bound_us(flops, gpu),
        memory_bound_us(bytes_moved, gpu),
        gpu.kernel_floor_us,
    )


def gemm_roofline_us(M: int, N: int, K: int, gpu: GpuSpec,
                     dtype_bytes: int = 2) -> float:
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * dtype_bytes
    return roofline_us(flops, bytes_moved, gpu)


def elementwise_roofline_us(bytes_moved: float, gpu: GpuSpec) -> float:
    return max(memory_bound_us(bytes_moved, gpu), gpu.kernel_floor_us)
