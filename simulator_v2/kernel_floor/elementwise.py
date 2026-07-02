# simulator_v2/kernel_floor/elementwise.py

"""Elementwise kernel cost (us): a memory-bound affine model per kernel,
`latency = floor_us + bytes_moved / eff_bw`, from a calibrated (floor_us, eff_bw)
pair (roofline fallback when uncalibrated). Native unit us."""

import json
from dataclasses import dataclass
from pathlib import Path

from simulator_v2.core.types import GpuConfig
from simulator_v2.core.mode import Mode, mode

# (reads, writes) per kernel -> bytes_moved = elements * dtype_bytes * (reads + writes).
# Intrinsic to each op's dataflow (e.g. silu_mul reads gate+up, writes one tensor).
_IO: dict[str, tuple[int, int]] = {
    "rmsnorm": (1, 1),
    "silu_mul": (2, 1),
    "rotary_emb": (1, 1),
    "residual_add": (2, 1),
}


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class ElementwiseTable:
    """Calibrated (floor_us, eff_bw_gb_s) per elementwise kernel.

    eff_bw_gb_s is an *effective* bandwidth (GB/s) that may exceed peak HBM —
    it's fitted to NCU latencies, so it soaks up whatever our (reads + writes)
    byte estimate over- or under-counts.
    """
    params: dict[str, tuple[float, float]]  # kernel -> (floor_us, eff_bw_gb_s)

    def elementwise_us(
        self, kernel: str, elements: int, gpu: GpuConfig, *, dtype_bytes: int = 2
    ) -> float:
        """Cost (us) for one elementwise call, in order of preference:

          1. kernel is calibrated -> floor_us + bytes / eff_bw  (affine model)
          2. otherwise            -> analytic roofline fallback

        We default back to the roofline (case 2) only when the kernel has no
        calibrated pair in the loaded file; bytes_moved uses the kernel's
        (reads, writes), or (1, 1) if the kernel is unknown to _IO.
        """
        reads, writes = _IO.get(kernel, (1, 1))
        bytes_moved = elements * dtype_bytes * (reads + writes)
        cal = self.params.get(kernel)
        if cal is None:
            eff_bw_bytes_s = gpu.peak_bw_bytes_per_s * gpu.util_bw
            return bytes_moved / eff_bw_bytes_s * 1e6
        floor_us, eff_bw_gb_s = cal
        return floor_us + bytes_moved / (eff_bw_gb_s * 1e3)


@mode(Mode.BACKTEST)
def load_elementwise_table(path: Path) -> ElementwiseTable:
    """Elementwise calibration JSON ({kernel: {floor_us, eff_bw_gb_s}}) ->
    ElementwiseTable. Kernels absent from the file price via roofline at lookup.
    """
    raw = json.loads(path.read_text())
    params = {
        kernel: (float(entry["floor_us"]), float(entry["eff_bw_gb_s"]))
        for kernel, entry in raw.items()
    }
    if not params:
        raise RuntimeError(f"no elementwise calibration entries in {path}")
    return ElementwiseTable(params=params)
