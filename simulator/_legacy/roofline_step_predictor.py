"""3D analytical roofline pricer for vLLM scheduler steps.

Replaces the profile-based ``CleanStepPredictor`` with a pure first-principles
3D roofline computation per step.  No ML, no per-workload fitting — just the
physical floor of GPU compute + HBM bandwidth for the work the step actually
schedules.

For a step that schedules ``N`` tokens of total work with decoded-request
context lengths ``{T_i}`` and prefill chunks ``{(U_j, P_j)}``::

    compute_ms   = 2 · n_params · N / (peak_flops · util_flops)
    bandwidth_ms = bytes_moved / (peak_bw · util_bw)
    step_ms      = max(compute_ms, bandwidth_ms)

Bytes moved per step::

    model_weights:    2 · n_params                       (loaded once per step)
    kv_read_decode:   sum(T_i) · kv_bytes_per_token      (each decode reads its
                                                          own full context KV)
    kv_read_prefill:  sum(P_j) · kv_bytes_per_token      (prefill chunk reads
                                                          its prefix KV once)
    kv_write_prefill: sum(U_j) · kv_bytes_per_token      (new prefill KV)
    kv_write_decode:  B · kv_bytes_per_token             (1 new KV per decode)

The ``util_flops`` / ``util_bw`` factors are calibrated from TWO independent
anchor measurements (one compute-bound, one bandwidth-bound) — NOT fit across
trace data.  They model effective GPU throughput after launch overhead,
splitK reductions, occupancy losses, and HBM channel inefficiency.  See
``profiling/data/roofline_params_H100_llama31_8b.json`` for the derivations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from simulator._legacy.vllm_scheduler_shape import VllmStepShape


@dataclass(frozen=True)
class RooflineParams:
    """Hardware + model constants + two utilization scalars.

    All fields are GPU/model specs except ``util_flops`` and ``util_bw``,
    which are derived from independent anchor measurements (one per GPU).
    """

    n_params: int = 8_030_000_000               # Llama-3.1-8B (Instruct)
    peak_flops_per_s: float = 989e12            # H100 BF16 dense
    peak_bw_bytes_per_s: float = 3.35e12        # H100 HBM3
    kv_bytes_per_token: float = 131072.0        # 2·L·n_kv·d·sizeof(bf16)
    util_flops: float = 0.65                    # anchor: c=40 t=12 step 1
    util_bw: float = 0.93                       # anchor: c=40 t=12 step 17-22
                                                # (includes weight + KV scan)
    bytes_per_param: float = 2.0                # bf16 weight read per step

    @classmethod
    def from_json(cls, path: Path) -> "RooflineParams":
        data = json.loads(path.read_text())
        # Only consume known fields so the JSON can include extra derivation
        # notes without breaking the loader.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class RooflineStepCost:
    """Breakdown for one scheduler step."""

    compute_ms: float
    bandwidth_ms: float
    total_ms: float
    classification: str  # "compute_bound", "bandwidth_bound", "empty"


class RooflineStepPredictor:
    """Pure analytical 3D roofline pricer.

    Construct with :class:`RooflineParams` or :meth:`from_json`.  Use
    :meth:`step_ms` to price one :class:`VllmStepShape`.
    """

    def __init__(self, params: RooflineParams) -> None:
        if params.peak_flops_per_s <= 0:
            raise ValueError("peak_flops_per_s must be > 0")
        if params.peak_bw_bytes_per_s <= 0:
            raise ValueError("peak_bw_bytes_per_s must be > 0")
        if not (0.0 < params.util_flops <= 2.0):
            raise ValueError(f"util_flops out of range: {params.util_flops}")
        if not (0.0 < params.util_bw <= 2.0):
            raise ValueError(f"util_bw out of range: {params.util_bw}")
        self._params = params

    @classmethod
    def from_json(cls, path: Path) -> "RooflineStepPredictor":
        return cls(RooflineParams.from_json(path))

    @property
    def params(self) -> RooflineParams:
        return self._params

    # ------------------------------------------------------------------ math

    def compute_ms(self, total_tokens: int) -> float:
        """2·n_params·N / (peak_flops·util_flops) in ms."""
        n = max(0, int(total_tokens))
        if n == 0:
            return 0.0
        p = self._params
        flops = 2.0 * float(p.n_params) * n
        return flops / (p.peak_flops_per_s * p.util_flops) * 1e3

    def bandwidth_ms(
        self,
        step: VllmStepShape,
        context_lens: Mapping[int, int],
    ) -> float:
        """Bytes moved through HBM this step, divided by effective BW.

        Includes:
          - model weights loaded once per step
          - KV reads: decode (per-request full ctx) + prefill chunk prefix
          - KV writes: prefill new tokens + 1 token per decoded request
        """
        p = self._params
        kv_bpt = float(p.kv_bytes_per_token)

        bytes_total = float(p.bytes_per_param) * float(p.n_params)  # weights

        # Decode KV reads (each request reads its own full context).
        for rid in step.decoded_request_ids:
            T = int(context_lens.get(rid, 0))
            if T > 0:
                bytes_total += T * kv_bpt
        # Decode KV writes (1 new KV per decoded request).
        bytes_total += step.decode_batch * kv_bpt

        # Prefill chunks: read prefix KV, write new KV.
        for chunk in step.prefill_chunks:
            U = max(0, int(chunk.scheduled_tokens))
            P = max(0, int(chunk.prefix_tokens))
            bytes_total += P * kv_bpt  # read prefix
            bytes_total += U * kv_bpt  # write new

        return bytes_total / (p.peak_bw_bytes_per_s * p.util_bw) * 1e3

    def step_ms(
        self,
        step: VllmStepShape,
        context_lens: Mapping[int, int],
    ) -> RooflineStepCost:
        """max(compute, bandwidth) per step.

        An empty step (no decode, no prefill) returns total_ms=0.
        """
        total_tokens = int(step.decode_batch) + int(step.prefill_tokens)
        if total_tokens == 0:
            return RooflineStepCost(
                compute_ms=0.0,
                bandwidth_ms=0.0,
                total_ms=0.0,
                classification="empty",
            )
        c_ms = self.compute_ms(total_tokens)
        bw_ms = self.bandwidth_ms(step, context_lens)
        if bw_ms >= c_ms:
            return RooflineStepCost(
                compute_ms=c_ms,
                bandwidth_ms=bw_ms,
                total_ms=bw_ms,
                classification="bandwidth_bound",
            )
        return RooflineStepCost(
            compute_ms=c_ms,
            bandwidth_ms=bw_ms,
            total_ms=c_ms,
            classification="compute_bound",
        )
