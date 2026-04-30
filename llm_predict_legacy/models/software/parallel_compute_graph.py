"""
Lightweight stand-ins for parallel compute graph utilities used by search tools.

The original implementation is unavailable in this repository snapshot.
These minimal classes provide just enough structure for the DP/TP search
utilities to run and return sane results for testing the PP search flow.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ParallelismType(str, Enum):
    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


@dataclass
class DeviceGroup:
    device_ids: List[int]
    parallelism_type: ParallelismType
    group_id: int


@dataclass
class ParallelismConfig:
    dp_size: int
    tp_size: int
    pp_size: int = 1

    @property
    def total_devices(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size

    def get_device_groups(self) -> Dict[str, List[List[int]]]:
        """
        Produce device groupings for DP and TP.
        Devices are assigned contiguously by DP then TP.
        """
        dp_groups: List[List[int]] = []
        current = 0
        for dp_idx in range(self.dp_size):
            group = list(range(current, current + self.tp_size))
            dp_groups.append(group)
            current += self.tp_size
        return {"dp_groups": dp_groups}


class ParallelComputeGraph:
    """
    Minimal compute graph to satisfy search tools.
    """

    def __init__(self, config: ParallelismConfig):
        self.config = config
        self.nodes = {}

    def add_compute_node(self, node_id: str, operator, device_group: DeviceGroup, input_tensors):
        # Store minimal metadata; no-op for simulation
        self.nodes[node_id] = {
            "operator": operator,
            "device_group": device_group,
            "inputs": input_tensors,
        }

    def simulate(self, system, compile_mode: str = "heuristic-GPU") -> Dict[str, float]:
        """
        Return a fixed latency per node to allow throughput estimation.
        The absolute value is less important than consistency across nodes.
        """
        return {node_id: 1.0 for node_id in self.nodes} or {"noop": 1.0}

    def get_total_latency(self, node_latencies: Dict[str, float]) -> float:
        """
        In this simplified model, total latency is the max node latency.
        """
        return max(node_latencies.values()) if node_latencies else 0.0

