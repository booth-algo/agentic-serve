"""
Compute Graph for Pipeline Parallelism with Device Clusters.

This module provides a compute graph representation for PP configurations where:
- Each PP stage has a device cluster (set of physical devices)
- Each PP stage can have different TP + DP arrangements
- Bandwidth between devices can vary (per-device-pair bandwidth matrix)

The graph can be used to configure OpenVLA models with heterogeneous parallelism.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from llm_predict.models.hardware.interconnect import InterConnectModule, LinkModule, TopologyType
from search.search_tools import find_best_configuration
from llm_predict.models.software.utils import DataType, data_type_dict


@dataclass
class PPStageConfig:
    """
    Configuration for a single Pipeline Parallelism (PP) stage.
    
    Attributes:
        stage_id: Unique identifier for this PP stage (0-indexed)
        device_cluster: List of device IDs (physical device IDs) in this stage's cluster
        tp_size: Tensor Parallelism size within this stage
        dp_size: Data Parallelism size within this stage
        layer_range: Tuple of (start_layer, end_layer) that this stage handles
        bandwidth_matrix: Optional per-device-pair bandwidth matrix for devices in this cluster.
                         Format: {device_i: {device_j: bandwidth_ratio}, ...}
                         bandwidth_ratio is a multiplier (0.0-1.0) applied to base bandwidth.
                         If None, assumes uniform bandwidth (1.0) for all pairs.
    """
    stage_id: int
    device_cluster: List[int]  # Physical device IDs in this cluster
    tp_size: int
    dp_size: int
    layer_range: tuple  # (start_layer, end_layer)
    bandwidth_matrix: Optional[Dict[int, Dict[int, float]]] = None
    
    def __post_init__(self):
        """Validate the configuration"""
        if self.tp_size < 1:
            raise ValueError(f"PP stage {self.stage_id}: tp_size must be >= 1")
        if self.dp_size < 1:
            raise ValueError(f"PP stage {self.stage_id}: dp_size must be >= 1")
        if len(self.device_cluster) < self.tp_size * self.dp_size:
            raise ValueError(
                f"PP stage {self.stage_id}: device_cluster size ({len(self.device_cluster)}) "
                f"must be >= tp_size * dp_size ({self.tp_size * self.dp_size})"
            )
        if len(self.layer_range) != 2:
            raise ValueError(f"PP stage {self.stage_id}: layer_range must be a tuple of (start, end)")
        start_layer, end_layer = self.layer_range
        if start_layer < 0 or start_layer >= end_layer:
            raise ValueError(
                f"PP stage {self.stage_id}: layer_range must have start < end, "
                f"got [{start_layer}, {end_layer})"
            )
        
        # Normalize bandwidth matrix: ensure all device pairs have entries
        if self.bandwidth_matrix is None:
            self.bandwidth_matrix = {}
        
        # Fill in missing entries with 1.0 (full bandwidth)
        for device_i in self.device_cluster:
            if device_i not in self.bandwidth_matrix:
                self.bandwidth_matrix[device_i] = {}
            for device_j in self.device_cluster:
                if device_i != device_j:
                    if device_j not in self.bandwidth_matrix[device_i]:
                        self.bandwidth_matrix[device_i][device_j] = 1.0
    
    def get_tp_device_groups(self) -> List[List[int]]:
        """
        Get TP device groups for this stage.
        
        Returns:
            List of device groups, where each group is a list of device IDs for one TP group.
            For DP > 1, there will be multiple TP groups (one per DP replica).
        """
        tp_groups = []
        
        # Organize devices into TP groups
        # Each TP group has tp_size devices
        # There are dp_size such groups (one per DP replica)
        devices_per_replica = self.tp_size
        
        for dp_idx in range(self.dp_size):
            start_idx = dp_idx * devices_per_replica
            end_idx = start_idx + devices_per_replica
            tp_group = self.device_cluster[start_idx:end_idx]
            if len(tp_group) != self.tp_size:
                raise ValueError(
                    f"PP stage {self.stage_id}: Cannot form TP group for DP replica {dp_idx}. "
                    f"Need {self.tp_size} devices, got {len(tp_group)}"
                )
            tp_groups.append(tp_group)
        
        return tp_groups
    
    def get_device_count(self) -> int:
        """Get total number of devices in this stage"""
        return len(self.device_cluster)


@dataclass
class ComputeGraph:
    """
    Compute graph representing a Pipeline Parallelism configuration.
    
    This graph contains:
    - Multiple PP stages, each with its own device cluster
    - TP + DP arrangements per stage
    - Per-device-pair bandwidth information
    
    Attributes:
        pp_stages: List of PP stage configurations
        base_link_bandwidth: Base bandwidth per link (bytes/second) for bandwidth matrix calculations
        link_latency: Base link latency (seconds)
        link_count_per_device: Number of links per device
    """
    pp_stages: List[PPStageConfig]
    base_link_bandwidth: float  # bytes/second
    link_latency: float = 8.92e-6  # seconds (default NVLink3 latency)
    link_count_per_device: int = 12
    topology: str = "FC"  # "FC" or "RING"
    
    def __post_init__(self):
        """Validate the compute graph"""
        if not self.pp_stages:
            raise ValueError("Compute graph must have at least one PP stage")
        
        # Check that layer ranges don't overlap and cover all layers
        all_layer_ranges = [stage.layer_range for stage in self.pp_stages]
        all_layer_ranges.sort(key=lambda x: x[0])
        
        # Check for gaps and overlaps
        covered_layers = set()
        for stage in self.pp_stages:
            start, end = stage.layer_range
            stage_layers = set(range(start, end))
            overlap = covered_layers & stage_layers
            if overlap:
                raise ValueError(
                    f"Layer overlap detected: stage {stage.stage_id} overlaps with previous stages "
                    f"at layers {overlap}"
                )
            covered_layers |= stage_layers
        
        # Check that stages are in order
        stage_ids = [stage.stage_id for stage in self.pp_stages]
        if sorted(stage_ids) != list(range(len(self.pp_stages))):
            raise ValueError(
                f"PP stage IDs must be consecutive starting from 0. Got: {stage_ids}"
            )
    
    def get_total_devices(self) -> int:
        """Get total number of unique devices across all PP stages"""
        all_devices = set()
        for stage in self.pp_stages:
            all_devices.update(stage.device_cluster)
        return len(all_devices)
    
