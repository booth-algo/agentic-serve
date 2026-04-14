"""
Compute Graph for Pipeline Parallelism with Device Clusters.

This module provides a compute graph representation for PP configurations where:
- Each PP stage has a device cluster (set of physical devices)
- Each PP stage can have different TP + DP arrangements
- Bandwidth between devices can vary (per-device-pair bandwidth matrix)

The graph can be used to configure OpenVLA models with heterogeneous parallelism.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from llmcompass.hardware_model.interconnect import InterConnectModule, LinkModule, TopologyType
from llmcompass.hardware_model.system import System
from search.search_tools import find_best_configuration
from llmcompass.software_model.utils import DataType, data_type_dict


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


def find_optimal_dp_tp(
        snapshot,
        snapshot_to_system_func,
        d_model: int,
        n_heads: int,
        batch_size: int = 1,
        seq_len: int = 512,
        data_type: DataType = data_type_dict["fp16"],
        compile_mode: str = "heuristic-GPU",
        metric: str = "throughput",
        verbose: bool = True,
    ) -> Tuple[int, int, Dict[str, float]]:
    """
    Find optimal DP+TP configuration for the current snapshot.
    
    Args:
        snapshot: Current system snapshot (SystemSnapshot object)
        snapshot_to_system_func: Function to convert snapshot to System object
        d_model: Hidden dimension
        n_heads: Number of attention heads
        batch_size: Batch size per DP group
        seq_len: Sequence length
        data_type: Data type
        compile_mode: Compilation mode
        metric: Metric to optimize ("throughput", "latency", "latency_per_sample")
        verbose: Whether to print progress
    
    Returns:
        Tuple of (best_dp_size, best_tp_size, best_result_dict)
    """
    # Convert snapshot to System object
    system = snapshot_to_system_func(snapshot)
    
    num_active_nodes = len(snapshot.active_node_ids)
    
    if verbose:
        # Calculate min bandwidth for display
        min_bw_ratio = 1.0
        for node_i in snapshot.active_node_ids:
            if node_i in snapshot.bandwidth_matrix:
                for node_j in snapshot.active_node_ids:
                    if node_i != node_j and node_j in snapshot.bandwidth_matrix[node_i]:
                        min_bw_ratio = min(min_bw_ratio, snapshot.bandwidth_matrix[node_i][node_j])
        
        print(f"\n{'=' * 80}")
        print(f"Finding optimal DP+TP configuration")
        print(f"{'=' * 80}")
        print(f"Active nodes: {snapshot.active_node_ids}")
        print(f"Total active devices: {num_active_nodes}")
        print(f"Bandwidth matrix (node_i -> node_j: ratio):")
        # Print bandwidth matrix in a readable format
        sorted_nodes = sorted(snapshot.active_node_ids)
        for node_i in sorted_nodes:
            if node_i in snapshot.bandwidth_matrix:
                bw_str = ", ".join([
                    f"{node_j}:{snapshot.bandwidth_matrix[node_i][node_j]:.2f}"
                    for node_j in sorted_nodes
                    if node_i != node_j and node_j in snapshot.bandwidth_matrix[node_i]
                ])
                print(f"  Node {node_i} -> {{{bw_str}}}")
        print(f"Minimum bandwidth ratio: {min_bw_ratio:.2f}")
        print(f"Effective bandwidth: {snapshot.base_bandwidth * min_bw_ratio:.2e} bytes/s")
        print(f"{'=' * 80}\n")
    
    # Use brute force search to find best configuration
    best_dp, best_tp, best_result = find_best_configuration(
        system=system,
        total_devices=num_active_nodes,
        d_model=d_model,
        n_heads=n_heads,
        batch_size=batch_size,
        seq_len=seq_len,
        data_type=data_type,
        compile_mode=compile_mode,
        metric=metric,
    )
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Optimal Configuration Found:")
        print(f"  DP size: {best_dp}")
        print(f"  TP size: {best_tp}")
        print(f"  Latency: {best_result['latency']:.6f} seconds")
        print(f"  Throughput: {best_result['throughput']:.6f} samples/second")
        print(f"  Latency per sample: {best_result['latency_per_sample']:.9f} seconds")
        print(f"{'=' * 80}\n")
    
    return (best_dp, best_tp, best_result)


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
    
    def to_vla_pp_config(self) -> List[Dict]:
        """
        Convert this compute graph to VLA PP configuration format.
        
        Returns:
            List of stage configs in format expected by OpenVLAForActionPredictionPP
        """
        vla_config = []
        for stage in self.pp_stages:
            # Get first TP group device IDs (for primary TP group)
            tp_groups = stage.get_tp_device_groups()
            primary_tp_group = tp_groups[0] if tp_groups else []
            
            stage_config = {
                'layer_range': stage.layer_range,
                'tp_size': stage.tp_size,
                'dp_size': stage.dp_size,
                'device_group': primary_tp_group,  # Device IDs for this TP group
            }
            vla_config.append(stage_config)
        
        return vla_config
    
    def create_interconnect_for_stage(
        self,
        stage: PPStageConfig,
        base_link_module: LinkModule,
    ) -> InterConnectModule:
        """
        Create an InterConnectModule for a specific PP stage.
        
        This uses the stage's bandwidth_matrix to create an interconnect
        with per-device-pair bandwidth.
        
        Args:
            stage: PP stage configuration
            base_link_module: Base link module with default bandwidth/latency
            
        Returns:
            InterConnectModule configured for this stage's device cluster
        """
        # Map logical device IDs (0, 1, 2, ...) to physical device IDs
        device_id_mapping = {logical_id: physical_id 
                           for logical_id, physical_id in enumerate(stage.device_cluster)}
        
        topology_type = TopologyType.FC if self.topology == "FC" else TopologyType.RING
        
        interconnect = InterConnectModule(
            device_count=len(stage.device_cluster),
            topology=topology_type,
            link_module=base_link_module,
            link_count_per_device=self.link_count_per_device,
            bandwidth_matrix=stage.bandwidth_matrix,
            device_id_mapping=device_id_mapping,
        )
        
        return interconnect
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the compute graph"""
        lines = []
        lines.append(f"Compute Graph Summary")
        lines.append(f"{'=' * 60}")
        lines.append(f"Total PP stages: {len(self.pp_stages)}")
        lines.append(f"Total unique devices: {self.get_total_devices()}")
        lines.append(f"Topology: {self.topology}")
        lines.append(f"Base link bandwidth: {self.base_link_bandwidth:.2e} bytes/s")
        lines.append("")
        
        for stage in self.pp_stages:
            lines.append(f"PP Stage {stage.stage_id}:")
            lines.append(f"  Layer range: [{stage.layer_range[0]}, {stage.layer_range[1]}) "
                        f"({stage.layer_range[1] - stage.layer_range[0]} layers)")
            lines.append(f"  Device cluster: {stage.device_cluster}")
            lines.append(f"  TP size: {stage.tp_size}")
            lines.append(f"  DP size: {stage.dp_size}")
            lines.append(f"  Total devices in stage: {stage.get_device_count()}")
            
            if stage.bandwidth_matrix:
                # Show bandwidth matrix summary
                unique_ratios = set()
                for device_i in stage.device_cluster:
                    if device_i in stage.bandwidth_matrix:
                        for device_j, ratio in stage.bandwidth_matrix[device_i].items():
                            if device_j in stage.device_cluster:
                                unique_ratios.add(ratio)
                
                if len(unique_ratios) == 0:
                    # Single device or no device pairs
                    lines.append(f"  Bandwidth: N/A (single device or no pairs)")
                elif len(unique_ratios) > 1:
                    lines.append(f"  Bandwidth ratios: min={min(unique_ratios):.2f}, "
                               f"max={max(unique_ratios):.2f}, "
                               f"avg={sum(unique_ratios)/len(unique_ratios):.2f}")
                else:
                    lines.append(f"  Bandwidth: uniform (ratio={list(unique_ratios)[0]:.2f})")
            
            lines.append("")
        
        return "\n".join(lines)

