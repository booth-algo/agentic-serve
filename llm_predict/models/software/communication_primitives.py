from llm_predict.models.hardware.device import Device
from llm_predict.models.hardware.interconnect import (
    LinkModule,
    InterConnectModule,
    TopologyType,
    interconnect_module_dict,
)
from llm_predict.models.software.utils import Tensor, DataType
from typing import Any, List, Optional
from llm_predict.utils import size
from math import ceil


class CommunicationPrimitive:
    def __init__(self, data_type: DataType) -> None:
        self.data_type = data_type
        # simulation results
        self.latency = None


class AllReduceMultiPCB(CommunicationPrimitive):
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor

    def simulate(self, interconnect_module: InterConnectModule, device_group: Optional[List[int]] = None) -> None:
        """
        Simulate AllReduce operation.
        
        Args:
            interconnect_module: Interconnect module with topology and bandwidth
            device_group: Optional list of device IDs participating in AllReduce.
                         If None, uses all devices (0 to device_count-1).
                         Useful for TP groups within a larger system.
        """
        if device_group is None:
            device_group = list(range(interconnect_module.device_count))
        
        device_count = len(device_group)
        
        # Early return if device_count is 1 (no communication needed)
        if device_count == 1:
            self.latency = 0.0
            return self.latency
        
        link_latency = interconnect_module.link_module.latency
        flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        data_size = size(self.input_shape) * self.data_type.word_size
        data_size_per_device = data_size / device_count
        effective_data_size_per_device = (
            header_size
            + ceil(data_size_per_device / max_payload_size) * header_size
            + data_size_per_device
        )
        
        if interconnect_module.topology == TopologyType.FC:
            # Check if we have per-device-pair bandwidth matrix
            if interconnect_module.bandwidth_matrix:
                # Use per-device-pair bandwidth for more accurate modeling
                # For FC topology, AllReduce can use all-to-all communication
                # The bottleneck is the minimum bandwidth among ALL pairs in the cluster
                # (not just along a ring path)
                
                # Stage 1: Ring reduce - find minimum bandwidth among all pairs in cluster
                min_reduce_bandwidth = float('inf')
                for i in range(len(device_group)):
                    for j in range(len(device_group)):
                        if i != j:
                            src = device_group[i]
                            dst = device_group[j]
                            bw = interconnect_module.get_bandwidth(src, dst, "both_direction")
                            min_reduce_bandwidth = min(min_reduce_bandwidth, bw)
                
                if min_reduce_bandwidth == float('inf'):
                    # Fallback to uniform bandwidth
                    min_reduce_bandwidth = (
                        interconnect_module.link_module.bandwidth_both_direction
                        * interconnect_module.link_count_per_device
                        / max(1, device_count - 1)
                    )
                
                # Stage 2: Broadcast - find minimum bandwidth among all pairs in cluster
                min_bcast_bandwidth = float('inf')
                for i in range(len(device_group)):
                    for j in range(len(device_group)):
                        if i != j:
                            src = device_group[i]
                            dst = device_group[j]
                            bw = interconnect_module.get_bandwidth(src, dst, "per_direction")
                            min_bcast_bandwidth = min(min_bcast_bandwidth, bw)
                
                if min_bcast_bandwidth == float('inf'):
                    # Fallback to uniform bandwidth
                    min_bcast_bandwidth = (
                        interconnect_module.link_module.bandwidth_per_direction
                        * interconnect_module.link_count_per_device
                        / max(1, device_count - 1)
                    )
                
                # Ring reduce stage
                latency = (
                    link_latency
                    + effective_data_size_per_device / min_reduce_bandwidth
                ) * (device_count - 1)
                
                # Broadcast stage
                latency += effective_data_size_per_device / min_bcast_bandwidth
            else:
                # Uniform bandwidth (original implementation)
                link_bandwidth_per_direction = (
                    interconnect_module.link_module.bandwidth_per_direction
                )
                link_bandwidth_both_direction = (
                    interconnect_module.link_module.bandwidth_both_direction
                )
                link_count_per_device = interconnect_module.link_count_per_device
                
                edge_bandwidth_per_direction = (
                    link_bandwidth_per_direction
                    * link_count_per_device
                    / max(1, device_count - 1)
                )
                edge_bandwidth_both_direction = (
                    link_bandwidth_both_direction
                    * link_count_per_device
                    / max(1, device_count - 1)
                )
                
                # stage 1: ring reduce
                latency = (
                    link_latency
                    + effective_data_size_per_device / edge_bandwidth_both_direction
                ) * (device_count - 1)
                # stage 2: broadcast
                latency += effective_data_size_per_device / edge_bandwidth_per_direction
            
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
            return latency
        elif interconnect_module.topology == TopologyType.RING:
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            per_transmission_latency = effective_data_size_per_device / edge_bandwidth
            latency = (edge_latency + per_transmission_latency) * (
                (device_count - 1) * 2
            )
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
        else:
            raise NotImplementedError
        return self.latency


class AllReduceHierarchical(CommunicationPrimitive):
    """
    Two-level hierarchical AllReduce for multi-node clusters.

    Algorithm:
    1. Intra-node ReduceScatter (fast NVLink)
    2. Inter-node AllReduce of partial results (slower InfiniBand)
    3. Intra-node AllGather (fast NVLink)

    This is the standard hierarchical approach used by NCCL for multi-node.
    For N nodes with G GPUs each:
      - Step 1: Each node does local reduce-scatter → G chunks, each GPU has 1/G of reduced data
      - Step 2: GPUs with same local rank across nodes do inter-node allreduce → 1 chunk fully reduced
      - Step 3: Each node does local allgather → all GPUs have full result

    Bandwidth-optimal: uses NVLink for bulk data, InfiniBand only for 1/G of data.
    """

    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor

    def simulate(self, interconnect_module: InterConnectModule, device_group=None) -> float:
        """
        Simulate hierarchical AllReduce.

        Falls back to flat AllReduce if no multi-node topology is detected.
        """
        if device_group is None:
            device_group = list(range(interconnect_module.device_count))

        device_count = len(device_group)
        if device_count == 1:
            self.latency = 0.0
            return self.latency

        # Check if this is a multi-node interconnect
        if not hasattr(interconnect_module, 'device_node_mapping') or not hasattr(interconnect_module, 'num_nodes'):
            # Fall back to flat AllReduce
            flat = AllReduceMultiPCB(self.data_type)
            flat(Tensor(self.input_shape, self.data_type))
            self.latency = flat.simulate(interconnect_module, device_group)
            return self.latency

        num_nodes = interconnect_module.num_nodes
        gpus_per_node = interconnect_module.gpus_per_node
        node_mapping = interconnect_module.device_node_mapping

        # If all devices are on the same node, use flat AllReduce
        nodes_in_group = set(node_mapping.get(d, 0) for d in device_group)
        if len(nodes_in_group) == 1:
            flat = AllReduceMultiPCB(self.data_type)
            flat(Tensor(self.input_shape, self.data_type))
            self.latency = flat.simulate(interconnect_module, device_group)
            return self.latency

        data_size = size(self.input_shape) * self.data_type.word_size

        # Get intra-node and inter-node bandwidths
        intra_link = interconnect_module.link_module
        intra_bw = intra_link.bandwidth_both_direction * interconnect_module.link_count_per_device
        intra_latency = intra_link.latency

        inter_link = interconnect_module.inter_node_link
        inter_bw = inter_link.bandwidth_both_direction * interconnect_module.inter_node_links_per_gpu
        inter_latency = inter_link.latency

        header_size = intra_link.header_size
        max_payload = intra_link.max_payload_size

        G = gpus_per_node  # GPUs per node
        N = len(nodes_in_group)  # Number of nodes participating

        # Step 1: Intra-node ReduceScatter
        # Each GPU sends (G-1)/G of data, receives 1/G chunk (fully reduced within node)
        # Ring algorithm: (G-1) steps, each step sends data_size/G bytes
        chunk_size = data_size / G
        effective_chunk = chunk_size + ceil(chunk_size / max_payload) * header_size + header_size
        # Intra-node bandwidth per link pair (for ring within node)
        intra_ring_bw = intra_bw / max(1, G - 1)  # share links among peers
        step1_latency = (intra_latency + effective_chunk / intra_ring_bw) * (G - 1)

        # Step 2: Inter-node AllReduce
        # Only 1 GPU per node participates (the one holding the relevant chunk)
        # Data volume = data_size / G (just one chunk)
        inter_chunk = chunk_size
        effective_inter_chunk = inter_chunk + ceil(inter_chunk / max_payload) * header_size + header_size
        # Ring allreduce across N nodes: 2*(N-1) steps
        step2_latency = (inter_latency + effective_inter_chunk / inter_bw) * 2 * (N - 1)

        # Step 3: Intra-node AllGather
        # Same cost as ReduceScatter (symmetric)
        step3_latency = step1_latency

        self.latency = step1_latency + step2_latency + step3_latency

        # Store breakdown for debugging
        self.latency_breakdown = {
            'intra_reduce_scatter': step1_latency,
            'inter_allreduce': step2_latency,
            'intra_allgather': step3_latency,
        }

        return self.latency


class P2P(CommunicationPrimitive):
    """
    Point-to-point communication primitive for pipeline parallelism.
    Sends data from source device to destination device.
    """
    def __init__(self, data_type: DataType, src_device: int, dst_device: int):
        super().__init__(data_type)
        self.src_device = src_device
        self.dst_device = dst_device

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor

    def simulate(self, interconnect_module: InterConnectModule) -> float:
        """
        Simulate P2P communication between two devices.
        Uses link bandwidth and latency for direct communication.
        """
        data_size = size(self.input_shape) * self.data_type.word_size
        link_bandwidth = interconnect_module.link_module.bandwidth_per_direction
        link_latency = interconnect_module.link_module.latency
        flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        
        # Account for packetization overhead
        num_packets = ceil(data_size / max_payload_size)
        effective_data_size = data_size + num_packets * header_size
        
        # Simple model: direct link communication
        # In reality, routing depends on topology, but this is a reasonable approximation
        transmission_latency = effective_data_size / link_bandwidth
        
        # For FC topology, assume direct link (might be optimistic)
        # For RING topology, might need to route through intermediate devices
        if interconnect_module.topology == TopologyType.RING:
            # Estimate hop count (simplified: assume devices are adjacent in ring)
            # In worst case, might need to go through (device_count/2) hops
            device_count = interconnect_module.device_count
            # Simplified: assume average hop distance
            avg_hops = min(abs(self.dst_device - self.src_device), 
                          device_count - abs(self.dst_device - self.src_device))
            transmission_latency *= max(1, avg_hops)
        
        self.latency = link_latency + transmission_latency
        return self.latency


class AllGather(CommunicationPrimitive):
    """
    AllGather communication primitive: gathers sharded tensors from all devices.
    Each device has a shard, all devices receive the full concatenated tensor.
    """
    def __init__(self, data_type: DataType):
        super().__init__(data_type)

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape  # Input is sharded tensor
        return tensor

    def simulate(self, interconnect_module: InterConnectModule) -> float:
        """
        Simulate AllGather operation.
        Similar to AllReduce but gathers instead of reduces.
        """
        device_count = interconnect_module.device_count
        
        # Early return if device_count is 1 (no communication needed)
        if device_count == 1:
            self.latency = 0.0
            return self.latency
        
        link_bandwidth = interconnect_module.link_module.bandwidth_per_direction
        link_bandwidth_both = interconnect_module.link_module.bandwidth_both_direction
        link_latency = interconnect_module.link_module.latency
        flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        link_count_per_device = interconnect_module.link_count_per_device
        
        # Input is sharded, output is full size
        shard_size = size(self.input_shape) * self.data_type.word_size
        full_size = shard_size * device_count
        
        if interconnect_module.topology == TopologyType.FC:
            # Similar to AllReduce but gathering instead of reducing
            edge_bandwidth = link_bandwidth * link_count_per_device / max(1, device_count - 1)
            edge_latency = link_latency
            effective_shard_size = (
                header_size
                + ceil(shard_size / max_payload_size) * header_size
                + shard_size
            )
            # Gather phase: each device sends its shard to all others
            # Simplified model: (device_count - 1) transmissions
            latency = (edge_latency + effective_shard_size / edge_bandwidth) * (device_count - 1)
            latency += full_size / interconnect_module.internal_link_bandwidth_per_direction
            self.latency = latency
            return latency
        elif interconnect_module.topology == TopologyType.RING:
            edge_bandwidth = link_bandwidth * link_count_per_device
            edge_latency = link_latency
            effective_shard_size = (
                header_size
                + ceil(shard_size / max_payload_size) * header_size
                + shard_size
            )
            # Ring-based allgather: propagate shards around the ring
            per_transmission_latency = effective_shard_size / edge_bandwidth
            # Each shard needs to travel (device_count - 1) hops
            latency = (edge_latency + per_transmission_latency) * (device_count - 1)
            latency += full_size / interconnect_module.internal_link_bandwidth_per_direction
            self.latency = latency
            return latency
        else:
            raise NotImplementedError(f"Topology {interconnect_module.topology} not supported for AllGather")
        return self.latency


class Broadcast(CommunicationPrimitive):
    """
    Broadcast communication primitive: broadcasts tensor from source device to all others.
    """
    def __init__(self, data_type: DataType, src_device: int = 0):
        super().__init__(data_type)
        self.src_device = src_device

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor

    def simulate(self, interconnect_module: InterConnectModule) -> float:
        """
        Simulate Broadcast operation.
        Source device sends data to all other devices.
        """
        device_count = interconnect_module.device_count
        link_bandwidth = interconnect_module.link_module.bandwidth_per_direction
        link_latency = interconnect_module.link_module.latency
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        
        data_size = size(self.input_shape) * self.data_type.word_size
        
        if device_count == 1:
            self.latency = 0.0
            return 0.0
        
        # Packetization overhead
        num_packets = ceil(data_size / max_payload_size)
        effective_data_size = data_size + num_packets * header_size
        
        if interconnect_module.topology == TopologyType.FC:
            # In FC, source can send to all destinations in parallel (simplified)
            # More accurate: would need to model fan-out tree
            transmission_latency = effective_data_size / link_bandwidth
            self.latency = link_latency + transmission_latency
            return self.latency
        elif interconnect_module.topology == TopologyType.RING:
            # In ring, need to propagate around the ring
            link_count_per_device = interconnect_module.link_count_per_device
            edge_bandwidth = link_bandwidth * link_count_per_device
            per_hop_latency = link_latency + effective_data_size / edge_bandwidth
            # Worst case: data needs to travel (device_count - 1) hops
            self.latency = per_hop_latency * (device_count - 1)
            return self.latency
        else:
            raise NotImplementedError(f"Topology {interconnect_module.topology} not supported for Broadcast")
        return self.latency

