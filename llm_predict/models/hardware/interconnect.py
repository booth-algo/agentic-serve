from enum import Enum, auto
from math import ceil
from typing import Dict, Optional


class TopologyType(Enum):
    RING = auto()
    FC = auto()


class LinkModule:
    def __init__(
        self,
        bandwidth_per_direction: float,  # B/s
        bandwidth_both_direction: float,  # B/s
        latency: float,  # s
        flit_size: int,  # B
        max_payload_size: int,  # B
        header_size: int,  # B
    ) -> None:
        self.bandwidth_per_direction = bandwidth_per_direction
        self.bandwidth_both_direction = bandwidth_both_direction
        self.latency = latency
        self.flit_size = flit_size
        self.max_payload_size = max_payload_size
        self.header_size = ceil(header_size / flit_size) * flit_size


link_module_dict = {
    "NVLinkV3": LinkModule(25e9, 50e9, 8.92e-6, 16, 256, 16),
    # H100: 18 NVLink V4 links × 25GB/s = 450 GB/s per direction, 900 GB/s bidi
    "NVLinkV4": LinkModule(25e9, 50e9, 5e-6, 16, 256, 16),
    # InfiniBand HDR (200 Gbps = 25 GB/s per direction)
    "InfiniBandHDR": LinkModule(25e9, 50e9, 1e-6, 16, 4096, 64),
    # InfiniBand NDR (400 Gbps = 50 GB/s per direction)
    "InfiniBandNDR": LinkModule(50e9, 100e9, 0.6e-6, 16, 4096, 64),
    # InfiniBand XDR (800 Gbps = 100 GB/s per direction)
    "InfiniBandXDR": LinkModule(100e9, 200e9, 0.5e-6, 16, 4096, 64),
    "TPUv3Link": LinkModule(81.25e9 / 2, 81.25e9, 150e-6, 16, 256, 16),
}
# we cannot find a way to measure TPU p2p latency, we also don't know TPU packet format


class InterConnectModule:
    def __init__(
        self,
        device_count: int,
        topology,
        link_module: LinkModule,
        link_count_per_device: int,
        internal_link_bandwidth_per_direction: float = float("inf"),
        bandwidth_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        device_id_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Initialize interconnect module.
        
        Args:
            device_count: Number of devices in the interconnect
            topology: Topology type (TopologyType.FC or TopologyType.RING)
            link_module: Link module with base bandwidth and latency
            link_count_per_device: Number of links per device
            internal_link_bandwidth_per_direction: Internal link bandwidth
            bandwidth_matrix: Optional per-device-pair bandwidth matrix.
                            Format: {device_i: {device_j: bandwidth_ratio}, ...}
                            bandwidth_ratio is a multiplier (0.0-1.0) applied to base bandwidth.
                            If None, assumes uniform bandwidth for all pairs.
            device_id_mapping: Optional mapping from logical device IDs to physical device IDs.
                             If provided, used to index bandwidth_matrix with physical IDs.
                             Format: {logical_id: physical_id}
        """
        self.device_count = device_count
        self.topology = topology
        self.link_module = link_module
        self.link_count_per_device = link_count_per_device
        self.internal_link_bandwidth_per_direction = (
            internal_link_bandwidth_per_direction
        )
        self.bandwidth_matrix = bandwidth_matrix or {}
        self.device_id_mapping = device_id_mapping or {i: i for i in range(device_count)}
        pass
    
    def get_bandwidth(self, device_i: int, device_j: int, direction: str = "per_direction") -> float:
        """
        Get bandwidth between two devices.
        
        Args:
            device_i: Source device ID (logical)
            device_j: Destination device ID (logical)
            direction: "per_direction" or "both_direction"
        
        Returns:
            Bandwidth in bytes/second
        """
        # Map logical IDs to physical IDs if mapping exists
        phys_i = self.device_id_mapping.get(device_i, device_i)
        phys_j = self.device_id_mapping.get(device_j, device_j)
        
        # Get bandwidth ratio from matrix, default to 1.0
        if phys_i in self.bandwidth_matrix and phys_j in self.bandwidth_matrix[phys_i]:
            ratio = self.bandwidth_matrix[phys_i][phys_j]
        else:
            ratio = 1.0
        
        # Apply ratio to base bandwidth
        if direction == "per_direction":
            base_bw = self.link_module.bandwidth_per_direction
        elif direction == "both_direction":
            base_bw = self.link_module.bandwidth_both_direction
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # For FC topology with multiple links, distribute bandwidth
        if self.topology == TopologyType.FC:
            # Distribute available links among (device_count - 1) connections
            effective_links = self.link_count_per_device / max(1, self.device_count - 1)
            return base_bw * ratio * effective_links
        else:
            # For other topologies, use all links
            return base_bw * ratio * self.link_count_per_device


# we treat the 2D torus interconnect of 8 TPU cores as 2 rings + internal link
interconnect_module_dict = {
    "NVLinkV3_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV3"], 12
    ),
    "TPUv3Link_8": InterConnectModule(
        4, TopologyType.RING, link_module_dict["TPUv3Link"], 2, 162.5e9
    ),
    # H100 single-node configs
    "H100_NVLink_2": InterConnectModule(
        2, TopologyType.FC, link_module_dict["NVLinkV4"], 18
    ),
    "H100_NVLink_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV4"], 18
    ),
    "H100_NVLink_8": InterConnectModule(
        8, TopologyType.FC, link_module_dict["NVLinkV4"], 18
    ),
}

# class InterConnectTorusModule:
#     def __init__(self) -> None:
#         pass


# class InterConnectUniRingModule:
#     def __init__(
#         self,
#         device_count,
#         link: LinkModule,
#     ) -> None:
#         pass


# class InterConnectMeshModule:
#     def __init__(self) -> None:
#         pass


# class InterConnectFCModule:
#     def __init__(self) -> None:
#         pass
