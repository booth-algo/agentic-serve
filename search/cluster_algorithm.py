"""
Clustering algorithm for device grouping based on communication bandwidth.

Given an nxn matrix representing communication bandwidth between n devices,
this module provides two clustering modes:
1. Fixed number of groups: Cluster devices into a specified number of groups
2. Automatic grouping: Automatically determine the optimal number of groups
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def cluster_devices_fixed_groups(
    bandwidth_matrix: np.ndarray,
    num_groups: int
) -> List[List[int]]:
    """
    Cluster devices into a fixed number of groups using a greedy algorithm.
    
    The algorithm:
    1. Starts by finding the highest bandwidth pairs
    2. Builds clusters greedily by adding devices that maximize intra-cluster bandwidth
    3. Ensures balanced cluster sizes when possible
    
    Args:
        bandwidth_matrix: nxn symmetric matrix where bandwidth_matrix[i][j] represents
                         the communication bandwidth between device i and device j.
                         Diagonal elements (self-communication) are typically 0 or ignored.
        num_groups: The desired number of clusters
        
    Returns:
        List of clusters, where each cluster is a list of device indices
    """
    n = bandwidth_matrix.shape[0]
    
    if num_groups <= 0:
        raise ValueError("Number of groups must be positive")
    if num_groups > n:
        raise ValueError(f"Number of groups ({num_groups}) cannot exceed number of devices ({n})")
    if num_groups == 1:
        return [list(range(n))]
    if num_groups == n:
        return [[i] for i in range(n)]
    
    # Create a copy and set diagonal to -inf to ignore self-connections
    matrix = bandwidth_matrix.copy()
    np.fill_diagonal(matrix, -np.inf)
    
    # Initialize clusters
    clusters: List[List[int]] = []
    assigned = set()
    
    # Step 1: Find the top num_groups pairs with highest bandwidth
    # These will serve as seeds for our clusters
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((matrix[i][j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize clusters with seed pairs
    seed_idx = 0
    for _ in range(num_groups):
        if seed_idx >= len(pairs):
            break
        _, i, j = pairs[seed_idx]
        # Skip if both devices are already assigned
        while seed_idx < len(pairs) and (i in assigned or j in assigned):
            seed_idx += 1
            if seed_idx < len(pairs):
                _, i, j = pairs[seed_idx]
        if seed_idx < len(pairs):
            clusters.append([i, j])
            assigned.add(i)
            assigned.add(j)
            seed_idx += 1
    
    # If we couldn't find enough seed pairs, create singleton clusters
    while len(clusters) < num_groups:
        for i in range(n):
            if i not in assigned:
                clusters.append([i])
                assigned.add(i)
                break
    
    # Step 2: Greedily assign remaining devices to clusters
    # For each unassigned device, assign it to the cluster that maximizes
    # the total bandwidth to devices in that cluster
    unassigned = [i for i in range(n) if i not in assigned]
    
    for device in unassigned:
        best_cluster_idx = 0
        best_total_bandwidth = -np.inf
        
        for cluster_idx, cluster in enumerate(clusters):
            # Calculate total bandwidth from device to all devices in cluster
            total_bandwidth = sum(matrix[device][member] for member in cluster)
            if total_bandwidth > best_total_bandwidth:
                best_total_bandwidth = total_bandwidth
                best_cluster_idx = cluster_idx
        
        clusters[best_cluster_idx].append(device)
        assigned.add(device)
    
    # Step 3: Balance clusters if needed (optional refinement)
    # Try to balance cluster sizes by moving devices from large clusters to small ones
    # if it doesn't significantly reduce intra-cluster bandwidth
    _balance_clusters(clusters, matrix)
    
    return clusters


def cluster_devices_automatic(
    bandwidth_matrix: np.ndarray,
    min_cluster_size: int = 1,
    max_cluster_size: Optional[int] = None,
    bandwidth_threshold_ratio: float = 0.5
) -> Tuple[List[List[int]], int]:
    """
    Automatically determine the number of groups and cluster devices.
    
    Uses a greedy agglomerative approach with a threshold-based stopping criterion.
    The algorithm merges clusters if the average inter-cluster bandwidth is above
    a threshold relative to the maximum bandwidth in the matrix.
    
    Args:
        bandwidth_matrix: n×n symmetric matrix where bandwidth_matrix[i][j] represents
                         the communication bandwidth between device i and device j.
        min_cluster_size: Minimum number of devices per cluster (default: 1)
        max_cluster_size: Maximum number of devices per cluster (default: None, no limit)
        bandwidth_threshold_ratio: Ratio of average bandwidth to max bandwidth for
                                  merging clusters (default: 0.5). Higher values create
                                  fewer, tighter clusters.
    
    Returns:
        Tuple of (list of clusters, number of clusters)
    """
    n = bandwidth_matrix.shape[0]
    
    if max_cluster_size is None:
        max_cluster_size = n
    
    # Create a copy and set diagonal to -inf
    matrix = bandwidth_matrix.copy()
    np.fill_diagonal(matrix, -np.inf)
    
    max_bandwidth = np.max(matrix)
    threshold = max_bandwidth * bandwidth_threshold_ratio
    
    # Start with each device in its own cluster
    clusters: List[List[int]] = [[i] for i in range(n)]
    
    # Greedily merge clusters
    while True:
        best_merge = None
        best_score = -np.inf
        
        # Try all pairs of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = clusters[i]
                cluster_j = clusters[j]
                
                # Check size constraints
                new_size = len(cluster_i) + len(cluster_j)
                if new_size > max_cluster_size:
                    continue
                
                # Calculate average bandwidth between the two clusters
                total_bandwidth = 0
                count = 0
                for dev_i in cluster_i:
                    for dev_j in cluster_j:
                        total_bandwidth += matrix[dev_i][dev_j]
                        count += 1
                
                if count > 0:
                    avg_bandwidth = total_bandwidth / count
                    # Score based on average bandwidth and cluster sizes
                    # Prefer merging clusters with high bandwidth and similar sizes
                    size_balance = 1.0 / (1.0 + abs(len(cluster_i) - len(cluster_j)))
                    score = avg_bandwidth * size_balance
                    
                    if score > best_score and avg_bandwidth >= threshold:
                        best_score = score
                        best_merge = (i, j)
        
        # If no good merge found, stop
        if best_merge is None:
            break
        
        # Merge the two clusters
        i, j = best_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    # Ensure minimum cluster size by merging small clusters
    small_clusters = [i for i, cluster in enumerate(clusters) if len(cluster) < min_cluster_size]
    if small_clusters and len(clusters) > 1:
        # Merge small clusters into the nearest larger cluster
        for small_idx in reversed(small_clusters):  # Reverse to maintain indices
            small_cluster = clusters[small_idx]
            if len(clusters) == 1:
                break
            
            best_target = None
            best_bandwidth = -np.inf
            
            for target_idx, target_cluster in enumerate(clusters):
                if target_idx == small_idx:
                    continue
                
                # Calculate average bandwidth
                total = sum(matrix[dev_i][dev_j] 
                           for dev_i in small_cluster 
                           for dev_j in target_cluster)
                count = len(small_cluster) * len(target_cluster)
                avg = total / count if count > 0 else 0
                
                if avg > best_bandwidth:
                    best_bandwidth = avg
                    best_target = target_idx
            
            if best_target is not None:
                clusters[best_target].extend(small_cluster)
                clusters.pop(small_idx)
    
    num_groups = len(clusters)
    return clusters, num_groups


def _balance_clusters(clusters: List[List[int]], matrix: np.ndarray) -> None:
    """
    Helper function to balance cluster sizes by moving devices between clusters.
    
    This is a refinement step that tries to improve cluster balance without
    significantly reducing intra-cluster bandwidth.
    """
    if len(clusters) <= 1:
        return
    
    # Calculate current cluster sizes
    sizes = [len(c) for c in clusters]
    avg_size = sum(sizes) / len(sizes)
    
    # Try to balance: move devices from large clusters to small ones
    max_iterations = 10
    for _ in range(max_iterations):
        improved = False
        
        # Find largest and smallest clusters
        largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        smallest_idx = min(range(len(clusters)), key=lambda i: len(clusters[i]))
        
        if len(clusters[largest_idx]) - len(clusters[smallest_idx]) <= 1:
            break  # Already balanced
        
        # Try moving each device from largest to smallest cluster
        largest_cluster = clusters[largest_idx]
        smallest_cluster = clusters[smallest_idx]
        
        best_device = None
        best_improvement = -np.inf
        
        for device in largest_cluster:
            if len(largest_cluster) <= 1:
                break
            
            # Calculate bandwidth loss from leaving largest cluster
            loss = sum(matrix[device][m] for m in largest_cluster if m != device)
            
            # Calculate bandwidth gain from joining smallest cluster
            gain = sum(matrix[device][m] for m in smallest_cluster)
            
            # Improvement is gain - loss, but also consider balance
            improvement = gain - loss * 0.5  # Weight loss less to encourage balance
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_device = device
        
        # Move device if it improves balance without too much bandwidth loss
        if best_device is not None and best_improvement > -np.inf:
            largest_cluster.remove(best_device)
            smallest_cluster.append(best_device)
            improved = True
        
        if not improved:
            break


def calculate_cluster_metrics(
    clusters: List[List[int]],
    bandwidth_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for the clustering result.
    
    Args:
        clusters: List of clusters (device indices)
        bandwidth_matrix: Original bandwidth matrix
        
    Returns:
        Dictionary with metrics:
        - intra_cluster_bandwidth: Average bandwidth within clusters
        - inter_cluster_bandwidth: Average bandwidth between clusters
        - cluster_sizes: List of cluster sizes
        - balance_score: How balanced the clusters are (1.0 = perfectly balanced)
    """
    n = bandwidth_matrix.shape[0]
    matrix = bandwidth_matrix.copy()
    np.fill_diagonal(matrix, 0)
    
    intra_bandwidths = []
    inter_bandwidths = []
    
    # Calculate intra-cluster bandwidths
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                intra_bandwidths.append(matrix[cluster[i]][cluster[j]])
    
    # Calculate inter-cluster bandwidths
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for dev_i in clusters[i]:
                for dev_j in clusters[j]:
                    inter_bandwidths.append(matrix[dev_i][dev_j])
    
    cluster_sizes = [len(c) for c in clusters]
    
    # Balance score: 1.0 if all clusters same size, 0.0 if very unbalanced
    if len(cluster_sizes) > 0:
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
        variance = sum((s - avg_size) ** 2 for s in cluster_sizes) / len(cluster_sizes)
        max_variance = (n / len(clusters)) ** 2  # Worst case: one cluster has all devices
        balance_score = 1.0 - min(variance / max_variance, 1.0) if max_variance > 0 else 1.0
    else:
        balance_score = 0.0
    
    return {
        'intra_cluster_bandwidth': np.mean(intra_bandwidths) if intra_bandwidths else 0.0,
        'inter_cluster_bandwidth': np.mean(inter_bandwidths) if inter_bandwidths else 0.0,
        'cluster_sizes': cluster_sizes,
        'balance_score': balance_score,
        'num_clusters': len(clusters)
    }


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a sample bandwidth matrix
    np.random.seed(42)
    n = 8
    
    # Create a matrix with some structure (devices 0-3 have high bandwidth, 4-7 have high bandwidth)
    matrix = np.random.rand(n, n) * 10
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    
    # Add structure: higher bandwidth within groups
    for i in range(4):
        for j in range(4):
            if i != j:
                matrix[i][j] += 50
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                matrix[i][j] += 50
    
    np.fill_diagonal(matrix, 0)
    
    print("Bandwidth Matrix:")
    print(matrix)
    print("\n" + "="*50 + "\n")
    
    # Test fixed number of groups
    print("Mode 1: Fixed number of groups (num_groups=2)")
    clusters_fixed = cluster_devices_fixed_groups(matrix, num_groups=2)
    print(f"Clusters: {clusters_fixed}")
    metrics_fixed = calculate_cluster_metrics(clusters_fixed, matrix)
    print(f"Metrics:")
    print(f"  Number of clusters: {metrics_fixed['num_clusters']}")
    print(f"  Cluster sizes: {metrics_fixed['cluster_sizes']}")
    print(f"  Average intra-cluster bandwidth: {metrics_fixed['intra_cluster_bandwidth']:.2f}")
    print(f"  Average inter-cluster bandwidth: {metrics_fixed['inter_cluster_bandwidth']:.2f}")
    print(f"  Balance score: {metrics_fixed['balance_score']:.2f}")
    print("\n" + "="*50 + "\n")
    
    # Test automatic grouping
    print("Mode 2: Automatic grouping")
    clusters_auto, num_groups_auto = cluster_devices_automatic(
        matrix, 
        min_cluster_size=1,
        max_cluster_size=None,
        bandwidth_threshold_ratio=0.3
    )
    print(f"Automatically determined number of groups: {num_groups_auto}")
    print(f"Clusters: {clusters_auto}")
    metrics_auto = calculate_cluster_metrics(clusters_auto, matrix)
    print(f"Metrics:")
    print(f"  Number of clusters: {metrics_auto['num_clusters']}")
    print(f"  Cluster sizes: {metrics_auto['cluster_sizes']}")
    print(f"  Average intra-cluster bandwidth: {metrics_auto['intra_cluster_bandwidth']:.2f}")
    print(f"  Average inter-cluster bandwidth: {metrics_auto['inter_cluster_bandwidth']:.2f}")
    print(f"  Balance score: {metrics_auto['balance_score']:.2f}")