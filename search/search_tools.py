"""
Search tools for evaluating DP+TP configurations.

This module provides functions to:
- Evaluate different DP+TP configurations
- Calculate performance metrics (latency, throughput)
- Compare configurations
"""

from typing import Dict, Tuple, List
from llm_predict.models.software.parallel_compute_graph import (
    ParallelComputeGraph,
    ParallelismConfig,
    DeviceGroup,
    ParallelismType,
)
from llm_predict.models.software.transformer import TransformerBlockInitComputationTP
from llm_predict.models.software.utils import Tensor, DataType, data_type_dict
from llm_predict.models.hardware.system import System


def evaluate_dp_tp_config(
    system: System,
    dp_size: int,
    tp_size: int,
    d_model: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
    data_type: DataType,
    compile_mode: str = "heuristic-GPU",
) -> Dict[str, float]:
    """
    Evaluate a specific DP+TP configuration.
    
    Args:
        system: System configuration
        dp_size: Data parallelism size
        tp_size: Tensor parallelism size (must satisfy dp_size * tp_size <= total devices)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        batch_size: Batch size per DP group
        seq_len: Sequence length
        data_type: Data type
        compile_mode: Compilation mode for simulation
    
    Returns:
        Dictionary with performance metrics:
        - 'latency': Latency per DP group (seconds)
        - 'throughput': Throughput (samples/second) = dp_size / latency
        - 'latency_per_sample': Latency per sample (seconds)
    """
    # Create parallelism configuration
    config = ParallelismConfig(dp_size=dp_size, tp_size=tp_size, pp_size=1)
    
    # Check if configuration is valid
    if config.total_devices > system.interconnect.device_count:
        return {
            'latency': float('inf'),
            'throughput': 0.0,
            'latency_per_sample': float('inf'),
            'valid': False,
        }
    
    # Create compute graph
    graph = ParallelComputeGraph(config)
    
    # Get DP groups
    groups = config.get_device_groups()
    dp_groups = [
        DeviceGroup(device_ids=ids, parallelism_type=ParallelismType.DATA_PARALLEL, group_id=i)
        for i, ids in enumerate(groups['dp_groups'])
    ]
    
    # Create one transformer instance per DP group
    for dp_idx, dp_group in enumerate(dp_groups):
        transformer = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=tp_size,  # TP within each DP group
            data_type=data_type
        )
        
        # Build computation graph
        input_tensor = Tensor([batch_size, seq_len, d_model], data_type)
        _ = transformer(input_tensor)
        
        # Add to graph
        node_id = f"transformer_dp{dp_idx}"
        graph.add_compute_node(
            node_id=node_id,
            operator=transformer,
            device_group=dp_group,
            input_tensors=[input_tensor],
        )
    
    # Simulate the graph
    node_latencies = graph.simulate(system, compile_mode)
    total_latency = graph.get_total_latency(node_latencies)
    
    # Calculate metrics
    # Latency per DP group (all DP groups run in parallel, so latency is the max)
    latency_per_dp_group = total_latency
    
    # Throughput: DP groups process in parallel, so total throughput = dp_size / latency
    throughput = dp_size / latency_per_dp_group if latency_per_dp_group > 0 else 0.0
    
    # Latency per sample (average across all DP groups)
    latency_per_sample = latency_per_dp_group / batch_size
    
    return {
        'latency': latency_per_dp_group,
        'throughput': throughput,
        'latency_per_sample': latency_per_sample,
        'dp_size': dp_size,
        'tp_size': tp_size,
        'valid': True,
    }


def compare_configurations(
    system: System,
    total_devices: int,
    d_model: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
    data_type: DataType = data_type_dict["fp16"],
    compile_mode: str = "heuristic-GPU",
    metric: str = "throughput",
) -> List[Dict[str, float]]:
    """
    Compare all possible DP+TP configurations for a given number of devices.
    
    Args:
        system: System configuration
        total_devices: Total number of devices
        d_model: Hidden dimension
        n_heads: Number of attention heads
        batch_size: Batch size per DP group
        seq_len: Sequence length
        data_type: Data type
        compile_mode: Compilation mode
        metric: Metric to optimize ("throughput", "latency", "latency_per_sample")
    
    Returns:
        List of evaluation results for all valid configurations, sorted by the metric
    """
    results = []
    
    # Generate all valid factorizations of total_devices
    # dp_size * tp_size = total_devices
    for dp_size in range(1, total_devices + 1):
        if total_devices % dp_size == 0:
            tp_size = total_devices // dp_size
            
            # Evaluate this configuration
            result = evaluate_dp_tp_config(
                system=system,
                dp_size=dp_size,
                tp_size=tp_size,
                d_model=d_model,
                n_heads=n_heads,
                batch_size=batch_size,
                seq_len=seq_len,
                data_type=data_type,
                compile_mode=compile_mode,
            )
            
            results.append(result)
    
    # Sort by metric (higher is better for throughput, lower is better for latency)
    if metric == "throughput":
        results.sort(key=lambda x: x.get('throughput', 0.0), reverse=True)
    elif metric in ["latency", "latency_per_sample"]:
        results.sort(key=lambda x: x.get(metric, float('inf')))
    
    return results


def find_best_configuration(
    system: System,
    total_devices: int,
    d_model: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
    data_type: DataType = data_type_dict["fp16"],
    compile_mode: str = "heuristic-GPU",
    metric: str = "throughput",
) -> Tuple[int, int, Dict[str, float]]:
    """
    Find the best DP+TP configuration using brute force search.
    
    Args:
        system: System configuration
        total_devices: Total number of devices
        d_model: Hidden dimension
        n_heads: Number of attention heads
        batch_size: Batch size per DP group
        seq_len: Sequence length
        data_type: Data type
        compile_mode: Compilation mode
        metric: Metric to optimize ("throughput", "latency", "latency_per_sample")
    
    Returns:
        Tuple of (best_dp_size, best_tp_size, best_result_dict)
    """
    results = compare_configurations(
        system=system,
        total_devices=total_devices,
        d_model=d_model,
        n_heads=n_heads,
        batch_size=batch_size,
        seq_len=seq_len,
        data_type=data_type,
        compile_mode=compile_mode,
        metric=metric,
    )
    
    # Filter valid results
    valid_results = [r for r in results if r.get('valid', False)]
    
    if not valid_results:
        raise ValueError(f"No valid configurations found for {total_devices} devices")
    
    best_result = valid_results[0]
    return (best_result['dp_size'], best_result['tp_size'], best_result)


def print_configuration_summary(results: List[Dict[str, float]], metric: str = "throughput"):
    """Print a summary table of all configurations."""
    print("\n" + "=" * 80)
    print(f"Configuration Comparison (sorted by {metric})")
    print("=" * 80)
    print(f"{'DP':<6} {'TP':<6} {'Latency (s)':<15} {'Throughput (1/s)':<18} {'Lat/sample (s)':<15}")
    print("-" * 80)
    
    for result in results:
        if result.get('valid', False):
            print(
                f"{result['dp_size']:<6} "
                f"{result['tp_size']:<6} "
                f"{result['latency']:<15.6f} "
                f"{result['throughput']:<18.6f} "
                f"{result['latency_per_sample']:<15.9f}"
            )
        else:
            print(
                f"{result.get('dp_size', 'N/A'):<6} "
                f"{result.get('tp_size', 'N/A'):<6} "
                f"{'INVALID':<15}"
            )
    
    print("=" * 80)

