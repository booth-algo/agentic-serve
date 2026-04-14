"""
Brute force search algorithm for finding optimal DP+TP configurations.

This module implements brute force search to find the best data parallelism
and tensor parallelism arrangement given a fixed number of devices.
"""

from typing import Dict, Tuple, List, Optional, Callable
from llmcompass.design_space_exploration.dse import template_to_system, read_architecture_template
from llmcompass.software_model.utils import DataType, data_type_dict
from llmcompass.hardware_model.system import System
from search.search_tools import (
    evaluate_dp_tp_config,
    compare_configurations,
    find_best_configuration,
    print_configuration_summary,
)
import os
import json
import time


class DPTPBruteForceSearch:
    """
    Brute force search for optimal DP+TP configuration.
    
    Given a fixed number of devices, this searches all possible factorizations
    to find the best DP and TP sizes.
    """
    
    def __init__(
        self,
        system: System,
        d_model: int,
        n_heads: int,
        batch_size: int = 1,
        seq_len: int = 512,
        data_type: DataType = data_type_dict["fp16"],
        compile_mode: str = "heuristic-GPU",
    ):
        """
        Initialize the search.
        
        Args:
            system: System configuration
            d_model: Hidden dimension
            n_heads: Number of attention heads
            batch_size: Batch size per DP group
            seq_len: Sequence length
            data_type: Data type
            compile_mode: Compilation mode for simulation
        """
        self.system = system
        self.d_model = d_model
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_type = data_type
        self.compile_mode = compile_mode
    
    def get_all_configurations(self, total_devices: int) -> List[Tuple[int, int]]:
        """
        Get all valid DP+TP configurations.
        
        Args:
            total_devices: Total number of devices
        
        Returns:
            List of (dp_size, tp_size) tuples
        """
        configurations = []
        for dp_size in range(1, total_devices + 1):
            if total_devices % dp_size == 0:
                tp_size = total_devices // dp_size
                configurations.append((dp_size, tp_size))
        return configurations
    
    def evaluate(self, dp_size: int, tp_size: int) -> Dict[str, float]:
        """
        Evaluate a specific DP+TP configuration.
        
        Args:
            dp_size: Data parallelism size
            tp_size: Tensor parallelism size
        
        Returns:
            Dictionary with performance metrics
        """
        return evaluate_dp_tp_config(
            system=self.system,
            dp_size=dp_size,
            tp_size=tp_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            data_type=self.data_type,
            compile_mode=self.compile_mode,
        )
    
    def search(
        self,
        total_devices: int,
        metric: str = "throughput",
        verbose: bool = True,
    ) -> Tuple[int, int, Dict[str, float], List[Dict[str, float]]]:
        """
        Brute force search for the best configuration.
        
        Args:
            total_devices: Total number of devices
            metric: Metric to optimize ("throughput", "latency", "latency_per_sample")
            verbose: Whether to print progress
        
        Returns:
            Tuple of (best_dp_size, best_tp_size, best_result, all_results)
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Brute Force Search: Finding best DP+TP configuration")
            print(f"{'=' * 80}")
            print(f"Total devices: {total_devices}")
            print(f"Model: d_model={self.d_model}, n_heads={self.n_heads}")
            print(f"Optimizing for: {metric}")
            print(f"{'=' * 80}\n")
        
        start_time = time.time()
        
        # Get all configurations
        configurations = self.get_all_configurations(total_devices)
        
        if verbose:
            print(f"Evaluating {len(configurations)} configurations...\n")
        
        # Evaluate all configurations
        all_results = []
        for i, (dp_size, tp_size) in enumerate(configurations):
            if verbose:
                print(f"[{i+1}/{len(configurations)}] Evaluating DP={dp_size}, TP={tp_size}...", end=" ")
            
            result = self.evaluate(dp_size, tp_size)
            all_results.append(result)
            
            if verbose:
                if result.get('valid', False):
                    if metric == "throughput":
                        print(f"Throughput: {result['throughput']:.6f} 1/s")
                    elif metric == "latency":
                        print(f"Latency: {result['latency']:.6f} s")
                    else:
                        print(f"Latency/sample: {result['latency_per_sample']:.9f} s")
                else:
                    print("INVALID")
        
        elapsed_time = time.time() - start_time
        
        # Find best configuration
        best_dp, best_tp, best_result = find_best_configuration(
            system=self.system,
            total_devices=total_devices,
            d_model=self.d_model,
            n_heads=self.n_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            data_type=self.data_type,
            compile_mode=self.compile_mode,
            metric=metric,
        )
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Search completed in {elapsed_time:.2f} seconds")
            print(f"{'=' * 80}")
            print(f"\nBest configuration:")
            print(f"  DP size: {best_dp}")
            print(f"  TP size: {best_tp}")
            print(f"  Latency: {best_result['latency']:.6f} seconds")
            print(f"  Throughput: {best_result['throughput']:.6f} samples/second")
            print(f"  Latency per sample: {best_result['latency_per_sample']:.9f} seconds")
            print(f"{'=' * 80}\n")
            
            print_configuration_summary(all_results, metric)
        
        return (best_dp, best_tp, best_result, all_results)
    
    def search_multiple_device_counts(
        self,
        device_counts: List[int],
        metric: str = "throughput",
        output_file: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[int, Tuple[int, int, Dict[str, float]]]:
        """
        Search for best configuration across multiple device counts.
        
        Args:
            device_counts: List of device counts to search
            metric: Metric to optimize
            output_file: Optional file to save results (JSON)
            verbose: Whether to print progress
        
        Returns:
            Dictionary mapping device_count -> (best_dp, best_tp, best_result)
        """
        results = {}
        
        for device_count in device_counts:
            if verbose:
                print(f"\n{'#' * 80}")
                print(f"Searching for {device_count} devices...")
                print(f"{'#' * 80}")
            
            best_dp, best_tp, best_result, _ = self.search(
                total_devices=device_count,
                metric=metric,
                verbose=verbose,
            )
            
            results[device_count] = (best_dp, best_tp, best_result)
        
        if output_file:
            # Save results to JSON
            output_data = {}
            for device_count, (dp, tp, result) in results.items():
                output_data[device_count] = {
                    'dp_size': dp,
                    'tp_size': tp,
                    'metrics': result,
                }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            if verbose:
                print(f"\nResults saved to {output_file}")
        
        return results


def main():
    """Example usage of the brute force search."""
    # Load system configuration
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "device_configs",
        "plena_sys_config.json"
    )
    
    if not os.path.exists(spec_path):
        print(f"System config not found at {spec_path}")
        print("Using default system configuration...")
        # You might want to create a default system here
        return
    
    specs = read_architecture_template(spec_path)
    system = template_to_system(specs)
    
    # Create search object
    search = DPTPBruteForceSearch(
        system=system,
        d_model=128,
        n_heads=8,
        batch_size=1,
        seq_len=512,
        data_type=data_type_dict["fp16"],
        compile_mode="heuristic-GPU",
    )
    
    # Search for best configuration with 8 devices
    best_dp, best_tp, best_result, all_results = search.search(
        total_devices=8,
        metric="throughput",
        verbose=True,
    )
    
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT:")
    print(f"  Best configuration: DP={best_dp}, TP={best_tp}")
    print(f"  Throughput: {best_result['throughput']:.6f} samples/second")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

