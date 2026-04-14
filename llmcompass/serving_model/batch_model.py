"""Batching model that predicts throughput and latency under different batch sizes
using LLMCompass per-layer analytical predictions."""

import math

from llmcompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
    pp_bubble_fraction,
)
from llmcompass.software_model.utils import Tensor, data_type_dict
from llmcompass.hardware_model.system import System


class BatchingModel:
    """Predict throughput and latency under different batch sizes using
    LLMCompass per-layer predictions."""

    def __init__(
        self,
        system: System,
        d_model: int,
        n_heads: int,
        n_layers: int,
        data_type,
        intermediate_size: int = None,
        n_kv_heads: int = None,
        device_count: int = 1,
        simulation_mode: str = "roofline",
        pp_size: int = 1,
        num_microbatches: int = None,
    ):
        """Store model config. Create transformer blocks for prefill + decode.

        Args:
            system: LLMCompass System (device + interconnect).
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            data_type: Data type from data_type_dict (e.g. fp16).
            intermediate_size: FFN intermediate size. Defaults to 4 * d_model.
            n_kv_heads: Number of KV heads (for GQA). Currently unused by
                LLMCompass transformer blocks which assume MHA, but stored
                for future extension.
            device_count: Number of TP devices.
            simulation_mode: "roofline" (fast analytical) or "compile" (slow
                but more accurate compile_and_simulate).
            pp_size: Pipeline parallelism degree. Layers are split evenly
                across stages; inter-stage P2P transfer overhead is added.
            num_microbatches: Number of microbatches used to fill the pipeline.
                Defaults to pp_size * 4, which gives good bubble efficiency.
        """
        self.system = system
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.data_type = data_type
        self.intermediate_size = intermediate_size or 4 * d_model
        self.n_kv_heads = n_kv_heads
        self.device_count = device_count
        self.simulation_mode = simulation_mode
        self.pp_size = pp_size
        self.num_microbatches = num_microbatches if num_microbatches is not None else pp_size * 4

        # Number of layers handled by each PP stage
        self.layers_per_stage = math.ceil(n_layers / pp_size)

        # Pre-build one prefill block and one decode block (reused across calls)
        self._prefill_block = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type,
            intermediate_size=self.intermediate_size,
        )
        self._decode_block = TransformerBlockAutoRegressionTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type,
            intermediate_size=self.intermediate_size,
        )

    def _prefill_latency_per_layer(self, batch_size: int, seq_len: int) -> float:
        """Get per-layer prefill latency in seconds for given batch/seq dimensions."""
        X = Tensor([batch_size, seq_len, self.d_model], self.data_type)
        _ = self._prefill_block(X)
        if self.simulation_mode == "compile":
            return self._prefill_block.compile_and_simulate(
                self.system, "heuristic-GPU"
            )
        return self._prefill_block.roofline_model(self.system)

    def _decode_latency_per_layer(self, batch_size: int, seq_len: int) -> float:
        """Get per-layer decode latency in seconds for given batch size and KV cache length."""
        x = Tensor([batch_size, 1, self.d_model], self.data_type)
        _ = self._decode_block(x, seq_len)
        if self.simulation_mode == "compile":
            return self._decode_block.compile_and_simulate(
                self.system, "heuristic-GPU"
            )
        return self._decode_block.roofline_model(self.system)

    def _inter_stage_latency_s(self) -> float:
        """Estimate one-way P2P activation transfer latency between PP stages.

        Transfers a [1, d_model] activation tensor (one token) across the
        inter-device link. Returns 0 when pp_size == 1.

        Uses the interconnect base link latency plus transfer time:
            latency = link.latency + activation_bytes / bandwidth_per_direction
        """
        if self.pp_size <= 1:
            return 0.0
        link = self.system.interconnect.link_module
        # Activation tensor: 1 token * d_model * bytes_per_element
        activation_bytes = self.d_model * self.data_type.word_size
        # Point-to-point bandwidth between adjacent PP stages (devices 0 and 1)
        bw = self.system.interconnect.get_bandwidth(0, 1, "per_direction")
        if bw <= 0:
            return link.latency
        return link.latency + activation_bytes / bw

    def predict_single_request(self, input_len: int, output_len: int) -> dict:
        """Predict TTFT, TPOT, E2E latency for a single request with no batching.

        With PP, layers are split evenly across stages. A single request must
        traverse all stages sequentially so total compute is unchanged, but
        (pp_size - 1) inter-stage P2P transfers are added to both prefill and
        each decode step.

        - TTFT = prefill_per_layer * layers_per_stage * pp_size
                 + inter_stage_latency * (pp_size - 1)
        - TPOT = decode_per_layer * layers_per_stage * pp_size
                 + inter_stage_latency * (pp_size - 1)
        - E2E  = TTFT + TPOT * output_len

        Returns:
            dict with ttft_s, tpot_s, e2e_s, ttft_ms, tpot_ms, e2e_ms
        """
        prefill_per_layer = self._prefill_latency_per_layer(1, input_len)
        # Per-stage compute + inter-stage transfers for one full forward pass
        p2p_overhead_s = self._inter_stage_latency_s() * (self.pp_size - 1)
        ttft_s = prefill_per_layer * self.layers_per_stage * self.pp_size + p2p_overhead_s

        # Decode: single token generation with KV cache of input_len
        decode_per_layer = self._decode_latency_per_layer(1, input_len)
        tpot_s = decode_per_layer * self.layers_per_stage * self.pp_size + p2p_overhead_s

        e2e_s = ttft_s + tpot_s * output_len

        return {
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "e2e_s": e2e_s,
            "ttft_ms": ttft_s * 1000,
            "tpot_ms": tpot_s * 1000,
            "e2e_ms": e2e_s * 1000,
        }

    def predict_batched(
        self, batch_size: int, avg_input_len: int, avg_output_len: int
    ) -> dict:
        """Predict throughput when processing a batch of requests together.

        With PP, each stage processes layers_per_stage layers. Inter-stage P2P
        transfers add latency per pass. Throughput is further reduced by the
        pipeline bubble fraction (idle stages during fill/drain).

        - Prefill: batch_size requests processed together across all stages
        - Decode: batch_size tokens generated per step
        - raw_throughput_tok_s = batch_size / decode_per_step_latency
        - effective_throughput = raw_throughput * (1 - bubble_fraction)

        Returns:
            dict with throughput_tok_s, per_request_tpot_ms, per_request_ttft_ms,
            per_request_e2e_ms, batch_size
        """
        p2p_overhead_s = self._inter_stage_latency_s() * (self.pp_size - 1)

        # Prefill the entire batch at once across all PP stages
        prefill_per_layer = self._prefill_latency_per_layer(batch_size, avg_input_len)
        batch_ttft_s = prefill_per_layer * self.layers_per_stage * self.pp_size + p2p_overhead_s

        # Decode: all batch_size requests generate one token per step
        decode_per_layer = self._decode_latency_per_layer(batch_size, avg_input_len)
        decode_step_s = decode_per_layer * self.layers_per_stage * self.pp_size + p2p_overhead_s

        # Each request sees the same TPOT (one decode step)
        per_request_tpot_s = decode_step_s

        # Raw throughput: batch_size tokens produced per decode step
        raw_throughput_tok_s = batch_size / decode_step_s if decode_step_s > 0 else 0.0

        # Apply PP bubble penalty to sustained throughput
        bubble = pp_bubble_fraction(self.pp_size, self.num_microbatches)
        throughput_tok_s = raw_throughput_tok_s * (1.0 - bubble)

        # Per-request E2E: prefill + output_len decode steps
        per_request_e2e_s = batch_ttft_s + per_request_tpot_s * avg_output_len

        # Request throughput with bubble penalty
        raw_request_throughput = batch_size / per_request_e2e_s if per_request_e2e_s > 0 else 0.0
        request_throughput = raw_request_throughput * (1.0 - bubble)

        return {
            "throughput_tok_s": throughput_tok_s,
            "per_request_tpot_ms": per_request_tpot_s * 1000,
            "per_request_ttft_ms": batch_ttft_s * 1000,
            "per_request_e2e_ms": per_request_e2e_s * 1000,
            "request_throughput": request_throughput,
            "batch_size": batch_size,
        }

    def predict_continuous_batching(
        self,
        arrival_rate: float,
        avg_input_len: int,
        avg_output_len: int,
        max_batch_size: int,
    ) -> dict:
        """Predict steady-state throughput and latency under continuous batching.

        Uses Little's Law: avg_concurrent = arrival_rate * avg_latency
        Iteratively solve: batch_size ~ min(avg_concurrent, max_batch_size)
        Then: throughput = batch_size / per_step_latency

        Args:
            arrival_rate: Requests per second arriving.
            avg_input_len: Average input sequence length.
            avg_output_len: Average output sequence length.
            max_batch_size: Maximum batch size the system supports.

        Returns:
            dict with throughput_tok_s, effective_batch_size, per_request_tpot_ms,
            per_request_ttft_ms, request_throughput, utilization
        """
        # Start with batch_size = 1 and iterate to find steady-state
        batch_size = 1
        for _ in range(20):  # converges quickly
            batched = self.predict_batched(batch_size, avg_input_len, avg_output_len)
            per_request_e2e_s = batched["per_request_e2e_ms"] / 1000

            # Little's Law: average number of concurrent requests
            avg_concurrent = arrival_rate * per_request_e2e_s
            new_batch_size = max(1, min(int(round(avg_concurrent)), max_batch_size))

            if new_batch_size == batch_size:
                break
            batch_size = new_batch_size

        # Final prediction at converged batch size
        final = self.predict_batched(batch_size, avg_input_len, avg_output_len)

        # Max throughput at max_batch_size for utilization calc
        max_result = self.predict_batched(max_batch_size, avg_input_len, avg_output_len)
        max_throughput = max_result["throughput_tok_s"]
        utilization = (
            final["throughput_tok_s"] / max_throughput if max_throughput > 0 else 0.0
        )

        return {
            "throughput_tok_s": final["throughput_tok_s"],
            "effective_batch_size": batch_size,
            "per_request_tpot_ms": final["per_request_tpot_ms"],
            "per_request_ttft_ms": final["per_request_ttft_ms"],
            "per_request_e2e_ms": final["per_request_e2e_ms"],
            "request_throughput": final["request_throughput"],
            "utilization": utilization,
        }
