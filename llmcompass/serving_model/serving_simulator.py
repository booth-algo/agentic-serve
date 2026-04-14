"""End-to-end serving prediction using LLMCompass analytical models."""

from llmcompass.serving_model.batch_model import BatchingModel
from llmcompass.serving_model.queuing_model import QueuingModel
from llmcompass.design_space_exploration.dse import template_to_system, read_architecture_template
from llmcompass.software_model.transformer import pp_bubble_fraction
from llmcompass.software_model.utils import data_type_dict
from llmcompass.hardware_model.system import System


class ServingSimulator:
    """End-to-end serving prediction using LLMCompass analytical models.

    Combines the BatchingModel (compute prediction) with the QueuingModel
    (waiting time under load) to predict realistic serving metrics.
    """

    def __init__(
        self,
        system: System,
        model_config: dict,
        tp_size: int = 1,
        dp_size: int = 1,
        pp_size: int = 1,
        num_microbatches: int = None,
    ):
        """Initialize the serving simulator.

        Args:
            system: LLMCompass System object (device + interconnect).
            model_config: Dict with model architecture parameters:
                {
                    'd_model': 4096, 'n_heads': 32, 'n_kv_heads': 8,
                    'n_layers': 32, 'intermediate_size': 14336,
                    'num_experts': None, 'top_k': None,
                }
            tp_size: Tensor parallelism degree.
            dp_size: Data parallelism degree (scales request throughput).
            pp_size: Pipeline parallelism degree. Layers are split evenly
                across stages; pipeline bubble overhead reduces throughput.
            num_microbatches: Number of microbatches used to fill the pipeline.
                Defaults to pp_size * 4 inside BatchingModel when None.
        """
        self.system = system
        self.model_config = model_config
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.num_microbatches = num_microbatches

        data_type = model_config.get("data_type", data_type_dict["fp16"])
        if isinstance(data_type, str):
            data_type = data_type_dict[data_type]

        self._batching_model = BatchingModel(
            system=system,
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"],
            data_type=data_type,
            intermediate_size=model_config.get("intermediate_size"),
            n_kv_heads=model_config.get("n_kv_heads"),
            device_count=tp_size,
            pp_size=pp_size,
            num_microbatches=num_microbatches,
        )

    def predict_single_request(
        self,
        batch_size: int,
        seq_len: int,
        output_len: int,
    ) -> dict:
        """Predict latency for a single request (no queuing, no batching).

        Delegates to BatchingModel.predict_single_request with PP overhead
        already factored in (inter-stage P2P transfers, layer-split compute).

        Args:
            batch_size: Number of requests in the batch (pass 1 for a true
                single request; larger values use batched prefill/decode).
            seq_len: Input sequence length.
            output_len: Output sequence length.

        Returns:
            dict with prefill_latency_ms, decode_latency_per_token_ms,
            e2e_latency_ms, and the full set from BatchingModel
            (ttft_ms, tpot_ms, e2e_ms).
        """
        result = self._batching_model.predict_single_request(seq_len, output_len)
        result["prefill_latency_ms"] = result["ttft_ms"]
        result["decode_latency_per_token_ms"] = result["tpot_ms"]
        result["e2e_latency_ms"] = result["e2e_ms"]
        return result

    def simulate(
        self,
        num_requests: int,
        arrival_rate: float,
        avg_input_len: int,
        avg_output_len: int,
        max_batch_size: int = 256,
    ) -> dict:
        """Run a serving simulation and return predicted metrics.

        Args:
            num_requests: Total number of requests (used for context, not
                directly in steady-state calculation).
            arrival_rate: Requests per second.
            avg_input_len: Average input sequence length.
            avg_output_len: Average output sequence length.
            max_batch_size: Maximum batch size for continuous batching.

        Returns:
            dict with:
                ttft_ms: Median TTFT including queuing delay.
                tpot_ms: Median TPOT.
                throughput_tok_s: Steady-state generation throughput.
                request_throughput: Requests per second.
                avg_batch_size: Average batch size under load.
                gpu_utilization: Fraction of peak compute used.
        """
        # Get continuous batching prediction
        cb_result = self._batching_model.predict_continuous_batching(
            arrival_rate=arrival_rate,
            avg_input_len=avg_input_len,
            avg_output_len=avg_output_len,
            max_batch_size=max_batch_size,
        )

        # Compute base TTFT (no queuing) from single request
        single = self._batching_model.predict_single_request(avg_input_len, avg_output_len)
        base_ttft_s = single["ttft_s"]

        # Add queuing delay to TTFT
        service_time_s = single["e2e_s"]
        # Effective service rate accounts for batching speedup and DP replicas
        effective_service_rate = (
            cb_result["request_throughput"] * self.dp_size
            if cb_result["request_throughput"] > 0
            else 1.0 / service_time_s * self.dp_size
        )
        effective_arrival = arrival_rate

        queuing = QueuingModel(
            service_rate=effective_service_rate,
            arrival_rate=effective_arrival,
        )
        ttft_with_queue_s = queuing.predict_ttft_with_queuing(base_ttft_s)

        # Scale throughput by DP
        total_throughput = cb_result["throughput_tok_s"] * self.dp_size
        total_request_throughput = cb_result["request_throughput"] * self.dp_size

        return {
            "ttft_ms": ttft_with_queue_s * 1000,
            "tpot_ms": cb_result["per_request_tpot_ms"],
            "throughput_tok_s": total_throughput,
            "request_throughput": total_request_throughput,
            "avg_batch_size": cb_result["effective_batch_size"],
            "gpu_utilization": cb_result["utilization"],
        }

    def sweep_concurrency(
        self,
        concurrency_levels: list,
        avg_input_len: int,
        avg_output_len: int,
    ) -> list:
        """Run simulation at multiple concurrency levels.

        For each concurrency level, derives an arrival rate assuming
        arrival_rate = concurrency / avg_e2e_latency (iteratively solved
        by the continuous batching model).

        Args:
            concurrency_levels: List of concurrency values to test.
            avg_input_len: Average input sequence length.
            avg_output_len: Average output sequence length.

        Returns:
            List of result dicts, one per concurrency level.
        """
        # Get single-request E2E to bootstrap arrival rate estimate
        single = self._batching_model.predict_single_request(avg_input_len, avg_output_len)
        base_e2e_s = single["e2e_s"]

        results = []
        for conc in concurrency_levels:
            # Estimate arrival rate from concurrency: arrival = conc / e2e
            arrival_rate = conc / base_e2e_s if base_e2e_s > 0 else conc

            result = self.simulate(
                num_requests=conc * 10,  # notional
                arrival_rate=arrival_rate,
                avg_input_len=avg_input_len,
                avg_output_len=avg_output_len,
            )
            result["concurrency"] = conc
            results.append(result)

        return results

    def sweep_batch_sizes(
        self,
        batch_sizes: list,
        avg_input_len: int,
        avg_output_len: int,
    ) -> list:
        """Predict metrics at fixed batch sizes (no queuing).

        Args:
            batch_sizes: List of batch sizes to evaluate.
            avg_input_len: Average input sequence length.
            avg_output_len: Average output sequence length.

        Returns:
            List of result dicts, one per batch size.
        """
        results = []
        for bs in batch_sizes:
            batched = self._batching_model.predict_batched(bs, avg_input_len, avg_output_len)
            results.append(batched)
        return results
