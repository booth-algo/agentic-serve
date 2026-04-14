"""Simple M/D/1 queuing model for request waiting time prediction."""


class QueuingModel:
    """Model request queuing and waiting times using M/D/1 queue theory.

    M/D/1: Markovian (Poisson) arrivals, Deterministic service time, 1 server.
    """

    def __init__(self, service_rate: float, arrival_rate: float):
        """Initialize the queuing model.

        Args:
            service_rate: Requests that can be served per second (1 / avg_service_time).
            arrival_rate: Requests arriving per second.
        """
        if service_rate <= 0:
            raise ValueError("service_rate must be positive")
        if arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")

        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.rho = arrival_rate / service_rate  # utilization

    def avg_queue_length(self) -> float:
        """Expected queue length E[Lq] = rho^2 / (2 * (1 - rho)).

        Returns 0 if arrival_rate is 0. Returns float('inf') if system is
        at or over capacity (rho >= 1).
        """
        if self.arrival_rate == 0:
            return 0.0
        if self.rho >= 1.0:
            return float("inf")
        return self.rho ** 2 / (2.0 * (1.0 - self.rho))

    def avg_wait_time(self) -> float:
        """Expected wait time in queue E[Wq] = E[Lq] / arrival_rate.

        Returns 0 if arrival_rate is 0. Returns float('inf') if system is
        at or over capacity.
        """
        if self.arrival_rate == 0:
            return 0.0
        lq = self.avg_queue_length()
        if lq == float("inf"):
            return float("inf")
        return lq / self.arrival_rate

    def predict_ttft_with_queuing(self, base_ttft: float, arrival_rate: float = None) -> float:
        """Predicted TTFT including queuing delay.

        Args:
            base_ttft: Base TTFT in seconds (no queuing).
            arrival_rate: Override arrival rate (uses stored value if None).

        Returns:
            TTFT = base_ttft + avg_wait_time (seconds).
        """
        if arrival_rate is not None and arrival_rate != self.arrival_rate:
            # Recompute with different arrival rate
            model = QueuingModel(self.service_rate, arrival_rate)
            return base_ttft + model.avg_wait_time()
        return base_ttft + self.avg_wait_time()
