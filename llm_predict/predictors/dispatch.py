"""
PredictorDispatch: tries per-kernel RandomForest first, falls back to
per-op XGBoost, then to analytical roofline.
"""

import os


class PredictorDispatch:
    """Dispatch latency predictions across available predictor tiers.

    Priority:
      1. Per-kernel RandomForest (paper direction — trained on real GPU profiles)
      2. Per-op XGBoost (trained on analytical features)
      3. Analytical roofline (always available, least accurate)
    """

    def __init__(self, gpu: str = "H100"):
        """
        Args:
            gpu: GPU name for selecting the profiles directory ('A100' or 'H100').
        """
        self.gpu = gpu
        self._kernel_predictor = None
        self._perop_predictor = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_kernel_predictor(self):
        if self._kernel_predictor is not None:
            return self._kernel_predictor
        from llm_predict.predictors.per_kernel.predictor import KernelPredictor
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profiles_dir = os.path.join(base, 'profiling', 'data', self.gpu)
        if not os.path.isdir(profiles_dir):
            return None
        predictor = KernelPredictor(
            profiles_dir=profiles_dir,
            cache_dir=os.path.join(profiles_dir, 'trained'),
        )
        try:
            predictor.train_all()
            self._kernel_predictor = predictor
        except Exception:
            pass
        return self._kernel_predictor

    def _load_perop_predictor(self):
        if self._perop_predictor is not None:
            return self._perop_predictor
        from llm_predict.predictors.per_op.predictor import PerOpPredictor
        p = PerOpPredictor()
        if p.load():
            self._perop_predictor = p
        return self._perop_predictor

    # ------------------------------------------------------------------
    # Public dispatch methods
    # ------------------------------------------------------------------

    def predict_gemm(self, M: int, N: int, K: int,
                     hbm_bandwidth: float = None, peak_flops: float = None) -> float:
        """Predict GEMM latency in seconds."""
        kp = self._load_kernel_predictor()
        if kp is not None and kp.is_trained():
            result = kp.predict_gemm(M, N, K)
            if result >= 0:
                return result

        # Analytical roofline fallback
        if hbm_bandwidth and peak_flops:
            return _roofline_gemm(M, N, K, hbm_bandwidth, peak_flops)
        return -1.0

    def predict_attention_prefill(self, batch: int, seq_len: int,
                                   n_heads: int, head_dim: int = 128,
                                   hbm_bandwidth: float = None,
                                   peak_flops: float = None) -> float:
        """Predict prefill attention latency in seconds."""
        kp = self._load_kernel_predictor()
        if kp is not None and kp.is_trained():
            result = kp.predict_attention_prefill(batch, seq_len, n_heads, head_dim)
            if result >= 0:
                return result

        if hbm_bandwidth and peak_flops:
            return _roofline_attn_prefill(batch, seq_len, n_heads, head_dim,
                                          hbm_bandwidth, peak_flops)
        return -1.0

    def predict_attention_decode(self, batch: int, kv_len: int,
                                  n_heads: int, head_dim: int = 128,
                                  hbm_bandwidth: float = None,
                                  peak_flops: float = None) -> float:
        """Predict decode attention latency in seconds."""
        kp = self._load_kernel_predictor()
        if kp is not None and kp.is_trained():
            result = kp.predict_attention_decode(batch, kv_len, n_heads, head_dim)
            if result >= 0:
                return result

        if hbm_bandwidth and peak_flops:
            return _roofline_attn_decode(batch, kv_len, n_heads, head_dim,
                                         hbm_bandwidth, peak_flops)
        return -1.0

    def kernel_predictor_available(self) -> bool:
        kp = self._load_kernel_predictor()
        return kp is not None and kp.is_trained()

    def perop_predictor_available(self) -> bool:
        return self._load_perop_predictor() is not None


# ------------------------------------------------------------------
# Analytical roofline helpers (fallback)
# ------------------------------------------------------------------

def _roofline_gemm(M, N, K, hbm_bandwidth, peak_flops):
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 2  # fp16
    return max(flops / peak_flops, bytes_moved / hbm_bandwidth)


def _roofline_attn_prefill(batch, seq_len, n_heads, head_dim, hbm_bandwidth, peak_flops):
    flops = 4 * batch * n_heads * seq_len * seq_len * head_dim
    mem = (2 * n_heads + 2 * n_heads) * batch * seq_len * head_dim * 2
    return max(flops / peak_flops, mem / hbm_bandwidth)


def _roofline_attn_decode(batch, kv_len, n_heads, head_dim, hbm_bandwidth, peak_flops):
    mem = 2 * batch * n_heads * kv_len * head_dim * 2
    flops = 4 * batch * n_heads * kv_len * head_dim
    return max(flops / peak_flops, mem / hbm_bandwidth)
