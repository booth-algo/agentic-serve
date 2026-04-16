"""
PredictorDispatch: tries per-kernel XGBoost first, falls back to per-category
RandomForest, then to analytical roofline.

Tiers from finest to coarsest:
  1. Per-kernel (XGBoost, shape-only, ncu ground truth) — trained in Phase A
     of the NeurIPS 2026 submission; held-out aggregate layer MAPE ~4%
  2. Per-category (RandomForest, isolated kernel benchmarks) — gemm +
     attn_prefill + attn_decode + elementwise RFs per GPU
  3. Analytical roofline — always available, least accurate
"""

import os


class PredictorDispatch:
    """Dispatch latency predictions across available predictor tiers.

    All public methods return latency in seconds.
    """

    def __init__(self, gpu: str = "H100"):
        """
        Args:
            gpu: GPU name for selecting the profiles directory ('A100' or 'H100').
        """
        self.gpu = gpu
        self._per_kernel_predictor = None
        self._category_predictor = None
        self._perop_predictor = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_per_kernel_predictor(self):
        if self._per_kernel_predictor is not None:
            return self._per_kernel_predictor
        from llm_predict.predictors.per_kernel.predictor import PerKernelPredictor
        p = PerKernelPredictor(gpu=self.gpu)
        if p.load():
            self._per_kernel_predictor = p
        return self._per_kernel_predictor

    def _load_category_predictor(self):
        if self._category_predictor is not None:
            return self._category_predictor
        from llm_predict.predictors.per_category.predictor import CategoryPredictor
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profiles_dir = os.path.join(base, 'profiling', 'data', self.gpu)
        if not os.path.isdir(profiles_dir):
            return None
        predictor = CategoryPredictor(
            profiles_dir=profiles_dir,
            cache_dir=os.path.join(profiles_dir, 'trained'),
        )
        try:
            predictor.train_all()
            self._category_predictor = predictor
        except Exception:
            pass
        return self._category_predictor

    def _load_perop_predictor(self):
        if self._perop_predictor is not None:
            return self._perop_predictor
        from llm_predict.predictors.per_op.predictor import PerOpPredictor
        p = PerOpPredictor()
        if p.load():
            self._perop_predictor = p
        return self._perop_predictor

    # ------------------------------------------------------------------
    # Public dispatch methods (all return seconds)
    # ------------------------------------------------------------------

    def predict_gemm(self, M: int, N: int, K: int,
                     hbm_bandwidth: float = None, peak_flops: float = None) -> float:
        """Predict GEMM latency in seconds."""
        # Tier 1: per-kernel (returns ms)
        pk = self._load_per_kernel_predictor()
        if pk is not None:
            result_ms = pk.predict_gemm(M, N, K, dtype='bf16')
            if result_ms >= 0:
                return result_ms / 1e3

        # Tier 2: per-category (returns ns)
        cp = self._load_category_predictor()
        if cp is not None and cp.is_trained():
            result_ns = cp.predict_gemm(M, N, K)
            if result_ns >= 0:
                return result_ns / 1e9

        # Tier 3: Analytical roofline (already seconds)
        if hbm_bandwidth and peak_flops:
            return _roofline_gemm(M, N, K, hbm_bandwidth, peak_flops)
        return -1.0

    def predict_attention_prefill(self, batch: int, seq_len: int,
                                   n_heads: int, head_dim: int = 128,
                                   hbm_bandwidth: float = None,
                                   peak_flops: float = None) -> float:
        """Predict prefill attention latency in seconds."""
        pk = self._load_per_kernel_predictor()
        if pk is not None:
            result_ms = pk.predict_attention_prefill(batch, seq_len, n_heads, head_dim)
            if result_ms >= 0:
                return result_ms / 1e3

        cp = self._load_category_predictor()
        if cp is not None and cp.is_trained():
            result_ns = cp.predict_attention_prefill(batch, seq_len, n_heads, head_dim)
            if result_ns >= 0:
                return result_ns / 1e9

        if hbm_bandwidth and peak_flops:
            return _roofline_attn_prefill(batch, seq_len, n_heads, head_dim,
                                          hbm_bandwidth, peak_flops)
        return -1.0

    def predict_attention_decode(self, batch: int, kv_len: int,
                                  n_heads: int, head_dim: int = 128,
                                  hbm_bandwidth: float = None,
                                  peak_flops: float = None) -> float:
        """Predict decode attention latency in seconds.

        Note: per-kernel predictor has no decode-attention family yet (training
        data was all prefill); falls through to per-category decode RF.
        """
        cp = self._load_category_predictor()
        if cp is not None and cp.is_trained():
            result_ns = cp.predict_attention_decode(batch, kv_len, n_heads, head_dim)
            if result_ns >= 0:
                return result_ns / 1e9

        if hbm_bandwidth and peak_flops:
            return _roofline_attn_decode(batch, kv_len, n_heads, head_dim,
                                         hbm_bandwidth, peak_flops)
        return -1.0

    def predict_elementwise(self, op_type: str, numel: int) -> float:
        """Predict elementwise kernel latency in seconds (per-kernel only for now)."""
        pk = self._load_per_kernel_predictor()
        if pk is not None:
            result_ms = pk.predict_elementwise(op_type, numel)
            if result_ms >= 0:
                return result_ms / 1e3
        return -1.0

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def per_kernel_predictor_available(self) -> bool:
        return self._load_per_kernel_predictor() is not None

    def category_predictor_available(self) -> bool:
        cp = self._load_category_predictor()
        return cp is not None and cp.is_trained()

    def perop_predictor_available(self) -> bool:
        return self._load_perop_predictor() is not None


# ------------------------------------------------------------------
# Analytical roofline helpers (fallback)
# ------------------------------------------------------------------

def _roofline_gemm(M, N, K, hbm_bandwidth, peak_flops):
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 2  # fp16/bf16
    return max(flops / peak_flops, bytes_moved / hbm_bandwidth)


def _roofline_attn_prefill(batch, seq_len, n_heads, head_dim, hbm_bandwidth, peak_flops):
    flops = 4 * batch * n_heads * seq_len * seq_len * head_dim
    mem = (2 * n_heads + 2 * n_heads) * batch * seq_len * head_dim * 2
    return max(flops / peak_flops, mem / hbm_bandwidth)


def _roofline_attn_decode(batch, kv_len, n_heads, head_dim, hbm_bandwidth, peak_flops):
    mem = 2 * batch * n_heads * kv_len * head_dim * 2
    flops = 4 * batch * n_heads * kv_len * head_dim
    return max(flops / peak_flops, mem / hbm_bandwidth)
