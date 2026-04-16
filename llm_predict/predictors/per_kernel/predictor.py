"""
Shape-only per-kernel latency predictor.

One XGBoost model per kernel family (gemm, flash_attn, elementwise, misc),
trained on ncu ground-truth latency with strictly shape-derivable features.
No runtime counters (cycles, throughput %) in the feature set.

Distinction from CategoryPredictor in per_category/:
  - CategoryPredictor: RandomForest trained on isolated kernel benchmarks
    timed via torch.cuda.Event (includes launch overhead)
  - PerKernelPredictor: XGBoost trained on kernels profiled via ncu during
    real model forward passes (pure kernel time, fuses matched to real
    cuBLAS dispatch behavior)

See also: llm_predict/predictors/per_kernel/README.md

Training artifacts: /Users/kev/gated/phase_a_results/phase_a3_validation_report.md
Paper: Section 4.4 of NeurIPS 2026 E&D submission.
"""

import math
import os
import pickle
from typing import Optional

import numpy as np


FAMILIES = ('gemm', 'flash_attn', 'elementwise', 'misc')
_PKL_PATTERN = 'perkernel_{family}_shape_v2.pkl'


class PerKernelPredictor:
    """Manages shape-only XGBoost predictors for four kernel families.

    Usage:
        p = PerKernelPredictor(gpu='A100')
        if p.load():
            t_ms = p.predict_gemm(M=512, N=4096, K=4096)
            t_ms = p.predict_attention_prefill(bs=1, seq=512, n_heads=32, head_dim=128)
            t_ms = p.predict_elementwise('rmsnorm', numel=512*4096)

    All predict_* methods return latency in milliseconds, or -1.0 if the
    model for that family is not loaded (e.g., missing pkl on disk).
    """

    def __init__(self, gpu: str = 'A100'):
        self.gpu = gpu
        self.models: dict = {}
        self.feature_cols: dict = {}
        self.metadata: dict = {}

    def _pkl_dir(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base, 'profiling', 'data', self.gpu, 'trained', 'per_kernel')

    def load(self) -> bool:
        """Load pkls from profiling/data/{gpu}/trained/per_kernel/. Returns True if any loaded."""
        pkl_dir = self._pkl_dir()
        if not os.path.isdir(pkl_dir):
            return False
        for family in FAMILIES:
            pkl_path = os.path.join(pkl_dir, _PKL_PATTERN.format(family=family))
            if not os.path.isfile(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.models[family] = data['model']
            self.feature_cols[family] = list(data['feature_cols'])
            self.metadata[family] = {
                'heldout_mape': data.get('heldout_mape'),
                'n_training': data.get('n_training'),
                'version': data.get('version'),
                'target': data.get('target', 'log_gpu_time_duration_ms'),
            }
        return len(self.models) > 0

    def is_loaded(self) -> bool:
        return len(self.models) > 0

    def families_loaded(self) -> list:
        return sorted(self.models.keys())

    # ------------------------------------------------------------------
    # Public predict methods (all return ms)
    # ------------------------------------------------------------------

    def predict_gemm(self, M: int, N: int, K: int, dtype: str = 'bf16') -> float:
        """Predict GEMM latency in milliseconds from shape.

        Args:
            M, N, K: GEMM dimensions (C[M,N] = A[M,K] @ B[K,N])
            dtype: element dtype ('bf16' or 'fp16' for 2 bytes; 'fp32' for 4)

        Returns:
            Latency in ms, or -1.0 if gemm model not loaded.
        """
        if 'gemm' not in self.models:
            return -1.0
        dtype_bytes = 2 if dtype in ('bf16', 'fp16') else 4
        M, N, K = float(M), float(N), float(K)
        flops = 2.0 * M * N * K
        bytes_val = (M * K + N * K + M * N) * dtype_bytes
        ai = flops / bytes_val if bytes_val > 0 else 0.0
        feats = {
            'M': M, 'N': N, 'K': K,
            'log_M': math.log1p(M), 'log_N': math.log1p(N), 'log_K': math.log1p(K),
            'analytical_flops': flops,
            'analytical_bytes': bytes_val,
            'analytical_ai': ai,
            'log_flops': math.log1p(flops),
            'log_bytes': math.log1p(bytes_val),
            'dtype_onehot_bf16': 1.0 if dtype == 'bf16' else 0.0,
        }
        return self._predict('gemm', feats)

    def predict_attention_prefill(self, bs: int, seq: int, n_heads: int,
                                    head_dim: int = 128,
                                    kv_heads: Optional[int] = None) -> float:
        """Predict FlashAttention prefill latency in ms.

        Training coverage is thin (16 kernels, Llama-8B + Mixtral only at
        bs=1, seq=128) so extrapolation to other GQA configs is unreliable;
        see phase_a_results/phase_a3_validation_report.md for caveats.
        """
        if 'flash_attn' not in self.models:
            return -1.0
        if kv_heads is None:
            kv_heads = n_heads
        bs_f, seq_f, h_f, hd_f = float(bs), float(seq), float(n_heads), float(head_dim)
        total_flops = 4.0 * bs_f * h_f * seq_f * seq_f * hd_f
        feats = {
            'bs': bs_f, 'seq': seq_f,
            'n_heads': h_f, 'head_dim': hd_f, 'kv_heads': float(kv_heads),
            'total_flops': total_flops,
            'log_seq': math.log1p(seq_f),
            'log_heads': math.log1p(h_f),
        }
        return self._predict('flash_attn', feats)

    def predict_elementwise(self, op_type: str, numel: int) -> float:
        """Predict elementwise (rmsnorm/silu/mul/residual/rope/neg/fill/compare/other) latency in ms."""
        if 'elementwise' not in self.models:
            return -1.0
        numel_f = float(numel)
        feats = {
            'numel': numel_f,
            'log_numel': math.log1p(numel_f),
        }
        # One-hot op_type (must match training: rmsnorm, silu, mul, residual, rope, neg, fill, compare, other)
        known_ops = ('rmsnorm', 'silu', 'mul', 'residual', 'rope', 'neg', 'fill', 'compare', 'other')
        op_key = op_type if op_type in known_ops else 'other'
        for op in known_ops:
            feats[f'op_type_onehot_{op}'] = 1.0 if op == op_key else 0.0
        return self._predict('elementwise', feats)

    def predict_misc(self, family: str, numel: int = 0,
                       M: int = 0, N: int = 0, K: int = 0) -> float:
        """Predict misc-kernel latency in ms.

        family: 'reduce' | 'splitk_reduce' | 'cast' | 'copy'
        numel: for reduce/cast/copy (element count)
        M, N, K: for splitk_reduce (3D shape)
        """
        if 'misc' not in self.models:
            return -1.0
        numel_f = float(numel) if numel > 0 else float(max(M * N, M * K, N * K, 1))
        Mf, Nf, Kf = float(M), float(N), float(K)
        feats = {
            'numel_or_shape_total': numel_f,
            'log_numel_or_shape': math.log1p(numel_f),
            'size_m': Mf, 'size_n': Nf, 'size_k': Kf,
            'log_size_m': math.log1p(Mf) if Mf > 0 else 0.0,
            'log_size_n': math.log1p(Nf) if Nf > 0 else 0.0,
            'log_size_k': math.log1p(Kf) if Kf > 0 else 0.0,
        }
        known_families = ('reduce', 'splitk_reduce', 'cast', 'copy')
        for fam in known_families:
            feats[f'kernel_family_onehot_{fam}'] = 1.0 if fam == family else 0.0
        return self._predict('misc', feats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(self, family: str, feats: dict) -> float:
        """Run the XGB model for `family`, returning latency in ms.
        Target is log_gpu_time_duration_ms, so we exp the output."""
        cols = self.feature_cols[family]
        X = np.array([[feats.get(c, 0.0) for c in cols]], dtype=float)
        log_pred = self.models[family].predict(X)[0]
        return float(np.exp(log_pred))


# --------------------------------------------------------------------------
# Module-level singleton accessor (parallels per_op / per_category patterns)
# --------------------------------------------------------------------------

_cache: dict = {}


def get_per_kernel_predictor(gpu: str = 'A100') -> Optional[PerKernelPredictor]:
    """Return cached PerKernelPredictor for `gpu` (loads lazily). None if no pickles found."""
    if gpu in _cache:
        return _cache[gpu]
    p = PerKernelPredictor(gpu=gpu)
    _cache[gpu] = p if p.load() else None
    return _cache[gpu]
