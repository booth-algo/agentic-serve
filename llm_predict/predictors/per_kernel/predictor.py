"""
Shape-only per-kernel latency predictor.

One XGBoost model per kernel family (gemm, flash_attn, elementwise, misc),
trained on ncu ground-truth latency with strictly shape-derivable features.
No runtime counters (cycles, throughput %) in the feature set.

Training data comes from kernels profiled via ncu during real model forward
passes, giving pure kernel time with fuses matched to real cuBLAS dispatch
behavior.

See also: llm_predict/predictors/per_kernel/README.md
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
        # For misc: optional per-subfamily submodels (reduce, splitk_reduce,
        # cast, copy). When present, predict_misc routes by subfamily.
        self.misc_submodels: dict = {}

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
            if family == 'misc' and isinstance(data.get('models'), dict):
                # Per-subfamily misc predictors override the single `model`.
                self.misc_submodels = dict(data['models'])
        # Phase 2: optional decode_attn pkl (trained from per-op decode data).
        da_path = os.path.join(pkl_dir, 'perkernel_decode_attn_shape_v2.pkl')
        if os.path.isfile(da_path):
            with open(da_path, 'rb') as f:
                da = pickle.load(f)
            self.models['decode_attn'] = da['model']
            self.feature_cols['decode_attn'] = list(da['feature_cols'])
            self.metadata['decode_attn'] = {
                'n_training': da.get('n_training'),
                'train_mape': da.get('train_mape'),
                'version': da.get('version'),
                'target': da.get('target', 'log_duration_us'),
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

    # Per-GPU HBM bandwidth in GB/s. Used by the Phase-1 decode-attention
    # approximation; Phase 2 retrains a learned decode pkl that supersedes this.
    # Sources: NVIDIA published specs, dtype-agnostic peak.
    _HBM_BW_GBPS: dict = {
        'A100': 1555.0,        # A100-40GB SXM4
        'A100-80GB': 2039.0,
        'RTX3090': 936.0,
        'RTX2080Ti': 616.0,
        'H100': 3350.0,        # H100-80GB SXM5
    }

    def predict_attention_decode(self, bs: int, kv_cache_len: int,
                                  n_heads: int, head_dim: int = 128,
                                  kv_heads: Optional[int] = None,
                                  dtype: str = 'bf16',
                                  efficiency: float = 0.65) -> float:
        """Predict ONE-step decode attention latency in ms (Phase 1 stub).

        Uses memory-bandwidth approximation: at decode step t, attn reads the
        full K and V cache (size bs * kv_cache_len * kv_heads * head_dim * 2
        bytes for bf16, doubled for K + V). Latency = bytes / (HBM_BW * eff)
        where eff is achieved fraction of peak (typically 60-70% for FlashAttn
        decode kernels).

        Phase 2 replaces this with a trained pkl matching the prefill
        flash_attn family but with `phase_onehot_decode=1` and `kv_cache_len`
        as features. Until then this returns a deterministic, kv-linear
        approximation that is correct in shape (linear in kv_cache_len) so
        the trapezoidal integration in serving_e2e.py is unbiased.

        Returns ms, or -1.0 if no model loaded and GPU not in bandwidth table.
        """
        if kv_heads is None:
            kv_heads = n_heads
        dtype_bytes = 2 if dtype in ('bf16', 'fp16') else 4

        # Learned model path (Phase 2): use trained decode_attn pkl.
        if 'decode_attn' in self.models:
            kv_bytes = 2.0 * bs * kv_cache_len * kv_heads * head_dim * dtype_bytes
            feats = {
                'bs': float(bs),
                'log_bs': math.log2(bs + 1),
                'kv_cache_len': float(kv_cache_len),
                'log_kv_cache_len': math.log2(kv_cache_len + 1),
                'n_heads': float(n_heads),
                'kv_heads': float(kv_heads),
                'head_dim': float(head_dim),
                'kv_bytes': kv_bytes,
                'log_kv_bytes': math.log2(kv_bytes + 1),
            }
            cols = self.feature_cols['decode_attn']
            X = np.array([[feats.get(c, 0.0) for c in cols]], dtype=float)
            log_pred = self.models['decode_attn'].predict(X)[0]
            # Target is log(duration_us); convert to ms.
            return float(np.exp(log_pred)) / 1000.0

        # Fallback: analytical bandwidth approximation.
        bw_gbps = self._HBM_BW_GBPS.get(self.gpu)
        if bw_gbps is None:
            return -1.0
        bytes_kv = 2.0 * bs * kv_cache_len * kv_heads * head_dim * dtype_bytes
        return bytes_kv / (bw_gbps * efficiency * 1e9) * 1000.0

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
        if 'misc' not in self.models and not self.misc_submodels:
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
        # Submodels trained on per-subfamily splits exist in pkl but have
        # fewer samples each and ended up worse on composition MAPE than the
        # monolithic model (2026-04-17 iter 6 regression). Always use the
        # monolithic model; keep `misc_submodels` loaded for future
        # experimentation.
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
# Module-level singleton accessor (parallels per_op pattern)
# --------------------------------------------------------------------------

_cache: dict = {}


def get_per_kernel_predictor(gpu: str = 'A100') -> Optional[PerKernelPredictor]:
    """Return cached PerKernelPredictor for `gpu` (loads lazily). None if no pickles found."""
    if gpu in _cache:
        return _cache[gpu]
    p = PerKernelPredictor(gpu=gpu)
    _cache[gpu] = p if p.load() else None
    return _cache[gpu]
