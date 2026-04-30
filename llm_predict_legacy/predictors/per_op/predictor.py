"""
Per-op XGBoost predictor for transformer layer latency.

Loads a pre-trained XGBoost model from the profiling data directory and
provides the PerOpPredictor class for inference. Prefers the trained
per-GPU `perop_v5_shape.pkl` and falls back to the legacy A100-only
`perop_analytical_v4.pkl` so callers without a per-GPU pkl still work.
"""

import os
import pickle

_perop_predictor_cache: dict[str, object] = {}


def _resolve_pkl(gpu: str | None) -> str | None:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if gpu:
        v5 = os.path.join(base, 'profiling', 'data', gpu, 'trained', 'per_op', 'perop_v5_shape.pkl')
        if os.path.isfile(v5):
            return v5
    if gpu in (None, 'A100'):
        v4 = os.path.join(base, 'profiling', 'data', 'A100', 'trained', 'perop_analytical_v4.pkl')
        if os.path.isfile(v4):
            return v4
    return None


def get_perop_predictor(gpu: str | None = None):
    """Lazy-load the per-op XGBoost predictor singleton for the given GPU."""
    key = gpu or 'A100'
    cached = _perop_predictor_cache.get(key)
    if cached is not None:
        return cached
    pkl_path = _resolve_pkl(gpu)
    if pkl_path is None:
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[per_op] loaded {pkl_path}")
    model = data['model']
    _perop_predictor_cache[key] = model
    return model


class PerOpPredictor:
    """Wraps the per-op XGBoost model for transformer layer latency prediction."""

    def __init__(self, gpu: str | None = None):
        self.gpu = gpu
        self._model = None

    def load(self):
        self._model = get_perop_predictor(self.gpu)
        return self._model is not None

    def is_loaded(self):
        return self._model is not None

    def predict(self, features):
        """Predict latency in microseconds for a single op feature vector."""
        import numpy as np
        if self._model is None:
            return None
        return self._model.predict(np.array([features]))[0]
