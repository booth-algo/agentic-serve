"""
Per-op XGBoost predictor for transformer layer latency.

Loads a pre-trained XGBoost model from the profiling data directory and
provides the PerOpPredictor class for inference.
"""

import os
import pickle

_perop_predictor_cache = None


def get_perop_predictor():
    """Lazy-load the per-op XGBoost predictor singleton."""
    global _perop_predictor_cache
    if _perop_predictor_cache is not None:
        return _perop_predictor_cache
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkl_path = os.path.join(base, 'profiling', 'data', 'A100', 'trained', 'perop_analytical_v4.pkl')
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    _perop_predictor_cache = data['model']
    return _perop_predictor_cache


class PerOpPredictor:
    """Wraps the per-op XGBoost model for transformer layer latency prediction."""

    def __init__(self):
        self._model = None

    def load(self):
        self._model = get_perop_predictor()
        return self._model is not None

    def is_loaded(self):
        return self._model is not None

    def predict(self, features):
        """Predict latency in microseconds for a single op feature vector."""
        import numpy as np
        if self._model is None:
            return None
        return self._model.predict(np.array([features]))[0]
