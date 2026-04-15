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


_bsseq_predictor_cache = None


def get_bsseq_predictor():
    """Lazy-load the bs×seq layer predictor (for BS>1 predictions)."""
    global _bsseq_predictor_cache
    if _bsseq_predictor_cache is not None:
        return _bsseq_predictor_cache
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkl_path = os.path.join(base, 'profiling', 'data', 'A100', 'trained', 'layer_predictor_bsseq.pkl')
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    _bsseq_predictor_cache = data['model']
    return _bsseq_predictor_cache


class PerOpPredictor:
    """Wraps the per-op XGBoost model for transformer layer latency prediction."""

    def __init__(self):
        self._model = None
        self._bsseq_model = None

    def load(self):
        self._model = _get_perop_predictor()
        self._bsseq_model = _get_bsseq_predictor()
        return self._model is not None

    def is_loaded(self):
        return self._model is not None

    def predict(self, features):
        """Predict latency in microseconds for a single op feature vector."""
        import numpy as np
        if self._model is None:
            return None
        return self._model.predict(np.array([features]))[0]

    def predict_bsseq(self, features):
        """Predict latency using the bs×seq model."""
        import numpy as np
        if self._bsseq_model is None:
            return None
        return self._bsseq_model.predict(np.array([features]))[0]
