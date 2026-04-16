"""
ML-based kernel latency predictor for LLMCompass.

Trains RandomForest models on GPU profiling data (CSV files) and provides
prediction functions for GEMM, attention (prefill/decode), and elementwise ops.

Inspired by LLMServingSim's profiling approach, clean implementation.
"""

import argparse
import hashlib
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Derived-feature helpers
# ---------------------------------------------------------------------------

def _add_derived_features_gemm(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for GEMM profiling data."""
    df = df.copy()
    df["log_mnk"] = np.log2(df["M"] * df["N"] * df["K"] + 1)
    df["is_compute_bound"] = (df["M"] * df["N"] * df["K"] > 1e9).astype(int)
    df["aspect_ratio"] = df["M"] / (df["N"] + 1)
    return df


def _add_derived_features_attn(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Add derived features for attention profiling data.

    phase: 'prefill' or 'decode'
    """
    df = df.copy()
    if phase == "prefill":
        df["total_flops"] = (
            4
            * df["batch_size"]
            * df["n_heads"]
            * df["seq_len"] ** 2
            * df["head_dim"]
        )
    elif phase == "decode":
        # memory-bound: bytes moved for KV cache (fp16 = 2 bytes)
        df["total_bytes"] = (
            2 * df["batch_size"] * df["n_heads"] * df["kv_len"] * df["head_dim"] * 2
        )
    else:
        raise ValueError(f"Unknown attention phase: {phase!r}. Expected 'prefill' or 'decode'.")
    return df


# ---------------------------------------------------------------------------
# MAPE scorer
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (0-100 scale)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _make_mape_scorer():
    return make_scorer(_mape, greater_is_better=False)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _csv_content_hash(path: str) -> str:
    """Return a short SHA256 hex digest of a CSV file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(cache_dir: str, key: str, csv_path: str) -> str:
    """Return the .pkl cache file path, embedding a content hash in the name."""
    content_hash = _csv_content_hash(csv_path)
    return os.path.join(cache_dir, f"{key}_{content_hash}.pkl")


def _load_cache(cache_path: str):
    """Load a pickled model; return None if the file does not exist."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(cache_path: str, model) -> None:
    """Persist a model to a pickle file."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(model, f)


# ---------------------------------------------------------------------------
# CategoryPredictor
# ---------------------------------------------------------------------------

class CategoryPredictor:
    """Manages trained ML models for different GPU kernel types.

    Attributes
    ----------
    profiles_dir : str
        Directory containing CSV profile files:
        - gemm_profiles.csv
        - attn_prefill_profiles.csv
        - attn_decode_profiles.csv
        - elementwise_profiles.csv
    cache_dir : str
        Where to save/load .pkl model files. Defaults to profiles_dir.
    models : dict
        Trained models keyed by 'gemm', 'attn_prefill', 'attn_decode',
        'elementwise'.
    """

    def __init__(self, profiles_dir: str, cache_dir: Optional[str] = None):
        self.profiles_dir = profiles_dir
        self.cache_dir = cache_dir or profiles_dir
        self.models: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_all(self, force_retrain: bool = False) -> None:
        """Train all models. Loads from cache when available."""
        self.train_gemm_model(force_retrain=force_retrain)
        self.train_attention_model("prefill", force_retrain=force_retrain)
        self.train_attention_model("decode", force_retrain=force_retrain)
        self.train_elementwise_model(force_retrain=force_retrain)

    def train_gemm_model(self, force_retrain: bool = False) -> RandomForestRegressor:
        """Train GEMM latency predictor and return the fitted model.

        Features: M, N, K, log2(M*N*K), is_compute_bound, aspect_ratio
        Target:   latency_ns
        """
        csv_path = os.path.join(self.profiles_dir, "gemm_profiles.csv")
        cache_key = "gemm"
        pkl_path = _cache_path(self.cache_dir, cache_key, csv_path)

        if not force_retrain:
            model = _load_cache(pkl_path)
            if model is not None:
                print(f"[CategoryPredictor] Loaded GEMM model from cache: {pkl_path}")
                self.models["gemm"] = model
                return model

        df = pd.read_csv(csv_path)
        df = df[df["latency_ns"] > 0].copy()
        df = _add_derived_features_gemm(df)

        feature_cols = ["M", "N", "K", "log_mnk", "is_compute_bound", "aspect_ratio"]
        X = df[feature_cols].values
        y = df["latency_ns"].values

        print(f"[CategoryPredictor] Training GEMM model on {len(X)} datapoints...")

        param_grid = {
            "n_estimators": [250, 500],
            "max_depth": [16, 24, 32],
            "min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring=_make_mape_scorer(),
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)
        model = gs.best_estimator_
        print(f"[CategoryPredictor] GEMM best params: {gs.best_params_}")

        _save_cache(pkl_path, model)
        self.models["gemm"] = model
        return model

    def train_attention_model(
        self, phase: str, force_retrain: bool = False
    ) -> RandomForestRegressor:
        """Train attention latency predictor (separate models for prefill/decode).

        Prefill features: batch_size, seq_len, n_heads, head_dim, total_flops
        Decode features:  batch_size, kv_len,  n_heads, head_dim, total_bytes
        Target: latency_ns
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")

        csv_name = f"attn_{phase}_profiles.csv"
        csv_path = os.path.join(self.profiles_dir, csv_name)
        cache_key = f"attn_{phase}"
        pkl_path = _cache_path(self.cache_dir, cache_key, csv_path)

        if not force_retrain:
            model = _load_cache(pkl_path)
            if model is not None:
                print(f"[CategoryPredictor] Loaded attn_{phase} model from cache: {pkl_path}")
                self.models[cache_key] = model
                return model

        df = pd.read_csv(csv_path)
        df = df[df["latency_ns"] > 0].copy()
        df = _add_derived_features_attn(df, phase)

        if phase == "prefill":
            feature_cols = ["batch_size", "seq_len", "n_heads", "head_dim", "total_flops"]
        else:
            feature_cols = ["batch_size", "kv_len", "n_heads", "head_dim", "total_bytes"]

        X = df[feature_cols].values
        y = df["latency_ns"].values

        print(
            f"[CategoryPredictor] Training attn_{phase} model on {len(X)} datapoints..."
        )

        param_grid = {
            "n_estimators": [250, 500],
            "max_depth": [16, 24, 32],
            "min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring=_make_mape_scorer(),
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)
        model = gs.best_estimator_
        print(f"[CategoryPredictor] attn_{phase} best params: {gs.best_params_}")

        _save_cache(pkl_path, model)
        self.models[cache_key] = model
        return model

    def train_elementwise_model(self, force_retrain: bool = False) -> RandomForestRegressor:
        """Train elementwise op latency predictor.

        Features: numel, op_type_encoded (ordinal)
        Target:   latency_ns
        """
        csv_path = os.path.join(self.profiles_dir, "elementwise_profiles.csv")
        cache_key = "elementwise"
        pkl_path = _cache_path(self.cache_dir, cache_key, csv_path)

        if not force_retrain:
            model = _load_cache(pkl_path)
            if model is not None:
                print(f"[CategoryPredictor] Loaded elementwise model from cache: {pkl_path}")
                self.models["elementwise"] = model
                return model

        df = pd.read_csv(csv_path)
        df = df[df["latency_ns"] > 0].copy()

        # Ordinal-encode op_type
        op_types = sorted(df["op_type"].unique())
        self._elementwise_op_types = op_types  # save for inference
        op_type_map = {op: i for i, op in enumerate(op_types)}
        df["op_type_encoded"] = df["op_type"].map(op_type_map)

        feature_cols = ["numel", "op_type_encoded"]
        X = df[feature_cols].values
        y = df["latency_ns"].values

        print(f"[CategoryPredictor] Training elementwise model on {len(X)} datapoints...")

        param_grid = {
            "n_estimators": [250, 500],
            "max_depth": [16, 24, 32],
            "min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring=_make_mape_scorer(),
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)
        model = gs.best_estimator_
        # Stash the op-type mapping inside the model object for portability
        model._elementwise_op_types = op_types
        print(f"[CategoryPredictor] elementwise best params: {gs.best_params_}")

        _save_cache(pkl_path, model)
        self.models["elementwise"] = model
        return model

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def predict_gemm(self, M: int, N: int, K: int) -> float:
        """Predict GEMM latency in seconds. Returns -1 if model not trained."""
        model = self.models.get("gemm")
        if model is None:
            return -1.0

        mnk = M * N * K
        features = np.array(
            [[M, N, K, np.log2(mnk + 1), int(mnk > 1e9), M / (N + 1)]]
        )
        latency_ns = model.predict(features)[0]
        return latency_ns / 1e9

    def predict_attention_prefill(
        self,
        batch: int,
        seq_len: int,
        n_heads: int,
        head_dim: int = 128,
    ) -> float:
        """Predict prefill attention latency in seconds. Returns -1 if model not trained."""
        model = self.models.get("attn_prefill")
        if model is None:
            return -1.0

        total_flops = 4 * batch * n_heads * seq_len**2 * head_dim
        features = np.array([[batch, seq_len, n_heads, head_dim, total_flops]])
        latency_ns = model.predict(features)[0]
        return latency_ns / 1e9

    def predict_attention_decode(
        self,
        batch: int,
        kv_len: int,
        n_heads: int,
        head_dim: int = 128,
    ) -> float:
        """Predict decode attention latency in seconds. Returns -1 if model not trained."""
        model = self.models.get("attn_decode")
        if model is None:
            return -1.0

        total_bytes = 2 * batch * n_heads * kv_len * head_dim * 2
        features = np.array([[batch, kv_len, n_heads, head_dim, total_bytes]])
        latency_ns = model.predict(features)[0]
        return latency_ns / 1e9

    def predict_elementwise(self, op_type: str, numel: int) -> float:
        """Predict elementwise op latency in seconds.

        op_type: 'rmsnorm', 'silu', 'residual_add'
        Returns -1 if model not trained or op_type unseen.
        """
        model = self.models.get("elementwise")
        if model is None:
            return -1.0

        op_types = getattr(model, "_elementwise_op_types", [])
        if op_type not in op_types:
            # Unknown op_type: use ordinal -1 (out-of-distribution, best effort)
            op_encoded = -1
        else:
            op_encoded = op_types.index(op_type)

        features = np.array([[numel, op_encoded]])
        latency_ns = model.predict(features)[0]
        return latency_ns / 1e9

    def is_trained(self) -> bool:
        """Returns True if all four models are trained/loaded."""
        required = {"gemm", "attn_prefill", "attn_decode", "elementwise"}
        return required.issubset(self.models)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def mape(y_true, y_pred) -> float:
        """Mean Absolute Percentage Error."""
        return _mape(y_true, y_pred)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RandomForest kernel-latency predictors on GPU profiling CSVs."
    )
    parser.add_argument(
        "--profiles-dir",
        required=True,
        help="Directory containing gemm_profiles.csv, attn_prefill_profiles.csv, etc.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached .pkl files and retrain from scratch.",
    )
    args = parser.parse_args()

    predictor = CategoryPredictor(profiles_dir=args.profiles_dir)
    predictor.train_all(force_retrain=args.force_retrain)

    # Accuracy report on held-out 20% test set
    print("\n" + "=" * 60)
    print("Accuracy Report (MAPE on held-out 20% test set)")
    print("=" * 60)

    kernel_specs = [
        {
            "key": "gemm",
            "csv": "gemm_profiles.csv",
            "feature_fn": lambda df: _add_derived_features_gemm(df)[
                ["M", "N", "K", "log_mnk", "is_compute_bound", "aspect_ratio"]
            ].values,
        },
        {
            "key": "attn_prefill",
            "csv": "attn_prefill_profiles.csv",
            "feature_fn": lambda df: _add_derived_features_attn(df, "prefill")[
                ["batch_size", "seq_len", "n_heads", "head_dim", "total_flops"]
            ].values,
        },
        {
            "key": "attn_decode",
            "csv": "attn_decode_profiles.csv",
            "feature_fn": lambda df: _add_derived_features_attn(df, "decode")[
                ["batch_size", "kv_len", "n_heads", "head_dim", "total_bytes"]
            ].values,
        },
        {
            "key": "elementwise",
            "csv": "elementwise_profiles.csv",
            "feature_fn": None,  # handled separately due to op_type encoding
        },
    ]

    for spec in kernel_specs:
        key = spec["key"]
        model = predictor.models.get(key)
        if model is None:
            print(f"  {key:<20} SKIPPED (model not trained)")
            continue

        csv_path = os.path.join(args.profiles_dir, spec["csv"])
        if not os.path.exists(csv_path):
            print(f"  {key:<20} SKIPPED (CSV not found: {csv_path})")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["latency_ns"] > 0].copy()

        if key == "elementwise":
            op_types = getattr(model, "_elementwise_op_types", [])
            op_type_map = {op: i for i, op in enumerate(op_types)}
            df["op_type_encoded"] = df["op_type"].map(op_type_map).fillna(-1).astype(int)
            X = df[["numel", "op_type_encoded"]].values
        else:
            X = spec["feature_fn"](df)

        y = df["latency_ns"].values

        if len(X) < 5:
            print(f"  {key:<20} SKIPPED (too few samples: {len(X)})")
            continue

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_pred = model.predict(X_test)
        mape_val = _mape(y_test, y_pred)
        print(f"  {key:<20} MAPE = {mape_val:.2f}%  (n_test={len(y_test)})")

    print("=" * 60)
    print(f"Models saved to: {predictor.cache_dir}")


if __name__ == "__main__":
    main()
