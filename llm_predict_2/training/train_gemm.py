"""Train XGBoost residual model for GEMM prediction.

Target: log(measured_us / roofline_us)
Features: log2(M), log2(N), log2(K), operational_intensity, log(roofline_us)
"""

import csv
import math
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from ..configs.gpu_specs import get_gpu
from ..kernels.roofline import gemm_roofline_us

DATA_DIR = Path(__file__).parent.parent / "data" / "gemm"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def load_data(gpu: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f"{gpu}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No GEMM data for {gpu} at {path}")

    gpu_spec = get_gpu(gpu)
    features, targets = [], []

    with open(path) as f:
        for row in csv.DictReader(f):
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            measured = float(row["latency_us"])
            dtype_bytes = int(row.get("dtype_bytes", 2))

            baseline = gemm_roofline_us(M, N, K, gpu_spec, dtype_bytes)
            if baseline < 1e-6 or measured < 1e-6:
                continue

            flops = 2.0 * M * N * K
            bytes_moved = (M * K + K * N + M * N) * dtype_bytes
            oi = flops / bytes_moved if bytes_moved > 0 else 0.0

            features.append([
                math.log2(max(M, 1)),
                math.log2(max(N, 1)),
                math.log2(max(K, 1)),
                oi,
                math.log(max(baseline, 1e-6)),
            ])
            targets.append(math.log(measured / baseline))

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def train(gpu: str, n_estimators: int = 200, max_depth: int = 6,
          learning_rate: float = 0.1) -> dict:
    X, y = load_data(gpu)
    gpu_spec = get_gpu(gpu)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        tree_method="hist",
    )
    model.fit(X, y)

    preds = model.predict(X)
    residuals = y - preds
    mae = float(np.mean(np.abs(residuals)))

    raw_measured = np.exp(y) * np.array([
        gemm_roofline_us(
            int(2**X[i, 0]), int(2**X[i, 1]), int(2**X[i, 2]),
            gpu_spec)
        for i in range(len(X))
    ])
    raw_predicted = np.exp(preds) * np.array([
        gemm_roofline_us(
            int(2**X[i, 0]), int(2**X[i, 1]), int(2**X[i, 2]),
            gpu_spec)
        for i in range(len(X))
    ])
    mape = float(np.mean(np.abs(raw_measured - raw_predicted) / raw_measured) * 100)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"gemm_{gpu}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    return {"gpu": gpu, "n_samples": len(X), "log_mae": mae, "mape_pct": mape,
            "model_path": str(out_path)}


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else "A100"
    result = train(gpu)
    print(f"Trained GEMM residual model for {gpu}:")
    for k, v in result.items():
        print(f"  {k}: {v}")
