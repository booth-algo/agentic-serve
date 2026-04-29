"""Train XGBoost residual model for flash attention prediction.

Target: log(measured_us / roofline_us)
Features: log2(seq_len), log2(kv_len), log2(n_heads), log2(head_dim),
          causal, operational_intensity, log(roofline_us)
"""

import csv
import math
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from ..configs.gpu_specs import get_gpu
from ..kernels.roofline import roofline_us
from ..kernels.flash_attn import _flash_attn_flops, _flash_attn_bytes

DATA_DIR = Path(__file__).parent.parent / "data" / "flash_attn"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def load_data(gpu: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f"{gpu}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No flash_attn data for {gpu} at {path}")

    gpu_spec = get_gpu(gpu)
    features, targets = [], []

    with open(path) as f:
        for row in csv.DictReader(f):
            seq_len = int(row["seq_len"])
            n_heads = int(row["n_heads"])
            head_dim = int(row["head_dim"])
            causal = int(row["causal"])
            measured = float(row["latency_us"])
            kv_len = int(row.get("kv_len", seq_len))

            flops = _flash_attn_flops(seq_len, kv_len, n_heads, head_dim)
            bytes_moved = _flash_attn_bytes(seq_len, kv_len, n_heads, head_dim)
            baseline = roofline_us(flops, bytes_moved, gpu_spec)

            if baseline < 1e-6 or measured < 1e-6:
                continue

            oi = flops / bytes_moved if bytes_moved > 0 else 0.0
            features.append([
                math.log2(max(seq_len, 1)),
                math.log2(max(kv_len, 1)),
                math.log2(max(n_heads, 1)),
                math.log2(max(head_dim, 1)),
                float(causal),
                oi,
                math.log(max(baseline, 1e-6)),
            ])
            targets.append(math.log(measured / baseline))

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def train(gpu: str, n_estimators: int = 200, max_depth: int = 6,
          learning_rate: float = 0.1) -> dict:
    X, y = load_data(gpu)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        tree_method="hist",
    )
    model.fit(X, y)

    preds = model.predict(X)
    mae = float(np.mean(np.abs(y - preds)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"flash_{gpu}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    return {"gpu": gpu, "n_samples": len(X), "log_mae": mae,
            "model_path": str(out_path)}


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else "A100"
    result = train(gpu)
    print(f"Trained flash_attn residual model for {gpu}:")
    for k, v in result.items():
        print(f"  {k}: {v}")
