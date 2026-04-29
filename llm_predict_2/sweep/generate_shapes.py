"""Generate canonical GEMM serving shapes from model configs.

For each model, enumerates all GEMM ops per layer (QKV, O, FFN, LM head)
with their (N, K) pairs. Then crosses each with a dense M grid to produce
the full sweep manifest.

Output: data/gemm/serving_shapes.csv
"""

import csv
from pathlib import Path

from ..configs.model_configs import MODEL_CONFIGS, ModelConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "gemm"

M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def enumerate_nk_pairs(cfg: ModelConfig) -> list[tuple[int, int, str]]:
    h = cfg.hidden_dim
    nh = cfg.n_heads
    nkv = cfg.n_kv_heads
    hd = cfg.head_dim
    ffn = cfg.intermediate_size
    vocab = cfg.vocab_size

    pairs = [
        (nh * hd, h, "q_proj"),
        (nkv * hd, h, "k_proj"),
        (nkv * hd, h, "v_proj"),
        (h, nh * hd, "o_proj"),
        (ffn, h, "gate_proj"),
        (ffn, h, "up_proj"),
        (h, ffn, "down_proj"),
        (vocab, h, "lm_head"),
    ]
    return pairs


def generate_shapes() -> list[dict]:
    all_nk: dict[tuple[int, int], set[str]] = {}

    for name, cfg in MODEL_CONFIGS.items():
        for N, K, op in enumerate_nk_pairs(cfg):
            key = (N, K)
            if key not in all_nk:
                all_nk[key] = set()
            all_nk[key].add(f"{name}/{op}")

    rows = []
    for (N, K) in sorted(all_nk):
        for M in M_GRID:
            rows.append({"M": M, "N": N, "K": K})

    return rows


def write_shapes(path: Path | None = None) -> Path:
    if path is None:
        path = DATA_DIR / "serving_shapes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_shapes()
    unique_nk = len({(r["N"], r["K"]) for r in rows})

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} shapes ({unique_nk} unique (N,K) x {len(M_GRID)} M values) to {path}")
    return path


if __name__ == "__main__":
    write_shapes()
