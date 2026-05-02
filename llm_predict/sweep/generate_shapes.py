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
FLASH_DATA_DIR = Path(__file__).parent.parent / "data" / "flash_attn"

M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
KV_GRID = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_BATCH_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def enumerate_nk_pairs(cfg: ModelConfig) -> list[tuple[int, int, str]]:
    h = cfg.hidden_dim
    nh = cfg.n_heads
    nkv = cfg.n_kv_heads
    hd = cfg.head_dim
    ffn = cfg.ffn_intermediate_size
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


def generate_attention_shapes() -> list[dict]:
    """Generate canonical flash-attention shapes for prefill/decode regimes."""
    rows: list[dict] = []
    seen: set[tuple[int, int, int, int, int, int, str]] = set()

    def add(q_len: int, kv_len: int, n_heads: int, n_kv_heads: int,
            head_dim: int, batch: int, phase: str) -> None:
        key = (q_len, kv_len, n_heads, n_kv_heads, head_dim, batch, phase)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "q_len": q_len,
            "kv_len": kv_len,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "batch": batch,
            "phase": phase,
        })

    for cfg in MODEL_CONFIGS.values():
        for tp in (1, 2, 4, 8):
            if cfg.n_heads % tp != 0 or cfg.n_kv_heads % tp != 0:
                continue
            n_heads = cfg.n_heads // tp
            n_kv_heads = cfg.n_kv_heads // tp
            for q_len in M_GRID:
                add(q_len, q_len, n_heads, n_kv_heads, cfg.head_dim, 1, "prefill")
            for kv_len in KV_GRID:
                for q_len in (1, 8, 32, 128, 512, 1024):
                    if q_len < kv_len:
                        add(q_len, kv_len, n_heads, n_kv_heads,
                            cfg.head_dim, 1, "cached_prefill")
                for batch in DECODE_BATCH_GRID:
                    add(1, kv_len, n_heads, n_kv_heads, cfg.head_dim,
                        batch, "decode")

    return sorted(
        rows,
        key=lambda row: (
            row["phase"],
            row["n_heads"],
            row["n_kv_heads"],
            row["head_dim"],
            row["q_len"],
            row["kv_len"],
            row["batch"],
        ),
    )


def write_shapes(path: Path | None = None) -> Path:
    if path is None:
        path = DATA_DIR / "serving_shapes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_shapes()
    unique_nk = len({(r["N"], r["K"]) for r in rows})

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} shapes ({unique_nk} unique (N,K) x {len(M_GRID)} M values) to {path}")
    return path


def write_attention_shapes(path: Path | None = None) -> Path:
    if path is None:
        path = FLASH_DATA_DIR / "serving_shapes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_attention_shapes()
    fieldnames = [
        "q_len", "kv_len", "n_heads", "n_kv_heads",
        "head_dim", "batch", "phase",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} attention shapes to {path}")
    return path


if __name__ == "__main__":
    write_shapes()
