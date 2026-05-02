"""FlashAttention serving-shape sweep under NCU.

Reads llm_predict/data/flash_attn/serving_shapes.csv, deduplicates unique
(q_len, kv_len, n_heads, n_kv_heads, head_dim, batch) tuples, samples
~300 representative shapes, and runs F.scaled_dot_product_attention under
NVTX ranges so NCU can attribute kernel latencies.

Usage (under ncu):
  ncu -o /tmp/flash --force-overwrite \\
      python sweep_flash_serving.py --gpu H100 --dtype bf16 \\
          --out-manifest /tmp/flash.manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import nvtx

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHAPES_CSV = _REPO_ROOT / "llm_predict" / "data" / "flash_attn" / "serving_shapes.csv"
_MAX_SHAPES = 200
_MAX_ALLOC_GB = 12  # skip shapes that would allocate more than this
_SEED = 42


def load_shapes(csv_path: Path) -> list[tuple[int, int, int, int, int, int]]:
    """Load and deduplicate unique (q, kv, h, kv_h, hd, batch) tuples."""
    seen: set[tuple[int, int, int, int, int, int]] = set()
    shapes: list[tuple[int, int, int, int, int, int]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = int(row["q_len"])
            kv = int(row["kv_len"])
            h = int(row["n_heads"])
            kv_h = int(row["n_kv_heads"])
            hd = int(row["head_dim"])
            bs = int(row["batch"])
            key = (q, kv, h, kv_h, hd, bs)
            if key not in seen:
                seen.add(key)
                shapes.append(key)
    return shapes


def sample_shapes(
    shapes: list[tuple[int, int, int, int, int, int]],
    max_n: int = _MAX_SHAPES,
    seed: int = _SEED,
) -> list[tuple[int, int, int, int, int, int]]:
    """Sample representative shapes, ensuring coverage of small/large dims."""
    if len(shapes) <= max_n:
        return shapes

    random.seed(seed)
    # Stratify: ensure we have small-Q (decode), large-Q (prefill), and
    # both small/large KV lengths.
    small_q = [s for s in shapes if s[0] <= 8]
    large_q = [s for s in shapes if s[0] > 8 and s[0] < 512]
    prefill_q = [s for s in shapes if s[0] >= 512]

    n_small = min(len(small_q), max_n // 3)
    n_large = min(len(large_q), max_n // 3)
    n_prefill = min(len(prefill_q), max_n - n_small - n_large)

    sampled = (
        random.sample(small_q, n_small)
        + random.sample(large_q, n_large)
        + random.sample(prefill_q, n_prefill)
    )
    random.shuffle(sampled)
    return sampled


def run_sweep(
    shapes: list[tuple[int, int, int, int, int, int]],
    dtype: torch.dtype,
    manifest_out: Path,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)

    print(f"[*] GPU: {name} sm_{cc[0]}{cc[1]}", flush=True)
    print(f"[*] {len(shapes)} shapes, dtype={dtype}", flush=True)

    manifest: list[dict] = []
    skipped = 0
    for idx, (q_len, kv_len, h, kv_h, hd, bs) in enumerate(shapes):
        if idx % 50 == 0:
            print(f"  {idx}/{len(shapes)}...", flush=True)

        tag = f"flash_q{q_len}_kv{kv_len}_h{h}_kvh{kv_h}_hd{hd}_b{bs}"

        # Memory safety: skip shapes that would allocate too much
        # Q/K/V tensors + expanded KV for GQA + output
        est_bytes = (bs * h * q_len * hd + 2 * bs * kv_h * kv_len * hd) * 2  # fp16
        if kv_h != h:
            est_bytes += 2 * bs * h * kv_len * hd * 2  # GQA-expanded KV (repeated)
        est_gb = est_bytes / (1024**3)
        if est_gb > _MAX_ALLOC_GB:
            skipped += 1
            continue

        nvtx.range_push(tag)

        # Build Q/K/V tensors
        q = torch.randn(bs, h, q_len, hd, dtype=dtype, device="cuda")
        k = torch.randn(bs, kv_h, kv_len, hd, dtype=dtype, device="cuda")
        v = torch.randn(bs, kv_h, kv_len, hd, dtype=dtype, device="cuda")

        # Expand KV heads for GQA if needed
        if kv_h != h:
            k = k.repeat_interleave(h // kv_h, dim=1)
            v = v.repeat_interleave(h // kv_h, dim=1)

        causal = q_len == kv_len and q_len > 1  # full-prefill is causal
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        except RuntimeError as e:
            print(f"  [skip] {tag}: {e}", flush=True)
            nvtx.range_pop()
            del q, k, v
            skipped += 1
            continue

        torch.cuda.synchronize()
        manifest.append({
            "idx": idx,
            "q_len": q_len, "kv_len": kv_len,
            "n_heads": h, "n_kv_heads": kv_h, "head_dim": hd,
            "batch": bs, "causal": causal,
            "dtype": str(dtype).replace("torch.", ""),
            "nvtx_range": tag,
        })
        nvtx.range_pop()
        del q, k, v, out

    torch.cuda.empty_cache()

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump({
            "gpu_name": name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "dtype": str(dtype),
            "n_invocations": len(manifest),
            "skipped": skipped,
            "invocations": manifest,
        }, f, indent=2)
    print(f"[+] {len(manifest)} invocations, {skipped} skipped → {manifest_out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100", help="GPU label (informational)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--shapes", type=Path, default=_SHAPES_CSV)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--max-shapes", type=int, default=_MAX_SHAPES)
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dt = dtype_map[args.dtype]

    all_shapes = load_shapes(args.shapes)
    print(f"[*] {len(all_shapes)} unique shapes from {args.shapes}", flush=True)

    sampled = sample_shapes(all_shapes, max_n=args.max_shapes)
    print(f"[*] sampled {len(sampled)} shapes", flush=True)

    run_sweep(sampled, dt, args.out_manifest)


if __name__ == "__main__":
    main()
