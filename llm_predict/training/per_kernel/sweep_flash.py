"""FlashAttention standalone sweep — varied (seq, n_heads, head_dim, kv_heads).

Closes the training-data gap for the per-kernel `flash_attn` predictor, which
currently has only ~112 training rows at fixed seq=128 / head_dim=128. This
sweep produces ~300 additional rows per GPU across bs=1, causal=True, varying
seq/heads/head_dim/GQA-ratio.

Torch's `F.scaled_dot_product_attention` auto-selects FA2 on sm_80+ when
head_dim ∈ {64, 128} and dtype is fp16/bf16. On sm_75 (RTX 2080Ti) FA2 is
unavailable and this script will likely dispatch to the CUDNN / math
fallback; do not run on 2080Ti for the flash training pool.

Run via:
    ncu --target-processes all \\
        --metrics "gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread" \\
        --kernel-name-base demangled \\
        --kernel-name "regex:flash_fwd|fmha|pytorch_flash" \\
        -o /tmp/ncu_flash_sweep --force-overwrite \\
        python sweep_flash.py --dtype bf16 --out-manifest /tmp/ncu_flash_sweep.manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import nvtx


_GQA_RATIOS = [1, 4, 8]  # kv_heads = n_heads / ratio (min 1)

# (n_heads, head_dim) pairs covering the dense-model landscape.
HEAD_CONFIGS = [
    (8,  64),
    (8,  128),
    (16, 64),
    (16, 128),
    (32, 128),
    (64, 128),
]

SEQ_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
REPS = 1  # One kernel call per config — primary-kernel dedup handles FA2 multi-kernel dispatch


def run_sweep(dtype: torch.dtype, manifest_out: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run on a GPU host.")
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)

    seen: set[tuple[int, int, int, int, int]] = set()
    configs: list[tuple[int, int, int, int, int]] = []
    for seq in SEQ_VALUES:
        for h, hd in HEAD_CONFIGS:
            for ratio in _GQA_RATIOS:
                kv = max(1, h // ratio)
                key = (1, seq, h, hd, kv)
                if key in seen:
                    continue
                seen.add(key)
                configs.append(key)

    print(f"[*] GPU: {name} sm_{cc[0]}{cc[1]}")
    print(f"[*] {len(configs)} (seq, h, hd, kv) configs × {REPS} reps "
          f"= {len(configs) * REPS} flash attentions, dtype={dtype}")

    manifest: list[dict] = []
    idx = 0
    # Don't restrict SDPA backends — let PyTorch pick flash / mem-efficient /
    # cuDNN as available. Manual-expand KV for GQA rather than rely on
    # `enable_gqa` (not supported on all FA2 builds / archs).
    for (bs, seq, h, hd, kv) in configs:
        tag = f"flash_bs{bs}_seq{seq}_h{h}_hd{hd}_kv{kv}"
        nvtx.range_push(tag)
        q = torch.randn(bs, h, seq, hd, dtype=dtype, device="cuda")
        k = torch.randn(bs, kv, seq, hd, dtype=dtype, device="cuda")
        v = torch.randn(bs, kv, seq, hd, dtype=dtype, device="cuda")
        if kv != h:
            k = k.repeat_interleave(h // kv, dim=1)
            v = v.repeat_interleave(h // kv, dim=1)
        # Warmup once; catch "No available kernel" and skip this config
        # (e.g. head_dim not supported by any FA2 variant on this arch).
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        except RuntimeError as e:
            print(f"[skip] {tag}: {e}")
            nvtx.range_pop()
            del q, k, v
            continue
        torch.cuda.synchronize()
        for _ in range(REPS):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            manifest.append({
                "idx": idx, "bs": bs, "seq": seq,
                "n_heads": h, "head_dim": hd, "kv_heads": kv,
                "dtype": str(dtype).replace("torch.", ""),
                "nvtx_range": tag,
            })
            idx += 1
        torch.cuda.synchronize()
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
            "configs": [
                {"bs": bs, "seq": seq, "n_heads": h, "head_dim": hd, "kv_heads": kv}
                for (bs, seq, h, hd, kv) in configs
            ],
            "reps": REPS,
            "invocations": manifest,
        }, f, indent=2)
    print(f"[+] manifest -> {manifest_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    run_sweep(dtype_map[args.dtype], Path(args.out_manifest))


if __name__ == "__main__":
    main()
