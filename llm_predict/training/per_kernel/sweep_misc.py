"""Synthetic misc-kernel sweep — covers shape-extrapolation gap for
held-out dense-model d values (8192 for Llama-70B / Qwen-72B).

Runs single-kernel ops at known shapes so labeling is iteration-order safe:
    reduce   : torch.sum(x, dim=-1)       # per-row reduce over d
    cast     : x.to(torch.float32)        # bf16 → fp32 cast
    copy     : x.clone()                  # contiguous copy
    softmax  : F.softmax(x, dim=-1)       # row-wise softmax

Each op × (M, d) config is called REPS times (default 1). Torch lowers each
op to exactly one primary CUDA kernel at these sizes, so iteration-order
labeling is 1:1 (unlike SDPA/FA2).

Run via:
    ncu --target-processes all \\
        --metrics "gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread" \\
        --kernel-name-base demangled \\
        --kernel-name "regex:reduce_kernel|softmax|copy_kernel|LoadWithCast|StoreWithCast|elementwise_kernel" \\
        -o /tmp/ncu_misc_sweep --force-overwrite \\
        python sweep_misc.py --dtype bf16 --out-manifest /tmp/ncu_misc_sweep.manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import nvtx


D_VALUES = [4096, 5120, 6144, 8192, 10240, 12288, 16384]
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
REPS = 1
OP_TYPES = ["reduce", "cast", "copy", "softmax"]


def _run_op(op: str, x: torch.Tensor) -> torch.Tensor:
    if op == "reduce":
        return torch.sum(x, dim=-1)
    if op == "cast":
        return x.to(torch.float32)
    if op == "copy":
        return x.clone()
    if op == "softmax":
        return F.softmax(x, dim=-1)
    raise ValueError(f"unknown op {op}")


def run_sweep(dtype: torch.dtype, manifest_out: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run on a GPU host.")
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    total = len(OP_TYPES) * len(M_VALUES) * len(D_VALUES) * REPS
    print(f"[*] GPU: {name} sm_{cc[0]}{cc[1]}")
    print(f"[*] {len(OP_TYPES)} ops × {len(M_VALUES)} M × {len(D_VALUES)} d × {REPS} reps "
          f"= {total} invocations, dtype={dtype}")

    manifest: list[dict] = []
    idx = 0
    for op in OP_TYPES:
        for M in M_VALUES:
            for d in D_VALUES:
                tag = f"misc_{op}_M{M}_d{d}"
                nvtx.range_push(tag)
                x = torch.randn(M, d, dtype=dtype, device="cuda")
                _run_op(op, x)  # warmup (not counted)
                torch.cuda.synchronize()
                for _ in range(REPS):
                    y = _run_op(op, x)
                    manifest.append({
                        "idx": idx, "op_type": op,
                        "M": M, "d": d, "numel": M * d,
                        "dtype": str(dtype).replace("torch.", ""),
                        "nvtx_range": tag,
                    })
                    idx += 1
                torch.cuda.synchronize()
                nvtx.range_pop()
                del x, y
    torch.cuda.empty_cache()

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump({
            "gpu_name": name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "dtype": str(dtype),
            "n_invocations": len(manifest),
            "op_types": OP_TYPES,
            "m_values": M_VALUES,
            "d_values": D_VALUES,
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
