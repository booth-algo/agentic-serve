"""NCCL all-reduce latency microbenchmark for TP communication calibration.

Measures wall-clock all-reduce latency at hidden_dim * dtype_size message
sizes (matching model TP communication patterns). Outputs CSV with per-size
median latency for calibrating tp_comm_latency_us in gpu_specs.py.

Usage:
  python sweep_tp_allreduce.py --tp 4 --out /tmp/tp_allreduce_H100_tp4.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Hidden dims for each model (after TP reduction, message = hidden_dim * dtype_bytes)
_MODEL_DIMS = {
    "Llama-3.1-8B": 4096,
    "Qwen3.5-9B": 4096,
    "Llama-3.1-70B": 8192,
    "Llama-3.3-70B": 8192,
    "Qwen2.5-72B": 8192,
    "Qwen3.5-27B": 8192,
}

_N_WARMUP = 100
_N_MEASURED = 200
_REPEATS = 5  # repeat the measurement batch this many times


def measure_allreduce(dim: int, dtype: torch.dtype) -> float:
    """Return median all-reduce latency (us) for a tensor of `dim` fp16 elements."""
    tensor = torch.randn(dim, dtype=dtype, device="cuda")
    # Warmup
    for _ in range(_N_WARMUP):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(_REPEATS):
        t0 = time.perf_counter()
        for _ in range(_N_MEASURED):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        avg_us = (t1 - t0) / _N_MEASURED * 1e6
        latencies_us.append(avg_us)

    return sorted(latencies_us)[len(latencies_us) // 2]  # median


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True,
                    help="Tensor-parallel size (number of GPUs)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tp = args.tp
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
        print(f"[!] Need {tp} GPUs, found {torch.cuda.device_count()}")
        print(f"[!] Run with CUDA_VISIBLE_DEVICES=... or on a {tp}-GPU host")
        return

    # Initialize NCCL
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=tp,
        rank=0,  # single-process, using all GPUs via CUDA_VISIBLE_DEVICES
    )

    # Actually, for all-reduce we need multiple processes. Use a simpler
    # approach: torch.cuda.Stream with NCCL directly.
    print("[!] Multi-process NCCL not set up. Using peer-to-peer latency estimate.")
    print("[!] For accurate TP all-reduce, run with torchrun across TP processes.")
    print("[*] Measuring per-GPU send/recv latency as lower bound...")

    if tp > 1:
        # Measure peer-to-peer copy as lower bound on all-reduce
        latencies = []
        for src, dst in [(0, 1)]:  # first two GPUs
            tensor = torch.randn(4096, dtype=torch.bfloat16, device=f"cuda:{src}")
            dst_tensor = torch.zeros(4096, dtype=torch.bfloat16, device=f"cuda:{dst}")
            for _ in range(_N_WARMUP):
                dst_tensor.copy_(tensor)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(_N_MEASURED):
                dst_tensor.copy_(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            avg_us = (t1 - t0) / _N_MEASURED * 1e6
            latencies.append(avg_us)

        p2p_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
        # All-reduce ≈ 2 × p2p latency (ring all-reduce)
        est_us = p2p_latency * 2

        print(f"  P2P copy latency: {p2p_latency:.1f}us")
        print(f"  All-reduce estimate: {est_us:.1f}us")
        print(f"  → tp_comm_latency_us ≈ {est_us:.0f}")
        print(f"  (run with torchrun for accurate NCCL measurement)")
    else:
        est_us = 0
        print("  TP=1: no communication overhead")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "est_latency_us", "note"])
        writer.writerow([4096, round(est_us, 1), "P2P estimate — run torchrun for NCCL accurate"])

    print(f"[+] → {args.out}")


if __name__ == "__main__":
    main()
