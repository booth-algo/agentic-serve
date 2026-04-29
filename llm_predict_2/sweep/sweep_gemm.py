"""GEMM sweep for llm_predict_2: profile serving shapes via ncu.

Reads serving_shapes.csv, runs each (M, N, K) through nn.Linear with
NVTX ranges so ncu can attribute timings per shape.

Usage:
    # On GPU host, with ncu:
    NCU=/usr/bin/ncu
    PY=~/miniconda3/envs/predictor/bin/python
    $NCU --target-processes all \
        --metrics "gpu__time_duration.sum" \
        --kernel-name-base "demangled" \
        --kernel-name "regex:gemm|GEMM|sgemm|hgemm" \
        -o /tmp/gemm_sweep --force-overwrite \
        $PY sweep_gemm.py --shapes serving_shapes.csv --dtype bf16

    # Export to CSV:
    $NCU --import /tmp/gemm_sweep.ncu-rep --csv > /tmp/gemm_sweep_raw.csv

    # Post-process:
    python post_process_gemm.py /tmp/gemm_sweep_raw.csv --out /tmp/gemm_A100.csv
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.cuda import nvtx


WARMUP = 10
REPS = 10


def run_sweep(shapes_path: Path, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    name = torch.cuda.get_device_name()
    print(f"[*] GPU: {name}, dtype: {dtype}")

    shapes = []
    with open(shapes_path) as f:
        for row in csv.DictReader(f):
            shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))

    print(f"[*] {len(shapes)} shapes x {REPS} reps = {len(shapes) * REPS} matmuls")

    for i, (M, N, K) in enumerate(shapes):
        tag = f"gemm_M{M}_N{N}_K{K}"
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        lin = torch.nn.Linear(K, N, bias=False).to(device="cuda", dtype=dtype)

        for _ in range(WARMUP):
            lin(a)
        torch.cuda.synchronize()

        nvtx.range_push(tag)
        for _ in range(REPS):
            lin(a)
        torch.cuda.synchronize()
        nvtx.range_pop()

        del a, lin
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(shapes)}]")
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    print(f"[+] sweep done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True, help="Path to serving_shapes.csv")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    run_sweep(Path(args.shapes), dtype_map[args.dtype])


if __name__ == "__main__":
    main()
