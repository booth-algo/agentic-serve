"""Minimal vLLM prefill step for Nsight Systems node-level CUDA graph tracing.

Run with:
  nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node \
    -o prefill_nodes \
    python prefill_nsys_trace.py --model ... --prefill-tokens 4096

This captures individual CUDA graph nodes (nvjet matmuls, flash attention,
elementwise kernels) so we can decompose prefill time into GEMM vs attention
vs KV write vs elementwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Remove script directory from sys.path — flash_attn.py in this directory
# shadows the flash_attn pip package that vLLM needs.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_sys_path_entries = [p for p in sys.path if p == _SCRIPT_DIR]
for _entry in _sys_path_entries:
    sys.path.remove(_entry)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    p.add_argument("--prefill-tokens", type=int, default=0)
    p.add_argument("--P", type=int, default=0, help="Cached prefix tokens (0 = full prefill)")
    p.add_argument("--U", type=int, default=4096, help="New suffix tokens")
    return p.parse_args()


def make_prompt(tokenizer, target_tokens: int) -> str:
    text = (" the" * target_tokens).strip()
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if encode is None or decode is None:
        return text
    token_ids = encode(text, add_special_tokens=False)
    while len(token_ids) < target_tokens:
        text = f"{text} {text}"
        token_ids = encode(text, add_special_tokens=False)
    return decode(token_ids[:target_tokens])


def main() -> None:
    args = parse_args()

    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM, SamplingParams

    print(f"Loading {args.model} (V0, CUDA graphs ON)...", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=32768,
        gpu_memory_utilization=0.70,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    if args.prefill_tokens > 0:
        # Full prefill mode
        N = args.prefill_tokens
        prompt = make_prompt(tokenizer, N)
        label = f"N={N}"
    else:
        # Cached prefill mode
        P = args.P
        U = args.U
        base = make_prompt(tokenizer, P)
        suffix = make_prompt(tokenizer, U)
        prompt = base + " " + suffix
        label = f"U={U},P={P}"
        # Warm KV cache by generating the base prefix first
        print(f"Warming KV cache (P={P})...", flush=True)
        for _ in range(2):
            llm.generate([base], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                          use_tqdm=False)
            torch.cuda.synchronize()

    # Warmup: trigger CUDA graph capture
    print(f"Warmup ({label})...", flush=True)
    for _ in range(3):
        llm.generate([prompt], params, use_tqdm=False)
        torch.cuda.synchronize()

    # Allow nsys to start capture
    torch.cuda.cudart().cudaProfilerStart()

    # Timed run
    print("Profiling...", flush=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    llm.generate([prompt], params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    torch.cuda.cudart().cudaProfilerStop()

    print(f"Prefill {label}: {elapsed*1000:.1f}ms wall time", flush=True)


if __name__ == "__main__":
    main()
