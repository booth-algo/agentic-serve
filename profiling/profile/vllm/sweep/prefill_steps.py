"""Profile pure prefill forward-pass step time on GPU #6 using vLLM.

Measures prefill latency by isolating the prefill step from decode: runs
two max_tokens settings with the same prompt, subtracts to remove decode
cost.  Uses vLLM (same runtime as decode profiling) for consistent kernel
composition validation.

Usage (on H100):
    TMPDIR=/data48/kevinlau/tmp CUDA_VISIBLE_DEVICES=6 \
    ~/miniconda3/envs/vllm/bin/python profiling/profile/vllm/sweep/prefill_steps.py \
    --output prefill_profile_H100.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


PREFILL_TOKENS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
DECODE_SAMPLE_TOKENS = 32
WARMUP_RUNS = 2
MEASURE_RUNS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    p.add_argument("--output", type=Path, default=Path("prefill_profile_H100.csv"))
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--prefill-tokens", nargs="*", type=int, default=PREFILL_TOKENS)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--gpu-label", default="H100")
    return p.parse_args()


def make_prompt(context_len: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    repeat = max(1, context_len // 5)
    return base * repeat


def time_generate(llm: LLM, prompt: str, max_tokens: int) -> float:
    """Return GPU wall time in ms for a single llm.generate call."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    torch.cuda.synchronize()
    start_ev.record()
    llm.generate([prompt], sampling_params, use_tqdm=False)
    end_ev.record()
    torch.cuda.synchronize()

    return start_ev.elapsed_time(end_ev)


def main() -> None:
    args = parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Warmup
    warmup_prompt = make_prompt(256)
    for _ in range(WARMUP_RUNS):
        time_generate(llm, warmup_prompt, max_tokens=32)

    results: list[dict] = []

    for n_tokens in args.prefill_tokens:
        prompt = make_prompt(n_tokens)

        # Measurement 1: prefill + DECODE_SAMPLE_TOKENS decode steps
        t32_samples = []
        for _ in range(MEASURE_RUNS):
            t32_samples.append(time_generate(llm, prompt, DECODE_SAMPLE_TOKENS))
        T32 = statistics.median(t32_samples)

        # Measurement 2: prefill + 1 decode step
        t1_samples = []
        for _ in range(MEASURE_RUNS):
            t1_samples.append(time_generate(llm, prompt, 1))
        T1 = statistics.median(t1_samples)

        # T32 = prefill(N) + 32 * d_avg
        # T1  = prefill(N) + d_first
        # Assume d_avg ≈ d_first = d (KV cache grows N→N+31, negligible)
        decode_steps_diff = DECODE_SAMPLE_TOKENS - 1
        d = (T32 - T1) / decode_steps_diff
        prefill_ms = T1 - d

        print(
            f"N={n_tokens:>5d}  prefill={prefill_ms:>8.2f}ms  "
            f"decode_ref={d:>6.2f}ms  (T32={T32:.1f} T1={T1:.1f})"
        )

        results.append({
            "gpu": "H100",
            "prefill_tokens": n_tokens,
            "prefill_ms": round(prefill_ms, 2),
            "decode_ref_ms": round(d, 2),
            "T32_ms": round(T32, 1),
            "T1_ms": round(T1, 1),
            "runs": MEASURE_RUNS,
        })

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gpu", "prefill_tokens", "prefill_ms", "decode_ref_ms",
                  "T32_ms", "T1_ms", "runs"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
