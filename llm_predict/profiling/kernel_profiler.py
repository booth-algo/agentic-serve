"""
kernel_profiler.py — Real GPU kernel latency profiler for LLMCompass.

Profiles three categories of operations that dominate transformer inference:
  1. GEMM         — matrix multiplications (Linear layers)
  2. Attention    — scaled dot-product attention (prefill and decode)
  3. Elementwise  — RMSNorm, SiLU activation, residual add

Results are written to CSV files in the specified output directory and are
consumed by the LLMCompass analytical cost model to provide data-driven
latency estimates rather than purely model-predicted ones.

Timing methodology
------------------
All timings use torch.cuda.Event with enable_timing=True.  This measures
elapsed GPU time directly and avoids CPU-side synchronisation overhead that
plagues time.time()-based approaches.  Each measurement follows the pattern:

    warmup  → discard first N iterations (JIT compilation, cache warm-up)
    repeat  → collect R samples, take the median (robust to outliers)

All tensors are fp16 to match typical LLM inference deployments.

Reference
---------
Used in the companion research paper:
    "LLMCompass: Enabling Efficient Hardware Design for Large Language Model
     Inference" — see docs/ for the full paper.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal timing helpers
# ---------------------------------------------------------------------------

def _time_kernel_ns(
    fn,
    warmup: int,
    repeat: int,
    device: str,
) -> float:
    """
    Time *fn* on *device* and return the median latency in nanoseconds.

    Parameters
    ----------
    fn:
        A zero-argument callable that executes the kernel once.
    warmup:
        Number of un-timed warm-up iterations.
    repeat:
        Number of timed iterations whose median is returned.
    device:
        The CUDA device string (e.g. "cuda:0").  Used only to pick the
        right device context for the CUDA Events.

    Returns
    -------
    float
        Median kernel latency in nanoseconds.
    """
    # Warm up: run without recording to heat caches and trigger JIT
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    # Timed iterations
    latencies_ms: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(repeat):
        torch.cuda.synchronize(device)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize(device)
        latencies_ms.append(start_event.elapsed_time(end_event))

    # elapsed_time returns milliseconds; convert to nanoseconds
    median_ms = statistics.median(latencies_ms)
    return median_ms * 1e6


# ---------------------------------------------------------------------------
# 1. GEMM profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_gemm(
    M_range: Optional[List[int]] = None,
    N_range: Optional[List[int]] = None,
    K_range: Optional[List[int]] = None,
    device: str = "cuda:0",
    warmup: int = 10,
    repeat: int = 30,
    output_dir: str = "profiler/profiles/H100",
) -> int:
    """
    Profile GEMM latency across a grid of (M, N, K) shapes.

    Performs C = A @ B where A is [M, K] fp16 and B is [K, N] fp16, which
    corresponds to the weight matrix multiplication in a Linear layer when
    processing a batch of M tokens.

    Shapes where M*N*K >= 1e12 are skipped to keep total profiling time
    reasonable (they would each take several seconds per call).

    Parameters
    ----------
    M_range:
        List of M (batch / sequence) dimension values to sweep.
        Defaults to a range covering 1 token (decode) through 16 384 tokens
        (long-context prefill).
    N_range:
        List of N (output feature) values.  Defaults to common d_model and
        intermediate FFN sizes used by Llama / Mistral / Qwen families.
    K_range:
        List of K (input feature) values.  Same defaults as N_range.
    device:
        CUDA device string.
    warmup:
        Warm-up iterations per shape (default 10).
    repeat:
        Timed iterations per shape; median is taken (default 30).
    output_dir:
        Directory where ``gemm_profiles.csv`` is written.

    Returns
    -------
    int
        Number of (M, N, K) combinations that were successfully profiled.
    """
    # Default ranges covering transformer GEMM shapes
    if M_range is None:
        M_range = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                   2048, 4096, 8192, 16384, 32768, 65536]
    if N_range is None:
        N_range = [128, 256, 512, 1024, 2048, 3072, 4096, 5120, 8192,
                   11008, 13824, 14336, 25600, 27648, 28672]
    if K_range is None:
        K_range = [128, 256, 512, 1024, 2048, 3072, 4096, 5120, 8192,
                   11008, 13824, 14336, 25600, 27648, 28672]

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gemm_profiles.csv")

    print(f"\n[GEMM] Profiling {len(M_range)} x {len(N_range)} x {len(K_range)} "
          f"= {len(M_range)*len(N_range)*len(K_range)} candidate shapes")
    print(f"       Results → {csv_path}")

    count = 0
    skipped = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "latency_ns"])

        for M in M_range:
            for N in N_range:
                for K in K_range:
                    # Skip shapes that are unrealistically large or would OOM
                    if M * N * K >= 1e13:
                        skipped += 1
                        continue

                    try:
                        A = torch.randn(M, K, dtype=torch.float16,
                                        device=device)
                        B = torch.randn(K, N, dtype=torch.float16,
                                        device=device)

                        latency_ns = _time_kernel_ns(
                            lambda: torch.mm(A, B),
                            warmup=warmup,
                            repeat=repeat,
                            device=device,
                        )
                        writer.writerow([M, N, K, f"{latency_ns:.2f}"])
                        count += 1

                    except torch.cuda.OutOfMemoryError:
                        writer.writerow([M, N, K, -1])
                        torch.cuda.empty_cache()
                        count += 1  # still recorded, just as OOM sentinel

                    finally:
                        # Release references so the allocator can reclaim
                        try:
                            del A, B
                        except NameError:
                            pass

                    if count % 50 == 0:
                        print(f"  [GEMM] {count} shapes profiled …")

    print(f"[GEMM] Done. {count} shapes profiled, {skipped} skipped (too large).")
    return count


# ---------------------------------------------------------------------------
# 2. Attention profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_attention(
    batch_sizes: Optional[List[int]] = None,
    seq_lens: Optional[List[int]] = None,
    n_heads_list: Optional[List[int]] = None,
    head_dim: int = 128,
    device: str = "cuda:0",
    warmup: int = 10,
    repeat: int = 30,
    output_dir: str = "profiler/profiles/H100",
) -> int:
    """
    Profile attention latency for both prefill and decode phases.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` which
    dispatches to FlashAttention on Ampere / Hopper GPUs when available,
    making it representative of production deployments.

    Prefill phase
    ~~~~~~~~~~~~~
    Q, K, V all have shape [batch, n_heads, seq_len, head_dim].
    This simulates processing a full prompt in a single pass.

    Decode phase
    ~~~~~~~~~~~~
    Q has shape [batch, n_heads, 1, head_dim] (one new token per sequence).
    K, V have shape [batch, n_heads, seq_len, head_dim] (full KV cache).
    ``seq_len`` here is treated as the current KV cache length.

    Parameters
    ----------
    batch_sizes:
        Batch sizes to sweep.
    seq_lens:
        Sequence / KV-cache lengths to sweep.
    n_heads_list:
        Number of attention heads *per device* (after tensor parallelism split).
    head_dim:
        Head dimension (default 128, standard for most open-weight models).
    device:
        CUDA device string.
    warmup:
        Warm-up iterations per shape.
    repeat:
        Timed iterations per shape.
    output_dir:
        Directory where CSV files are written.

    Returns
    -------
    int
        Total number of (prefill + decode) shapes profiled.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048, 4096]
    if n_heads_list is None:
        n_heads_list = [8, 16, 32, 40, 64]

    os.makedirs(output_dir, exist_ok=True)
    prefill_path = os.path.join(output_dir, "attn_prefill_profiles.csv")
    decode_path  = os.path.join(output_dir, "attn_decode_profiles.csv")

    total_shapes = (len(batch_sizes) * len(seq_lens) * len(n_heads_list))
    print(f"\n[ATTN] Profiling {total_shapes} prefill + {total_shapes} decode shapes")
    print(f"       Prefill → {prefill_path}")
    print(f"       Decode  → {decode_path}")

    count = 0

    with open(prefill_path, "w", newline="") as pf, \
         open(decode_path,  "w", newline="") as df:

        prefill_writer = csv.writer(pf)
        decode_writer  = csv.writer(df)
        prefill_writer.writerow(["batch_size", "seq_len",  "n_heads", "head_dim", "latency_ns"])
        decode_writer.writerow( ["batch_size", "kv_len",   "n_heads", "head_dim", "latency_ns"])

        for bs in batch_sizes:
            for seq in seq_lens:
                for nh in n_heads_list:

                    # ---- Prefill ----------------------------------------
                    try:
                        Q = torch.randn(bs, nh, seq,  head_dim,
                                        dtype=torch.float16, device=device)
                        K = torch.randn(bs, nh, seq,  head_dim,
                                        dtype=torch.float16, device=device)
                        V = torch.randn(bs, nh, seq,  head_dim,
                                        dtype=torch.float16, device=device)

                        lat_ns = _time_kernel_ns(
                            lambda: F.scaled_dot_product_attention(Q, K, V),
                            warmup=warmup,
                            repeat=repeat,
                            device=device,
                        )
                        prefill_writer.writerow(
                            [bs, seq, nh, head_dim, f"{lat_ns:.2f}"])

                    except torch.cuda.OutOfMemoryError:
                        prefill_writer.writerow([bs, seq, nh, head_dim, -1])
                        torch.cuda.empty_cache()

                    finally:
                        try:
                            del Q, K, V
                        except NameError:
                            pass

                    # ---- Decode -----------------------------------------
                    try:
                        Qd = torch.randn(bs, nh, 1,   head_dim,
                                         dtype=torch.float16, device=device)
                        Kd = torch.randn(bs, nh, seq, head_dim,
                                         dtype=torch.float16, device=device)
                        Vd = torch.randn(bs, nh, seq, head_dim,
                                         dtype=torch.float16, device=device)

                        lat_ns = _time_kernel_ns(
                            lambda: F.scaled_dot_product_attention(Qd, Kd, Vd),
                            warmup=warmup,
                            repeat=repeat,
                            device=device,
                        )
                        decode_writer.writerow(
                            [bs, seq, nh, head_dim, f"{lat_ns:.2f}"])

                    except torch.cuda.OutOfMemoryError:
                        decode_writer.writerow([bs, seq, nh, head_dim, -1])
                        torch.cuda.empty_cache()

                    finally:
                        try:
                            del Qd, Kd, Vd
                        except NameError:
                            pass

                    count += 2  # one prefill + one decode
                    if count % 50 == 0:
                        print(f"  [ATTN] {count} shapes profiled …")

    print(f"[ATTN] Done. {count} shapes profiled (prefill + decode).")
    return count


# ---------------------------------------------------------------------------
# 3. Elementwise profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_elementwise(
    sizes: Optional[List[int]] = None,
    device: str = "cuda:0",
    warmup: int = 10,
    repeat: int = 30,
    output_dir: str = "profiler/profiles/H100",
) -> int:
    """
    Profile elementwise / normalisation operations.

    Covers the three most common non-GEMM operations in modern LLM layers:

    * **rmsnorm**  — Root Mean Square Layer Normalisation (used by Llama,
                      Mistral, Qwen, etc. in place of LayerNorm).
                      Implemented as ``x / rms(x) * weight`` using PyTorch
                      primitives, matching the pattern seen in production.

    * **silu**     — Sigmoid Linear Unit activation used in SwiGLU FFN blocks.
                      ``F.silu(x)`` → ``x * sigmoid(x)``.

    * **residual** — Element-wise residual addition ``x + residual``.

    Parameters
    ----------
    sizes:
        List of tensor element counts (``numel``) to sweep.  Each tensor is
        1-D fp16 for simplicity; real-world shapes do not significantly affect
        elementwise throughput for the same ``numel``.
    device:
        CUDA device string.
    warmup:
        Warm-up iterations per (op, size) pair.
    repeat:
        Timed iterations per (op, size) pair.
    output_dir:
        Directory where ``elementwise_profiles.csv`` is written.

    Returns
    -------
    int
        Number of (op_type, numel) combinations profiled.
    """
    if sizes is None:
        # Covers 1 token at d_model=4096 up to ~16 M elements (large batch)
        sizes = [
            4_096, 8_192, 16_384, 32_768, 65_536,
            131_072, 262_144, 524_288, 1_048_576,
            2_097_152, 4_194_304, 8_388_608, 16_777_216,
        ]

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "elementwise_profiles.csv")

    ops = ["rmsnorm", "silu", "residual"]
    print(f"\n[ELEM] Profiling {len(ops)} ops x {len(sizes)} sizes "
          f"= {len(ops)*len(sizes)} shapes")
    print(f"       Results → {csv_path}")

    count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op_type", "numel", "latency_ns"])

        for numel in sizes:
            x = torch.randn(numel, dtype=torch.float16, device=device)

            # RMSNorm — learnable weight parameter (ones for profiling)
            weight = torch.ones(numel, dtype=torch.float16, device=device)

            def _rmsnorm():
                # rms = sqrt(mean(x^2) + eps); scale by weight
                rms = x.float().pow(2).mean().add(1e-6).sqrt()
                return (x.float() / rms * weight.float()).half()

            def _silu():
                return F.silu(x)

            residual = torch.randn(numel, dtype=torch.float16, device=device)

            def _residual_add():
                return x + residual

            kernel_map = {
                "rmsnorm": _rmsnorm,
                "silu":    _silu,
                "residual": _residual_add,
            }

            for op_name, fn in kernel_map.items():
                try:
                    lat_ns = _time_kernel_ns(
                        fn,
                        warmup=warmup,
                        repeat=repeat,
                        device=device,
                    )
                    writer.writerow([op_name, numel, f"{lat_ns:.2f}"])

                except torch.cuda.OutOfMemoryError:
                    writer.writerow([op_name, numel, -1])
                    torch.cuda.empty_cache()

                count += 1
                if count % 50 == 0:
                    print(f"  [ELEM] {count} shapes profiled …")

            del x, weight, residual

    print(f"[ELEM] Done. {count} (op, size) pairs profiled.")
    return count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.  Example usage:

        # Full profile run on the default H100 system
        python -m llm_predict.profiling.kernel_profiler --device cuda:0

        # Quick smoke-test with a reduced grid (useful in CI or on smaller GPUs)
        python -m llm_predict.profiling.kernel_profiler --quick --device cuda:0

        # Custom output directory
        python -m llm_predict.profiling.kernel_profiler --output-dir profiling/data/A100
    """
    parser = argparse.ArgumentParser(
        description="Profile GPU kernel latencies and save CSV lookup tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to run profiling on (e.g. 'cuda:0', 'cuda:1').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiler/profiles/H100",
        help="Directory where CSV profile files are written.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a reduced shape grid for quick smoke-testing.  "
            "Reduces GEMM M-range to 4 values, seq_lens to 3, etc."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  kernel_profiler requires a CUDA-capable GPU."
        )

    print("=" * 70)
    print("LLMCompass GPU Kernel Profiler")
    print("=" * 70)
    print(f"Device      : {args.device} ({torch.cuda.get_device_name(args.device)})")
    print(f"Output dir  : {args.output_dir}")
    print(f"Quick mode  : {args.quick}")
    print("=" * 70)

    t0 = time.perf_counter()

    if args.quick:
        # Reduced grids for fast smoke-testing / CI
        gemm_count = profile_gemm(
            M_range=[1, 32, 256, 2048],
            N_range=[4096, 8192, 14336],
            K_range=[4096, 8192, 14336],
            device=args.device,
            warmup=2,
            repeat=5,
            output_dir=args.output_dir,
        )
        attn_count = profile_attention(
            batch_sizes=[1, 4, 16],
            seq_lens=[128, 512, 2048],
            n_heads_list=[8, 32],
            device=args.device,
            warmup=2,
            repeat=5,
            output_dir=args.output_dir,
        )
        elem_count = profile_elementwise(
            sizes=[4096, 65536, 1048576, 16777216],
            device=args.device,
            warmup=2,
            repeat=5,
            output_dir=args.output_dir,
        )
    else:
        # Full production grids
        gemm_count = profile_gemm(
            device=args.device,
            output_dir=args.output_dir,
        )
        attn_count = profile_attention(
            device=args.device,
            output_dir=args.output_dir,
        )
        elem_count = profile_elementwise(
            device=args.device,
            output_dir=args.output_dir,
        )

    elapsed = time.perf_counter() - t0
    total = gemm_count + attn_count + elem_count

    print("\n" + "=" * 70)
    print("Profiling complete.")
    print(f"  GEMM shapes       : {gemm_count}")
    print(f"  Attention shapes  : {attn_count}")
    print(f"  Elementwise pairs : {elem_count}")
    print(f"  Total data points : {total}")
    print(f"  Wall-clock time   : {elapsed:.1f}s")
    print(f"  Output directory  : {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
