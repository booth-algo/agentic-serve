"""Serving-framework overhead calibration.

At short ISL and low concurrency, measured TTFT is dominated by
framework-level costs that are *not* kernel-time:

  - vLLM scheduler: request enqueue, worker IPC, lock contention
  - Tokenizer: input tokenization (CPU-side)
  - cuBLAS / cuDNN kernel dispatch latency
  - Python-side event loop / async handoff
  - CUDA graph instantiation (first request only, amortized out at steady state)

Our per-kernel, per-category, and per-op predictors all estimate pure
kernel time. To compare to measured TTFT, add a framework-overhead term:

    predicted_TTFT = kernel_time + framework_overhead(gpu, conc, isl)

Constants here are calibrated from Step 5 H100 findings + A100 sweep results
at conc=1, ISL ∈ [64, 512].

TODO: fit per-(gpu, backend) constants properly once we have enough measured
data points across the full sweep. Current values are rough estimates from
observed prediction gaps at small ISL.
"""


# Rough per-GPU additive constants in milliseconds at conc=1 prefill.
# These are placeholders; update with a proper calibration fit from the
# step4/step5 validation CSVs in /Users/kev/gated/step4_results/.
_FRAMEWORK_OVERHEAD_MS = {
    ("A100", "vllm"):  5.0,   # observed ~32ms measured vs ~27ms predicted at short ISL
    ("A100", "sglang"): 5.0,
    ("H100", "vllm"):  15.0,  # observed ~30ms measured vs ~15ms predicted on Llama-8B H100
    ("H100", "sglang"): 15.0,
}


def framework_overhead_ms(
    gpu: str = "A100",
    backend: str = "vllm",
    concurrency: int = 1,
    isl: int = 0,
) -> float:
    """Return additive framework overhead in milliseconds.

    Currently a simple (gpu, backend) lookup. Future versions may scale with
    concurrency (more scheduling work) or decrease for large ISL (amortized).

    Args:
        gpu: 'A100', 'H100', '3090', '2080ti'
        backend: 'vllm' or 'sglang'
        concurrency: server concurrency (unused for now)
        isl: input sequence length (unused for now)

    Returns:
        Overhead in ms. Zero if unknown (gpu, backend).
    """
    return _FRAMEWORK_OVERHEAD_MS.get((gpu, backend), 0.0)
