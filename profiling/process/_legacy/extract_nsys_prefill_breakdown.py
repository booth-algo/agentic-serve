"""Extract per-N prefill kernel breakdown from nsys sqlite captures.

Reads ``prefill_N{X}.sqlite`` files produced by
``profiling/profile/vllm/cuda_events/prefill_nsys_trace.py``.  Classifies
GPU kernels into buckets using ``CUPTI_ACTIVITY_KIND_KERNEL`` joined to
``StringIds`` (both ``shortName`` and ``demangledName`` — FA3 attention shows
up as ``device_kernel`` shortName and only demangles to a FlashAttn signature).

The captures contain multiple forward passes plus warmup activity.  Absolute
per-step times don't fall out cleanly because (a) some captures contain
decode-only steps that overlap on multiple streams, and (b) the
``cudaProfilerStart`` bracket is sometimes ignored by nsys.  What IS stable
across N is the **proportion** of non-attention time spent in each bucket.

This extractor therefore follows the YAML-documented "proportions from nsys,
scale from ground truth" path:

  non_attention_budget(N) = vllm_prefill_total(N) - FA3(N)
  gemm_compiled(N)        = non_attention_budget × share_gemm(N)
  elementwise(N)          = non_attention_budget × share_elementwise(N)
  kv_write(N)             = non_attention_budget × share_kv_write(N)
  other(N)                = non_attention_budget × share_other(N)
  FA3(N)                  = from profile_data/kernels/fa3_prefill_H100.csv (independent)

The shares are computed within the non-attention bucket sums from the trace
and are dimensionless — they're invariant to how many iterations the trace
contains.  The absolute scale comes from independent vLLM step timing.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NSYS_DIR = Path("/mnt/100g/nsys_h100/prefill")
DEFAULT_VLLM_PREFILL = Path("profile_data/kernels/prefill_profile_H100_dense.csv")
DEFAULT_FA3_PREFILL = Path("profile_data/kernels/fa3_prefill_H100.csv")
DEFAULT_OUTPUT = Path("profile_data/kernels/prefill_compiled_breakdown_H100.csv")

# Kernel classification.  shortName carries cuBLAS/Triton kernels directly,
# but FA3 reports as "device_kernel" shortName — disambiguate via demangledName.
_GEMM_RE = re.compile(r"^(nvjet|cublasLt|splitKreduce)")
_KV_WRITE_RE = re.compile(
    r"(reshape_and_cache|_compute_slot_mapping|prepare_varlen_num_blocks)"
)
_ELEMENTWISE_RE = re.compile(
    r"(triton_(poi|red)_fused|vectorized_elementwise|unrolled_elementwise"
    r"|elementwise_kernel|index_elementwise)"
)
_SAMPLING_RE = re.compile(r"(_topk_topp|cunn_SoftMax|argmax)")
_ATTENTION_DEM_RE = re.compile(r"(FlashAttnFwd|flash_attn_varlen|memory_efficient_attention)")


@dataclass(frozen=True)
class PerNTrace:
    prefill_tokens: int
    gemm_ns: int
    attention_ns: int
    elementwise_ns: int
    kv_write_ns: int
    sampling_ns: int
    other_ns: int


def classify(short: str, demangled: str) -> str:
    if _ATTENTION_DEM_RE.search(demangled):
        return "attention"
    if _GEMM_RE.search(short):
        return "gemm_linear"
    if _KV_WRITE_RE.search(short):
        return "kv_write"
    if _ELEMENTWISE_RE.search(short):
        return "elementwise"
    if _SAMPLING_RE.search(short):
        return "sampling"
    return "other"


def extract_trace(sqlite_path: Path) -> PerNTrace:
    n_match = re.search(r"prefill_N(\d+)\.sqlite$", sqlite_path.name)
    if not n_match:
        raise ValueError(f"cannot parse prefill_N<int> from {sqlite_path}")
    n = int(n_match.group(1))
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute(
            """
            SELECT s_short.value, s_dem.value, SUM(k.end - k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s_short ON s_short.id = k.shortName
            JOIN StringIds s_dem ON s_dem.id = k.demangledName
            GROUP BY s_short.value, s_dem.value
            """
        ).fetchall()
    finally:
        con.close()
    sums: dict[str, int] = defaultdict(int)
    for short, dem, total_ns in rows:
        sums[classify(short or "", dem or "")] += int(total_ns or 0)
    return PerNTrace(
        prefill_tokens=n,
        gemm_ns=sums["gemm_linear"],
        attention_ns=sums["attention"],
        elementwise_ns=sums["elementwise"],
        kv_write_ns=sums["kv_write"],
        sampling_ns=sums["sampling"],
        other_ns=sums["other"],
    )


def load_reference_dense(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                n = int(row["prefill_tokens"])
                ms = float(row["prefill_ms"])
            except (KeyError, ValueError):
                continue
            out[n] = ms
    return out


def load_fa3(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                n = int(row["prefill_tokens"])
                ms = float(row["flash_full_model_ms"])
            except (KeyError, ValueError):
                continue
            out[n] = ms
    return out


def log_linear_at(samples: dict[int, float], target: int) -> float | None:
    if not samples:
        return None
    if target in samples:
        return samples[target]
    points = sorted(samples.items())
    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]
    lo, hi = points[0], points[-1]
    for i in range(len(points) - 1):
        if points[i][0] <= target <= points[i + 1][0]:
            lo, hi = points[i], points[i + 1]
            break
    log_lo, log_hi, log_t = math.log(lo[0]), math.log(hi[0]), math.log(target)
    frac = (log_t - log_lo) / (log_hi - log_lo) if log_hi > log_lo else 0.0
    return lo[1] + (hi[1] - lo[1]) * frac


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsys-dir", type=Path, default=DEFAULT_NSYS_DIR)
    p.add_argument(
        "--reference-vllm-prefill", type=Path, default=DEFAULT_VLLM_PREFILL,
        help="CSV with prefill_tokens,prefill_ms (independent vLLM step timing).",
    )
    p.add_argument(
        "--fa3-prefill", type=Path, default=DEFAULT_FA3_PREFILL,
        help="CSV with prefill_tokens,flash_full_model_ms (independent FA3 timing).",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--warn-band", type=float, default=0.15,
        help="Sanity-ratio band; sums outside [1-band, 1+band] log a warning.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.nsys_dir.exists():
        raise SystemExit(f"nsys directory missing: {args.nsys_dir}")
    dense = load_reference_dense(args.reference_vllm_prefill)
    fa3 = load_fa3(args.fa3_prefill)
    if not dense:
        raise SystemExit(f"reference dense CSV empty: {args.reference_vllm_prefill}")

    traces = sorted(
        (extract_trace(p) for p in args.nsys_dir.glob("prefill_N*.sqlite")),
        key=lambda t: t.prefill_tokens,
    )
    if not traces:
        raise SystemExit(f"no prefill_N*.sqlite found in {args.nsys_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prefill_tokens",
        "gemm_compiled_ms",
        "elementwise_ms",
        "kv_write_ms",
        "other_ms",
        "fa3_ms",
        "total_ms",
        "reference_vllm_prefill_ms",
        "sanity_ratio",
        "share_gemm",
        "share_elementwise",
        "share_kv_write",
        "share_other",
    ]
    rows_out: list[dict[str, str]] = []
    warnings: list[str] = []
    for t in traces:
        non_attn_ns = t.gemm_ns + t.elementwise_ns + t.kv_write_ns + t.other_ns
        if non_attn_ns <= 0:
            warnings.append(f"N={t.prefill_tokens}: non-attention trace sum is 0; skipping")
            continue
        share_gemm = t.gemm_ns / non_attn_ns
        share_elem = t.elementwise_ns / non_attn_ns
        share_kv = t.kv_write_ns / non_attn_ns
        share_other = t.other_ns / non_attn_ns

        ref_ms = log_linear_at(dense, t.prefill_tokens)
        fa3_ms = log_linear_at(fa3, t.prefill_tokens)
        if ref_ms is None:
            warnings.append(f"N={t.prefill_tokens}: missing reference vllm_prefill; skipping")
            continue
        if fa3_ms is None:
            warnings.append(f"N={t.prefill_tokens}: missing FA3 reference; skipping")
            continue
        non_attention_budget = max(0.0, ref_ms - fa3_ms)
        gemm_ms = non_attention_budget * share_gemm
        elem_ms = non_attention_budget * share_elem
        kv_ms = non_attention_budget * share_kv
        other_ms = non_attention_budget * share_other
        total_ms = gemm_ms + elem_ms + kv_ms + other_ms + fa3_ms
        sanity = total_ms / ref_ms if ref_ms > 0 else 0.0
        if not (1.0 - args.warn_band <= sanity <= 1.0 + args.warn_band):
            warnings.append(
                f"N={t.prefill_tokens}: sanity_ratio={sanity:.3f} outside "
                f"[{1.0-args.warn_band:.2f}, {1.0+args.warn_band:.2f}]"
            )
        rows_out.append({
            "prefill_tokens": str(t.prefill_tokens),
            "gemm_compiled_ms": f"{gemm_ms:.6f}",
            "elementwise_ms": f"{elem_ms:.6f}",
            "kv_write_ms": f"{kv_ms:.6f}",
            "other_ms": f"{other_ms:.6f}",
            "fa3_ms": f"{fa3_ms:.6f}",
            "total_ms": f"{total_ms:.6f}",
            "reference_vllm_prefill_ms": f"{ref_ms:.6f}",
            "sanity_ratio": f"{sanity:.6f}",
            "share_gemm": f"{share_gemm:.6f}",
            "share_elementwise": f"{share_elem:.6f}",
            "share_kv_write": f"{share_kv:.6f}",
            "share_other": f"{share_other:.6f}",
        })
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {args.output} ({len(rows_out)} rows)")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
