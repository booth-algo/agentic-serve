"""Post-process ncu CSV export into clean (M, N, K, latency_us) format.

Supports two modes:
1. NVTX ranges present: map via gemm_M{M}_N{N}_K{K} range names
2. No NVTX: map by sequential kernel index using serving_shapes.csv order

The sweep runs shapes sequentially with REPS=10 each, so kernel i maps to
shape[i // reps] with rep index i % reps.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


NVTX_PATTERN = re.compile(r"gemm_M(\d+)_N(\d+)_K(\d+)")
REPS = 10


def _parse_duration(row: dict) -> float | None:
    for key in ["gpu__time_duration.sum", "gpu__time_duration.sum [ns]",
                "GPU Time Duration (ns)", "gpu__time_duration.sum [nsecond]"]:
        if key in row and row[key]:
            try:
                return float(row[key].replace(",", "").replace('"', ''))
            except ValueError:
                continue
    return None


def post_process(raw_csv: Path, out_path: Path,
                 shapes_csv: Path | None = None) -> None:
    with open(raw_csv) as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        print("[!] Empty CSV")
        return

    # Try NVTX mode first
    shape_times: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    nvtx_found = False

    for row in raw_rows:
        nvtx_range = None
        for key in ["NVTX Range", "NvtxRange", "nvtx_range",
                     "NVTX Range:gemm_M"]:
            if key in row and row[key]:
                nvtx_range = row[key]
                break

        if nvtx_range:
            match = NVTX_PATTERN.search(nvtx_range)
            if match:
                nvtx_found = True
                M = int(match.group(1))
                N = int(match.group(2))
                K = int(match.group(3))
                dur = _parse_duration(row)
                if dur is not None:
                    shape_times[(M, N, K)].append(dur / 1000.0)

    if not nvtx_found:
        # Sequential mode: map kernels to shapes by index
        if shapes_csv is None:
            shapes_csv = Path(__file__).parent.parent / "data" / "gemm" / "serving_shapes.csv"
        if not shapes_csv.exists():
            print(f"[!] No NVTX ranges and no shapes CSV at {shapes_csv}")
            return

        shapes = []
        with open(shapes_csv) as f:
            for r in csv.DictReader(f):
                shapes.append((int(r["M"]), int(r["N"]), int(r["K"])))

        # Filter to only GEMM kernel rows (skip headers, non-kernel rows)
        durations = []
        for row in raw_rows:
            dur = _parse_duration(row)
            if dur is not None:
                durations.append(dur / 1000.0)

        expected = len(shapes) * REPS
        # ncu may capture warmup kernels too (10 warmup per shape)
        # Total kernels = shapes * (WARMUP + REPS) = shapes * 20
        # But ncu profiles ALL kernel launches, warmup included
        kernels_per_shape = len(durations) // len(shapes) if shapes else 0

        print(f"[*] Sequential mode: {len(durations)} kernels, {len(shapes)} shapes, "
              f"~{kernels_per_shape} kernels/shape")

        if kernels_per_shape < REPS:
            print(f"[!] Too few kernels ({len(durations)}) for {len(shapes)} shapes")
            return

        for i, (M, N, K) in enumerate(shapes):
            start = i * kernels_per_shape
            # Take the last REPS kernels (skip warmup)
            rep_start = start + kernels_per_shape - REPS
            for j in range(REPS):
                idx = rep_start + j
                if idx < len(durations):
                    shape_times[(M, N, K)].append(durations[idx])

    # Aggregate: median per shape
    rows = []
    for (M, N, K), times in sorted(shape_times.items()):
        median = sorted(times)[len(times) // 2]
        rows.append({"M": M, "N": N, "K": K, "latency_us": round(median, 3)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "latency_us"])
        w.writeheader()
        w.writerows(rows)

    print(f"[+] {len(rows)} shapes -> {out_path}")
    if rows:
        print(f"    M range: {min(r['M'] for r in rows)} - {max(r['M'] for r in rows)}")
        print(f"    latency range: {min(r['latency_us'] for r in rows):.1f} - "
              f"{max(r['latency_us'] for r in rows):.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", help="Raw ncu CSV export")
    ap.add_argument("--out", required=True, help="Output clean CSV path")
    ap.add_argument("--shapes", default=None, help="serving_shapes.csv for sequential mode")
    args = ap.parse_args()
    shapes = Path(args.shapes) if args.shapes else None
    post_process(Path(args.raw_csv), Path(args.out), shapes)


if __name__ == "__main__":
    main()
