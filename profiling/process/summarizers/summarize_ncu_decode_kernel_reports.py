#!/usr/bin/env python3
"""Summarize NCU raw CSV exports for Experiment 2 decode-kernel profiles."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
QKV_OUT_FEATURES = 6144
VOCAB_SIZE = 128256
NUM_LAYERS = 32

GEMM_SHAPES = {
    "qkv_fused": (QKV_OUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS),
    "o_proj": (HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS),
    "gate_up_fused": (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, NUM_LAYERS),
    "down_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS),
    "lm_head": (VOCAB_SIZE, HIDDEN_SIZE, 1),
}

FUSED_BUCKETS = {
    "rms_norm": ("norm_residual", 2 * NUM_LAYERS),
    "silu_and_mul": ("ffn_activation", NUM_LAYERS),
    "rotary_embedding": ("kv_rope_cache", NUM_LAYERS),
    "kv_cache_write": ("kv_rope_cache", NUM_LAYERS),
    "sampling_topk": ("sampling_logits", 1),
    "sampling_logits": ("sampling_logits", 1),
}


@dataclass(frozen=True)
class ReportSummary:
    kernel_count: int
    gpu_time_ms_sum: float
    gpu_time_ms_max: float
    dram_read_mbytes_sum: float
    dram_write_mbytes_sum: float
    top_kernel_examples: str


@dataclass(frozen=True)
class FlashCudaEventRow:
    flash_layer_ms_median: float
    flash_full_model_ms_median: float


@dataclass(frozen=True)
class FusedMetadata:
    implementation: str
    source_status: str


TIME_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}

BYTES_TO_MBYTES = {
    "byte": 1e-6,
    "bytes": 1e-6,
    "b": 1e-6,
    "kbyte": 1e-3,
    "kbytes": 1e-3,
    "kb": 1e-3,
    "mbyte": 1.0,
    "mbytes": 1.0,
    "mb": 1.0,
    "gbyte": 1e3,
    "gbytes": 1e3,
    "gb": 1e3,
    "tbyte": 1e6,
    "tbytes": 1e6,
    "tb": 1e6,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize raw NCU CSV exports into compact latency rows."
    )
    parser.add_argument("--kind", choices=("gemm", "fused", "flash"), required=True)
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern. Defaults to gemm_*.csv, fused_*.csv, or flash_attn_*.csv.",
    )
    parser.add_argument(
        "--cuda-event-flash",
        type=Path,
        help=(
            "Optional standalone CUDA-event flash CSV for companion sanity columns. "
            "Do not pass CUDA-event CSVs recorded inside an NCU run."
        ),
    )
    parser.add_argument("--cuda-event-gpu", default="H100")
    parser.add_argument("--top-kernels", type=int, default=3)
    return parser.parse_args(argv)


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    stripped = value.strip()
    if not stripped:
        return 0.0
    return float(stripped)


def normalize_unit(value: str | None) -> str:
    return "" if value is None else value.strip().lower()


def convert_time_to_ms(value: str | None, unit: str) -> float:
    return parse_float(value) * TIME_TO_MS.get(normalize_unit(unit), 1e-3)


def convert_bytes_to_mbytes(value: str | None, unit: str) -> float:
    return parse_float(value) * BYTES_TO_MBYTES.get(normalize_unit(unit), 1.0)


def compact_kernel_name(name: str, max_len: int = 180) -> str:
    compact = " ".join(name.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def summarize_report(path: Path, top_kernels: int) -> ReportSummary:
    totals_by_kernel: dict[str, float] = defaultdict(float)
    counts_by_kernel: dict[str, int] = defaultdict(int)
    kernel_count = 0
    gpu_time_sum = 0.0
    gpu_time_max = 0.0
    dram_read_mbytes_sum = 0.0
    dram_write_mbytes_sum = 0.0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        units = {
            "gpu__time_duration.sum": "us",
            "dram__bytes_read.sum": "Mbyte",
            "dram__bytes_write.sum": "Mbyte",
        }
        for row in reader:
            kernel_name = row.get("Kernel Name", "").strip()
            if not kernel_name:
                for field in units:
                    value = row.get(field)
                    if value and not value.strip().replace(".", "", 1).isdigit():
                        units[field] = value.strip()
                continue
            # NCU raw CSV exports an unlabeled units row after the header. The
            # memory units can change per file, e.g. Mbyte for smaller rows and
            # Gbyte for larger rows, so normalize before aggregating.
            gpu_time_ms = convert_time_to_ms(
                row.get("gpu__time_duration.sum"),
                units["gpu__time_duration.sum"],
            )
            dram_read_mbytes = convert_bytes_to_mbytes(
                row.get("dram__bytes_read.sum"),
                units["dram__bytes_read.sum"],
            )
            dram_write_mbytes = convert_bytes_to_mbytes(
                row.get("dram__bytes_write.sum"),
                units["dram__bytes_write.sum"],
            )
            kernel_count += 1
            gpu_time_sum += gpu_time_ms
            gpu_time_max = max(gpu_time_max, gpu_time_ms)
            dram_read_mbytes_sum += dram_read_mbytes
            dram_write_mbytes_sum += dram_write_mbytes
            totals_by_kernel[kernel_name] += gpu_time_ms
            counts_by_kernel[kernel_name] += 1

    top_parts = []
    for kernel_name, time_ms in sorted(
        totals_by_kernel.items(), key=lambda item: item[1], reverse=True
    )[:top_kernels]:
        count = counts_by_kernel[kernel_name]
        label = compact_kernel_name(kernel_name)
        top_parts.append(f"{label} ({count}x, {time_ms:.6g} ms)")

    return ReportSummary(
        kernel_count=kernel_count,
        gpu_time_ms_sum=gpu_time_sum,
        gpu_time_ms_max=gpu_time_max,
        dram_read_mbytes_sum=dram_read_mbytes_sum,
        dram_write_mbytes_sum=dram_write_mbytes_sum,
        top_kernel_examples=" | ".join(top_parts),
    )


def parse_gemm_tag(path: Path) -> dict[str, object]:
    match = re.fullmatch(r"gemm_(?P<op>.+)_B(?P<batch>\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected GEMM NCU filename: {path.name}")
    op_name = match.group("op")
    batch_size = int(match.group("batch"))
    if op_name not in GEMM_SHAPES:
        raise ValueError(f"Unknown GEMM op in NCU filename: {path.name}")
    n, k, calls_per_decode_step = GEMM_SHAPES[op_name]
    return {
        "batch_size": batch_size,
        "op_name": op_name,
        "m": batch_size,
        "n": n,
        "k": k,
        "calls_per_decode_step": calls_per_decode_step,
    }


def parse_fused_tag(path: Path) -> dict[str, object]:
    match = re.fullmatch(
        r"fused_(?P<kernel>.+)_B(?P<batch>\d+)_T(?P<context>\d+)", path.stem
    )
    if not match:
        raise ValueError(f"Unexpected fused-kernel NCU filename: {path.name}")
    kernel_name = match.group("kernel")
    batch_size = int(match.group("batch"))
    context_len = int(match.group("context"))
    bucket, calls_per_decode_step = FUSED_BUCKETS.get(
        kernel_name, ("other_unattributed", 1)
    )
    return {
        "batch_size": batch_size,
        "context_len": context_len,
        "bucket": bucket,
        "kernel_name": kernel_name,
        "calls_per_decode_step": calls_per_decode_step,
    }


def parse_flash_tag(path: Path) -> dict[str, object]:
    match = re.fullmatch(
        r"(?:flash|flash_attn|attention)_B(?P<batch>\d+)_T(?P<context>\d+)(?:_.+)?",
        path.stem,
    )
    if not match:
        raise ValueError(f"Unexpected flash-attention NCU filename: {path.name}")
    return {
        "batch_size": int(match.group("batch")),
        "context_len": int(match.group("context")),
        "bucket": "attention",
        "kernel_name": "flash_attn",
        "calls_per_decode_step": NUM_LAYERS,
    }


def load_flash_cuda_events(
    path: Path | None,
    *,
    gpu: str,
) -> dict[tuple[int, int], FlashCudaEventRow]:
    if path is None:
        return {}
    if "cuda_events_under_ncu" in path.parts:
        raise SystemExit(
            "--cuda-event-flash must point to a standalone CUDA-event flash sweep. "
            f"Refusing NCU-wrapped CUDA-event file: {path}"
        )

    rows: dict[tuple[int, int], FlashCudaEventRow] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_gpu = (row.get("gpu") or gpu).strip()
            if row_gpu and row_gpu != gpu:
                continue
            batch_size = int(row["batch_size"])
            context_len = int(row["context_len"])
            layer_ms = parse_float(row.get("flash_ms_median"))
            full_model_ms = parse_float(row.get("flash_full_model_ms_median"))
            if full_model_ms <= 0.0 and layer_ms > 0.0:
                full_model_ms = layer_ms * NUM_LAYERS
            rows[(batch_size, context_len)] = FlashCudaEventRow(
                flash_layer_ms_median=layer_ms,
                flash_full_model_ms_median=full_model_ms,
            )
    return rows


def format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def load_fused_metadata(ncu_csv: Path) -> FusedMetadata:
    event_csv = ncu_csv.parent.parent / "cuda_events_under_ncu" / ncu_csv.name
    if not event_csv.exists():
        return FusedMetadata(implementation="", source_status="unknown")
    with event_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            implementation = (row.get("implementation") or "").strip()
            if implementation == "vllm":
                source_status = "source_of_truth"
            elif implementation:
                source_status = "diagnostic"
            else:
                source_status = "unknown"
            return FusedMetadata(
                implementation=implementation,
                source_status=source_status,
            )
    return FusedMetadata(implementation="", source_status="unknown")


def write_rows(args: argparse.Namespace) -> None:
    default_patterns = {
        "gemm": "gemm_*.csv",
        "fused": "fused_*.csv",
        "flash": "flash_attn_*.csv",
    }
    pattern = args.pattern or default_patterns[args.kind]
    csv_paths = sorted(args.ncu_dir.glob(pattern))
    if not csv_paths:
        raise SystemExit(f"No NCU CSV files matched {args.ncu_dir / pattern}")

    common_fields = [
        "tag",
        "kernel_count",
        "ncu_gpu_time_ms_sum",
        "ncu_gpu_time_ms_max",
        "ncu_dram_read_mbytes_sum",
        "ncu_dram_write_mbytes_sum",
        "top_kernel_examples",
        "ncu_csv",
    ]
    if args.kind == "gemm":
        fields = [
            "batch_size",
            "op_name",
            "m",
            "n",
            "k",
            "calls_per_decode_step",
        ] + common_fields
        parse_tag = parse_gemm_tag
    elif args.kind == "fused":
        fields = [
            "batch_size",
            "context_len",
            "bucket",
            "kernel_name",
            "implementation",
            "source_status",
            "calls_per_decode_step",
        ] + common_fields
        parse_tag = parse_fused_tag
    else:
        fields = [
            "batch_size",
            "context_len",
            "bucket",
            "kernel_name",
            "calls_per_decode_step",
            "ncu_flash_layer_ms_sum",
            "ncu_flash_full_model_ms_sum",
            "cuda_event_flash_layer_ms_median",
            "cuda_event_flash_full_model_ms_median",
            "ncu_minus_cuda_event_full_model_ms",
            "ncu_vs_cuda_event_full_model_pct",
        ] + common_fields
        parse_tag = parse_flash_tag

    flash_event_rows = load_flash_cuda_events(
        args.cuda_event_flash if args.kind == "flash" else None,
        gpu=args.cuda_event_gpu,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in csv_paths:
            tag_fields = parse_tag(path)
            if args.kind == "fused":
                metadata = load_fused_metadata(path)
                tag_fields = {
                    **tag_fields,
                    "implementation": metadata.implementation,
                    "source_status": metadata.source_status,
                }
            summary = summarize_report(path, args.top_kernels)
            row = {
                **tag_fields,
                "tag": path.stem,
                "kernel_count": summary.kernel_count,
                "ncu_gpu_time_ms_sum": f"{summary.gpu_time_ms_sum:.9g}",
                "ncu_gpu_time_ms_max": f"{summary.gpu_time_ms_max:.9g}",
                "ncu_dram_read_mbytes_sum": f"{summary.dram_read_mbytes_sum:.9g}",
                "ncu_dram_write_mbytes_sum": f"{summary.dram_write_mbytes_sum:.9g}",
                "top_kernel_examples": summary.top_kernel_examples,
                "ncu_csv": str(path),
            }
            if args.kind == "flash":
                batch_size = int(tag_fields["batch_size"])
                context_len = int(tag_fields["context_len"])
                ncu_full_model_ms = summary.gpu_time_ms_sum * NUM_LAYERS
                event = flash_event_rows.get((batch_size, context_len))
                event_layer_ms = None if event is None else event.flash_layer_ms_median
                event_full_model_ms = (
                    None if event is None else event.flash_full_model_ms_median
                )
                diff_ms = (
                    None
                    if event_full_model_ms is None
                    else ncu_full_model_ms - event_full_model_ms
                )
                pct_diff = (
                    None
                    if event_full_model_ms is None or event_full_model_ms <= 0.0
                    else diff_ms / event_full_model_ms * 100.0
                )
                row.update({
                    "ncu_flash_layer_ms_sum": f"{summary.gpu_time_ms_sum:.9g}",
                    "ncu_flash_full_model_ms_sum": f"{ncu_full_model_ms:.9g}",
                    "cuda_event_flash_layer_ms_median": format_optional_float(
                        event_layer_ms
                    ),
                    "cuda_event_flash_full_model_ms_median": format_optional_float(
                        event_full_model_ms
                    ),
                    "ncu_minus_cuda_event_full_model_ms": format_optional_float(
                        diff_ms
                    ),
                    "ncu_vs_cuda_event_full_model_pct": format_optional_float(
                        pct_diff
                    ),
                })
            writer.writerow(row)


def main() -> None:
    write_rows(parse_args())


if __name__ == "__main__":
    main()
