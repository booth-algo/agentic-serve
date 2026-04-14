"""
Parse profiler output (PyTorch Profiler JSON or ncu CSV) into structured
data for roofline plotting.

For PyTorch profiler output: extracts per-kernel timing and FLOPs.
Computes theoretical arithmetic intensity for GEMM kernels from known
model architecture parameters.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.roofline.roofline_config import classify_kernel, H100_SXM


@dataclass
class KernelRecord:
    name: str
    category: str
    cuda_time_us: float
    calls: int
    flops: float
    # For roofline: theoretical estimates
    theoretical_bytes: float      # estimated DRAM bytes from tensor shapes
    arithmetic_intensity: float   # FLOPs / byte
    achieved_tflops: float


def parse_shape_dims(input_shapes_str: str) -> list[list[int]]:
    """Parse input_shapes string like '[[4, 512, 4096], [4096, 14336]]' into list of dim lists."""
    try:
        dims = re.findall(r'\[[\d,\s]+\]', input_shapes_str)
        return [[int(x.strip()) for x in d.strip('[]').split(',')] for d in dims]
    except Exception:
        return []


def tensor_numel(dims: list[int]) -> int:
    """Total number of elements in a tensor."""
    n = 1
    for d in dims:
        n *= d
    return n


# FLOPs per element for common elementwise ops
ELEMENTWISE_FLOPS: dict[str, float] = {
    "aten::mul": 1, "aten::add": 1, "aten::neg": 1, "aten::sub": 1,
    "aten::pow": 1, "aten::rsqrt": 1, "aten::mean": 1,
    "aten::silu": 4, "aten::gelu": 4,  # activation: exp + mul + add + ...
    "aten::cos": 8, "aten::sin": 8,    # transcendental
}


def estimate_elementwise(name: str, input_shapes_str: str, cuda_time_us: float) -> tuple[float, float]:
    """Estimate FLOPs and bytes for elementwise/unary/reduction ops from shapes.

    Returns (flops, bytes).
    """
    shapes = parse_shape_dims(input_shapes_str)
    if not shapes:
        # Fallback: estimate from time at peak bandwidth
        est_bytes = cuda_time_us * H100_SXM.hbm_bandwidth_tb_s * 1e6
        return est_bytes * 0.5, est_bytes

    # Use first input tensor to estimate size
    numel = tensor_numel(shapes[0])
    bytes_per_elem = 2  # BF16

    # FLOPs: look up by op name
    flops_per_elem = 1.0
    for op, fpe in ELEMENTWISE_FLOPS.items():
        if op in name:
            flops_per_elem = fpe
            break

    # Bytes: read all inputs + write output
    # Most ops: read 1-2 inputs, write 1 output
    num_inputs = len(shapes)
    total_read = sum(tensor_numel(s) for s in shapes if s) * bytes_per_elem
    total_write = numel * bytes_per_elem  # output same size as first input
    total_bytes = total_read + total_write
    total_flops = numel * flops_per_elem

    return total_flops, total_bytes


def estimate_gemm_bytes(input_shapes_str: str) -> float:
    """Estimate DRAM bytes for a GEMM from input shapes.

    For aten::mm with inputs [M,K] and [K,N]:
    - Read A: M*K elements
    - Read B: K*N elements
    - Write C: M*N elements
    All in BF16 (2 bytes per element).
    """
    # Parse shapes like "[[4, 512, 4096], [4096, 14336]]" or "[[4, 512, 4096], [14336, 4096]]"
    try:
        # Find all dimension lists
        dims = re.findall(r'\[[\d,\s]+\]', input_shapes_str)
        if len(dims) < 2:
            return 0

        def parse_dims(s):
            return [int(x.strip()) for x in s.strip('[]').split(',')]

        a_dims = parse_dims(dims[0])
        b_dims = parse_dims(dims[1])

        # Flatten batch dims: last 2 dims are the matrix dims
        if len(a_dims) >= 2:
            M = 1
            for d in a_dims[:-1]:
                M *= d
            K_a = a_dims[-1]
        else:
            return 0

        if len(b_dims) >= 2:
            K_b = b_dims[-2] if len(b_dims) == 2 else b_dims[-2]
            N = b_dims[-1]
        else:
            return 0

        # BF16: 2 bytes per element
        bytes_a = M * K_a * 2
        bytes_b = K_b * N * 2
        bytes_c = M * N * 2
        return bytes_a + bytes_b + bytes_c

    except Exception:
        return 0


def compute_attention_flops(model_info: dict, batch_size: int, seq_len: int, phase: str, num_layers: int) -> float:
    """Compute flash attention FLOPs analytically per layer.

    FLOPs = 2 * batch * num_heads * seq_len * kv_len * head_dim * 2 (fwd)
    For prefill: kv_len = seq_len
    For decode: kv_len = seq_len (KV cache length), query_len = 1
    """
    num_heads = model_info.get("num_attention_heads", 32)
    head_dim = model_info.get("head_dim", 128)

    if phase == "decode":
        # decode: Q is [batch, 1, heads, dim], K/V are [batch, seq_len, kv_heads, dim]
        # FLOPs ≈ 2 * batch * num_heads * 1 * seq_len * head_dim * 2
        return 4 * batch_size * num_heads * seq_len * head_dim
    else:
        # prefill: Q,K,V are all [batch, seq_len, heads, dim]
        # FLOPs ≈ 2 * batch * num_heads * seq_len * seq_len * head_dim * 2
        return 4 * batch_size * num_heads * seq_len * seq_len * head_dim


def parse_torch_profiler_json(data: dict) -> list[KernelRecord]:
    """Parse a PyTorch profiler JSON output into KernelRecords."""
    records = []
    model_info = data.get("model_info", {})
    batch_size = data.get("batch_size", 1)
    seq_len = data.get("seq_len", 512)
    phase = data.get("phase", "prefill")
    num_layers = data.get("num_layers_profiled", 2)

    # Pre-compute attention FLOPs per layer (analytically)
    attn_flops_per_layer = compute_attention_flops(model_info, batch_size, seq_len, phase, num_layers)

    for k in data.get("kernel_summary", []):
        name = k["name"]
        category = classify_kernel(name)
        if category == "profiler-overhead":
            continue

        # Skip low-level CUDA GEMM kernels (nvjet, cublas) — they are child
        # kernels of aten::mm which already has correct FLOPs and timing.
        # Including both would double-count.
        if category == "gemm" and not name.startswith("aten::"):
            continue

        cuda_time_us = k["cuda_time_us"]
        calls = k["calls"]
        flops = k["flops"]

        # Compute bytes and AI
        theoretical_bytes = 0

        if category == "gemm":
            # Use actual tensor shapes from profiler
            theoretical_bytes = estimate_gemm_bytes(k.get("input_shapes", ""))
            if theoretical_bytes == 0 and flops > 0:
                # Fallback: estimate from FLOPs assuming typical large GEMM AI
                theoretical_bytes = flops / 150  # conservative estimate

        elif category == "attention":
            # Compute FLOPs analytically from model architecture
            flops = attn_flops_per_layer  # total across profiled layers
            # Flash attention memory: reads Q,K,V + writes O, all in BF16
            # Per head: Q=[bs,seq,dim], K=[bs,kv_len,dim], V=[bs,kv_len,dim], O=[bs,seq,dim]
            num_heads = model_info.get("num_attention_heads", 32)
            head_dim = model_info.get("head_dim", 128)
            if phase == "decode":
                q_len = 1
                kv_len = seq_len
            else:
                q_len = seq_len
                kv_len = seq_len
            # Bytes: read Q + K + V + write O, all BF16 (2 bytes)
            theoretical_bytes = 2 * batch_size * num_heads * head_dim * (q_len + 2 * kv_len + q_len)

        elif category in ("elementwise", "activation", "reduce", "layernorm", "rope"):
            # Estimate from tensor shapes when available
            flops, theoretical_bytes = estimate_elementwise(
                name, k.get("input_shapes", ""), cuda_time_us)

        elif category == "memory" or category == "indexing":
            # Pure data movement — estimate bytes from duration
            theoretical_bytes = cuda_time_us * H100_SXM.hbm_bandwidth_tb_s * 1e6
            flops = 0

        # Compute derived metrics
        if theoretical_bytes > 0 and flops > 0:
            arithmetic_intensity = flops / theoretical_bytes
        elif category in ("memory", "indexing"):
            arithmetic_intensity = 0.01
        else:
            arithmetic_intensity = 0

        if cuda_time_us > 0 and flops > 0:
            achieved_tflops = flops / (cuda_time_us * 1e-6) / 1e12
        else:
            achieved_tflops = 0

        records.append(KernelRecord(
            name=name,
            category=category,
            cuda_time_us=cuda_time_us,
            calls=calls,
            flops=flops,
            theoretical_bytes=theoretical_bytes,
            arithmetic_intensity=arithmetic_intensity,
            achieved_tflops=achieved_tflops,
        ))

    return records


def summarize_by_category(records: list[KernelRecord]) -> dict:
    """Aggregate kernel records by category."""
    cats: dict[str, dict] = {}
    total_time = sum(r.cuda_time_us for r in records)

    for r in records:
        if r.category not in cats:
            cats[r.category] = {
                "total_cuda_time_us": 0,
                "total_flops": 0,
                "total_bytes": 0,
                "kernel_count": 0,
                "call_count": 0,
                "duration_pct": 0,
            }
        c = cats[r.category]
        c["total_cuda_time_us"] += r.cuda_time_us
        c["total_flops"] += r.flops
        c["total_bytes"] += r.theoretical_bytes
        c["kernel_count"] += 1
        c["call_count"] += r.calls

    for cat, c in cats.items():
        c["duration_pct"] = (c["total_cuda_time_us"] / total_time * 100) if total_time > 0 else 0
        if c["total_bytes"] > 0:
            c["avg_arithmetic_intensity"] = c["total_flops"] / c["total_bytes"]
        else:
            c["avg_arithmetic_intensity"] = 0
        if c["total_cuda_time_us"] > 0:
            c["avg_achieved_tflops"] = c["total_flops"] / (c["total_cuda_time_us"] * 1e-6) / 1e12
        else:
            c["avg_achieved_tflops"] = 0

    return cats


def parse_file(input_path: Path) -> dict:
    """Parse a profiler output JSON file."""
    data = json.loads(input_path.read_text())

    records = parse_torch_profiler_json(data)
    summary = summarize_by_category(records)

    return {
        "model": data.get("model", "unknown"),
        "phase": data.get("phase", "unknown"),
        "batch_size": data.get("batch_size", 0),
        "seq_len": data.get("seq_len", 0),
        "num_layers_profiled": data.get("num_layers_profiled", 0),
        "profiler": data.get("profiler", "torch"),
        "source_file": input_path.name,
        "total_kernels": len(records),
        "kernels": [asdict(r) for r in records],
        "category_summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse profiler output for roofline analysis")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file or glob pattern")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/roofline/parsed/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "roofline" / "parsed"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if "*" in args.input:
        files = sorted(input_path.parent.glob(input_path.name))
    else:
        files = [input_path]

    for f in files:
        if not f.exists() or f.suffix != ".json" or "trace" in f.name:
            continue

        print(f"Parsing: {f.name}...")
        try:
            result = parse_file(f)
            out_path = output_dir / f.name
            out_path.write_text(json.dumps(result, indent=2))

            # Print summary
            print(f"  {result['total_kernels']} kernels")
            for cat, info in sorted(result["category_summary"].items(), key=lambda x: -x[1]["duration_pct"]):
                print(f"    {cat:20s}: {info['duration_pct']:5.1f}% | "
                      f"AI={info['avg_arithmetic_intensity']:.1f} FLOP/B | "
                      f"{info['avg_achieved_tflops']:.2f} TFLOPS")
            print(f"  Output: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Parsed {len(files)} file(s).")


if __name__ == "__main__":
    main()
