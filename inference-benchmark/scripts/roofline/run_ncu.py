"""
Per-kernel roofline profiler using NVIDIA Nsight Compute (ncu).

Orchestrates ncu runs across models, batch sizes, and phases (prefill/decode).
Each run invokes _ncu_target.py as a subprocess under ncu.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow imports from the roofline package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.roofline.roofline_config import (
    MODEL_REGISTRY,
    NCU_PATH,
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN,
    DEFAULT_NUM_LAYERS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / "_ncu_target.py"
PYTHON = os.environ.get("PYTHON", sys.executable)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_ncu_command(
    model_name: str,
    batch_size: int,
    seq_len: int,
    phase: str,
    num_layers: int,
    device: str,
    output_csv: Path,
) -> list[str]:
    """Build the ncu command line."""
    spec = MODEL_REGISTRY[model_name]

    env_vars = {
        "ROOFLINE_MODEL_PATH": spec.path,
        "ROOFLINE_LAYERS": str(num_layers),
        "ROOFLINE_BATCH_SIZE": str(batch_size),
        "ROOFLINE_SEQ_LEN": str(seq_len),
        "ROOFLINE_PHASE": phase,
        "ROOFLINE_DEVICE": device,
        "ROOFLINE_TRUST_REMOTE_CODE": "true" if spec.trust_remote_code else "false",
    }

    # Build env prefix for the subprocess
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())

    ncu_cmd = [
        NCU_PATH,
        "--set", "full",
        "--csv",
        "--profile-from-start", "no",  # wait for cudaProfilerStart()
        "--target-processes", "all",
        "-o", str(output_csv.with_suffix("")),  # ncu adds .ncu-rep
        "-f",  # overwrite existing
        PYTHON, "-B", str(TARGET_SCRIPT),
    ]

    return ncu_cmd, env_vars


def run_profile(
    model_name: str,
    batch_size: int,
    seq_len: int,
    phase: str,
    num_layers: int,
    device: str,
    output_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Run a single ncu profiling pass."""
    output_csv = output_dir / f"{model_name}_{phase}_bs{batch_size}.ncu-rep"

    if output_csv.exists():
        print(f"  SKIP (exists): {output_csv.name}")
        return True

    ncu_cmd, env_vars = build_ncu_command(
        model_name, batch_size, seq_len, phase, num_layers, device, output_csv,
    )

    # Prepare environment
    env = os.environ.copy()
    env.update(env_vars)
    env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

    cmd_str = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " " + " ".join(ncu_cmd)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True

    print(f"  Running: {model_name} {phase} bs={batch_size}...")
    print(f"  Output:  {output_csv}")

    try:
        result = subprocess.run(
            ncu_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
            # Print last 20 lines of stderr
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"    {line}", file=sys.stderr)
            return False

        print(f"  Done: {output_csv.name}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1 hour", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}", file=sys.stderr)
        return False


def export_csv(ncu_rep_path: Path, output_dir: Path) -> Path:
    """Export an .ncu-rep file to CSV for parsing."""
    csv_path = output_dir / ncu_rep_path.with_suffix(".csv").name

    cmd = [
        NCU_PATH, "--import", str(ncu_rep_path),
        "--csv",
        "--page", "raw",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        csv_path.write_text(result.stdout)
        print(f"  Exported: {csv_path.name}")
        return csv_path
    else:
        print(f"  CSV export failed: {result.stderr[:200]}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Per-kernel roofline profiler")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name. Available: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,4,8,16,32,64)")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Prefill sequence length (default: 512)")
    parser.add_argument("--phases", type=str, default="prefill,decode",
                        help="Comma-separated phases: prefill,decode")
    parser.add_argument("--layers", type=int, default=DEFAULT_NUM_LAYERS,
                        help="Number of decoder layers to profile (default: 2)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU device (default: cuda:0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/roofline/raw/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ncu commands without executing")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export .ncu-rep files to CSV after profiling")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else DEFAULT_BATCH_SIZES
    phases = [p.strip() for p in args.phases.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "roofline" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = MODEL_REGISTRY[args.model]
    print(f"═══ Roofline Profile: {args.model} ═══")
    print(f"  Model path: {spec.path}")
    print(f"  TP size:    {spec.tp_size}")
    print(f"  MoE:        {spec.is_moe}")
    print(f"  Layers:     {args.layers}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Phases:     {phases}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {output_dir}")
    print()

    successes = 0
    failures = 0

    for phase in phases:
        print(f"── {phase.upper()} ──")
        for bs in batch_sizes:
            ok = run_profile(
                model_name=args.model,
                batch_size=bs,
                seq_len=args.seq_len,
                phase=phase,
                num_layers=args.layers,
                device=args.device,
                output_dir=output_dir,
                dry_run=args.dry_run,
            )
            if ok:
                successes += 1
            else:
                failures += 1

    print(f"\n═══ Summary: {successes} succeeded, {failures} failed ═══")

    # Export to CSV if requested
    if args.export_csv and not args.dry_run:
        print("\n── Exporting to CSV ──")
        csv_dir = PROJECT_ROOT / "results" / "roofline" / "raw"
        for rep_file in output_dir.glob(f"{args.model}_*.ncu-rep"):
            export_csv(rep_file, csv_dir)


if __name__ == "__main__":
    main()
