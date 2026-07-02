# simulator_v2/kernel_floor/validate/decode.py

"""
Validate the composed decode-step floor against measured ground truth.

Runs `KernelFloor.decode_step_ms(batch, ctx)` over every (batch, context_len)
cell in the H100 wide-summary trace and reports the per-cell absolute percent
error (APE) plus the overall MAPE. This is the foundation check before building
prefill: if the decode floor doesn't track the measured decode step within
~10-15%, the per-kernel tables / composition are wrong and prefill would inherit it.

We compare the kernel sum against the measured `decode_step_ms` (the event-timed
step). The floor is a pure kernel sum (no scheduler/host overhead), so a small
*positive* bias in the measurement (floor slightly under) is expected and fine;
large or non-monotonic errors are the red flag.

Run:  python -m simulator_v2.kernel_floor.validate.decode
"""

import csv
from dataclasses import replace
from pathlib import Path

from simulator_v2.configs.config_loader import load_gpu_config, load_model_config
from simulator_v2.kernel_floor.gemm import load_gemm_table
from simulator_v2.kernel_floor.sum_kernels import load_kernel_floor
from simulator_v2.core.mode import Mode, mode

_REPO = Path(__file__).resolve().parents[3]
_GPU_YAML = _REPO / "simulator_v2/configs/gpu_configs/h100.yaml"
_MODEL_YAML = _REPO / "simulator_v2/configs/model_configs/llama3.1-8b.yaml"
_GEMM_CSV = _REPO / "profile_data/kernels/forward_pass/gemm/H100.csv"
_FLASH_CSV = _REPO / "profile_data/kernels/flash_attn/H100.csv"
_PREFILL_FLASH_CSV = _REPO / "profile_data/kernels/fa3_prefill_H100.csv"
_ELEM_JSON = _REPO / "profile_data/kernels/elementwise/H100.json"
_GROUND_TRUTH = _REPO / "profile_data/results/decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv"


@mode(Mode.BACKTEST)
def _score(floor, label: str) -> None:
    rows = []
    with _GROUND_TRUTH.open() as f:
        for r in csv.DictReader(f):
            batch = int(r["batch_size"])
            ctx = int(r["context_len"])
            measured = float(r["decode_step_ms"])
            predicted = floor.decode_step_ms(batch, ctx)
            ape = abs(predicted - measured) / measured * 100.0
            rows.append((batch, ctx, measured, predicted, ape))

    rows.sort(key=lambda x: (x[0], x[1]))
    header = f"{'batch':>5} {'ctx':>6} {'measured_ms':>12} {'floor_ms':>10} {'ratio':>6} {'APE%':>7}"
    print(f"\n### {label}")
    print(header)
    print("-" * len(header))
    for batch, ctx, measured, predicted, ape in rows:
        ratio = predicted / measured
        print(f"{batch:>5} {ctx:>6} {measured:>12.4f} {predicted:>10.4f} {ratio:>6.2f} {ape:>7.1f}")

    apes = [r[4] for r in rows]
    ratios = [r[3] / r[2] for r in rows]
    mape = sum(apes) / len(apes)
    median_ape = sorted(apes)[len(apes) // 2]
    print("-" * len(header))
    print(f"cells={len(rows)}  MAPE={mape:.1f}%  median_APE={median_ape:.1f}%  "
          f"mean_ratio={sum(ratios) / len(ratios):.2f}  "
          f"min_ratio={min(ratios):.2f}  max_ratio={max(ratios):.2f}")


@mode(Mode.BACKTEST)
def main() -> None:
    gpu = load_gpu_config(_GPU_YAML)
    model = load_model_config(_MODEL_YAML)

    # Backtest floor: composed purely from isolated per-kernel microbenches (GEMM table
    # + isolated FA grid + elementwise), GEMM duplicates min-reduced. This is a
    # true bottom-up composition -- NO term is read off the decode-step trace
    # we're validating against. The validated default; mean_ratio ~1.0, MAPE ~10%.
    floor = load_kernel_floor(model, gpu, _GEMM_CSV, _FLASH_CSV, _PREFILL_FLASH_CSV, _ELEM_JSON)
    _score(floor, "composed floor: isolated leaves, GEMM min-reduced (default)")

    # Reference only: same composition with GEMM duplicates MEAN-reduced, to show
    # the ~14% hot bias that min removes (still pure composition, not a fit).
    mean_gemm = replace(floor, gemm=load_gemm_table(_GEMM_CSV, reduce="mean"))
    _score(mean_gemm, "reference: GEMM mean-reduced (pre-fix, ~14% hot)")


if __name__ == "__main__":
    main()
