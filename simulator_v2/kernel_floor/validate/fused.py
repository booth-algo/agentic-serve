# simulator_v2/kernel_floor/validate/fused.py

"""
Validate the fused-step floor (decodes + a prefill chunk in one forward pass)
against the measured mixed-step profile.

The profile (mixed_step_profile_llama31_8b_H100.csv) is a (prefill_tokens,
decode_batch) -> step_ms grid with NO decode context column, and its absolute
level sits in a heavier regime than our CUDA-graphed per-kernel tables (pure-decode
rows read ~13.5 ms vs our ~6.5 ms floor -- looks eager / per-step overhead). So
absolute MAPE here would conflate a regime offset with composition error.

PRIMARY metric -- the PREFILL INCREMENT:  delta = step(p, b) - step(0, b).
Subtracting the same-batch pure-decode row cancels BOTH confounds:
  * the fixed per-step regime overhead (present in both terms), and
  * the decode-attention term decode_us(ctx, b) (identical in both) -> so the
    unknown decode context drops out entirely.
What's left is the chunk's marginal cost on top of b decodes -- i.e. the
piggyback: how much a prefill chunk adds when it rides with decodes. Our
prediction is the same increment of fused_step_ms, which is decode_ctx-independent
by the same cancellation.

This directly tests the insight: a SMALL chunk should add almost nothing (its
linear work is ~free atop the already-loaded weights -- memory-bound), while a
LARGE chunk should add its full compute (compute-bound). The transition is the
piggyback.

Run:  python -m simulator_v2.kernel_floor.validate.fused
"""

import csv
from pathlib import Path

from simulator_v2.configs.config_loader import load_gpu_config, load_model_config
from simulator_v2.kernel_floor.sum_kernels import load_kernel_floor
from simulator_v2.core.mode import Mode, mode

_REPO = Path(__file__).resolve().parents[3]
_GPU_YAML = _REPO / "simulator_v2/configs/gpu_configs/h100.yaml"
_MODEL_YAML = _REPO / "simulator_v2/configs/model_configs/llama3.1-8b.yaml"
_GEMM_CSV = _REPO / "profile_data/kernels/forward_pass/gemm/H100.csv"
_FLASH_CSV = _REPO / "profile_data/kernels/flash_attn/H100.csv"
_PREFILL_FLASH_CSV = _REPO / "profile_data/kernels/fa3_prefill_H100.csv"
_ELEM_JSON = _REPO / "profile_data/kernels/elementwise/H100.json"
_MIXED = _REPO / "profile_data/kernels/mixed_step_profile_llama31_8b_H100.csv"

# Cancels out of every prefill-increment below; only labels the optional absolute view.
_DECODE_CTX = 2048


@mode(Mode.BACKTEST)
def main() -> None:
    gpu = load_gpu_config(_GPU_YAML)
    model = load_model_config(_MODEL_YAML)
    floor = load_kernel_floor(model, gpu, _GEMM_CSV, _FLASH_CSV, _PREFILL_FLASH_CSV, _ELEM_JSON)

    data: dict[tuple[int, int], float] = {}
    with _MIXED.open() as f:
        for r in csv.DictReader(f):
            data[(int(r["prefill_tokens"]), int(r["decode_batch"]))] = float(r["step_ms"])
    prefills = sorted({p for p, _ in data if p > 0})
    batches = sorted({b for _, b in data})

    print("### prefill increment  delta = step(p,b) - step(0,b)   [regime- & ctx-independent]")
    header = f"{'p':>5} {'b':>4} {'meas_delta':>11} {'pred_delta':>11} {'ratio':>6} {'APE%':>7}"
    print(header)
    print("-" * len(header))
    apes = []
    for p in prefills:
        for b in batches:
            if (0, b) not in data or (p, b) not in data:
                continue
            meas = data[(p, b)] - data[(0, b)]
            pred = floor.fused_step_ms(p, b, _DECODE_CTX) - floor.fused_step_ms(0, b, _DECODE_CTX)
            if meas <= 0.5:  # near-zero / noisy small chunks: report, don't score
                print(f"{p:>5} {b:>4} {meas:>11.2f} {pred:>11.2f} {'--':>6} {'(noise)':>7}")
                continue
            ape = abs(pred - meas) / meas * 100.0
            apes.append(ape)
            print(f"{p:>5} {b:>4} {meas:>11.2f} {pred:>11.2f} {pred / meas:>6.2f} {ape:>7.1f}")
    print("-" * len(header))
    if apes:
        print(f"scored cells={len(apes)}  MAPE={sum(apes) / len(apes):.1f}%")

    print("\n### piggyback: marginal cost of decodes at a fixed chunk  (free when small, costly when large)")
    hdr2 = f"{'p':>5} {'meas d(b1->b256)':>16} {'pred(linear-only)':>18}"
    print(hdr2)
    print("-" * len(hdr2))
    gemm, elem = floor._pricers()
    for p in prefills:
        if (p, 1) not in data or (p, 256) not in data:
            continue
        meas_d = data[(p, 256)] - data[(p, 1)]
        # linear-only marginal of +255 decode tokens (ctx-independent piggyback core)
        lin_d = (floor._block_linear_us(p + 256, gemm, elem)
                 - floor._block_linear_us(p + 1, gemm, elem)) * model.n_layers / 1000.0
        print(f"{p:>5} {meas_d:>16.2f} {lin_d:>18.2f}")


if __name__ == "__main__":
    main()
