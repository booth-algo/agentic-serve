# simulator_v2/kernel_floor/validate/prefill.py

"""
Validate the composed prefill-step floor against measured prefill.

PRIMARY (full-compute): compares our prefill LINEAR floor (everything that scales
~linearly in the chunk's token count -- GEMMs, elementwise, kv-write, lm_head;
i.e. prefill_step_ms minus the O(N^2) causal-attention term) against the measured
per-step GEMM intercept from the prefix-caching-DISABLED util sweep
(prefill_gemm_util_H100.json, from profiling/.../prefill_util_sweep.py). That
sweep runs the real engine with enable_prefix_caching=False so every step does its
full compute -- the only valid ground truth for a physical full-compute prefill.
Its util_sim plateaus at ~0.64-0.75 (NO ramp to 1.0, NEVER sub-physical).

We compare the LINEAR part (not the total) because the sweep's per-step attention
runs with a GROWING kv context (a long prompt chunked over many steps), which is
cached-prefill attention, not the q_len=kv_len=N full-causal attention our
PrefillAttnGrid models. The GEMM intercept isolates the m-linear compute, which is
the apples-to-apples target. Prefill attention is separately grounded in the FA3
q=kv=N measurement (fa3_prefill_H100.csv) that feeds PrefillAttnGrid.

DIAGNOSTIC (why not the dense profile): the H100 dense prefill profile
(prefill_profile_H100_dense.csv) goes SUB-PHYSICAL at large N -- the implied FLOP
util (xpeak) exceeds 1.0 (impossible), i.e. those rows are prefix-cached/chunked,
not full-compute. We print xpeak to show exactly where that file stops being a
valid target.

Run:  python -m simulator_v2.kernel_floor.validate.prefill
"""

import csv
import json
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
_FULL_COMPUTE_JSON = _REPO / "profile_data/kernels/prefill_gemm_util_H100.json"
_DENSE_PROFILE = _REPO / "profile_data/kernels/prefill_profile_H100_dense.csv"


@mode(Mode.BACKTEST)
def _validate_full_compute(floor, model, gpu) -> None:
    anchors = json.loads(_FULL_COMPUTE_JSON.read_text())["anchors"]
    print("### full-compute sweep (prefix-caching OFF): our LINEAR floor vs measured GEMM intercept")
    header = f"{'m':>6} {'our_lin_ms':>10} {'meas_ms':>9} {'ratio':>6} {'util_sim':>8} {'APE%':>7}"
    print(header)
    print("-" * len(header))
    apes = []
    for a in anchors:
        m = int(a["m_tokens"])
        measured = float(a["gemm_intercept_ms"])
        total = floor.prefill_step_ms(m)
        attn = floor.prefill_attn.prefill_us(
            m, gpu, n_heads=model.n_heads, n_kv_heads=model.kv_heads, head_dim=model.head_dim,
        ) / 1000.0
        linear = total - attn
        ape = abs(linear - measured) / measured * 100.0
        apes.append(ape)
        print(f"{m:>6} {linear:>10.2f} {measured:>9.2f} {linear / measured:>6.2f} "
              f"{float(a['util_sim']):>8.3f} {ape:>7.1f}")
    print("-" * len(header))
    print(f"cells={len(apes)}  MAPE={sum(apes) / len(apes):.1f}%  (linear prefill compute vs full-compute sweep)")


@mode(Mode.BACKTEST)
def _diagnose_dense_profile(floor, model, gpu) -> None:
    peak, nparams = gpu.peak_flops_per_s, model.n_params
    print("\n### diagnostic: dense profile is cache-contaminated at large N (xpeak>1 = impossible)")
    header = f"{'N':>6} {'meas_ms':>9} {'floor_ms':>9} {'ratio':>6} {'xpeak':>6}  flag"
    print(header)
    print("-" * len(header))
    with _DENSE_PROFILE.open() as f:
        for r in csv.DictReader(f):
            n = int(r["prefill_tokens"])
            measured = float(r["prefill_ms"])
            predicted = floor.prefill_step_ms(n)
            xpeak = (2.0 * nparams * n) / (measured / 1000.0) / peak
            flag = "SUB-PHYSICAL" if xpeak > 1.0 else ""
            print(f"{n:>6} {measured:>9.2f} {predicted:>9.2f} {predicted / measured:>6.2f} {xpeak:>6.2f}  {flag}")


@mode(Mode.BACKTEST)
def main() -> None:
    gpu = load_gpu_config(_GPU_YAML)
    model = load_model_config(_MODEL_YAML)
    floor = load_kernel_floor(model, gpu, _GEMM_CSV, _FLASH_CSV, _PREFILL_FLASH_CSV, _ELEM_JSON)
    _validate_full_compute(floor, model, gpu)
    _diagnose_dense_profile(floor, model, gpu)


if __name__ == "__main__":
    main()
