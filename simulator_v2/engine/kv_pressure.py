# simulator_v2/engine/kv_pressure.py

"""KV-pressure readout for the queue sim.

Per (profile, concurrency) cell, reports:
  * demand pressure -- the peak active-herd resident KV / pool. > 1.0 means the
    in-flight herd alone can't fit, so the engine MUST trim active prefixes
    (tier-2 eviction) -> recompute. This is the analytic onset of the climb.
  * recomp/pool -- the realized re-prefilled (evicted-prefix) tokens / pool, from
    the sim (0 below pressure 1.0, grows above it).
  * the TTFT pred/meas ratio, so you can see the recompute regime (pressure > 1)
    line up with where the prediction is accurate vs the sub-saturation regime
    (pressure < 1), where we currently under-predict (no recompute to drive it).

Run:  python -m simulator_v2.engine.kv_pressure
"""

import glob
import statistics as st
from pathlib import Path

from simulator_v2.core.mode import Mode, mode
from simulator_v2.engine.queue_sim import _build_cohort, predict_ttft
from simulator_v2.getters.hardware import load_kernel_composition_hardware
from simulator_v2.getters.workload import load_benchmark

_REPO = Path(__file__).resolve().parents[2]
_GPU = _REPO / "simulator_v2/configs/gpu_configs/h100.yaml"
_MODEL = _REPO / "simulator_v2/configs/model_configs/llama3.1-8b.yaml"
_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm")


@mode(Mode.BACKTEST)
def peak_demand_pressure(cohort, pool_tokens: float) -> float:
    """Max over turn-index of the active herd's full resident KV / pool. The herd
    at turn t = sessions whose trajectory reaches t; each holds its turn-t context
    plus its generated output (isl + osl) at peak. > 1.0 => the herd overflows the
    pool on its own, forcing active-prefix eviction (recompute)."""
    max_turns = max((len(s.turns) for s in cohort), default=0)
    peak = 0.0
    for t in range(max_turns):
        demand = sum(
            s.turns[t].isl_tokens + s.turns[t].osl_tokens
            for s in cohort if len(s.turns) > t
        )
        peak = max(peak, demand / pool_tokens)
    return peak


@mode(Mode.BACKTEST)
def main() -> None:
    hw = load_kernel_composition_hardware(_GPU, _MODEL, tp=1)
    pool = hw.kv_pool_blocks * hw.cache_block_size
    print(f"KV pool = {hw.kv_pool_blocks} blocks x {hw.cache_block_size} = {pool:,} tokens\n")

    rows = []
    for f in sorted(glob.glob(str(_BASE / "*_conc*.json"))):
        if "_per_turn" in f:
            continue
        cell = load_benchmark(Path(f))
        if not cell.turns or not cell.ground_truth:
            continue
        cohort = _build_cohort(cell.turns, cell.concurrency, cell.trajectories)
        pressure = peak_demand_pressure(cohort, pool)
        stats: dict = {}
        preds = predict_ttft(
            hw, cell.turns, cell.concurrency, trajectories=cell.trajectories, stats=stats
        )
        gt = [g.ttft_ms for g in cell.ground_truth]
        n = min(len(preds), len(gt))
        mp, mm = st.mean(preds[:n]), st.mean(gt[:n])
        rows.append((
            cell.profile, cell.concurrency, pressure,
            stats.get("recompute_tokens", 0.0) / pool, mp, mm, mp / mm if mm else 0.0,
        ))

    rows.sort(key=lambda r: (r[0], r[1]))
    hdr = f"{'profile':30} {'conc':>4} {'pressure':>8} {'recomp/pool':>11} {'pred':>7} {'meas':>7} {'ratio':>5}"
    print(hdr)
    print("-" * len(hdr))
    last = None
    for prof, c, pressure, recomp, mp, mm, ratio in rows:
        if last is not None and prof != last:
            print()
        last = prof
        cross = " <== crosses 1.0" if pressure > 1.0 else ""
        print(f"{prof[:30]:30} {c:>4} {pressure:>8.2f} {recomp:>11.2f} {mp:>7.1f} {mm:>7.1f} {ratio:>5.2f}{cross}")


if __name__ == "__main__":
    main()
