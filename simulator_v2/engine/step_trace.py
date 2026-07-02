# simulator_v2/engine/step_trace.py

"""Per-step decomposition of the TTFT queue sim, for one cell.

Answers: when the sim under-predicts TTFT, is it draining the barrier herd in
*too few steps* (admission too parallel -- prefill packed to the token budget in
~2 giant steps) or is each step *too cheap* (per-step overhead / util)? Step cost
is the kernel floor (measured), so if the cost is trustworthy the gap must be in
the step *count* -> a scheduler-serialization fidelity gap, not a pricing one.

For each turn-index herd it prints:
  * prefill phase: #steps, mean step_ms, mean prefill tokens/step, makespan
  * decode phase:  #steps, mean step_ms
  * sim vs measured per-turn TTFT, and `implied_steps` = measured / mean_prefill
    step_ms (how many sim-priced steps the *measured* makespan corresponds to).

Run:  python -m simulator_v2.engine.step_trace [profile] [conc]
"""

import statistics as st
import sys
from pathlib import Path

import simulator_v2.engine.queue_sim as qs
from simulator_v2.core.mode import Mode, mode
from simulator_v2.getters.hardware import load_kernel_composition_hardware
from simulator_v2.getters.workload import load_benchmark

_REPO = Path(__file__).resolve().parents[2]
_GPU = _REPO / "simulator_v2/configs/gpu_configs/h100.yaml"
_MODEL = _REPO / "simulator_v2/configs/model_configs/llama3.1-8b.yaml"
_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm")


@mode(Mode.BACKTEST)
def _install_trace(hw) -> list[dict]:
    """Monkeypatch _price_step to append one record per engine step. Returns the
    (initially empty) trace list, filled as the sim runs."""
    trace: list[dict] = []
    orig = qs._price_step

    def traced(state) -> float:
        turn = state.current_turn
        n_pref, n_run, n_wait = len(state.prefilling), len(state.running), len(state.waiting)
        step_ms = orig(state)  # sets req.chunk on each prefilling req
        total_chunk = sum(r.chunk for r in state.prefilling.values())
        decode_ctx = (
            st.mean(r.kv_tokens for r in state.running.values()) if state.running else 0.0
        )
        decode_ms = hw.decode_step_ms(n_run, decode_ctx) if n_run else 0.0
        prefill_gpu_ms = hw.fused_step_ms(int(total_chunk), 0, 0) if total_chunk > 0 else 0.0
        trace.append(dict(
            turn=turn, clock=state.clock, step_ms=step_ms, total_chunk=total_chunk,
            n_pref=n_pref, n_run=n_run, n_wait=n_wait,
            decode_ms=decode_ms, prefill_gpu_ms=prefill_gpu_ms,
        ))
        return step_ms

    qs._price_step = traced
    return trace


@mode(Mode.BACKTEST)
def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "chat-multiturn-synth"
    conc = sys.argv[2] if len(sys.argv) > 2 else "120"
    path = _BASE / f"{profile}_conc{conc}.json"
    if not path.exists():
        raise SystemExit(f"missing cell: {path}")

    hw = load_kernel_composition_hardware(_GPU, _MODEL, tp=1)
    cell = load_benchmark(path)
    sched = hw.sched
    budget = sched.max_num_batched_tokens or 8192
    print(f"{profile} conc{conc}  |  {len(cell.turns)} turns  |  "
          f"max_batched={budget}  max_seqs={sched.max_num_seqs}  "
          f"long_prefill={sched.long_prefill_token_threshold}")
    print(f"KV pool = {hw.kv_pool_blocks} blocks  |  full-budget step = "
          f"{hw.fused_step_ms(int(budget), 0, 0):.1f}ms  "
          f"host_floor(request_overhead) = {hw.request_overhead_ms:.1f}ms\n")

    trace = _install_trace(hw)
    preds = qs.predict_ttft(
        hw, cell.turns, cell.concurrency,
        trajectories=cell.trajectories, shared_prefix_tokens=cell.shared_prefix_tokens,
    )
    meas = [g.ttft_ms for g in cell.ground_truth]

    # Group trace steps by the herd (turn index) in flight, split prefill vs decode.
    turns = sorted({r["turn"] for r in trace})
    hdr = (f"{'turn':>4} {'pf_steps':>8} {'pf_ms/st':>8} {'pf_tok/st':>9} "
           f"{'pf_make':>8} {'dc_steps':>8} {'dc_ms/st':>8} "
           f"{'sim_ttft':>8} {'meas':>7} {'ratio':>5} {'impl_st':>7}")
    print(hdr)
    print("-" * len(hdr))
    for t in turns:
        steps = [r for r in trace if r["turn"] == t]
        pf = [r for r in steps if r["total_chunk"] > 0.0]
        dc = [r for r in steps if r["total_chunk"] <= 0.0]
        pf_ms = st.mean([r["step_ms"] for r in pf]) if pf else 0.0
        pf_tok = st.mean([r["total_chunk"] for r in pf]) if pf else 0.0
        pf_make = sum(r["step_ms"] for r in pf)
        dc_ms = st.mean([r["step_ms"] for r in dc]) if dc else 0.0
        sim = preds[t] if t < len(preds) else 0.0
        m = meas[t] if t < len(meas) else 0.0
        ratio = sim / m if m else 0.0
        impl = m / pf_ms if pf_ms else 0.0
        print(f"{t:>4} {len(pf):>8} {pf_ms:>8.1f} {pf_tok:>9.0f} {pf_make:>8.1f} "
              f"{len(dc):>8} {dc_ms:>8.2f} {sim:>8.1f} {m:>7.1f} {ratio:>5.2f} {impl:>7.1f}")

    # Aggregate read: is the gap step-count or step-cost?
    all_pf = [r for r in trace if r["total_chunk"] > 0.0]
    print(f"\nprefill steps total: {len(all_pf)}  |  "
          f"mean prefill tokens/step: {st.mean([r['total_chunk'] for r in all_pf]):.0f} "
          f"(budget {budget})")
    print(f"mean prefill step_ms: {st.mean([r['step_ms'] for r in all_pf]):.1f}  |  "
          f"steps where prefill branch dominated: "
          f"{sum(1 for r in all_pf if r['prefill_gpu_ms'] >= r['decode_ms'])}/{len(all_pf)}")


if __name__ == "__main__":
    main()
