## Notes

### Handwavy
- yaml numbers are handwavy and need to be remeasured

### Structure

main.py parses args and dispatches by --mode (thin proxies, MODES dict):

  backtest -> engine/backtest.py::run
    getters/hardware.load_roofline_hardware()   # roofline today; kernel HW still a stub
    getters/workload.load_benchmark(path)        -> Cell (turns + ground_truth)
    engine/predict.predict(hw, turns, conc)
    _score()                                      -> mean_ape vs ground truth (MAPE)

  forward  -> engine/forward.py::run
    getters/hardware.load_roofline_hardware()
    getters/workload.load_distribution(path)      # NotImplementedError (stub)
    engine/predict.predict(hw, turns, conc)
    _report()                                     -> per-turn predictions (no ground truth)

Shared engine (mode-agnostic) - engine/predict.py::predict:
  _tpot_ms() -> hw.decode_step_ms(batch, ctx)     # per-turn TPOT
  _ttft_ms() -> hw.prefill_ms(new, cached, batch) # per-turn TTFT
  compose    -> e2el = ttft + tpot * osl_tokens

Notes:
- hardware is NOT yet mode-selected: both modes use roofline (load_kernel_hardware is a stub).
- scoring/reporting live in the drivers (backtest/forward), not in engine/predict.

### Need to build

getters/hardware -> load_kernel_hardware

kernel_floor/ builds floor.
Then kv_wall/ builds ceiling.

### Kernel notes

-- Decode --
GEMM fused kernels basically stack the GEMMs on top of each other, increasing M,N,K dimensions.

We can extrapolate them instead of defaulting to a roofline.

-- Prefill --
vLLM splits into bite size.

### Basically

-- Decode --

Kernel_Floor::sum_kernels --> TPOT floor
KV_Wall --> TPOT ceiling

-- Prefill --

Kernel_Floor::sum_kernels (prefill_step_ms / fused_step_ms) --> per-step prefill cost
[queue sim] --> TTFT   (NOT BUILT — schedules chunks across the cohort: barrier
                        arrival, chunked-prefill budget, KV eviction/recompute)

NB prefill has no floor/ceiling pair like decode. TTFT = the queue sim running the
per-step prefill cost across the cohort; saturation emerges from the sim (queue +
eviction backlog), not a separate ceiling artifact.

stub now: predict._ttft_ms = hw.prefill_ms(new, cached, batch)  (one per-turn
estimate, no queue) -- the queue sim is the unported piece (v1 ttft_queue_sim.py).