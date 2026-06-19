# Forward predictor — ISL:OSL distribution + hardware → TTFT/TPOT/E2EL

Predict serving latency for a workload you have **not** benchmarked, on a hardware config you may
not own. This is the forward counterpart to the batch backtester
(`profiling/process/build_simulator_rows.py`), which can only score workloads that were already run.
Background + the dependency map: [`prefill_comm_compute_decomposition.md`](prefill_comm_compute_decomposition.md).

## CLI

```bash
python -m profiling.process.predict_forward \
    --gpu A100 --tp 4 --engine vllm --model Llama-3.1-70B \
    --concurrency 40 --isl-osl dist.json
```

`--isl-osl` is the client's trace as either:
- a **JSON** list of `[isl, osl]` pairs: `[[1800, 210], [2400, 64], ...]`, or
- a **CSV** / whitespace file with two columns `isl, osl` per line (a header row is skipped).

Each `(isl, osl)` is one single-turn request; the cohort reflects the whole distribution.
Flags: `--engine vllm|sglang`, `--shared-prefix <tokens>` (cross-session APC prefix), `--json`.

Example output:
```
A100 tp4 vllm / Llama-3.1-70B  @ concurrency 40
  workload : 5 (isl,osl) samples; median isl=1800  osl=210
  TTFT     :    8367.7 ms
  TPOT     :      41.14 ms/token
  E2EL     :   17006.3 ms
  CONFIDENCE: MEASURED  (a100_tp4_vllm_analytic_roofline_firstcut)
```

## Library

```python
from simulator.forward import predict_forward
res = predict_forward(gpu="A100", model="Llama-3.1-70B", tp=4, engine="vllm",
                      concurrency=40, isl_osl_samples=[(1800, 210), (2400, 64)])
res.ttft_ms, res.tpot_ms, res.e2el_ms, res.calibration_status
```

## Confidence (`calibration_status`)

Derived from the **actual measured artifacts** backing the hardware, not a label:
- **measured** — a calibrated deployment with both a measured decode grid AND saturated-ITL ceiling.
- **partial** — a deployment exists but inherits one of those (analytic decode or default ceiling).
- **extrapolated** — no deployment: spec-sheet roofline (datasheet FLOPs/BW) + analytically-derived
  KV pool + H100-borrowed utilizations + inherited ceiling. A first-cut **lower-bound estimate**;
  the saturation knee, tp-comm at load, and decode floor are the parts that need a run on that GPU.

## How it works (one line)

Resolves `RooflineParams` + measured artifacts for `(gpu, tp, engine, model)` (reusing a calibrated
deployment, else `configs.loader.compose_roofline` + analytic `configs.kv_pool`), builds a cohort
from the client trace, and calls the same predictors the dashboard uses —
`kernel_tpot.predict_cell_tpot` (TPOT) and `ttft_queue_sim.predict_cell_ttft_qsim(cohort=…)` (TTFT) —
with no ground-truth file. Verified self-consistent with `build_row` to <1e-3 ms (`test_forward.py`).

## Current limits (planned)

- Single-turn ISL:OSL only; multi-turn trajectories + quantile-summary input are a follow-up.
- Reports the cohort-median TTFT/TPOT/E2EL; per-request percentiles (p90/p99) are a follow-up
  (the queue sim aggregates to the median internally).
- The `(…)` detail after CONFIDENCE is the deployment's own `calibration_status` string, which can be
  stale relative to the artifacts actually present — the coarse word (MEASURED/PARTIAL/EXTRAPOLATED) is
  the authoritative signal.
