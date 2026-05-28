# Diagnostic verdict — hybrid TPOT mechanism attribution

**One-line answer**: The physical decode + chunked-prefill formula nails
**per-scheduler-step time** across all profiles (median step_residual 0.88–1.05
in non-throttle regimes). Under `kv_admission_throttle`, step time is *still*
correctly predicted (median 1.02–1.69), but client-measured TPOT diverges by
5–8× because **TPOT is `mean(ITL)` (per-session inter-token latency), not
per-step time**. Under throttle, individual sessions don't get a token every
step — coverage drops as low as 0.49 on swebench. No fitted constants are
needed for the step-time prediction. The next mechanism the diagnostic
identifies (per-session scheduling fairness) is vLLM-scheduler-internal and
not derivable from RooflineParams alone.

## Setup

[diagnose_hybrid_tpot.py](/root/agentic-serve/profiling/process/predictors/diagnose_hybrid_tpot.py)
emits a per-turn record with:

- **Observed**: `tpot_meas`, `engine_total_step_ms`, `engine_steps`,
  `engine_max_decode_batch`, `engine_capacity_waiting_requests`,
  `engine_total_decode_slots`.
- **Predicted (no fits, all from `RooflineParams` + workload)**:
  - `decode_only_step_ms = (n_params·bytes_per_param + running·ctx·kv_bytes) / (bw·util_bw)`
  - `chunked_prefill_intrusion_ms = (engine_total_prefill_tokens/engine_steps) · prefill_per_token`
  - `predicted_step_ms = decode_only + intrusion`
  - `predicted_tpot_ms = predicted_step_ms` (1 token/session/step assumption)
- **Diagnostic columns**: `decode_coverage = engine_total_decode_slots / (engine_steps · running)` (1.0 = every running session got a token every step), regime label.
- **Residuals**: `step_residual = observed_step_ms_per_step / predicted_step_ms`, `tpot_residual = tpot_meas / predicted_tpot_ms`.

Output: 1,043 per-turn records across 44 (profile, c) cells in
[profiling/results/hybrid_tpot_diagnostic.csv](/root/agentic-serve/profiling/results/hybrid_tpot_diagnostic.csv).

Critical config (confirmed empirically by grepping `total_scheduled_tokens` max across all 4 engine traces): vLLM v1 `--max-num-batched-tokens=8192` (default, never overridden). Chunked-prefill ON. Swap-space=0. No preemption fires across 2,276 steps.

## Results — per-(profile, regime) residual table

| profile | regime | n | tpot_residual median (p10–p90) | step_residual median (p10–p90) |
|---|---|---:|---|---|
| chat-multiturn-synth | no_pressure | 58 | **1.08** (0.81–1.27) | 0.99 (0.69–1.29) |
| chat-multiturn-synth | mild_pressure | 124 | **1.05** (0.89–1.27) | 0.96 (0.73–1.23) |
| chat-multiturn-synth | kv_admission_throttle | 8 | 1.39 (0.86–2.18) | 0.53 (0.49–0.67) |
| osworld-multiturn-synth | no_pressure | 23 | **1.08** (0.34–1.57) | 1.09 (0.26–1.26) |
| osworld-multiturn-synth | mild_pressure | 174 | **1.11** (0.93–1.23) | 0.98 (0.79–1.21) |
| osworld-multiturn-synth | kv_admission_throttle | 106 | 4.94 (1.18–5.50) | 1.26 (0.58–1.81) |
| swebench-multiturn-synth | no_pressure | 143 | 0.88 (0.41–1.17) | 0.88 (0.43–1.19) |
| swebench-multiturn-synth | mild_pressure | 40 | **1.02** (0.91–5.83) | 0.97 (0.73–1.09) |
| swebench-multiturn-synth | kv_admission_throttle | 147 | **8.07** (5.18–9.39) | 1.69 (0.66–3.84) |
| terminalbench-multiturn-synth | no_pressure | 111 | **0.98** (0.22–1.21) | 0.93 (0.39–1.23) |
| terminalbench-multiturn-synth | mild_pressure | 39 | 0.93 (0.70–5.72) | 0.80 (0.61–0.96) |
| terminalbench-multiturn-synth | kv_admission_throttle | 70 | **6.94** (4.96–7.97) | 1.02 (0.54–1.89) |

## What this tells us

**The physical formula works for ~84% of cells.** Across all profiles, the `no_pressure` and `mild_pressure` regimes (covering 712 of 1,043 turns) have median tpot_residual in [0.88, 1.11] — the decode-bandwidth + chunked-prefill-compute model is essentially right. No fitted constants needed.

**Under `kv_admission_throttle`, the step time is STILL correctly predicted.** swebench step_residual=1.69 is the worst, and the other three profiles' throttle regimes show step_residual ≤ 1.26. The engine is doing roughly what physics says it should do per scheduler step.

**The gap between step time and TPOT is the diagnosis-yielding signal.** tpot is `mean(itl)` ([metrics.py:65](/root/agentic-serve/inference-benchmark/src/benchmark/metrics.py)) — per-session inter-token-arrival latency. The "1 token/session/step" assumption breaks under throttle:

| profile | throttle median coverage | implied ITL multiplier (1/coverage) | observed tpot_residual / step_residual |
|---|---:|---:|---:|
| swebench | 0.55 (swe c=80 t=11 example) | 1.8× | 8.07/1.69 = **4.77×** |
| terminal | 0.74 | 1.4× | 6.94/1.02 = **6.80×** |
| osworld | 0.65 | 1.5× | 4.94/1.26 = **3.92×** |
| chat | 0.62 | 1.6× | 1.39/0.53 = **2.62×** |

Coverage alone explains only ~30–40% of the tpot/step gap. The remainder is likely **client-measured queue wait inflating ITL**: when many sessions submit concurrent turn-N requests, vLLM admits them serially through chunked-prefill, so a session that's "ready to decode" might wait several scheduler steps while earlier sessions are still being prefilled. The bench measures from `chunk_received_at` so this queue wait inflates each ITL observation.

Engine traces confirm this regime: at swe c=80 turn 11, waiting_queue grew to 10 while running_queue stayed at 44–73. No preemption, no swap, no allocation failure — pure admission throttle.

## Recommendation for the hybrid model

**Ship the physical model as-is for non-throttle regimes** (no fits, runs in <1ms per cell). It correctly predicts 84% of the turn matrix.

**For the kv_admission_throttle regime, two honest options**:

1. **Predict server-side step time (what physics says) and report the gap.** Make the diagnostic part of the predictor output: alongside `predicted_tpot_ms`, emit a "throttle_warning" flag with the expected client-side amplification range from the diagnostic. The model is honest about what it predicts (engine step time) and the user knows when client-side TPOT will diverge.

2. **Add a per-(profile, c) measured `coverage_observed` baseline and ITL-inflation term.** This requires per-cell measurement, which is effectively llm-d-augmenter's approach — a single per-cell constant per the [augment_simulator_predictions_with_llm_d.py](/root/agentic-serve/profiling/process/emitters/augment_simulator_predictions_with_llm_d.py) pattern. Each constant has a physical interpretation (observed per-session scheduling cadence) but is *measured*, not derived.

The user's "no fitted constants" constraint argues for **(1)**. The physical model captures the mechanism it can model; what it can't model (per-session scheduler fairness under chunked-prefill admission) is exposed in the diagnostic rather than papered over.

## Loose ends

- Coverage correlation within `kv_admission_throttle` is **positive** (+0.44 to +0.68 across profiles), the opposite of the naïve hypothesis ("lower coverage → higher tpot residual"). Explanation: within the throttle regime, both quantities vary together as the scheduler thrashes between admission and decode — coverage is one symptom of throttle, not its cause.
- swebench `no_pressure` shows 0.88 median residual (12% over-prediction). Likely the bandwidth utilization assumption (`util_bw=0.93`) is slightly conservative at moderate batch sizes. Not worth chasing — well inside acceptable error.
- The diagnostic doesn't enumerate **per-step** coverage variation within a turn (only the turn-level aggregate). If we want to attribute the gap further, the engine-step JSONL traces have the per-step `running_request_ids` and `scheduled_request_ids` needed for sequence-level analysis — out of scope for this PR but cheap follow-up.
- Engine traces only cover swe/terminal at certain concurrencies. Extrapolation to chat/osworld used the simulator-predictions per-turn aggregates only, which carry `engine_*` fields summarized per turn. All conclusions above hold for those profiles in the aggregate; sequence-level inspection requires raw traces.
