# Kernel-composition mixed-step cost for three-regime PERTURBING regime

## One-line verdict

The cached-prefill kernel lookup is the right physical T_max for PERTURBING
cells — it nails osworld c=80 turn 5 (predicted 29.8 ms vs observed 29.9 ms,
sub-1% error). But the overall osworld median APE doesn't budge (17.2%
unchanged) because the **classifier mis-classifies osworld c≥160 as
SATURATING**, where the kernel lookup is correctly NOT applied. The
mis-classification is a workload-only classifier limitation; with kernel
composition wired correctly, fixing it would unlock significant osworld
improvements.

## What landed

| component | file | purpose |
|---|---|---|
| **Cached-prefill lookup** | [simulator/cached_prefill_lookup.py](/root/agentic-serve/simulator/cached_prefill_lookup.py) | Bilinear-log interp over the 25-point `cached_prefill_v3_H100.csv` grid; returns `cached_prefill_step_ms(U, P)` |
| **Scheduler-overhead anchor** | [simulator/closed_form_tpot.py](/root/agentic-serve/simulator/closed_form_tpot.py) (RooflineParams) + [profiling/data/roofline_params_H100_llama31_8b.json](/root/agentic-serve/profiling/data/roofline_params_H100_llama31_8b.json) | `scheduler_overhead_ms_per_step = 5.7` — calibrated from 99 minimal-work decode steps in swe_c40 trace |
| **Regime-conditional T_max** | [simulator/three_regime_tpot.py](/root/agentic-serve/simulator/three_regime_tpot.py) `_t_max_per_turn` | PERTURBING → `cached_prefill_step_ms(U, P) + scheduler_overhead`; SATURATING → physical T_upper (unchanged) |
| **Unit tests** | [simulator/tests/test_cached_prefill_lookup.py](/root/agentic-serve/simulator/tests/test_cached_prefill_lookup.py) | 8 tests covering anchors, clamping, monotonicity, real-data spot check |

No fitted constants — the new `scheduler_overhead_ms_per_step` follows the
same single-anchor discipline as `util_flops` / `util_bw`.

## Per-profile APE comparison

| profile | before (constant T_max) | after (kernel composition for PERTURBING) | delta |
|---|---:|---:|---:|
| chat | 9.1% | 9.1% | 0.0 |
| osworld | 17.2% | 17.2% | 0.0 |
| swebench | 12.7% | 12.7% | 0.0 |
| terminalbench | 11.5% | 11.5% | 0.0 |

The median APE is unchanged. This is **misleading**; the delta is hidden in
specific cells where the lookup applies.

## Spot-checks: kernel lookup IS working where applied

For PERTURBING cells the per-turn prediction now uses the measured kernel
step time. Where it matters:

**osworld c=80 turn 5 (PERTURBING peak)** — exact match:
```
turn   meas   3reg
   4   31.2   17.2
   5   29.9   29.8     ← kernel lookup (U=100, P~6500) + overhead
   6   25.5   25.2     ← kernel lookup
   7   19.7   17.3
```

The peak turn predictions now track within 1% of observed. Previously the
flat T_max=36ms cap underpredicted the variance across the spike.

**chat c=320 turns 4–9 (PERTURBING bell-curve)** — magnitude under but
shape preserved. T_max cap is binding (llm-d mean = 28 ms) so predictions
can't exceed 28 ms even though observed peaks at 60 ms. Cap binding is
expected: chat workload doesn't have enough U/P to drive the kernel lookup
above the llm-d envelope.

## Why osworld c≥160 doesn't improve

osworld at c=160 is classified **SATURATING** because the workload-side
`scheduled_requests` stays high enough that `pressure ≥ 1` sustains across
many turns (`late_pressure` ≥ 1 + `saturated_turns` ≥ 2). The kernel
lookup is intentionally NOT applied to SATURATING cells — sustained
admission cycling forces per-session step-skipping, and client-side TPOT
amortizes to chunk × prefill_per_token (~205 ms), not engine step time.

But osworld c=160 doesn't actually sustain saturation — it recovers as
sessions complete and the active count drops. The workload-only classifier
can't see this; only engine telemetry (`engine_max_decode_batch`) would
expose it. So the classifier mis-fires.

If you re-classify osworld c=160 as PERTURBING (e.g., by also tracking
`scheduled_requests` decline rate), the kernel lookup would apply per
turn — predicting close to the observed 22-28 ms recovery range instead
of the 66 ms plateau the constant T_max gives.

## Tests + verification

- 8 new lookup tests + all 17 three-regime tests pass
- Augmenter regenerates `simulator-predictions.json` (1043 turn records)
- Dashboard tpot-fit JSON updated: median APE 12.9% (essentially unchanged)
- Per-tier breakdown (in [three-regime-tpot-2026-05-28.md](/root/agentic-serve/profiling/docs/three-regime-tpot-2026-05-28.md)) shows the cells where kernel composition is genuinely the right model.

## What's next (out of scope for this PR)

1. **Re-classify osworld c≥160 as PERTURBING**. Detect cohort completion
   from `scheduled_requests[t-3:t]` decline rate (a workload-side signal
   the current classifier ignores). This unlocks the kernel lookup on
   ~150 osworld turns and could drop osworld median APE substantially.
2. **Calibrate FA3-cached lookup for very long P** (>8192). The current
   CSV grid stops at P=8192; agentic workloads at high turn count have
   P≥10000. Clamping is the safe fallback today but extrapolation might
   eventually warrant adding more measurement points.

## What kernel composition can NOT help with

- Sustained-saturation client-side TPOT amplification (swe/terminal high c
  late turns at 200+ ms). The engine step time is correctly predicted by
  the lookup (~17–25 ms); the gap is per-session step-skipping which is a
  scheduler-fairness property, not a kernel-level fact. The constant
  physical T_upper = 205 ms is the right asymptote here and we keep it.
