# Three-regime TPOT predictor — validation verdict

Per-profile MAPE (mean) and median APE across all turns × concurrencies. Three-regime uses no fitted constants — every input is `RooflineParams`, the empirically-confirmed `max_num_batched_tokens=8192`, per-turn workload, or the existing `tpot_pred` / `tpot_pred_llm_d` anchors. **MAPE is what the dashboard's TPOT MAPE badge displays**; median APE is robust to outliers (heavy tail on extreme cells).

## Per-profile MAPE (mean) — dashboard metric

| profile | turns | roofline | llm-d | two-roofline | **three-regime** |
|---|---:|---:|---:|---:|---:|
| chat-multiturn-synth | 190 | 13.8% | 14.2% | 18.3% | **13.4%** |
| osworld-multiturn-synth | 303 | 32.1% | 63.2% | 35.8% | **23.3%** |
| swebench-multiturn-synth | 330 | 44.4% | 76.3% | 36.3% | **22.4%** |
| terminalbench-multiturn-synth | 220 | 46.2% | 118.9% | 33.0% | **30.7%** |

## Per-profile median APE — tail-robust view

| profile | turns | roofline | llm-d | two-roofline | **three-regime** |
|---|---:|---:|---:|---:|---:|
| chat-multiturn-synth | 190 | 9.1% | 8.4% | 19.0% | **9.1%** |
| osworld-multiturn-synth | 303 | 18.1% | 25.7% | 18.3% | **13.9%** |
| swebench-multiturn-synth | 330 | 48.6% | 46.1% | 19.2% | **14.2%** |
| terminalbench-multiturn-synth | 220 | 42.9% | 51.2% | 19.6% | **14.1%** |

## Per-profile tail of the three-regime error distribution

| profile | turns | median | MAPE | p90 | max |
|---|---:|---:|---:|---:|---:|
| chat-multiturn-synth | 190 | 9.1% | 13.4% | 27.0% | 52.8% |
| osworld-multiturn-synth | 303 | 13.9% | 23.3% | 61.4% | 87.1% |
| swebench-multiturn-synth | 330 | 14.2% | 22.4% | 60.9% | 240.2% |
| terminalbench-multiturn-synth | 220 | 14.1% | 30.7% | 78.6% | 185.0% |

**Overall (median across all 4 profiles):**

- `roofline`: median APE = 19.5%
- `llm_d`: median APE = 30.9%
- `two_roofline`: median APE = 18.7%
- `three_regime`: median APE = 13.7%

## Per-profile breakdown by concurrency tier

| profile | c tier | turns | roofline | llm-d | two-roofline | **three-regime** |
|---|---|---:|---:|---:|---:|---:|
| chat-multiturn-synth | low (≤20) | 64 | 4.3% | 1.5% | 21.8% | **4.3%** |
| chat-multiturn-synth | mid (21–80) | 36 | 8.3% | 9.5% | 16.4% | **8.3%** |
| chat-multiturn-synth | high (>80) | 90 | 20.4% | 20.4% | 12.6% | **20.4%** |
| osworld-multiturn-synth | low (≤20) | 93 | 2.9% | 17.8% | 17.4% | **2.9%** |
| osworld-multiturn-synth | mid (21–80) | 60 | 15.3% | 86.8% | 9.6% | **14.9%** |
| osworld-multiturn-synth | high (>80) | 150 | 71.5% | 23.7% | 75.2% | **41.4%** |
| swebench-multiturn-synth | low (≤20) | 120 | 7.4% | 21.8% | 9.4% | **7.4%** |
| swebench-multiturn-synth | mid (21–80) | 60 | 47.6% | 77.0% | 42.8% | **33.6%** |
| swebench-multiturn-synth | high (>80) | 150 | 72.3% | 46.3% | 55.2% | **17.1%** |
| terminalbench-multiturn-synth | low (≤20) | 80 | 6.8% | 30.8% | 13.1% | **6.8%** |
| terminalbench-multiturn-synth | mid (21–80) | 40 | 38.6% | 87.3% | 36.0% | **38.6%** |
| terminalbench-multiturn-synth | high (>80) | 100 | 83.1% | 51.4% | 49.7% | **42.2%** |

## Regime classification: predicted vs detected

Predicted regime (from physics + workload) vs detected regime (from `tpot_meas` trajectory: peak/baseline ≥ 1.5 and last/peak > 0.6 ⇒ saturating; ≤ 0.6 ⇒ perturbing; otherwise flat). One vote per (profile, c) cell.

| predicted ↓ / detected → | FLAT | PERTURBING | SATURATING |
|---|---:|---:|---:|
| **FLAT** | 13 | 3 | 7 |
| **PERTURBING** | 0 | 5 | 0 |
| **SATURATING** | 0 | 0 | 16 |

Diagonal (correct): 34 / 44 = 77%

## Spot checks (motivating cells)

| profile | c | regime (predicted) | turn-0 obs / pred | turn-mid obs / pred | turn-last obs / pred |
|---|---:|---|---|---|---|
| chat-multiturn-synth | 5 | FLAT | 6.9 / 6.7 | 7.1 / 6.9 | 7.0 / 6.8 |
| chat-multiturn-synth | 320 | PERTURBING | 13.2 / 10.6 | 60.2 / 28.4 | 21.7 / 17.3 |
| osworld-multiturn-synth | 160 | PERTURBING | 22.3 / 15.3 | 23.4 / 20.4 | 24.3 / 20.6 |
| swebench-multiturn-synth | 80 | SATURATING | 17.1 / 12.9 | 220.8 / 70.0 | 247.9 / 204.7 |
| terminalbench-multiturn-synth | 80 | SATURATING | 8.6 / 11.5 | 30.0 / 17.3 | 156.2 / 25.1 |

## Reading

**Where three-regime wins**: cells where the regime classification matches reality. SATURATING shape (swe/terminal mid-high c) is captured by the linear T_min → T_max ramp from the predicted jump turn; FLAT cells reduce to the existing roofline (so identical to roofline APE in low-c regimes).

**Documented limitations**:

1. **osworld c≥160 misclassified as SATURATING**: the workload's `scheduled_requests` stays high enough that pressure ≥ 1 sustains in the model, but vLLM's chunked-prefill scheduler actually throttles admission, so observed `tpot_meas` recovers by turn 14+. Pure-forward prediction (workload only, no engine telemetry) can't capture vLLM throttle dynamics. This is the documented trade-off vs. the two-roofline predictor which uses `engine_max_decode_batch`.

2. **Perturbing-regime magnitude**: regime-2 cells (chat c=256/320, osworld c=80/120) spike to `T_max` at perturbation turns. Observed magnitudes are often smaller than `T_max` (e.g. chat c=320 turn 8 observed 50 ms vs predicted T_max ~28 ms — predicted ceiling capped by llm-d mean). Modeling spike magnitude < T_max needs an additional term; deferred.

3. **SATURATING ramp shape**: linear T_min → T_max across (jump_turn .. last_turn) was the simplest choice. Real curves often climb faster early, then plateau. A sigmoid ramp would fit better but adds tunable steepness.
