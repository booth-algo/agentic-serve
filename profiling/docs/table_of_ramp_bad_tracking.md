# Ramp bad-tracking table — where the predictor lags the measured saturation rise

**Analysis doc (not a spec).** Generated 2026-05-29 from
`inference-benchmark/dashboard/public/simulator-predictions.json`. Lists the cells
where the current predictors (`kernel`, `kernel+hint`) fail to track the saturation
**ramp** — they get the plateau magnitude roughly right but rise several turns late,
so the high-error turns are on the rising edge.

## Diagnosis (why the ramp is mistracked)

The saturation jump is the **KV-pool eviction watermark crossing**, not the
pool-full crossing. Across every real-jump cell the measured jump fires at
**pressure ≈ 0.88–1.05** (pool ~88–92% committed) — i.e. vLLM begins
preemption/recompute *before* the 27250-block pool is full. For short-output coding
families (swe/terminal) the turn-space rise is nearly a **step** (1–2 turns); for
osworld it ramps ~2–4 turns then recovers as the cohort drains.

The current ramp model (`session_regime_classifier` window `pcross(0.85)→pcross(2.0)`,
consumed by `kernel_tpot_hint`) is **too gradual and ~1–2 turns too late**: it keys
the rise off pressure crossing 0.85→2.0, but the measured ITL is already ~70% of
plateau by pressure 0.92. Result: a 4–5 turn phase lag on the worst cells.

`pressure[t] = scheduled[t]·ceil((cached+new_prefill+0.5·output)/16)/27250`;
`deficit = scheduled·blocks − 27250` (blocks). `kPlat`/`hPlat` = `kernel` /
`kernel+hint` mean APE over plateau turns (tpot_meas > 100 ms). Measured ramp
window `[start,end]`: start = first turn meas > base+0.25·(plateau−base);
end = first turn meas ≥ 0.9·plateau.

## Cells with bad ramp tracking (kernel+hint plateau MAPE, worst first)

| profile | c | nPlat | kPlat% | hPlat% | measRamp [start,end] | width | worstLag @turn | base→plateau (ms) |
|---|---|---|---|---|---|---|---|---|
| terminalbench | 80  | 6  | 68.2 | 64.8 | [14,15] | 1 | +168 @ t15 | 8 → 219 |
| swebench      | 40  | 6  | 62.5 | 61.0 | [22,27] | 5 | +148 @ t27 | 8 → 215 |
| terminalbench | 120 | 10 | 25.8 | 23.5 | [9,12]  | 3 | +129 @ t10 | 8 → 230 |
| osworld       | 160 | 6  | 28.7 | 21.4 | [2,4]   | 2 | +29  @ t8  | 17 → 124 |
| osworld       | 320 | 28 | 17.6 | 19.2 | [2,3]   | 1 | +101 @ t3  | 30 → 149 |
| osworld       | 200 | 14 | 25.4 | 17.1 | [2,4]   | 2 | +35  @ t24 | 19 → 132 |
| terminalbench | 160 | 13 | 17.0 | 17.0 | [6,9]   | 3 | +140 @ t7  | 8 → 237 |
| terminalbench | 200 | 15 | 15.2 | 15.2 | [5,7]   | 2 | +96  @ t6  | 10 → 233 |

Notes:
- **terminalbench c=80 / swebench c=40** are the worst — a short, sharp, *late* jump
  the prediction almost entirely misses on the rising edge.
- **osworld c=320**: `kernel+hint` (19.2%) is slightly **worse** than `kernel`
  (17.6%) — the hint over-lifts the early plateau on the fastest-drain cell.
- `worstLag` = max(meas − pred_hint) over the plateau, i.e. the worst single-turn
  under-prediction.

## Case study: terminalbench c=120 (the turn-11 lag)

```
 t   meas   kernel  hint   pressure  deficit   note
 8   34.3   23.7    23.7    0.82     -4810
 9   66.0   31.4    31.4    0.89     -3050    measured rise begins
10  167.2   38.6    38.6    0.92     -2060    JUMP to 70% of plateau — pred still flat
11  172.6   65.6    67.5    1.02      +580    pool crosses 100%; pred only 67   ← turn 11
12  209.4  119.7   119.7    1.17     +4540
13  219.9  143.2   153.2    1.23     +6134
14  213.6  166.2   190.9    1.28     +7739
15  219.3  193.6   220.7    1.36     +9815    pred finally catches up
16  224.8  225.7   236.7    1.48    +12965
```

The measured jump is at **t10 (pressure 0.92, deficit still −2060)**; the prediction
does not reach the plateau until ~t15. The eviction watermark (~0.88–0.92) leads the
deficit-zero crossing (t11) by 1–2 turns.

## Per-turn eviction trajectory (other worst cells)

terminalbench c=80 — jump at t14 (pressure 0.88, deficit −3376):
```
 t13 meas  41.9  pressure 0.81  deficit -5239
 t14 meas 155.4  pressure 0.88  deficit -3376    JUMP
 t15 meas 211.0  pressure 0.93  deficit -1886
 t16 meas 224.6  pressure 0.98  deficit  -458
```

swebench c=40 — jump at t22 (pressure 0.99, deficit −262):
```
 t21 meas  36.2  pressure 0.97  deficit  -808
 t22 meas  83.3  pressure 0.99  deficit  -262    JUMP begins
 t24 meas 142.6  pressure 1.00  deficit   -42
 t27 meas 214.6  pressure 1.01  deficit  +315
```

**Takeaway for the ramp predictor:** anchor the ramp onset at the eviction watermark
(`pressure ≈ 0.88`, `deficit/capacity ≈ −0.12`), make the rise steep for short-output
families, and forecast the `[start, width]` from the forward eviction-deficit
trajectory rather than the `pcross(0.85→2.0)` window. See
`/root/.claude/plans/write-the-canonical-extraction-splendid-peacock.md`.
