# Per-turn TPOT

TPOT (inter-token latency, ITL) for one turn is a **measured decode-kernel step
ramped toward a measured saturation ceiling by KV pressure**:

```
ITL[t] = kernel_step  +  (w · sustain) · (T_upper − kernel_step)
         └── floor ──┘    └─ weight ─┘    └────── ceiling ──────┘
```

Both ends are measured + interpolated; the blend is derived physics. Nothing is
MAPE-fit. The "interpolation" is just linear interpolation between two measured
surfaces (the decode grid and the plateau anchors), weighted by a computed
KV-pressure term.

## The three terms

**Floor — `kernel_step = decode_step_ms(b_eff, ctx)`**
The composed decode-step floor (our `kernel_floor/`). This is the physically
correct lower bound: below KV saturation, ITL ≈ kernel_step (~58% of cells in
v1). Inputs are workload-derived:
- `ctx = cached + new + 0.5·output` — resident KV at the decode midpoint.
- `per_session_blocks = ceil(ctx / block_size)`.
- `capacity_batch = available_kv_blocks / per_session_blocks`.
- `b_eff = min(scheduled, capacity_batch)` — the KV-throttled running batch.

**Ceiling — `T_upper = max(kernel_step, saturated_ceiling_ms(output))`**
The saturated-ITL plateau: the median benchmark ITL once KV pressure is high
(≥ ~2.5), read from measured anchors `{output_tokens: plateau_ms}` and linearly
interpolated IN OUTPUT LENGTH. NOT kernel-composed — an emergent benchmark
property. Short output saturates higher (clamped outside the measured range).

**Weight — `w · sustain ∈ [0, 1]`**
Where between floor and ceiling this turn sits.
- `pressure = scheduled · per_session_blocks / available_kv_blocks` — median-
  session resident demand / pool.
- `z = pressure · context_spread` — distribution-integrated demand / pool
  (`context_spread`, v1's `qbar`, = the cohort's measured context-size spread).
- `w = _overflow_weight(...)` — the eviction-recompute duty cycle: 0 below
  pool-full (`pressure < 1` or `z ≤ 1`), ramping to 1 when every step's budget
  is recompute-filled (the regime where `T_upper` was anchored). Computed from
  the LIFO-evicted fraction `(1 − 1/z)` re-prefilled in chunked-prefill steps;
  zero tuned constants.
- `sustain = smoothstep(output; 9, 24)` — a turn too short to co-reside through
  the eviction buildup cannot reach the ceiling.

At `pressure < 1` or `z ≤ 1`: `w = 0` → `ITL = kernel_step` (the raw floor).

## Why TPOT is decode-only (no fused prefill step)

TPOT is an inter-token (decode-step) quantity by definition. Prefill's effect on
TPOT is NOT a within-step prefill+decode overlap — it is an emergent,
cohort-level, CROSS-TURN scheduling phenomenon (KV eviction → LIFO preemption →
recompute → queue rotation), captured by `w`. You cannot compose that bottom-up
from one step's kernels; it depends on the whole cohort's KV-pressure trajectory
across turns. So it is modeled as the floor→ceiling amplifier, not a kernel sum.

The fused/mixed prefill+decode step (`fused_step_ms`, `max()` piggyback) belongs
to the **TTFT** path (v1's `ttft_queue_sim._price_step`), not TPOT. See
`scrap.md`.

## Status

| Piece | Where | State |
|---|---|---|
| Floor — `decode_step_ms(b_eff, ctx)` | `kernel_floor/sum_kernels.py` | ✅ done, validated 9.7% |
| Ceiling — plateau anchors + interp | `kv_wall/saturated_ceiling.py` | ✅ done (measured reader) |
| Wire floor + ceiling into the hardware getter | `getters/hardware.py` | ✅ done (`KernelComposition`) |
| Amplifier — `floor + w·sustain·(ceiling−floor)` | `engine/amplifier.py` | ✅ done (B1: stateless onset+drain+rotation; B2 deferred) |

The hardware getter the engine loads (`KernelComposition`) composes the validated
`KernelFloor` and the measured `SaturatedCeiling`, and implements
`saturated_ceiling_ms(output)`. `Roofline` (forward) raises `NotImplementedError`
for it — a forward-mode ceiling without measured anchors is unsolved (see below).

`engine/amplifier.py` (`@mode SHARED`) computes the per-turn law: per-turn
`pressure = scheduled·per_session_blocks/kv_pool_blocks`, `z = pressure·context_spread`,
floor at the KV-throttled `b_eff = min(scheduled, capacity)`, ceiling from
`kv_wall`, weight from the fit-free onset+chunked-drain+rotation core. It is
STATELESS per turn (B1): the cell-level refinements that fix onset *timing* —
firing-gate hysteresis, development clock, fresh-crossing damping (v1
`predict_cell_tpot`) — are deferred to B2, to add only if backtest MAPE needs the
jump-turn accuracy. `context_spread` (v1's `qbar`) defaults to 1.0 (no-spread
fallback); wiring it to the measured `context_scale_quantiles` is future work. In the unsaturated regime
(w=0) the ceiling is never read, so forward runs there; it raises only on a
genuinely saturated turn.

## kv_wall — the measured ceiling (and why there's no analytic one)

`SaturatedCeiling` (`kv_wall/saturated_ceiling.py`) reads the measured plateau:
the median benchmark `tpot_ms` in the overloaded regime, bucketed into output
anchors and interpolated in output. Artifact:
`profile_data/kernels/saturated_ceiling/{gpu}.json`. H100/8B = {28→243.1, 86→134.9}
ms. This is the validated, backtest path.

There is deliberately NO analytic forward ceiling. The obvious candidate — a
memory-bandwidth wall (one step reading weights + the full KV pool) — lands ~10×
too low (~24 ms vs measured 243 ms on H100/8B), because saturation is
recompute/queue-bound, not single-step-read-bound. It was dropped rather than ship
a misleading number. Forward mode for an uncalibrated config should INHERIT the
nearest measured anchors (v1's approach); that wiring is not done yet.

## Build order

1. ✅ Wire `KernelFloor` into the hardware getter the engine loads.
2. ✅ Build `kv_wall/` — the saturated ceiling.
3. ✅ Build the amplifier (B1) in `engine/amplifier.py` — per-turn `b_eff`,
   `pressure`, `z`, fit-free `_overflow_weight`, `sustain`, blend floor → ceiling;
   `_tpot_ms` now uses per-turn `scheduled_requests`, not raw concurrency.
4. Score backtest MAPE against GT cells. **Result: B1 = 16.6% on H100/8B (1043
   turns), vs v1's 15.2% — near parity, so B1 stands.** Gap is concentrated in
   chat (v2 9.5 vs v1 5.4), where v1's cell-level state helps most.
5. `context_spread` (v1's `qbar`; `cohort_context_spread`, SHARED) is ported but
   NOT wired as a default: plugging v1's measured per-profile values into B1
   slightly *hurts* (16.6→16.9) — its earlier onset over-predicts without B2's
   hysteresis to temper it. Revisit it together with B2 (cell-level hysteresis /
   development clock / fresh-crossing damping), the only remaining lever to close
   the ~1.4pt to v1.

## Reference (v1)

`simulator/kernel_tpot.py` (the law), `simulator/kernel_step_cost.py` (the floor
grid), `profiling/docs/prediction_construction.md` § TPOT (full provenance).
