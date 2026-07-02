# Provenance: why TPOT is decode-floor-only and the fused step is a TTFT thing

The per-turn predictors are NOT a single kernel sum. v1 splits the problem by
PROVENANCE — what is kernel-composable vs what is an emergent benchmark property —
and TPOT vs TTFT use the leaf kernels very differently. This is deliberate, and it
reframes the `fused_step_ms` / `mixed_step` work: that work belongs to TTFT, not TPOT.

## The TPOT law (kernel_tpot.py)

    ITL[t] = kernel_step  +  (w · sustain) · (T_upper − kernel_step)
             └── floor ──┘    └─ weight ─┘    └────── ceiling ──────┘

- floor   = decode_step_ms(b_eff, ctx)  -- measured decode kernel grid, bilinear
            interp. THIS is what kernel_floor/ builds. Valid (≈ITL) below KV
            saturation, which is ~58% of all cells.
- ceiling = saturated_ceiling_ms(output) -- measured saturated-ITL plateau anchors,
            linearly interpolated IN OUTPUT LENGTH. NOT kernel-composed; it is an
            emergent benchmark property (median ITL once KV pressure ≥ ~2.5). This
            is what kv_wall/ must build.
- weight  = w·sustain ∈ [0,1] -- WHERE between floor and ceiling this turn sits.
            w = eviction-recompute duty cycle from KV pressure z = pressure·qbar
            (_overflow_weight); sustain = smoothstep(output; 9, 24). Computed
            physics, zero tuned constants.

Nothing is MAPE-fit: both ends are measured+interpolated, the blend is derived
pressure. The "interpolation" we kept remembering is just linear interpolation
between two measured surfaces (decode grid + plateau anchors).

## Why a fused per-step kernel sum is the WRONG tool for TPOT

TPOT is inter-token latency -- a DECODE-step quantity by definition. Prefill's
effect on TPOT is NOT a within-step prefill+decode overlap; it is an emergent,
cohort-level, CROSS-TURN scheduling phenomenon: KV eviction → LIFO preemption →
recompute → queue rotation (captured by _overflow_weight). You cannot compose that
bottom-up from one step's kernels -- it depends on the whole cohort's KV-pressure
trajectory across turns. So v1 models it as the floor→ceiling amplifier, not a sum.

## Where the fused/mixed step actually lives (TTFT)

The mixed prefill+decode step is in v1's queue sim (ttft_queue_sim._price_step),
and it combines the phases with max(), NOT an additive fused sum:

    step_ms = max(decode_ms + scheduler_overhead, prefill_ms)

The cheaper phase rides free under the more expensive one (same forward pass).
That is the piggyback. Our v2 fused_step_ms now matches this: pure-leaf decode and
pure-leaf prefill composed independently, combined with max(). With one population
it collapses to the validated corner (decode 9.7%, prefill 3.4%).

NOTE on validating the fused step against mixed_step_profile_*.csv: that file is the
WRONG yardstick in absolute terms -- its pure-decode rows are ~2× our CUDA-graphed
floor (flat ~13.5 ms, batch-independent → eager-mode / big host overhead) and its
prefill cells sit at a flat ~0.68× regime/util offset (it runs prefill ~0.47–0.52
util vs the 0.75 our prefill floor validated against). max() gets the SHAPE right
(p=4096 row is dead-flat across batch 1→256 = correct piggyback), but the absolute
gap is a regime offset, not a composition error. To pin the absolute number we'd
need a CUDA-graphed mixed sweep that records the decode context.

## Implication for v2 (the actual path to per-turn prediction)

TPOT per-turn needs three things:
  1. decode floor  -- decode_step_ms(b_eff, ctx)        DONE (kernel_floor/, 9.7%)
  2. saturated ceiling -- measured plateau anchors + interp   PENDING (kv_wall/)
  3. the amplifier law -- floor + w·sustain·(ceiling−floor)   PENDING (where?)

The fused step is NOT on the TPOT critical path. It's for the TTFT queue sim. Don't
block TPOT on validating the fused interaction against mixed_step.

---

ITL[t] = kernel_step  +  (w · sustain) · (T_upper − kernel_step)
         └── floor ──┘    └─ weight ─┘    └────── ceiling ──────┘
