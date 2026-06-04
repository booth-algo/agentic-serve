# Fitted constants — replacement ledger

Quick-reference table of every **fitted** (least-squares) or **tuned-knob** (hand-chosen, no measured
anchor) constant in the tp1 headline TTFT/TPOT/E2EL flow. Full narrative + provenance in
[fitted_constants_audit.md](fitted_constants_audit.md) and [prefill_law_defit_trace.md](prefill_law_defit_trace.md).
Status: ✅ done (de-fitted / measured-anchored) · ⛔ open. `kernel_tpot.py` constants drive TPOT+E2EL;
`ttft_queue_sim.py` constants drive TTFT+E2EL.

## FITTED — least-squares regressions

| Status | Constant (value) | Where | Anchor / replacement |
|---|---|---|---|
| ✅ | `SATURATED_BASE_MS` (118.7) | kernel_tpot (removed) | measured plateau anchors, per output cluster |
| ✅ | `SATURATED_TURN_OVERHEAD_MS` (3263) | kernel_tpot (removed) | same artifact (`build_saturated_ceiling.py`) |
| ✅ | `PREFILL_NEW_MS_PER_TOKEN` (0.0310) | ttft_queue_sim | GEMM roofline 0.02498 DERIVED + **CONFIRMED** (microbench 06-03: offline ttft.new 25.3 ms/1k ≈ roofline 25). Residual ≈ serving 31 − offline 25.3 ≈ 5.7 = serving-stack delta (validated; not offline-removable). |
| ✅ | `PREFILL_FLOOR_MS` (22.5→**26.0**) | ttft_queue_sim:119 | measured min pure-prefill TTFT (c1 turn-0, cached≈0 ≈26.07 ms; microbench intercept 19–26 confirms); fit sat ~4 ms low → TTFT 33.01→32.89 |
| ✅ | `PREFILL_HOST_SHARED…` (→**0.00305**) | ttft_queue_sim:126 | **LIVE-MEASURED 06-03.** Live vLLM server reproduces the cached rate (5.89≈6.1); concurrency sweep gives ~40-54% shared → **50/50** within range maximizes the gate (was imported 57/43). Cached decomposes: model 2.4 + IPC 0.7 + HTTP 2.8 ms/1k — a real serving-stack cost (not model physics). |
| ✅ | `PREFILL_HOST_PERREQ…` (→**0.00305**) | ttft_queue_sim:127 | per-request half (0.50×6.103e-3). The B-slope (~3.5 ms/1k per added concurrent req < the 5.89 c1 rate) shows the cost partly amortizes. Offline batch-CSV's 12/88 was wrong (lacked the serving stack + regressed). |

> One c1 regression `22.5 + 0.0310·new + 0.006103·cached` (R²=0.963) fanned into named constants — now ALL
> de-fitted/characterized. The prefill stage-split microbench + **live vLLM-server measurement** (06-03, see
> `prefill_stage_split_results.md`) showed: offline `ttft.new`=25.3≈GEMM roofline, `tokenize`≈1.3 ms/1k,
> FLOOR≈26; and the **live server reproduces** the serving rates (cached 5.89≈6.1, new 29≈31) where offline
> can't. The cached 6.1 decomposes (measured): **model 2.4 + engine-IPC 0.7 + HTTP-frontend 2.8 ms/1k**, with
> a live-concurrency-measured **40/60** shared/perreq split. It's a real *serving-system* cost (HTTP/IPC, not
> model physics) — correctly an empirical serving constant, now fully characterized rather than a blind fit.

## TUNED-KNOB — hand-chosen (all DE-FITTED 2026-06-03 → measured anchors)

Replaced the "knees retuned vs the data" amplifier-ramp constants with measured anchors. Knees **interact** —
the jointly-measured band is what holds. Gate: TPOT 15.89→**15.42**, swe-plateau 8.64→8.65 (held), E2EL
19.77→**19.32**, no profile regresses.

| Status | Constant (value) | Anchor / replacement |
|---|---|---|
| ✅ | `P_HI_SHORT` (1.6→**1.22**) | eviction-watermark cluster max (ramp_tpot `DEF_HI`, pressure 1.22) |
| ✅ | `OUT_KNEE_LO` (40→**28**) | short-output saturated-ceiling cluster (28 tok) |
| ✅ | `OUT_KNEE_HI` (80→**86**) | long-output ceiling cluster (86 tok) |
| ✅ | `SAT_SUSTAIN_LO` (10→**9**) | p5 output of saturated turns (tpot>100 ms = 9 tok) |

## Anchored — measured / kernel-derived (no action)

| Status | Constant (value) | Anchor |
|---|---|---|
| ✅ | `P_LO` (0.8→**0.88**) | watermark onset (ramp_tpot `DEF_LO`, pool ~88%) |
| ✅ | `P_HI_LONG` (2.5→**2.0**) | 2× pool commit (ramp_tpot `DEF_SAT`, pressure 2.0) |
| ✅ | `SAT_SUSTAIN_HI` (24) | 22-tok plateau-min + 2 (structural offset) |
| ✅ | `PREFILL_FA3_MS_PER_TOKEN2` (8.31e-7) | kernel-derived `FA3(8192)/(8192²/2)` |

## Note on the de-fitted ceiling
The ceiling (`SATURATED_*`) is **fit-free but still empirical** — it reads the measured plateau, doesn't
derive it. The from-physics roofline route failed (plateau is ~259 ms queueing/recompute vs ~27 ms compute);
a fully-derived `T_upper` needs an eviction/queue model (extend `ttft_queue_sim`).
