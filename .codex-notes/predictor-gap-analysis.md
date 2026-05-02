# Predictor Gap Analysis — Why TP=1 Predictions Are Off

Date: 2026-05-02

## Primary Finding: Flash Attention Has No Empirical Data

The flash attention predictor at `llm_predict/kernels/flash_attn.py` falls through
to pure roofline because:

- Per-GPU CSV tables (`data/flash_attn/H100.csv`, `A100.csv`, etc.) → **don't exist**
- XGBoost models (`data/models/flash_H100.pkl`, etc.) → **don't exist**
- `data/flash_attn/serving_shapes.csv` (14,617 rows) has no latency column — shape
  registry only, not consumed by predictor

Contrast: GEMM has per-GPU `.csv` tables (430-463 rows, 14 M values, 33 (N,K)
shapes, actual ncu latencies) + per-GPU `.pkl` XGBoost models (500-600KB each).

## Missing Physics in Composer

`composer.py` models 14 kernel calls per Llama layer (7 GEMMs, 1 flash_attn,
6 elementwise). What's missing:

- No `lm_head` (final vocab projection)
- No embedding lookup
- No KV cache write/update bandwidth
- No CUDA graph / framework scheduler overhead
- QK transpose, softmax, PV multiply folded into one fused "flash attention" call
  (not decomposed into sub-kernels)

## Per-Regime Error Profile

| Regime | GEMM fraction | Attn fraction | Source of error |
|---|---|---|---|
| Decode (short KV) | 91-92% | 3-5% | H100: bad GEMM model. A100: negligible |
| Decode (long KV) | 84-85% | 10-11% | Roofline-only attention + GEMM compounds |
| Prefill (TTFT) | 76-77% | 8% | Roofline attention + missing KV write + overhead |

## Per-GPU Kernel Predictor Quality

| GPU | GEMM held-out MAPE | Flash attn | Training rows | Verdict |
|---|---|---|---|---|
| A100 | 1-2% (good) | Roofline-only | 61K | TPOT good, TTFT off |
| H100 | 90-137% (bad) | Roofline-only | 5K (very sparse) | Both TTFT and TPOT suffer |
| RTX3090 | 0.1-2% (good) | Roofline-only | 55K | TPOT good, TTFT off |
| RTX2080Ti | 166% (poor) | Missing (no FA2) | 19K (sparse) | Everything off |

## Error Magnitudes (C=1, single-GPU, dense models, 43 rows)

| GPU | TTFT mean err | TPOT mean err | E2EL mean err |
|---|---|---|---|
| A100 | 40.9% | **4.2%** | 8.1% |
| H100 | 49.8% | **65.2%** | 71.5% |
| RTX3090 | 67.6% | 42.7% | 44.0% |
| Overall | 51.0% (med 6.9%) | 56.9% (med 9.7%) | 62.4% (med 10.3%) |

Worst rows: H100 × 70B/72B × chat-singleturn → 350-390% error.
These are all `low_confidence` calibrations — sparse H100 70B data.

## Known Issue: H100 GEMM Data Was Corrupt

Old `gemm_serving_ncu_H100.csv` had alternating M values differing by 10-15x.
Clean re-sweeps done. Some stale artifacts may linger.

## Aaron's Key Insight

"GEMM and flash attention are the two major kernels. If you aggregate them
properly, you should get good predictions without a large training set."

CONFIRMED: flash attention is the gap. GEMM is decent (except H100).
Fix is to ncu-profile flash attention shapes the same way GEMM was.

## Also: Correction Factors Removed

Per Aaron meeting direction, framework alpha/beta TTFT correction was removed
(dirty worktree). TTFT is now raw kernel + queue factor. See
`.codex-notes/from-aaron-meeting-2026-04-30.md`.
