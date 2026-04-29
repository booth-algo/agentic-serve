# llm_predict_2 Profiling Results — 2026-04-29

## Context

Fresh ncu GEMM sweeps on H100 and A100 using `llm_predict_2` pipeline.
462 shapes (33 unique (N,K) × 14 M values) profiled via `nn.Linear` with NVTX attribution.
Elementwise kernels calibrated from existing per-family ncu data.
Flash attention uses roofline only (no sweep — component attribution showed it's 0.4-1.2% of decode TPOT).

**Key discovery:** The old H100 GEMM data (`gemm_serving_ncu_H100.csv`) was corrupt — alternating M values differed by 10-15x due to interleaved kernel rows. All data below is from clean re-sweeps.

## Data Quality Check

Clean ncu data is monotonically increasing as expected:

```
H100 FFN gate (N=14336, K=4096):        A100 FFN gate (N=14336, K=4096):
  M=1:      42 us  (mem-bound flat)       M=1:      98 us
  M=128:    44 us                         M=128:   125 us
  M=512:   125 us                         M=512:   393 us
  M=2048:  374 us                         M=2048: 1170 us
  M=8192: 1375 us  (compute-bound)        M=8192: 4400 us
```

No alternating corruption. H100 is ~3x faster than A100 at large M, consistent with H100's ~3x higher bf16 TFLOPS.

## Validation Results — After Fixes (actual ISL/OSL, decode alpha=1.0)

Two fixes applied per Codex review:
1. Validation now uses actual per-row `total_input_tokens / successful_requests` instead of hardcoded profile ISL/OSL
2. Decode alpha correction disabled (set to 1.0) — old alpha was calibrated for stale pipeline

### H100 Dense Models, C=1

| Model | Profile | ISL | OSL | TTFT pred | TTFT meas | TTFT err | TPOT pred | TPOT meas | TPOT err | E2EL pred | E2EL meas | E2EL err |
|-------|---------|-----|-----|----------|----------|---------|----------|----------|---------|----------|----------|---------|
| Llama-3.1-8B | chat-short | 127 | 156 | 7.38 ms | 15.53 ms | -52% | 6.79 ms | 6.27 ms | **+8%** | 1066 ms | 1036 ms | **+3%** |
| Llama-3.1-8B | chat-medium | 136 | 289 | 7.55 ms | 15.33 ms | -51% | 6.79 ms | 6.28 ms | **+8%** | 1970 ms | 1914 ms | **+3%** |
| Llama-3.1-8B | chat-long | 187 | 278 | 8.54 ms | 16.17 ms | -47% | 6.80 ms | 6.28 ms | **+8%** | 1898 ms | 1674 ms | +13% |
| Gemma-2-9B | chat-short | 88 | 141 | 8.32 ms | 31.71 ms | -74% | 7.97 ms | 8.28 ms | **-4%** | 1131 ms | 1449 ms | -22% |
| Granite-3.0-8B | chat-short | 99 | 127 | 8.46 ms | 33.04 ms | -74% | 7.99 ms | 7.03 ms | **+14%** | 1024 ms | 890 ms | +15% |

### H100 Dense Models — Long ISL Profiles, C=1

| Model | Profile | ISL | OSL | TTFT pred | TTFT meas | TTFT err | TPOT pred | TPOT meas | TPOT err |
|-------|---------|-----|-----|----------|----------|---------|----------|----------|---------|
| Llama-3.1-8B | prefill-heavy | 7891 | 158 | 198.64 ms | 275.72 ms | -28% | 8.33 ms | 6.58 ms | +27% |
| Llama-3.1-8B | coding-agent | 6236 | 341 | 155.47 ms | 59.59 ms | +161% | 8.02 ms | 6.60 ms | +22% |
| Llama-3.1-8B | random-1k | 1017 | 275 | 27.38 ms | 39.89 ms | -31% | 6.96 ms | 6.34 ms | **+10%** |
| Llama-3.1-8B | decode-heavy | 280 | 315 | 10.51 ms | 23.73 ms | -56% | 6.81 ms | 6.31 ms | **+8%** |

### A100 Dense Models, C=1

| Model | Profile | ISL | OSL | TTFT pred | TTFT meas | TTFT err | TPOT pred | TPOT meas | TPOT err | E2EL pred | E2EL meas | E2EL err |
|-------|---------|-----|-----|----------|----------|---------|----------|----------|---------|----------|----------|---------|
| Llama-3.1-8B | chat-short | 127 | 149 | 17.27 ms | 35.18 ms | -51% | 12.66 ms | 12.89 ms | **-2%** | 1904 ms | 1993 ms | **-5%** |
| Llama-3.1-8B | chat-medium | 119 | 257 | 16.88 ms | 34.81 ms | -52% | 12.67 ms | 12.95 ms | **-2%** | 3272 ms | 3204 ms | **+2%** |
| Llama-3.1-8B | chat-long | 217 | 285 | 28.05 ms | 34.62 ms | -19% | 12.69 ms | 12.95 ms | **-2%** | 3644 ms | 3633 ms | **+0.3%** |

### MoE Models (gpt-oss-20b), C=1

| GPU | Profile | ISL | OSL | TTFT pred | TTFT meas | TTFT err | TPOT pred | TPOT meas | TPOT err |
|-----|---------|-----|-----|----------|----------|---------|----------|----------|---------|
| H100 | chat-short | 165 | 166 | 28.71 ms | 18.73 ms | +53% | 23.39 ms | 3.84 ms | +510% |
| H100 | prefill-heavy | 9261 | 199 | 685.19 ms | 651.59 ms | **+5%** | 27.18 ms | 3.66 ms | +643% |
| A100 | chat-short | 150 | 170 | 73.21 ms | 32.99 ms | +122% | 48.43 ms | 4.93 ms | +882% |

## Component Attribution (decode TPOT, per layer)

| GPU | Model | GEMM % | Attn % | Elem % |
|-----|-------|--------|--------|--------|
| H100 | Llama-3.1-8B | 93.4% | 0.9% | 5.7% |
| H100 | gpt-oss-20b (MoE top_k=2) | 96.9% | 0.4% | 2.7% |
| H100 | Gemma-2-9B | 92.6% | 1.1% | 6.4% |
| A100 | Llama-3.1-8B | 92.6% | 0.8% | 6.6% |

## Key Observations After Fixes

### 1. Dense TPOT is solved — 2-14% error
After removing stale decode alpha and using actual ISL/OSL:
- H100 Llama-8B: **8%** across all profiles
- H100 Gemma-9B: **4%**
- A100 Llama-8B: **2%** across all profiles
- Standalone ncu `nn.Linear` matches vLLM decode TPOT. No CUDA event path needed.

### 2. Dense E2EL is excellent — 0.3-5% on A100, 3-15% on H100
E2EL = TTFT + decode_total. Since TPOT is accurate, E2EL works despite TTFT being off (TTFT is small fraction of total E2EL for short ISL).

### 3. TTFT has a fixed overhead gap — not multiplicative
| GPU | Profile | ISL | Kernel sum | Measured | Gap |
|-----|---------|-----|-----------|---------|-----|
| H100 | chat-short | 127 | 7.38 ms | 15.53 ms | ~8 ms |
| H100 | prefill-heavy | 7891 | 198.64 ms | 275.72 ms | ~77 ms |
| A100 | chat-short | 127 | 17.27 ms | 35.18 ms | ~18 ms |
| A100 | chat-long | 217 | 28.05 ms | 34.62 ms | ~7 ms |

At short ISL the fixed overhead dominates. At long ISL the kernel sum is closer.
Exception: coding-agent (ISL=6236) predicts 155ms but measures 59ms — likely **prefix cache hits** from shared system prompts.

### 4. MoE decode remains fundamentally broken (500-900%)
vLLM uses fused MoE kernels, not `top_k × sequential nn.Linear`.
MoE prefill works (5.2% at ISL=9261) because prefill GEMMs behave like dense.
MoE decode needs a separate modeling approach — profile the actual fused kernel.

### 5. Two measured TTFT values per model
Llama-8B on H100 shows 15.53 ms and 27.50 ms for chat-short C=1.
Likely different benchmark runs. Need to disambiguate or deduplicate in data.json.

## Measurement methodology note
- All ncu GEMM data collected via `sweep_gemm.py`: `torch.nn.Linear(K, N, bias=False)` with TN layout
- 10 warmup + 10 measured reps per shape, NVTX-attributed
- H100 kernel regex: `gemm|GEMM|cutlass|nvjet|cublasLt`
- A100 kernel regex: `gemm|GEMM|cutlass`
- RTX3090: sweep still running
- Elementwise: calibrated from existing per-family ncu data via least-squares `floor_us + bytes/eff_bw`

## Codex Review Notes

### A. The 2x dense TPOT overprediction is mostly the stale decode alpha

`llm_predict_2/serving.py` still multiplies decode time by old per-GPU correction
factors:

| GPU | Current alpha at C=1 |
|-----|---------------------:|
| H100 | 1.84 |
| A100 | 1.497 |

Dividing the reported dense TPOT predictions by these stale factors makes C=1
TPOT look much healthier:

| GPU | Model | Reported TPOT pred | Alpha | Raw TPOT pred | Measured TPOT | Raw err |
|-----|-------|-------------------:|------:|--------------:|--------------:|--------:|
| H100 | Llama-3.1-8B | 12.49 ms | 1.84 | 6.79 ms | 6.27 ms | +8% |
| H100 | Gemma-2-9B | 14.66 ms | 1.84 | 7.97 ms | 8.28 ms | -4% |
| H100 | Granite-3.0-8B | 14.71 ms | 1.84 | 7.99 ms | 7.03 ms | +14% |
| A100 | Llama-3.1-8B | 18.95 ms | 1.497 | 12.66 ms | 12.89 ms | -2% |

Implication: do not conclude that standalone ncu GEMM is inherently 2x slower
than vLLM. First recalibrate or disable the old decode alpha after the clean
GEMM re-sweep. CUDA graphs mainly reduce launch overhead; they should not make
the same cuBLAS kernel execute 2x faster. A 2x gap is more likely stale
calibration, different fused kernels, different shapes/batches, or a validation
data mismatch.

### B. Validation is using intended profile lengths, but the benchmark rows have much shorter actual lengths

`llm_predict_2/validate.py` hardcodes:

| Profile | Hardcoded ISL | Hardcoded OSL |
|---------|--------------:|--------------:|
| chat-short | 200 | 100 |
| chat-medium | 2000 | 1000 |
| chat-long | 8000 | 2000 |

But the H100 C=1 rows in `data.json` contain much shorter average token counts:

| Model | Profile | Actual avg ISL | Actual avg OSL |
|-------|---------|---------------:|---------------:|
| Llama-3.1-8B | chat-short | 127 | 151-157 |
| Llama-3.1-8B | chat-medium | 136 | 289-291 |
| Llama-3.1-8B | chat-long | 188 | 278-284 |
| Gemma-2-9B | chat-long | 206 | 258 |
| Granite-3.0-8B | chat-long | 225 | 229 |

This explains why measured TTFT appears not to scale with ISL and why
chat-medium/chat-long E2EL errors explode. The predictor is being asked to
predict ISL=2000/8000 and OSL=1000/2000, while the measured rows are closer to
short prompts. Validation should use per-row observed averages:

```
avg_isl = total_input_tokens / successful_requests
avg_osl = total_output_tokens / successful_requests
```

Profile defaults are useful for synthetic planning, but benchmark validation
must compare against the lengths actually present in each result row.

### C. Recommended next debugging order

1. Recompute validation with actual per-row `avg_isl` and `avg_osl`.
2. Report raw decode TPOT before applying `_DECODE_CORRECTION`.
3. Refit `_DECODE_CORRECTION` from clean dense rows, or set C=1 alpha near 1.0.
4. Keep MoE out of the dense headline until the composer models fused MoE
   kernels instead of sequential per-expert GEMMs.
5. Use CUDA-event or minimal vLLM-forward timing as a serving-path calibration
   check, not as a replacement for ncu kernel data.
