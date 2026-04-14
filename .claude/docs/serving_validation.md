---
# Serving-Level Validation: LLMCompass vs SGLang Measured
---

## Benchmark Setup
- Hardware: 2x NVIDIA H100 80GB HBM3 (RunPod, NVLink)
- Framework: SGLang (latest)
- Workload: 20 sequential requests, ~30 token prompt, 128 output tokens, temperature=0
- Configurations: TP=2 and PP=2

## Llama-70B (80 layers, d=8192) — PRIMARY VALIDATION

| Metric | Predicted | Measured | Ratio |
|--------|-----------|----------|-------|
| TP=2 TTFT | 46.8ms | 57.3ms | 0.82x |
| TP=2 TPOT | 25.89ms | 26.38ms | **0.98x** |
| TP=2 E2E | 3361ms | 3408ms | **0.99x** |
| PP=2 TTFT | 59.5ms | 62.1ms | **0.96x** |
| PP=2 TPOT | 55.61ms | 49.94ms | 1.11x |
| PP=2 E2E | 7177ms | 6405ms | 1.12x |

**All within 0.82-1.12x.** TP=2 E2E prediction is 0.99x — near-perfect.

## Llama-8B (32 layers, d=4096) — SMALL MODEL TEST

| Metric | Predicted (CG) | Measured | Ratio |
|--------|----------------|----------|-------|
| TP=2 TPOT | 6.05ms | 4.19ms | 1.44x |
| PP=2 TPOT | 12.92ms | 15.08ms | 0.86x |
| PP=2 E2E | 1667ms | 1967ms | 0.85x |

8B is harder: per-layer compute is tiny (~130us), framework overhead dominates.
70B is the realistic use case and predictions are excellent.

## Measured: TP vs PP Comparison

### Llama-70B
| Metric | TP=2 | PP=2 | PP/TP |
|--------|------|------|-------|
| TTFT | 57.3ms | 62.1ms | 1.08x |
| TPOT | 26.38ms | 49.94ms | 1.89x |
| E2E | 3408ms | 6405ms | 1.88x |

**TP=2 is 1.88x faster than PP=2** for Llama-70B on 2x H100 with NVLink.
PP only makes sense when the model doesn't fit with TP alone.

### Llama-8B
| Metric | TP=2 | PP=2 | PP/TP |
|--------|------|------|-------|
| TTFT | 24.8ms | 51.8ms | 2.09x |
| TPOT | 4.19ms | 15.08ms | 3.60x |
| E2E | 556ms | 1967ms | 3.54x |

## Model Settings Used
- LLMCompass: `use_ml_predictor=True`, `use_cuda_graph=True` (decode),
  allreduce overlap factor 0.75
- Prefill: no CUDA graph (dynamic shape)
- PP: no CUDA graph, no allreduce overlap (inter-stage P2P instead)
