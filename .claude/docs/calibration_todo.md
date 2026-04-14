---
# LLMCompass Calibration Status
---

## Current State (after ML hybrid predictor)

### Decode — SOLVED via ML predictor
- Previous: 0.43-0.67x (analytical underpredicts badly)
- Now: **0.81-1.00x** with `use_ml_predictor=True`
- Root cause was analytical model assuming 100% HBM bandwidth; RF model learns real ~60-70%

### Prefill — Analytical is already good
- Analytical: 0.86-1.00x (well-calibrated for compute-bound GEMMs)
- ML: 1.03-1.51x (overpredicts, especially at medium batch sizes)
- Root cause: SDPA attention profiling dispatches differently than fused flash_attn

## Recommended Configuration

```python
# Decode: use ML (0.81-1.00x)
block = TransformerBlockAutoRegressionTP(..., use_ml_predictor=True)

# Prefill: use analytical (0.86-1.00x)
block = TransformerBlockInitComputationTP(..., use_ml_predictor=False)
```

## Remaining Improvements

1. **Prefill attention accuracy**: Profile with `flash_attn` package instead of PyTorch SDPA
2. **Phase-aware auto-selection**: Automatically pick ML for decode, analytical for prefill
3. **More hardware**: Profile A100, L40S for cross-hardware support
4. **Cross-node modeling**: Add InfiniBand latency/bandwidth for virtual GPU clusters
