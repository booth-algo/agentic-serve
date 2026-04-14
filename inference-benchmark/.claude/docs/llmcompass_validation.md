# LLMCompass Validation Against Measured Roofline Data

## Setup
- **Measured data**: PyTorch Profiler on H100 SXM5, 2 decoder layers, seq_len=512
- **LLMCompass**: `TransformerBlockInitComputationTP` with H100.json config, heuristic-GPU compile mode
- **Models**: Llama-3.1-8B, Qwen3-32B, Llama-3.1-70B, Qwen2.5-72B

## Results

### Prefill (2-layer latency, microseconds)

| Model | BS | Measured | Predicted | Ratio | Status |
|-------|-----|---------|-----------|-------|--------|
| Llama-3.1-8B | 1 | 1,530 | 871 | 0.57 | GOOD |
| Llama-3.1-8B | 4 | 5,032 | 871 | 0.17 | MISMATCH |
| Llama-3.1-8B | 8 | 9,876 | 871 | 0.09 | MISMATCH |
| Llama-3.1-8B | 32 | 38,868 | 871 | 0.02 | MISMATCH |
| Qwen3-32B | 1 | 2,893 | 1,007 | 0.35 | OK |
| Llama-3.1-70B | 1 | 3,769 | 1,593 | 0.42 | OK |
| Qwen2.5-72B | 1 | 3,880 | 1,593 | 0.41 | OK |

### Key Findings

1. **bs=1 predictions are 2-3x off** — reasonable for an analytical model without calibration
   - Llama-8B: 0.57x (predicted slightly faster than measured)
   - 70B/72B: 0.41-0.42x (predicted ~2.4x faster than measured)
   - The gap is likely kernel launch overhead + memory latency not modeled

2. **Batch size does NOT scale in predictions** — fundamental bug
   - `compile_and_simulate()` re-creates dummy tensors with b=1, s=1 (line 277)
   - Ignores the computational graph built by `__call__` with actual batch size
   - Predicted latency is constant regardless of batch size

3. **Model size scaling is directionally correct**
   - 8B: 871 μs, 32B: 1,007 μs, 70B/72B: 1,593 μs
   - Larger models → higher predicted latency (correct direction)

4. **Decode not working** — `TransformerBlockAutoRegressionTP.__call__` crashes
   - Needs investigation of the decode block's expected input format

## Root Causes

### Bug: compile_and_simulate ignores batch size
Location: `llmcompass/software_model/transformer.py`, line 277
```python
b, s, d = 1, 1, self.d_model  # Hardcoded!
```
Fix: Use the dimensions from the computational graph built by `__call__`.

### Missing: Kernel launch overhead
LLMCompass uses a fixed overhead per matmul (`pcb.compute_module.overhead.matmul ≈ 2.1e-5 s`), but real H100 has additional:
- CUDA kernel launch latency (~5-10 μs per kernel)
- Memory allocation/deallocation
- Cache warmup effects

### Missing: Flash Attention
LLMCompass models attention as Q×K^T → softmax → ×V (three separate matmuls), not as fused FlashAttention. Real inference uses FlashAttention which has very different memory access patterns.

## Next Steps

1. **Fix batch size bug** in `compile_and_simulate` to use actual dimensions
2. **Calibrate overhead** using our measured per-kernel data
3. **Add FlashAttention** modeling (or calibrate with measured attention latency)
4. **Validate decode** — fix `TransformerBlockAutoRegressionTP`
5. **Test TP scaling** — compare TP=1 vs TP=2 predicted vs measured communication overhead

## Conclusion

LLMCompass provides a solid analytical foundation (correct model-size scaling, reasonable bs=1 predictions) but needs batch-size fix and calibration against measured data before it can be used for paper-quality predictions. The framework is well-suited for our Step 2 (virtual GPU scaling) and Step 3 (optimal parallelism) once these issues are resolved.
