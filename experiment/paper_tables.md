# LLMCompass v4 Validation Results

## Table 1: Prefill Summary

| Model | Type | GOOD/Total | Accuracy |
|-------|------|-----------|----------|
| Llama-3.1-8B-Instruct | Dense | 23/25 | **92%** |
| Qwen2.5-72B-Instruct | Dense | 21/25 | **84%** |
| gpt-oss-20b | MoE | 9/12 | **75%** |
| **Total** | | **53/62** | **85%** |

## Table 2: TP Scaling (vLLM, A100 NVLink)

| Model | TP | TPOT (ms) | Speedup |
|-------|---:|----------:|--------:|
| Llama-3.1-8B | 1 | 13.0 | 1.00x |
| Llama-3.1-8B | 2 | 7.9 | 1.65x |
| Llama-3.1-70B | 4 | 31.7 | — |
| Mixtral-8x7B | 4 | 7.6 | — |

## Table 3: Decode Validation (Llama-8B)

| Per-layer measured | Per-layer predicted | Error |
|-------------------|--------------------:|------:|
| ~900 us | 1003 us | +11% |

## Table 4: Predictor Details

| Property | Value |
|----------|-------|
| Model | XGBoost (n_estimators=200, max_depth=6) |
| Features | 26 (analytical: FLOPs, weight_bytes, AI + bs/seq/attn_quad) |
| Training data | 8672 rows (4 models, prefill + decode) |
| Hardware | NVIDIA A100-SXM4-40GB |