# H100 Validation Summary (2026-04-15)

## Setup
- Hardware: 4x NVIDIA H100-SXM5-80GB
- Profiling: CUDA events, isolated single-layer forward pass (2-layer models)
- Predictor: H100 RandomForest ML (trained on 3992 H100 kernel profiles)
- Kernel predictor MAPE: GEMM 1.04%, attn prefill 2.93%, attn decode 0.60%

## Key Fix: GH100.json Memory Config
- `memory_protocol` was `HBM2e` (A100), changed to `HBM3`
- `bandwidth_per_pin_bit` was `3.2e9` (2.05 TB/s = A100), changed to `5.24e9` (3.35 TB/s = H100)
- Also found: A100 XGBoost `perop_analytical_v4.pkl` was silently taking priority over H100 RF predictor

## Results

### Dense Models (H100 RF predictor, A100 XGBoost disabled)

| Model | GOOD/Total | Prefill Range | Decode Range |
|-------|-----------|---------------|-------------|
| Llama-3.1-8B | **18/24 (75%)** | 0.97-1.33x | 1.10-1.12x |

Decode: near-perfect (10-12% error). Prefill: excellent at large batch*seq, ~30% over at small batch.

### MoE Models (H100 RF predictor)

| Model | Experts | GOOD/Total | Decode bs=8 | Prefill Range |
|-------|---------|-----------|-------------|---------------|
| Mixtral-8x7B | 8, top-2 | **5/21 (24%)** | 0.94-1.00x | 1.56-1.93x |
| gpt-oss-20b | 32, top-4 | **0/21 (0%)** | 1.22-1.29x | 3.23-3.98x |

### MoE Analysis

Prefill overprediction scales with num_experts:
- 8 experts: ~1.8x overprediction
- 32 experts: ~3.5x overprediction

Root cause: LLMCompass models expert FFNs as sequential per-expert GEMMs.
Real H100 execution parallelizes expert dispatch much more effectively:
- Mixtral MLP (8 experts): 628-1535us for tok=1-256 (sub-linear scaling)
- gpt-oss MLP (32 experts): 509-1092us for tok=1-256 (sub-linear scaling)
- Both show MLP latency only 2x increase for 256x more tokens

Decode at bs=8 is accurate because: (a) top-k experts is small (2-4),
(b) per-expert GEMM at M=1 is memory-bound and well-predicted by RF.

## Sub-module Profile Data (H100, CUDA events)

### Mixtral-8x7B (d=4096, h=32, kv=8, ffn=14336, E=8, k=2)
| Tokens | Total (us) | Attention (us) | MLP/MoE (us) | Norms (us) |
|-------:|-----------:|---------------:|-------------:|-----------:|
| 1 | 1094 | 317 | 629 | 220 |
| 16 | 1913 | 314 | 1458 | 180 |
| 64 | 1902 | 375 | 1460 | 179 |
| 256 | 1992 | 320 | 1536 | 178 |

### gpt-oss-20b (d=2880, h=64, kv=8, ffn=2880, E=32, k=4)
| Tokens | Total (us) | Attention (us) | MLP/MoE (us) | Norms (us) |
|-------:|-----------:|---------------:|-------------:|-----------:|
| 1 | 1206 | 526 | 510 | 222 |
| 16 | 1541 | 542 | 853 | 195 |
| 64 | 1674 | 530 | 997 | 193 |
| 256 | 1764 | 524 | 1092 | 192 |

## Paper Implications

1. **Dense model prediction on H100**: RF predictor trained on H100 CUDA events achieves 75% GOOD (within 20%). Suitable for design space exploration.

2. **MoE prediction gap**: The sequential expert GEMM model in LLMCompass fundamentally mismatches H100's parallel expert execution. This is a known limitation that scales with num_experts.

3. **Contribution opportunity**: A fused/parallel expert dispatch model could close the 1.8-3.5x gap. The sub-module profiles provide ground truth for calibrating such a model.

4. **Cross-hardware insight**: The same MoE model was 0.34-0.48x (underprediction) on A100 and 1.2-3.5x (overprediction) on H100, suggesting the simulator's MoE model is calibrated for neither — it sits between A100 and H100 actual behavior.

## Files Produced
- `experiment/dense_validation_llama_8b_h100.json`
- `experiment/moe_validation_gpt_oss_20b_h100.json`
- `experiment/moe_validation_mixtral_8x7b_h100.json`
- `experiment/profiles_h100/Mixtral_8x7B_Instruct_v0_1_moe_profile.csv`
- `experiment/profiles_h100/gpt_oss_20b_moe_profile.csv`
- `experiment/validate_moe_h100.py`
- `experiment/validate_dense_h100.py`
- `experiment/profile_moe_h100.py`
- `device_configs/GH100.json` (fixed memory bandwidth)
