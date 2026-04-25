# Session: H100 Validation + PDD Experiment + llm_predict Restructure

**Date**: 2026-04-15 to 2026-04-16
**Hardware**: 4x NVIDIA H100-SXM5-80GB (RunPod)

## Accomplishments

### 1. H100 ML Predictor Trained
Per-kernel RandomForest on 3992 H100 CUDA event profiles:
- GEMM: **1.04% MAPE** (n=3529, p95=3.10%)
- Attn Prefill: 2.93% MAPE (n=210)
- Attn Decode: **0.60% MAPE** (n=210)

RF vs XGBoost comparison: RF wins for GEMM, XGBoost wins for attn prefill (4.74% vs 9.71%).

### 2. Critical Bugs Fixed
- **GH100.json had A100 memory bandwidth** — `memory_protocol: HBM2e → HBM3`, `bandwidth_per_pin_bit: 3.2e9 → 5.24e9` (2.05 → 3.35 TB/s)
- **A100 XGBoost silently overrode H100 RF** — `_get_perop_predictor()` hardcoded A100 pkl path, caused 1.6-3x overprediction on H100 dense until disabled

### 3. Validation Results
**Dense (Llama-8B, H100 RF, TP=1):** 18/24 GOOD (75%), decode 10-12% error

**MoE key finding**: overprediction scales linearly with num_experts
| Model | Experts | Prefill over |
|-------|---------|-------------|
| Mixtral | 8 top-2 | 1.6-1.9x |
| gpt-oss | 32 top-4 | 3.2-4.0x |

LLMCompass models expert FFNs as sequential but H100 parallelizes. Sub-module profiling confirms MoE MLP only 2x slower for 256x more tokens.

### 4. PDD Experiment — Paper Headline
**Chunked prefill tradeoff** (gpt-oss-20b, prefill-heavy, conc=200):
- chunk128: 12.8x better TPOT but 16.6x worse TTFT
- No single setting resolves the tradeoff

**Real PDD** (SGLang + nixl, Llama-8B, GPU 0,1 prefill + GPU 2,3 decode):
| Profile | Baseline E2EL | PDD E2EL | Speedup |
|---------|--------------|----------|---------|
| chat-short conc=200 | 1731ms | 325ms | **5.3x** |
| prefill-heavy conc=200 | 27900ms | 4041ms | **6.9x** |
| random-1k conc=200 | 4719ms | 754ms | **6.3x** |

Uses `nixl` transfer backend over NVLink P2P (no RDMA needed).

### 5. Codebase Restructure
`llmcompass/` → `llm_predict/` with clean predictor organization:
```
llm_predict/
├── predictors/
│   ├── per_kernel/predictor.py   # KernelPredictor (RF)
│   ├── per_op/{predictor,features,corrections}.py  # PerOpPredictor (XGBoost)
│   └── dispatch.py               # per-kernel first → per-op fallback
├── models/{software,hardware,serving,cost}/
├── profiling/{data,kernel_profiler.py}
└── dse/
```

Key wins: predictor dispatch extracted from 2972-line transformer.py; `compute_perop_features()` defined once; CSVs vs .pkl separated.

### 6. Benchmark Coverage Filled
- Mixtral coding-agent: 0/20 → 20/20
- Qwen3.5-27B conc 200-320: 0/42 → 42/42
- Llama-70B TP2 high-conc: partially (SGLang has failures, needs rerun)
- Qwen2.5-72B TP2 SGLang: 35/70 → 65/70

### 7. R2 Migration
All GCS-dependent data sources now on R2 (~8.5 GB / 10 GB free tier):
- coding_agent_prompts.jsonl (13 MB)
- swebench_trajectories.jsonl (1.6 GB)
- terminalbench_trajectories.jsonl (2.0 GB)
- osworld_trajectories.jsonl (19 MB)
- Trained .pkl models (292 MB)
- All benchmark results

## Outstanding Issues

**Critical (need fix):**
1. Llama-70B TP2 SGLang — 310 failed requests on high-conc prefill-heavy/decode-heavy/coding-agent (OOM timeouts). Need rerun with `--mem-fraction-static 0.80` and shorter context.
2. NOSTREAM (36 cases) — Llama-70B TP2 SGLang low-conc returning 0 output tokens.

**Known (backlog):**
3. conc=120/160 underloaded (201 cases) — still have pre-fix `nreq=200`
4. gpt-oss MXFP4 TPOT drops (14 cases) — chunked streaming artifact, not fixable in vLLM 0.18.1
5. Qwen2.5-72B TP2 vLLM — only 4/10 concs per profile

## Files Created

**Experiment scripts:**
- `experiment/validate_moe_h100.py`
- `experiment/validate_dense_h100.py`
- `experiment/profile_moe_h100.py`
- `experiment/h100_validation_summary.md`
- `experiment/pdd_experiment_summary.md`

**New predictor modules** in `llm_predict/predictors/`:
- `per_kernel/predictor.py` (moved from old `profiler/ml_predictor.py`)
- `per_op/{predictor,features,corrections}.py` (extracted from `transformer.py`)
- `dispatch.py`

**Fixed:**
- `device_configs/GH100.json` (HBM3 bandwidth)
- `experiment/validate_model.py` (removed hardcoded paths)

## Commits

- `b0974d1` — H100 validation + PDD experiment
- `e006172` — llmcompass → llm_predict restructure (202 files)
- `50cfa59` — Move trained .pkl to R2

## Paper Contributions Confirmed

1. **Per-kernel prediction on H100**: 1-3% MAPE, publishable
2. **MoE overprediction scales with num_experts**: systematic finding
3. **PDD delivers 5-7x E2EL improvement**: core paper result
4. **Agentic workloads break colocated serving**: prefill-heavy @ conc=200 → 28s E2EL baseline
5. **Chunked prefill is insufficient**: TPOT/TTFT tradeoff unresolvable
