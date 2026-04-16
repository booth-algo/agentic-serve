---
# Multi-Node Communication Model for LLMCompass
---

## What Was Built

An analytical multi-node communication model replacing ASTRA-Sim's event-driven approach.

### New Files / Modules

1. **`llm_predict/profiling/category_profiler.py`** — Real GPU per-category profiler (GEMM, attention, elementwise)
2. **`llm_predict/predictors/per_category/predictor.py`** — RandomForest CategoryPredictor trained on profile data
3. **`llm_predict/profiling/data/H100/`** — profiled data points + trained .pkl models (stored on R2)

### Modified Files

4. **`llmcompass/hardware_model/interconnect.py`** — Added:
   - `NVLinkV4`, `InfiniBandHDR/NDR/XDR` link modules
   - `create_multi_node_interconnect()` — builds heterogeneous bandwidth matrices
   - `H100_NVLink_2/4/8` single-node configs

5. **`llmcompass/software_model/communication_primitives.py`** — Added:
   - `AllReduceHierarchical` — 3-step NCCL-style: intra ReduceScatter → inter AllReduce → intra AllGather
   - Falls back to flat AllReduce for single-node

6. **`llmcompass/software_model/transformer.py`** — Added:
   - `use_ml_predictor` flag for hybrid ML+analytical prediction
   - ML path: RF for GEMMs + attention, analytical for elementwise, 15us/kernel overhead

## Architecture

```
Per-Layer Prediction = ML(GEMM) + ML(Attention) + Analytical(Elementwise) + Kernel_Overhead + Communication

Communication (single-node) = AllReduceMultiPCB with NVLink bandwidth
Communication (multi-node)  = AllReduceHierarchical:
  Step 1: Intra-node ReduceScatter (NVLink, ~900 GB/s)
  Step 2: Inter-node AllReduce (InfiniBand, ~50-100 GB/s)
  Step 3: Intra-node AllGather (NVLink, ~900 GB/s)
```

## Validation Results

### Per-Layer Accuracy: 18/18 GOOD (0.83-1.14x)

ML hybrid vs measured on real H100:
- Prefill: 0.88-1.14x across 3 models × 3 batch sizes
- Decode: 0.83-1.09x across 3 models × 3 batch sizes
- Previous analytical: 0.44-1.00x (decode was terrible)

### Multi-Node Scaling (Llama-70B, BS=8, seq=512)

| Config | Prefill/layer | Decode/layer |
|--------|--------------|--------------|
| 1 node TP=1 | 8.52ms | 0.74ms |
| 1 node TP=2 | 4.78ms | 0.54ms |
| 1 node TP=4 | 3.17ms | 0.45ms |
| 1 node TP=8 | 2.87ms | 0.43ms |
| 2 nodes TP=8 | 12.86ms | 0.43ms |
| 4 nodes TP=16 | 23.95ms | 0.49ms |

## Usage

```python
from llmcompass.hardware_model.interconnect import create_multi_node_interconnect
from llmcompass.hardware_model.system import System

# Create 2-node cluster with 4 H100s per node
ic = create_multi_node_interconnect(
    num_nodes=2, gpus_per_node=4,
    intra_node_link="NVLinkV4",
    inter_node_link="InfiniBandNDR",
)

# Create system
system = System(h100_device, ic)

# Predict with ML hybrid
block = TransformerBlockInitComputationTP(
    d_model=8192, n_heads=64, n_kv_heads=8,
    device_count=8, data_type=fp16,
    use_ml_predictor=True,
)
```

## Comparison: ASTRA-Sim vs Our Approach

| Aspect | ASTRA-Sim | Our Model |
|--------|-----------|-----------|
| Method | Event-driven C++ subprocess | Analytical Python |
| Overhead | 1.41x end-to-end | ~0% (no subprocess) |
| Accuracy | Packet-level (overkill for LLM) | Ring formula + hierarchical |
| Setup | Chakra traces + C++ build | Single function call |
| Multi-node | Full topology routing | Bandwidth matrix + hierarchical AR |

## Next Steps

1. Wire `AllReduceHierarchical` into transformer.py (currently uses flat AllReduceMultiPCB)
2. Add pipeline parallelism (PP) bubble modeling for PP+TP configs
3. Validate multi-node predictions against real 2-node H100 measurements
4. Profile InfiniBand actual latency/bandwidth on multi-node cluster
