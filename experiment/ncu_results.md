## ncu Hardware Counters for Key GEMM Shapes (A100-SXM4-40GB, bf16)

| Kernel | Shape (M×N×K) | Time (µs) | DRAM Read (MB) | DRAM Write (MB) | TFLOPS | %Peak | HBM BW (GB/s) | %Peak | AI |
|--------|--------------|----------:|---------------:|----------------:|-------:|------:|--------------:|------:|---:|
| Llama-8B QKV prefill | 512×4096×4096 | 112.7 | 38.5 | 2.9 | 152.4 | 49% | 367 | 18% | 415 |
| Llama-8B QKV decode | 1×4096×4096 | 32.7 | 33.6 | 2.2 | 1.0 | 0% | 1094 | 54% | 1 |
| Llama-8B FFN gate | 512×14336×4096 | 388.9 | 129.1 | 9.4 | 154.6 | 50% | 356 | 17% | 434 |
| Llama-70B QKV | 512×8192×8192 | 401.5 | 159.3 | 15.8 | 171.2 | 55% | 436 | 21% | 393 |
| Mixtral expert FFN | 64×14336×4096 | 91.2 | 118.0 | 2.4 | 82.4 | 26% | 1320 | 65% | 62 |
| Decode FFN | 1×14336×4096 | 93.4 | 117.5 | 2.4 | 1.3 | 0% | 1284 | 63% | 1 |
| Large prefill | 4096×4096×4096 | 652.5 | 164.6 | 30.7 | 210.6 | 68% | 299 | 15% | 704 |

**Hardware**: NVIDIA A100-SXM4-40GB, bf16 peak = 312 TFLOPS, HBM2e = 2039 GB/s
**Profiler**: ncu 2024.3.2 (CUDA 12.6), --set full

### Key Observations
1. Large prefill GEMMs achieve 49-68% of peak compute — firmly compute-bound (AI > 200)
2. Decode GEMMs (M=1) achieve 54-63% HBM BW — firmly memory-bound (AI ≈ 1)
3. Mixtral expert FFN (M=64) is memory-bound at 65% BW — even moderate batch sizes dont make MoE compute-bound
4. The compute/memory boundary on A100 is at AI ≈ 153 (312 TFLOPS / 2039 GB/s). Kernels above this are compute-bound.
5. Decode FFN reads ~118MB (full weight matrix) in 93µs — confirms memory-bound regime for all decode operations
