# Prefill latency: the three-layer comm/compute decomposition (2026-06-19)

Diagnosis of why TTFT (and thus E2EL) is mispredicted under concurrency, separating what is a
tensor-parallel **communication** problem from what is a **compute** problem and what is an
**expert-parallel** gap. Established by ablation + an adversarial code audit (see end).

## TL;DR

The under-load prefill error is **not one thing**. There are three structurally distinct issues;
only one is TP comm, and for MoE the dominant one is **not comm at all**.

| # | Issue | Kind | Scope | Status today |
|---|-------|------|-------|--------------|
| 1 | TP all-reduce comm over-charge (`(tp−1)` linear extrapolation) | comm | dense **and** MoE, every tp>1 cell | **active** error |
| 2 | MoE prefill compute uses **total** instead of **active** expert params | compute | MoE only | **active, dominant** for MoE |
| 3 | MoE expert-parallel **all-to-all** (dispatch/combine) comm | comm | MoE under EP only | **latent** (no EP in GT yet) |

## Evidence

### The c=1 control: per-request tp>1 prefill cost is already correct
A100×4 tp4: Llama-3.1-70B c1 TTFT 167 vs 173 ms (15%), Qwen2.5-72B 248 vs 242 (8%). The compute
roofline shards by tp fit-free; the per-request model is fine. The error is purely **under load**.

### #1 — TP comm over-charge (ablation, big-pool 8B cells, no admission confound)
`_prefill_gemm_per_tok_loaded` adds `PREFILL_TP_COMM_MS_PER_TOKEN·(tp−1)` per token. Ablating it:

```
a100 8B tp4 TTFT  Δms(pred-meas)/MAPE      c80          c160          c320
base (comm on)                          +431/53%     +1666/52%     +5949/43%
comm_off                                +261/38%      +325/33%     +1590/23%   ← removes ~70%
util0.754 (cap ablation)                +444/54%     +1766/53%     +6275/45%   ← cap is NOT it
```
Comm is the dominant tp>1 load over-charge. But it should not be *zeroed*: h100 8B tp4 comm_off
*under*-shoots (−262 at c320). The bug is the **linear `(tp−1)`** law — ring all-reduce scales
sub-linearly (`2·(tp−1)/tp` → tp4 ≈ 1.5× tp2, not 3×) and the per-rank rate is GPU-specific. The
util-saturation cap (1.0 vs measured 0.754) is exonerated.
**Fix:** measure prefill comm at tp4 per GPU (existing `build_tp_comm.py` stage-split) → pin
`prefill_tp_comm_ms_per_token` per deployment. Interim, zero-GPU: replace `(tp−1)` with the
ring law `2·(tp−1)/tp` anchored at the measured tp2 point.

### #2 — MoE prefill compute: total vs active experts (the dominant MoE error, NOT comm)
Prefill GEMM is `2·(n_params/tp)·tokens/(peak_flops·util)` with `n_params = total`. MoE prefill
runs only top-k experts → FLOPs ∝ **active** params. Over-charge ratio = total/active, and the
data tracks it exactly:

| MoE cell | total → active | total/active | TTFT error |
|----------|----------------|--------------|------------|
| gpt-oss-120b tp4 | 117B → 5.1B | ~23× | **+26.7 s / 1330%** @ c320 |
| gpt-oss-20b tp1 | 21B → 3.6B | ~5.8× | 38–141% — **tp1 has zero comm** |
| Mixtral-8x7B tp4 | 46.7B → 12.9B | ~3.6× | 39%, mildest (flips negative) |

The **tp1 cell proves it is not comm** (no inter-GPU comm, still 38–141% off). This is the prefill
analog of the active-expert *decode* read on `moe-active-expert-decode`; prefill was never given the
active-params treatment, so the deployed predictor over-charges MoE prefill by total/active.
**Fix:** use active params (`n_active_params`) in the prefill compute roofline when the model is MoE.
Decode (memory-bound weight read) is a separate axis handled on the decode branch.

### #3 — EP all-to-all comm: a real but latent separate gap
Adversarial repo-wide audit could **not** find any modeling of MoE expert-parallel all-to-all
(dispatch/combine) communication anywhere. The MoE config fields that exist (`is_moe`, `n_experts`,
`moe_overhead_us` in `simulator/configs/model_configs.py`) are **dead code** (imported by nothing).
The only inter-GPU comm modeled is TP all-reduce (`prefill_tp_comm_*`, `gpu_specs.tp_comm_latency_us`).

It does **not** bite today: GT used **pure TP** for MoE (no `--enable-expert-parallel` anywhere — the
only `EP=` match is a shell var for the R2 endpoint). Under TP, experts are sharded and the FFN is
all-reduced like dense, so the GT has no all-to-all to fit. The gap becomes a live, *additive* comm
error the moment we run/predict expert-parallel configs (the natural scaling path for gpt-oss-120b /
Mixtral / DeepSeek-class across nodes). All-to-all ≠ all-reduce, so the #1 TP fix does not touch it.
**Build when we run EP**, gated off by topology (byte-identical when EP disabled).

## Priority
1. **#2** active-vs-total prefill params — dominant, current, cheap, not comm.
2. **#1** TP comm — shared dense+MoE; ring-law interim now, measured tp4 comm to lock it.
3. **#3** EP all-to-all — real but latent; model it gated-off, activate when EP GT exists.

## Method note
Ablation: `/tmp/ablate_comm.py` (monkeypatch `_prefill_gemm_per_tok_loaded`, big-pool 8B cells).
Audit: 5-agent workflow `ep-moe-comm-audit` (adversarial finder survived; topology + data checked
directly). `ttft_err` in the predictions JSON is **abs MAPE**; signed direction via `pred-meas`.
