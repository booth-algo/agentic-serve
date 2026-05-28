# H100 Flash Decode Residual Safety Plan

## Summary

The H100 flash attention profiling pipeline now produces `flash_attn/H100.csv`
and `models/flash_H100.pkl`, but serving validation shows the trained residual
overpredicts some decode shapes. The residual was trained from PyTorch
`scaled_dot_product_attention` NCU measurements, while serving uses vLLM-native
kernels, so decode extrapolation can inflate TPOT by 3.5-4x roofline.

Implement a decode-only safety cap on extrapolated flash residual predictions.
Preserve exact table hits and keep prefill behavior unchanged.

## Key Changes

- In `llm_predict/kernels/flash_attn.py`, cap only extrapolated decode
  predictions.
- Treat decode as `phase == "decode"` or `seq_len == 1`.
- Keep exact `_table` hits returning measured table latency.
- If `_xgb` is used for decode, return
  `min(residual_pred, roofline_baseline * 2.0)`.
- Keep prefill and cached-prefill predictions on the existing residual path.
- Keep the generated H100 artifacts:
  - `llm_predict/data/flash_attn/H100.csv`
  - `llm_predict/data/models/flash_H100.pkl`

## Test Plan

- Add focused tests for `FlashAttnPredictor`:
  - Exact table lookup bypasses the cap and returns the CSV value.
  - Decode XGBoost extrapolation is capped at `2.0 * roofline`.
  - Prefill XGBoost extrapolation is not decode-capped.
- Run existing suite:
  - `python3 -m pytest llm_predict/tests`
  - `python3 -m py_compile llm_predict/kernels/flash_attn.py`
- Run focused H100/Llama-8B synthetic multi-turn validation and report MAPE for:
  - `chat-multiturn-synth`
  - `osworld-multiturn-synth`
  - `swebench-multiturn-synth`
  - `terminalbench-multiturn-synth`

## Assumptions

- Default cap is `2.0x` roofline because quick validation showed uncapped
  residual severely overpredicts chat TPOT, while roofline-only underpredicts
  long-context workloads.
- This is a safety guard, not final calibration. The better long-term fix is
  profiling vLLM's native flash attention kernel directly.
- Do not change TTFT queuing, KV eviction, or per-turn aggregation in this pass.
