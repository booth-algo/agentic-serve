# Session Summary - Simulator - 2026-05-14

## Goal

Start over on the serving predictor with a new standalone `simulator/` package
outside `llm_predict/`, focused only on H100 + Llama-3.1-8B until the modeling
approach is accurate and explainable.

## High-Level Outcome

- Added a standalone simulator package.
- Added H100 + Llama-3.1-8B kernel composition and turn-level prediction.
- Trained GEMM and flash XGBoost residual artifacts from existing CSV data.
- Exported simulator predictions for the dashboard.
- Added a Simulator dashboard page locked to H100 + Llama-3.1-8B.
- Added row-end median TTFT, TPOT, and E2EL error columns.
- Documented the next backend-emulation plan in
  `docs/simulator-backend-emulation-plan-2026-05-14.md`.

No GPU profiling was run in this session. Existing profile/benchmark artifacts
were consumed locally. If future H100 profiling is needed, it must explicitly
target GPU 6.

## Files Added

| File | Purpose |
|---|---|
| `simulator/__init__.py` | Public simulator exports |
| `simulator/models.py` | Llama-3.1-8B model specification |
| `simulator/roofline.py` | H100 roofline helpers |
| `simulator/kernel_models.py` | GEMM/flash table lookup, XGBoost residuals, roofline fallback |
| `simulator/provider.py` | Kernel-composed prefill/decode latency provider |
| `simulator/turns.py` | Dashboard per-turn summary adapter |
| `simulator/predictor.py` | Turn-level and row-level prediction logic |
| `simulator/train.py` | CLI for training kernel residual artifacts |
| `simulator/predict.py` | CLI for exporting dashboard simulator predictions |
| `simulator/tests/test_kernel_models.py` | Kernel model tests |
| `simulator/tests/test_scheduler.py` | Scheduler semantic tests |
| `simulator/tests/test_turns_and_predictor.py` | Turn adapter and predictor tests |
| `simulator/README.md` | Usage and scope notes |
| `inference-benchmark/dashboard/public/simulator-predictions.json` | Dashboard payload for simulator rows |
| `docs/simulator-backend-emulation-plan-2026-05-14.md` | Next implementation plan |

## Files Updated

| File | Purpose |
|---|---|
| `inference-benchmark/dashboard/src/App.tsx` | Added Simulator page route and H100/Llama-3.1-8B focus |
| `inference-benchmark/dashboard/src/components/ServingPredictionsPage.tsx` | Added prediction URL override, focus mode, row median columns, target summary |
| `inference-benchmark/dashboard/src/dataUrls.ts` | Added simulator predictions JSON URL |

## Simulator Behavior Implemented

The v1 simulator does the following:

1. Loads existing H100 GEMM and flash-attention kernel data.
2. Trains XGBoost residual models over roofline baselines.
3. Converts Llama-3.1-8B prefill/decode work into kernel shapes.
4. Uses observed per-turn benchmark summaries:
   - turn count,
   - successful requests,
   - total context tokens,
   - new prefill tokens,
   - cached context tokens,
   - cache hit rate,
   - output tokens.
5. Predicts per-turn TTFT, TPOT, and E2EL.
6. Aggregates per-turn predictions into row-level dashboard output.

## Generated Artifacts

Kernel residual artifacts were generated under `/tmp/simulator-artifacts/H100`:

| Artifact | Samples | Train residual MAPE |
|---|---:|---:|
| `gemm_H100.pkl` | 462 | 0.619% |
| `flash_H100.pkl` | 198 | 1.517% |

The dashboard simulator export contains:

- GPU: `H100`
- Model: `Llama-3.1-8B`
- Rows: `110`
- Profiles: `5`

## Validation

Commands run:

```bash
python3 -m pytest simulator/tests
npm run lint
npm run build
python3 -m simulator.predict \
  --data inference-benchmark/dashboard/public/data.synthetic_distributional.json \
  --kernel-data llm_predict/data \
  --artifacts /tmp/simulator-artifacts/H100 \
  --output /tmp/simulator-predictions-check.json \
  --gpu H100 \
  --model Llama-3.1-8B
```

Results:

- `8` simulator tests passed.
- Dashboard lint passed.
- Dashboard production build passed.
- Simulator export regenerated successfully with `H100: 110`.
- Local dashboard smoke checks returned HTTP `200`.

## Current Accuracy Caveat

The simulator is implemented and inspectable, but v1 accuracy is not final.
Kernel residual training looks good, but serving-level errors are still uneven.
The biggest missing piece is backend runtime emulation:

- vLLM/SGLang continuous batching,
- prefill/decode scheduling order,
- chunked prefill limits,
- decode batch efficiency,
- scheduler overhead,
- KV cache residency and eviction,
- prefix-cache/radix-cache realization.

The next step is to implement a lightweight backend event-loop emulator rather
than keep tuning aggregate formulas.

## Next Step

Implement the plan in
`docs/simulator-backend-emulation-plan-2026-05-14.md`, starting with:

1. `BackendSpec` for vLLM and SGLang knobs.
2. vLLM event-loop scheduler for one turn.
3. Trace output that explains TTFT, TPOT, and E2EL from scheduler steps.
4. Calibration of small backend parameters only after scheduler semantics are
   in place.

