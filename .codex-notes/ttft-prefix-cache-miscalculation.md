# TTFT Prefix-Cache Miscalculation

Date: 2026-05-02

## Problem

Current serving TTFT predictions are badly wrong for the active benchmark data,
especially prefix-cache affected workloads. The biggest failure mode is that
some rows are still predicted as full-prefill work even though the serving
engine is launched with prefix caching and chunked prefill enabled.

This should not be treated as a calibration problem. The intended predictor
should model the actual prefill regime:

- Full prefill: QKV/O/MLP GEMMs plus attention and KV writes over the new
  prompt.
- Prefix-cached prefill: GEMM/GEMV-like work for only the new suffix plus
  attention over cached context, KV prefix reads, and KV writes for new tokens.

When the new suffix is small and the cached context is large, TTFT can be
dominated by cached-KV loading / attention memory traffic rather than full
prompt GEMM work.

## Evidence From Current Prediction Data

Source files inspected:

- `inference-benchmark/dashboard/public/serving-predictions.json`
- `inference-benchmark/dashboard/public/data.json`

Current rows with TTFT error:

```text
all current rows: n=107, avg TTFT error=310.9%, p50=76.5%, p90=924.8%, p99=3255.9%
all current rows have ttft_validation_scope=prefix_cache_affected
prefix_cache_contention_applied=false for all 107 current rows
```

Cache-aware split:

```text
cache_aware_applied=false:
  n=15
  avg TTFT error=1615.5%
  p50=1199.1%
  p90=3255.9%
  p99=3786.2%

cache_aware_applied=true:
  n=92
  avg TTFT error=98.2%
  p50=71.0%
  p90=96.2%
  p99=1268.1%
```

Worst current profile:

```text
coding-singleturn:
  n=15
  avg TTFT error=1615.5%
  p50=1199.1%
  p90=3255.9%
```

Worst example rows:

```text
RTX3090 gpt-oss-20b vLLM coding-singleturn c=20:
  predicted TTFT=3653.11 ms
  measured TTFT=92.68 ms
  error=3841.7%
  cache_aware_applied=false
  new_prefill_tokens=6092
  total_context_tokens=6092
  cache_hit_rate=0
  calibration=fallback_static

RTX3090 gpt-oss-20b vLLM coding-singleturn c=40:
  predicted TTFT=6706.05 ms
  measured TTFT=172.56 ms
  error=3786.2%

RTX3090 gpt-oss-20b SGLang coding-singleturn c=20:
  predicted TTFT=8583.85 ms
  measured TTFT=255.78 ms
  error=3255.9%
```

Conclusion from data: yes, the catastrophic TTFT error is mainly because
prefix-cache affected rows are not being modeled as cached prefill. The clearest
case is `coding-singleturn`, where the predictor records `cache_aware_applied`
as false and treats `new_prefill_tokens == total_context_tokens`.

There is also a second, broader failure mode: even cache-aware rows are still
off because the current cached-prefill model is too coarse. No current row has
`prefix_cache_contention_applied=true`, and the repo records
`engine_cache_telemetry: "not_available"`, so exact cache residency is inferred
from metadata and prediction/measurement gaps rather than measured directly.

## Relevant Code Paths

- `llm_predict/composer.py`
  - `predict_ttft_ms(...)` sums per-layer prefill kernel predictions.
  - It accepts `seq_len` and `kv_len`, but the flash/KV part is too coarse for
    cached-prefix regimes.

- `llm_predict/serving.py`
  - `predict_serving(...)` computes TTFT from `composer.predict_ttft_ms(...)`,
    then applies framework correction and TTFT queue factors.
  - Optional prefix-cache contention correction exists, but it is not applied to
    the current rows.

- `llm_predict/cache_aware.py`
  - Multi-turn rows with `perTurn` can derive `new_prefill_tokens`,
    `cached_context_tokens`, and `cache_hit_rate`.
  - Single-turn real trace rows do not have `perTurn`, so
    `coding-singleturn` currently falls back to full-prefill behavior.

- `llm_predict/export_serving_predictions.py`
  - Cache-aware prediction is used for prefix-cache affected rows only when
    `perTurn` or a usable prefix-cache prior is present.
  - Otherwise the exporter falls back to regular `predict_serving(...)`.

- `inference-benchmark/src/benchmark/runner.py`
  - Prediction metadata records prefix/chunked state, but engine cache telemetry
    is marked unavailable.

- `inference-benchmark/scripts/run_one_bench.sh`
  - vLLM is launched with `--enable-prefix-caching` and
    `--enable-chunked-prefill`.

## Fix Direction

The local and CI rebuild path should produce a physics-style serving prediction,
not a calibration-first prediction.

1. Split TTFT into explicit full-prefill and cached-prefill regimes.
2. Add an explicit cached-prefix cost term:
   - new suffix GEMM/GEMV work,
   - attention over `total_context_tokens`,
   - cached KV read/load bandwidth,
   - new KV write bandwidth.
3. For rows without `perTurn`, derive or record a prefix-cache prior from the
   real trace prompts instead of assuming `new_prefill_tokens ==
   total_context_tokens`.
4. Keep generated/static calibration separate from the primary prediction path,
   or label it as an overlay rather than the model's core explanation.
5. In dashboard validation, flag `prefix_cache_affected &&
   cache_aware_applied=false` as unsupported or low-confidence until a cache
   prior exists.
6. Add regression tests around the `coding-singleturn` case so exported
   predictions cannot silently fall back to full-prefill behavior again.

## Caveat

The current dataset does not expose direct engine-level cache hit/residency
telemetry. The diagnosis is strongly supported by metadata, per-row prediction
fields, and measured-vs-predicted TTFT gaps, but a final fix should either add
telemetry to the benchmark runner or compute a reproducible trace-level prefix
prior from the prompt dataset.
