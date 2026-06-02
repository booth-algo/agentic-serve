# configs/ — model · GPU · deployment configs + per-config data tracking

Modular home for the prediction stack's configuration. Replaces the hardcoded `CONFIGS` tables in
`profiling/process/build_simulator_rows.py` + `build_saturated_ceiling.py`, and supersedes the ad-hoc
`profiling/docs/*.yaml` as the per-config **data tracker**.

## Layout
```
configs/
  models/<model>.json        # n_params, kv_bytes_per_token, kv_heads, bytes_per_param, cache_block_size
  gpus/<gpu>.json             # peak_flops_per_s, peak_bw_bytes_per_s, util_flops, util_bw, scheduler_overhead_ms_per_step
  deployments/<name>.json     # one GPU×model×TP run: references a model + gpu, + tp/kv/bench/ground_truth + a `data` manifest
  loader.py                   # composes model+gpu+deployment -> a Deployment (RooflineParams + resolved artifact paths)
  coverage_report.py          # python3 -m configs.coverage_report -> the config × data-input → status matrix
```

A **deployment** references a `model` + `gpu` by name and adds the run-specific fields
(`tp`, `available_kv_blocks`, `bench_dir`, `backend`, `calibration_status`, `ground_truth`) plus a
`data` manifest. `loader.py` composes the gpu + model into a `RooflineParams` and resolves which measured
artifacts the deployment OWNS (decode grid, saturated ceiling) vs INHERITS (→ the in-code H100 default).

## The `data` manifest (per-input provenance + coverage)
Each logical predictor input (`decode_grid, kv_pool, saturated_ceiling, roofline_peak, util_flops, util_bw,
scheduler_overhead, cached_prefill_grid, fa3_grid, ground_truth`) gets an entry with a **status**:

| status | meaning |
|---|---|
| `measured` | produced from this config's own GPU runs / `/mnt/100g` data (has a `path`) |
| `derived` | computed from this config's measured inputs |
| `inherited` | borrowed from another config (`"from": "H100"`) — e.g. H100x2/A100 reusing the H100 cached-prefill grid |
| `placeholder` | a default-valued stand-in not yet validated for this config (e.g. A100 `util_flops`) |
| `missing` | required by the predictor but no source exists → coverage report flags it |

The loader uses an artifact's `path` only when its status is `measured`/`derived` (the config OWNS it);
otherwise it falls back to the in-code H100 default. So the manifest is both **documentation** and the
**wiring** — they can't drift.

## Consumers
- `build_simulator_rows.py`: `CONFIGS = all_deployments()`; per config swaps the decode grid + saturated
  ceiling and uses `cfg.roofline`. Add a GPU/TP → add a `deployments/*.json`, no code change.
- `build_saturated_ceiling.py`: generates a ceiling for every deployment that OWNS one (status measured/derived).
- `coverage_report.py`: prints the `config × input → status` matrix, per-config counts, and a drift check
  (every `measured` path must exist). Run it to see what each config has / inherits / is missing.

## Add a new config
1. `models/<model>.json` + `gpus/<gpu>.json` if new.
2. `deployments/<gpu>_<model>_tp<N>.json` referencing them, with the `data` manifest (mark owned artifacts
   `measured` with paths, others `inherited`/`placeholder`/`missing`).
3. `python3 -m configs.coverage_report` to verify; `python3 -m profiling.process.build_simulator_rows` to regenerate.
