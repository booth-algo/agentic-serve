# llm_predict

Current cache-aware serving predictor for AgentServe-Bench.

This package owns the serving prediction path used for dashboard validation:

- kernel predictors under `kernels/`
- model/GPU configs under `configs/`
- serving composition in `composer.py` and `serving.py`
- cache-aware multi-turn handling in `cache_aware.py`
- calibration/export/validation scripts in `training/`, `validate.py`, and
  `export_serving_predictions.py`

The older LLMCompass-derived simulator and per-op/per-kernel training stack has
been removed from the repository. Historical predictor/profiling dashboard JSON
artifacts remain published under the dashboard data feed.
