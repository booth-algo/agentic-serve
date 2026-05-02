"""Validate predictions against benchmark data.json results."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from llm_predict.configs.model_configs import get_model, MODEL_CONFIGS
    from llm_predict.cache_aware import predict_multiturn_from_per_turn
    from llm_predict.composer import Composer
    from llm_predict.serving import predict_serving
    from llm_predict.framework_corrections import (
        get_calibration_status,
        ttft_validation_scope,
    )
else:
    from .configs.model_configs import get_model, MODEL_CONFIGS
    from .cache_aware import predict_multiturn_from_per_turn
    from .composer import Composer
    from .serving import predict_serving
    from .framework_corrections import (
        get_calibration_status,
        ttft_validation_scope,
    )

BENCH_DATA = Path(__file__).parent.parent / "inference-benchmark" / "dashboard" / "public" / "data.json"

_MODEL_PATH_MAP = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "gpt-oss-20b": "gpt-oss-20b",
    "gemma-2-9b-it": "Gemma-2-9B",
    "granite-3.0-8b-instruct": "Granite-3.0-8B",
    "Yi-1.5-34B-Chat": "Yi-1.5-34B",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen3.5-27B",
    "gpt-oss-120b": "gpt-oss-120b",
    "Mixtral-8x7B-Instruct": "Mixtral-8x7B",
}

_HW_MAP = {
    "A100-40GB": "A100",
    "3090": "RTX3090",
    "2080Ti": "RTX2080Ti",
    "H100": "H100",
}

_PROFILE_ALIASES: dict[str, str] = {}

def _actual_isl_osl(summary: dict) -> tuple[int, int]:
    """Derive actual avg ISL and OSL from benchmark summary token counts."""
    n = summary.get("successful_requests", 0)
    if n <= 0:
        return 200, 100
    isl = int(summary.get("total_input_tokens", 0) / n)
    osl = int(summary.get("total_output_tokens", 0) / n)
    return max(isl, 1), max(osl, 1)


def _resolve_model(bench_model: str) -> str | None:
    basename = bench_model.rsplit("/", 1)[-1]
    return _MODEL_PATH_MAP.get(basename)


def validate(gpu: str, profile: str = "chat-singleturn",
             concurrency: int = 1) -> list[dict]:
    profile = _PROFILE_ALIASES.get(profile, profile)
    if not BENCH_DATA.exists():
        raise FileNotFoundError(f"No benchmark data at {BENCH_DATA}")

    with open(BENCH_DATA) as f:
        data = json.load(f)

    composer = Composer(gpu)
    results = []

    for entry in data:
        cfg_block = entry.get("config", {})
        hw_raw = entry.get("hardware", "")
        hw_mapped = _HW_MAP.get(hw_raw, hw_raw)
        if hw_mapped != gpu:
            continue
        if cfg_block.get("profile") != profile:
            continue
        if cfg_block.get("concurrency") != concurrency:
            continue

        model_key = _resolve_model(cfg_block.get("model", ""))
        if model_key is None or model_key not in MODEL_CONFIGS:
            continue

        summary = entry.get("summary", {})
        isl, osl = _actual_isl_osl(summary)

        cfg = get_model(model_key)
        backend = cfg_block.get("backend", "")
        backend_version = entry.get("engineVersion")
        validation_scope = ttft_validation_scope(profile, cfg_block.get("mode"))
        pred = None
        missing_prefix_cache_features = False
        if validation_scope == "prefix_cache_affected" and entry.get("perTurn"):
            pred = predict_multiturn_from_per_turn(
                composer, cfg, gpu, entry.get("perTurn"), concurrency,
                backend=backend,
                backend_version=backend_version,
                model_key=model_key,
                profile=profile,
            )
        elif validation_scope == "prefix_cache_affected":
            missing_prefix_cache_features = True
        if pred is None:
            pred = predict_serving(
                composer, cfg, gpu, isl, osl, concurrency,
                backend=backend,
                backend_version=backend_version,
                model_key=model_key,
                profile=profile,
                cache_feature_source="missing" if missing_prefix_cache_features else None,
                cache_prediction_regime=(
                    "unknown_prefix_cache" if missing_prefix_cache_features else None
                ),
                ttft_prediction_supported=not missing_prefix_cache_features,
                unsupported_reason=(
                    "missing_prefix_cache_features"
                    if missing_prefix_cache_features
                    else None
                ),
            )

        measured_ttft = summary.get("median_ttft_ms")
        measured_tpot = summary.get("median_tpot_ms")
        measured_e2el = summary.get("median_e2el_ms")

        row: dict = {"model": model_key, "concurrency": concurrency,
                     "profile": profile, "backend": backend,
                     "backend_version": backend_version,
                     "mode": cfg_block.get("mode"),
                     "isl": isl, "osl": osl,
                     "calibration_status": pred.calibration_status,
                     "ttft_kernel_ms": round(pred.ttft_kernel_ms, 2),
                     "ttft_base_ms": round(pred.ttft_base_ms, 2),
                     "total_context_tokens": pred.total_context_tokens,
                     "new_prefill_tokens": pred.new_prefill_tokens,
                     "cached_context_tokens": pred.cached_context_tokens,
                     "cache_hit_rate": round(pred.cache_hit_rate, 4),
                     "cache_aware_applied": pred.cache_aware_applied,
                     "cache_feature_source": pred.cache_feature_source,
                     "cache_prediction_regime": pred.cache_prediction_regime,
                     "ttft_prediction_supported": pred.ttft_prediction_supported,
                     "unsupported_reason": pred.unsupported_reason,
                     "ttft_validation_scope": validation_scope}
        if pred.multiturn_prediction_mode:
            row["multiturn_prediction_mode"] = pred.multiturn_prediction_mode
            row["predicted_turn_count"] = pred.predicted_turn_count
            row["total_successful_turn_requests"] = pred.total_successful_turn_requests
            row["mean_predicted_turn_ttft_ms"] = round(
                pred.mean_predicted_turn_ttft_ms, 2
            )
            row["mean_predicted_turn_tpot_ms"] = round(
                pred.mean_predicted_turn_tpot_ms, 2
            )
            row["multiturn_turn_predictions"] = pred.multiturn_turn_predictions
        row["calibration_lookup_status"] = get_calibration_status(
            gpu, backend, backend_version, model_key
        )

        if measured_ttft and measured_ttft > 0 and pred.ttft_prediction_supported:
            row["ttft_pred"] = round(pred.ttft_ms, 2)
            row["ttft_meas"] = round(measured_ttft, 2)
            row["ttft_err_pct"] = round(abs(pred.ttft_ms - measured_ttft) / measured_ttft * 100, 1)
        if measured_tpot and measured_tpot > 0:
            row["tpot_pred"] = round(pred.tpot_ms, 2)
            row["tpot_meas"] = round(measured_tpot, 2)
            row["tpot_err_pct"] = round(abs(pred.tpot_ms - measured_tpot) / measured_tpot * 100, 1)
        if measured_e2el and measured_e2el > 0 and pred.ttft_prediction_supported:
            row["e2el_pred"] = round(pred.e2el_ms, 2)
            row["e2el_meas"] = round(measured_e2el, 2)
            row["e2el_err_pct"] = round(abs(pred.e2el_ms - measured_e2el) / measured_e2el * 100, 1)

        results.append(row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate llm_predict serving predictions against dashboard data."
    )
    parser.add_argument("gpu", nargs="?", default="A100",
                        help="GPU name, e.g. H100, A100, RTX3090")
    parser.add_argument("profile", nargs="?", default="chat-singleturn",
                        help="Benchmark profile to validate")
    parser.add_argument("concurrency", nargs="?", type=int, default=1,
                        help="Concurrency level")
    args = parser.parse_args()

    gpu = args.gpu
    profile = args.profile
    conc = args.concurrency

    rows = validate(gpu, profile, conc)
    if not rows:
        print(f"No matching entries for {gpu} / {profile} / C={conc}")
        sys.exit(0)

    header = ["model", "backend", "backend_version", "calibration_status",
              "ttft_validation_scope",
              "ttft_kernel_ms", "ttft_base_ms",
              "cache_aware_applied", "cache_hit_rate",
              "cache_feature_source", "cache_prediction_regime",
              "ttft_prediction_supported", "unsupported_reason",
              "new_prefill_tokens", "total_context_tokens",
              "multiturn_prediction_mode", "predicted_turn_count",
              "ttft_pred", "ttft_meas", "ttft_err_pct",
              "tpot_pred", "tpot_meas", "tpot_err_pct",
              "e2el_pred", "e2el_meas", "e2el_err_pct"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r.get(h, "")) for h in header))
