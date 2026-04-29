"""Export llm_predict_2 validation rows for the dashboard predictor tab.

The dashboard view is intentionally scoped to current paper-relevant rows:
high-concurrency serving, multi-turn/cache-affected workloads, and a small
canonical single-turn/stress comparator set. Legacy ShareGPT variants
(`chat-short`, `chat-medium`, `chat-long`) are excluded.
"""

import argparse
import json
from pathlib import Path

from .cache_aware import predict_multiturn_from_per_turn
from .configs.model_configs import MODEL_CONFIGS, get_model
from .composer import Composer
from .framework_corrections import (
    get_calibration_status,
    get_correction_note,
    get_correction_params,
    prefix_cache_prior,
    ttft_validation_scope,
)
from .serving import predict_serving
from .validate import BENCH_DATA, _HW_MAP, _actual_isl_osl, _resolve_model


DEFAULT_GPUS = ["H100", "A100", "RTX3090", "RTX2080Ti"]
DEFAULT_PROFILES = [
    "chat-singleturn",
    "coding-agent",
    "prefill-heavy",
    "decode-heavy",
    "random-1k",
    "chat-multiturn-short",
    "chat-multiturn-medium",
    "chat-multiturn-long",
    "swebench-multiturn-short",
    "swebench-multiturn-medium",
    "swebench-multiturn-long",
    "terminalbench-multiturn-short",
    "terminalbench-multiturn-medium",
    "terminalbench-multiturn-long",
    "osworld-multiturn-short",
    "osworld-multiturn-medium",
    "osworld-multiturn-long",
]
DEFAULT_CONCURRENCIES = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]
LEGACY_EXCLUDED_PROFILES = {"chat-short", "chat-medium", "chat-long"}
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "inference-benchmark"
    / "dashboard"
    / "public"
    / "serving-predictions.json"
)


def _prediction_row(entry: dict, composer: Composer, gpu: str) -> dict | None:
    cfg_block = entry.get("config", {})
    model_key = _resolve_model(cfg_block.get("model", ""))
    if model_key is None or model_key not in MODEL_CONFIGS:
        return None

    summary = entry.get("summary", {})
    isl, osl = _actual_isl_osl(summary)
    cfg = get_model(model_key)
    backend = cfg_block.get("backend", "")
    backend_version = entry.get("engineVersion")
    profile = cfg_block.get("profile", "")
    concurrency = int(cfg_block.get("concurrency", 1))

    validation_scope = ttft_validation_scope(profile, cfg_block.get("mode"))
    pred = None
    if validation_scope == "prefix_cache_affected" and entry.get("perTurn"):
        pred = predict_multiturn_from_per_turn(
            composer, cfg, gpu, entry.get("perTurn"), concurrency,
            backend=backend,
            backend_version=backend_version,
            model_key=model_key,
            profile=profile,
        )
    elif validation_scope == "prefix_cache_affected":
        prior_new_tokens, prior_hit_rate, prior_applied = prefix_cache_prior(
            gpu, backend, backend_version, model_key, profile, isl
        )
        if prior_applied:
            pred = predict_serving(
                composer, cfg, gpu, isl, osl, concurrency,
                backend=backend,
                backend_version=backend_version,
                model_key=model_key,
                profile=profile,
                total_context_tokens=isl,
                new_prefill_tokens=prior_new_tokens,
                cache_hit_rate=prior_hit_rate,
            )
    if pred is None:
        pred = predict_serving(
            composer, cfg, gpu, isl, osl, concurrency,
            backend=backend,
            backend_version=backend_version,
            model_key=model_key,
            profile=profile,
        )

    row: dict = {
        "model": model_key,
        "backend": backend,
        "backend_version": backend_version,
        "profile": profile,
        "mode": cfg_block.get("mode"),
        "concurrency": concurrency,
        "isl": isl,
        "osl": osl,
        "calibration_status": pred.calibration_status,
        "calibration_lookup_status": get_calibration_status(
            gpu, backend, backend_version, model_key
        ),
        "ttft_validation_scope": validation_scope,
        "ttft_kernel_ms": round(pred.ttft_kernel_ms, 2),
        "ttft_base_ms": round(pred.ttft_base_ms, 2),
        "ttft_queue_factor": round(pred.ttft_queue_factor, 3),
        "ttft_correction_applied": pred.ttft_correction_applied,
        "ttft_queue_applied": pred.ttft_queue_applied,
        "decode_correction_factor": round(pred.decode_correction_factor, 3),
        "decode_correction_applied": pred.decode_correction_applied,
        "moe_decode_correction_applied": pred.moe_decode_correction_applied,
        "total_context_tokens": pred.total_context_tokens,
        "new_prefill_tokens": pred.new_prefill_tokens,
        "cached_context_tokens": pred.cached_context_tokens,
        "cache_hit_rate": round(pred.cache_hit_rate, 4),
        "cache_aware_applied": pred.cache_aware_applied,
        "prefix_cache_ttft_factor": round(pred.prefix_cache_ttft_factor, 3),
        "prefix_cache_decode_factor": round(pred.prefix_cache_decode_factor, 3),
        "prefix_cache_contention_applied": pred.prefix_cache_contention_applied,
    }
    corr_params = get_correction_params(gpu, backend, backend_version, model_key)
    if corr_params:
        row["ttft_correction_alpha"] = corr_params[0]
        row["ttft_correction_beta_ms"] = corr_params[1]
    corr_note = get_correction_note(gpu, backend, backend_version, model_key)
    if corr_note:
        row["ttft_correction_note"] = corr_note

    measured_ttft = summary.get("median_ttft_ms")
    measured_tpot = summary.get("median_tpot_ms")
    measured_itl = summary.get("median_itl_ms")
    measured_e2el = summary.get("median_e2el_ms")
    if measured_ttft and measured_ttft > 0:
        row["ttft_pred"] = round(pred.ttft_ms, 2)
        row["ttft_meas"] = round(measured_ttft, 2)
        row["ttft_err"] = round(abs(pred.ttft_ms - measured_ttft) / measured_ttft * 100, 1)
    if measured_tpot and measured_tpot > 0:
        row["tpot_pred"] = round(pred.tpot_ms, 2)
        row["tpot_meas"] = round(measured_tpot, 2)
        row["tpot_err"] = round(abs(pred.tpot_ms - measured_tpot) / measured_tpot * 100, 1)
    if measured_itl and measured_itl > 0:
        row["itl_meas"] = round(measured_itl, 2)
    if measured_e2el and measured_e2el > 0:
        row["e2el_pred"] = round(pred.e2el_ms, 2)
        row["e2el_meas"] = round(measured_e2el, 2)
        row["e2el_err"] = round(abs(pred.e2el_ms - measured_e2el) / measured_e2el * 100, 1)
    return {key: value for key, value in row.items() if value is not None}


def _dashboard_row(row: dict) -> dict:
    out = {
        "model": row["model"],
        "backend": row.get("backend", ""),
        "profile": row["profile"],
        "concurrency": row["concurrency"],
        "isl": row["isl"],
        "osl": row["osl"],
        "calibration_status": row.get("calibration_status"),
        "ttft_validation_scope": row.get("ttft_validation_scope"),
        "ttft_kernel_ms": row.get("ttft_kernel_ms"),
        "ttft_base_ms": row.get("ttft_base_ms"),
        "ttft_queue_factor": row.get("ttft_queue_factor"),
        "decode_correction_factor": row.get("decode_correction_factor"),
        "itl_meas": row.get("itl_meas"),
        "total_context_tokens": row.get("total_context_tokens"),
        "new_prefill_tokens": row.get("new_prefill_tokens"),
        "cached_context_tokens": row.get("cached_context_tokens"),
        "cache_hit_rate": row.get("cache_hit_rate"),
        "cache_aware_applied": row.get("cache_aware_applied"),
        "prefix_cache_ttft_factor": row.get("prefix_cache_ttft_factor"),
        "prefix_cache_decode_factor": row.get("prefix_cache_decode_factor"),
        "prefix_cache_contention_applied": row.get("prefix_cache_contention_applied"),
    }
    for metric in ("ttft", "tpot", "e2el"):
        out[f"{metric}_pred"] = row.get(f"{metric}_pred")
        out[f"{metric}_meas"] = row.get(f"{metric}_meas")
        out[f"{metric}_err"] = row.get(f"{metric}_err")
    return {key: value for key, value in out.items() if value is not None}


def export_serving_predictions(output: Path = DEFAULT_OUTPUT,
                               gpus: list[str] | None = None,
                               profiles: list[str] | None = None,
                               concurrencies: list[int] | None = None) -> dict[str, list[dict]]:
    gpus = gpus or DEFAULT_GPUS
    profiles = profiles or DEFAULT_PROFILES
    concurrencies = concurrencies or DEFAULT_CONCURRENCIES
    gpu_set = set(gpus)
    profile_set = set(profiles) - LEGACY_EXCLUDED_PROFILES
    concurrency_set = set(concurrencies)

    payload: dict[str, list[dict]] = {}
    composers: dict[str, Composer] = {}
    with open(BENCH_DATA) as f:
        data = json.load(f)

    for gpu in gpus:
        payload[gpu] = []

    for entry in data:
        cfg_block = entry.get("config", {})
        profile = cfg_block.get("profile")
        if profile not in profile_set:
            continue
        concurrency = int(cfg_block.get("concurrency", 1))
        if concurrency not in concurrency_set:
            continue
        gpu = _HW_MAP.get(entry.get("hardware", ""), entry.get("hardware", ""))
        if gpu not in gpu_set:
            continue
        composer = composers.setdefault(gpu, Composer(gpu))
        row = _prediction_row(entry, composer, gpu)
        if row is not None:
            payload[gpu].append(_dashboard_row(row))

    for gpu_rows in payload.values():
        gpu_rows.sort(
            key=lambda row: (
                row["profile"],
                row["concurrency"],
                row["model"],
                row.get("backend", ""),
                row["isl"],
                row["osl"],
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export serving prediction validation rows for dashboard."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, action="append",
                        help="Concurrency to export. May be repeated. Defaults to canonical sweep levels.")
    args = parser.parse_args()

    payload = export_serving_predictions(
        output=args.output,
        concurrencies=args.concurrency,
    )
    counts = ", ".join(f"{gpu}: {len(rows)}" for gpu, rows in payload.items())
    print(f"Wrote {args.output} ({counts})")


if __name__ == "__main__":
    main()
