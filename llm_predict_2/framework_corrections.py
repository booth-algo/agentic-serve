"""Framework wall-clock correction for serving predictions.

The kernel predictor outputs raw kernel execution time. Real serving
frameworks (vLLM, SGLang) add overhead from scheduling, KV cache
allocation, CUDA graph setup, and per-kernel launch overhead accumulating
across ~224 kernel calls per forward pass.

TTFT correction model: corrected_ttft = alpha * kernel_ttft + beta_ms

Fitted from C=1 dense-model benchmark data (2026-04-29). Queue and
decode factors are empirical serving-system corrections fitted from
dense single-turn sweeps; they intentionally live outside the kernel
composer.
"""

from dataclasses import dataclass

from .serving_calibration import find_calibration


@dataclass(frozen=True)
class FrameworkCorrection:
    alpha: float
    beta_ms: float


@dataclass(frozen=True)
class CorrectionParams:
    ttft: FrameworkCorrection
    notes: str = ""


_BACKEND_DEFAULTS: dict[str, FrameworkCorrection] = {
    "vllm":   FrameworkCorrection(alpha=1.3, beta_ms=18.0),
    "sglang": FrameworkCorrection(alpha=1.3, beta_ms=6.0),
}

_CORRECTIONS: dict[tuple[str, str], CorrectionParams] = {
    # H100 has enough C=1 dense single-turn coverage across short and long ISL.
    ("H100", "vllm"): CorrectionParams(
        FrameworkCorrection(alpha=1.35, beta_ms=17.15),
        "fit on H100 vLLM dense C=1 single-turn profiles",
    ),
    ("H100", "sglang"): CorrectionParams(
        FrameworkCorrection(alpha=1.29, beta_ms=5.78),
        "fit on H100 SGLang dense C=1 single-turn profiles",
    ),

    # A100 C=1 rows are short-ISL only, so the fit is floor-dominated.
    ("A100", "vllm"): CorrectionParams(
        FrameworkCorrection(alpha=0.10, beta_ms=33.12),
        "short-ISL A100 vLLM fit; refresh once long-prefill A100 rows exist",
    ),

    # Consumer RTX vLLM runs are faster than standalone nn.Linear ncu timings.
    ("RTX3090", "vllm"): CorrectionParams(
        FrameworkCorrection(alpha=0.22, beta_ms=26.43),
        "RTX3090 vLLM fit; standalone GEMM sweep over-predicts",
    ),
    ("RTX2080Ti", "vllm"): CorrectionParams(
        FrameworkCorrection(alpha=1.00, beta_ms=15.55),
        "RTX2080Ti vLLM beta-only fit from two C=1 rows",
    ),
}

_TTFT_QUEUE_FACTORS: dict[tuple[str, str], dict[int, float]] = {
    ("H100", "vllm"): {
        1: 1.00, 10: 1.04, 20: 1.23, 40: 1.51, 80: 1.77,
        120: 8.31, 160: 11.08, 200: 9.15, 256: 11.22, 320: 13.15,
    },
    ("H100", "sglang"): {
        1: 1.00, 10: 1.36, 20: 1.37, 40: 1.51, 80: 1.87,
        120: 12.91, 160: 16.96, 200: 17.47, 256: 26.55, 320: 41.97,
    },
    ("A100", "vllm"): {
        1: 1.00, 10: 1.31, 20: 1.49, 40: 2.15,
        80: 7.51, 160: 7.25, 200: 8.74, 256: 9.61,
    },
    ("RTX3090", "vllm"): {
        1: 1.00, 10: 1.99, 20: 2.40, 40: 4.50, 80: 10.36,
    },
    ("RTX2080Ti", "vllm"): {
        1: 1.00, 10: 2.33, 20: 2.61, 40: 4.14,
    },
}

_DECODE_FACTORS: dict[tuple[str, str], dict[int, float]] = {
    ("H100", "vllm"): {
        1: 0.92, 10: 1.02, 20: 1.07, 40: 1.15, 80: 1.25,
        120: 1.29, 160: 1.29, 200: 1.63, 256: 1.45, 320: 1.45,
    },
    ("H100", "sglang"): {
        1: 0.93, 10: 1.04, 20: 1.09, 40: 1.23, 80: 1.35,
        120: 1.40, 160: 1.30, 200: 1.67, 256: 1.49, 320: 1.62,
    },
    ("A100", "vllm"): {
        1: 0.99, 10: 1.05, 20: 1.13, 40: 1.09,
        80: 1.34, 160: 1.46, 200: 1.26, 256: 1.19,
    },
    ("RTX3090", "vllm"): {
        1: 0.91, 10: 1.07, 20: 1.12, 40: 1.04, 80: 0.99,
    },
    ("RTX2080Ti", "vllm"): {
        1: 1.20, 10: 1.51, 20: 1.30, 40: 1.36,
    },
}

_PREFIX_CACHE_PROFILES = {
    "coding-agent",
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
}

_APPLY_CALIBRATION_STATUSES = {"high_confidence", "medium_confidence"}


def _artifact_for(gpu: str, backend: str,
                  backend_version: str | None = None,
                  model: str | None = None) -> dict | None:
    return find_calibration(gpu, backend, backend_version, model)


def _artifact_status_allows_apply(item: dict | None) -> bool:
    if item is None:
        return False
    return item.get("calibration_status") in _APPLY_CALIBRATION_STATUSES


def _params_for(gpu: str, backend: str,
                backend_version: str | None = None,
                model: str | None = None) -> CorrectionParams | None:
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is not None:
        if not _artifact_status_allows_apply(artifact):
            return None
        ttft = artifact.get("ttft_correction", {})
        alpha = ttft.get("alpha")
        beta = ttft.get("beta_ms")
        if alpha is None or beta is None:
            return None
        return CorrectionParams(
            FrameworkCorrection(alpha=float(alpha), beta_ms=float(beta)),
            artifact.get("notes", "generated serving calibration"),
        )

    params = _CORRECTIONS.get((gpu, backend))
    if params is not None:
        return params
    default = _BACKEND_DEFAULTS.get(backend)
    if default is None:
        return None
    return CorrectionParams(default, f"backend default for {backend}")


def framework_correction(gpu: str, backend: str,
                         kernel_ttft_ms: float,
                         backend_version: str | None = None,
                         model: str | None = None) -> tuple[float, bool]:
    """Apply framework wall-clock correction to raw kernel TTFT.

    Returns (corrected_ttft_ms, correction_applied).
    Unknown backends return the raw value with correction_applied=False.
    """
    params = _params_for(gpu, backend, backend_version, model)
    if params is None:
        return kernel_ttft_ms, False
    corr = params.ttft
    return corr.alpha * kernel_ttft_ms + corr.beta_ms, True


def get_correction_params(gpu: str, backend: str,
                          backend_version: str | None = None,
                          model: str | None = None) -> tuple[float, float] | None:
    """Return (alpha, beta_ms) for a GPU/backend, or None if unknown."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is not None:
        ttft = artifact.get("ttft_correction", {})
        alpha = ttft.get("alpha")
        beta = ttft.get("beta_ms")
        if alpha is not None and beta is not None:
            return float(alpha), float(beta)

    params = _params_for(gpu, backend, backend_version, model)
    if params is None:
        return None
    return params.ttft.alpha, params.ttft.beta_ms


def get_correction_note(gpu: str, backend: str,
                        backend_version: str | None = None,
                        model: str | None = None) -> str | None:
    """Return calibration provenance for dashboard/debug output."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is not None:
        return artifact.get("notes")

    params = _params_for(gpu, backend, backend_version, model)
    if params is None:
        return None
    return params.notes


def get_calibration_status(gpu: str, backend: str,
                           backend_version: str | None = None,
                           model: str | None = None) -> str:
    """Return generated calibration confidence/status for a backend pair."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is None:
        if _CORRECTIONS.get((gpu, backend)) or _BACKEND_DEFAULTS.get(backend):
            return "fallback_static"
        return "missing"
    return str(artifact.get("calibration_status", "missing"))


def _interpolate_factor(table: dict[int, float] | None,
                        concurrency: int) -> tuple[float, bool]:
    if not table:
        return 1.0, False
    points = sorted(table)
    if concurrency <= points[0]:
        return table[points[0]], True
    if concurrency >= points[-1]:
        return table[points[-1]], True
    for left, right in zip(points, points[1:]):
        if left <= concurrency <= right:
            span = right - left
            weight = (concurrency - left) / span if span else 0.0
            factor = table[left] + weight * (table[right] - table[left])
            return factor, True
    return 1.0, False


def _factor_table_from_block(raw: dict | None) -> dict[int, float] | None:
    if not isinstance(raw, dict):
        return None
    table: dict[int, float] = {}
    for key, value in raw.items():
        try:
            conc = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            factor = value.get("factor")
        else:
            factor = value
        if factor is not None:
            table[conc] = float(factor)
    return table or None


def _factor_table_from_artifact(item: dict | None, field: str) -> dict[int, float] | None:
    if not _artifact_status_allows_apply(item):
        return None
    raw = item.get(field, {}) if item else {}
    return _factor_table_from_block(raw)


def _profile_factor_table_from_artifact(item: dict | None, field: str,
                                        profile: str | None) -> dict[int, float] | None:
    if profile is None or not _artifact_status_allows_apply(item):
        return None
    raw = item.get(field, {}) if item else {}
    profile_block = raw.get(profile)
    if not isinstance(profile_block, dict):
        return None
    return _factor_table_from_block(profile_block)


def ttft_queue_factor(gpu: str, backend: str,
                      concurrency: int,
                      backend_version: str | None = None,
                      model: str | None = None,
                      profile: str | None = None) -> tuple[float, bool]:
    """Empirical TTFT scheduler/queueing factor for corrected base TTFT."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is not None and not _artifact_status_allows_apply(artifact):
        return 1.0, False
    profile_table = _profile_factor_table_from_artifact(
        artifact, "ttft_queue_factors_by_profile", profile
    )
    if profile_table:
        return _interpolate_factor(profile_table, concurrency)
    artifact_table = _factor_table_from_artifact(
        artifact, "ttft_queue_factors"
    )
    if artifact_table:
        return _interpolate_factor(artifact_table, concurrency)
    return _interpolate_factor(_TTFT_QUEUE_FACTORS.get((gpu, backend)),
                               concurrency)


def decode_correction_factor(gpu: str, backend: str,
                             concurrency: int,
                             backend_version: str | None = None,
                             model: str | None = None,
                             profile: str | None = None) -> tuple[float, bool]:
    """Empirical decode correction factor for TPOT/decode-total predictions."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if artifact is not None and not _artifact_status_allows_apply(artifact):
        return 1.0, False
    profile_table = _profile_factor_table_from_artifact(
        artifact, "decode_factors_by_profile", profile
    )
    if profile_table:
        return _interpolate_factor(profile_table, concurrency)
    artifact_table = _factor_table_from_artifact(
        artifact, "decode_factors"
    )
    if artifact_table:
        return _interpolate_factor(artifact_table, concurrency)
    return _interpolate_factor(_DECODE_FACTORS.get((gpu, backend)),
                               concurrency)


def _prefix_factor_table_from_artifact(item: dict | None,
                                       profile: str | None,
                                       factor_key: str) -> dict[int, float] | None:
    if profile is None or not _artifact_status_allows_apply(item):
        return None
    raw = item.get("prefix_cache_factors_by_profile", {}) if item else {}
    profile_block = raw.get(profile)
    if not isinstance(profile_block, dict):
        return None

    table: dict[int, float] = {}
    for key, value in profile_block.items():
        try:
            conc = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        factor = value.get(factor_key)
        if factor is not None:
            table[conc] = float(factor)
    return table or None


def prefix_cache_contention_factors(gpu: str, backend: str,
                                    backend_version: str | None,
                                    model: str | None,
                                    concurrency: int,
                                    profile: str | None = None) -> tuple[float, float, bool]:
    """Empirical factors for high-concurrency prefix-cache contention.

    Returns (ttft_factor, decode_factor, applied). These are separate from
    the cache-aware token accounting: they model queueing/cache residency
    effects visible in multi-turn benchmark rows but not exposed directly by
    the current benchmark schema.
    """
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if not _artifact_status_allows_apply(artifact):
        return 1.0, 1.0, False

    ttft_table = _prefix_factor_table_from_artifact(
        artifact, profile, "ttft_factor"
    )
    decode_table = _prefix_factor_table_from_artifact(
        artifact, profile, "decode_factor"
    )

    ttft_factor, ttft_applied = _interpolate_factor(ttft_table, concurrency)
    decode_factor, decode_applied = _interpolate_factor(decode_table, concurrency)
    return ttft_factor, decode_factor, ttft_applied or decode_applied


def prefix_cache_prior(gpu: str, backend: str,
                       backend_version: str | None,
                       model: str | None,
                       profile: str | None,
                       total_context_tokens: int) -> tuple[int, float, bool]:
    """Return a calibrated cache prior for prefix-cache rows without perTurn."""
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if profile is None or not _artifact_status_allows_apply(artifact):
        return total_context_tokens, 0.0, False
    raw = artifact.get("prefix_cache_priors_by_profile", {}) if artifact else {}
    profile_block = raw.get(profile)
    if not isinstance(profile_block, dict):
        return total_context_tokens, 0.0, False
    new_tokens = profile_block.get("new_prefill_tokens")
    if new_tokens is None:
        new_fraction = profile_block.get("new_prefill_fraction")
        if new_fraction is not None:
            new_tokens = round(total_context_tokens * float(new_fraction))
    if new_tokens is None:
        return total_context_tokens, 0.0, False
    prefill_tokens = max(1, min(int(round(float(new_tokens))), total_context_tokens))
    cache_hit_rate = max(0.0, (total_context_tokens - prefill_tokens) / total_context_tokens)
    return prefill_tokens, cache_hit_rate, True


def moe_decode_correction_factor(gpu: str, backend: str,
                                 backend_version: str | None,
                                 model: str | None,
                                 concurrency: int,
                                 profile: str | None = None) -> tuple[float, bool]:
    """Empirical total decode correction for fused MoE kernels."""
    if model is None:
        return 1.0, False
    artifact = _artifact_for(gpu, backend, backend_version, model)
    if not _artifact_status_allows_apply(artifact):
        return 1.0, False
    if profile is not None:
        by_profile = artifact.get("moe_decode_factors_by_profile", {}) if artifact else {}
        model_profiles = by_profile.get(model)
        if isinstance(model_profiles, dict):
            profile_block = model_profiles.get(profile)
            if isinstance(profile_block, dict):
                table = _factor_table_from_block(profile_block)
                if table:
                    return _interpolate_factor(table, concurrency)

    raw = artifact.get("moe_decode_factors", {}) if artifact else {}
    model_block = raw.get(model)
    if not isinstance(model_block, dict):
        return 1.0, False
    table = _factor_table_from_block(model_block)
    if not table:
        table = {}
        for key, value in model_block.items():
            try:
                conc = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                factor = value.get("factor")
            else:
                factor = value
            if factor is not None:
                table[conc] = float(factor)
    if not table:
        return 1.0, False
    return _interpolate_factor(table, concurrency)


def ttft_validation_scope(profile: str, mode: str | None = None) -> str:
    """Label TTFT rows whose measured latency is affected by prefix cache."""
    if mode == "multi-turn" or profile in _PREFIX_CACHE_PROFILES:
        return "prefix_cache_affected"
    return "full_prefill"
