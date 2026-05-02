"""Serving metadata helpers for analytical serving predictions.

The predictor intentionally stays analytical: calibration artifacts may
describe coverage and raw error, but they must not supply empirical
multipliers for TTFT, TPOT, MoE decode, or prefix-cache behavior.
"""

from .serving_calibration import find_calibration

_PREFIX_CACHE_PROFILES = {
    "coding-singleturn",
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


def get_calibration_status(gpu: str, backend: str,
                           backend_version: str | None = None,
                           model: str | None = None) -> str:
    """Return generated calibration confidence/status for a backend pair."""
    artifact = find_calibration(gpu, backend, backend_version, model)
    if artifact is None:
        return "missing"
    return str(artifact.get("calibration_status", "missing"))


def ttft_validation_scope(profile: str, mode: str | None = None) -> str:
    """Label TTFT rows whose measured latency is affected by prefix cache."""
    if mode == "multi-turn" or profile in _PREFIX_CACHE_PROFILES:
        return "prefix_cache_affected"
    return "full_prefill"
