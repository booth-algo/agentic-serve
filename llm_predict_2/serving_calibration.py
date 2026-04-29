"""Load generated serving calibration artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CALIBRATION_PATH = Path(__file__).parent / "data" / "serving_calibration.json"


@lru_cache(maxsize=1)
def load_serving_calibration() -> dict[str, Any]:
    if not CALIBRATION_PATH.exists():
        return {}
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def find_calibration(gpu: str, backend: str,
                     backend_version: str | None = None,
                     model: str | None = None) -> dict[str, Any] | None:
    payload = load_serving_calibration()
    calibrations = payload.get("calibrations", [])
    if not isinstance(calibrations, list):
        return None

    exact_model = None
    exact_generic = None
    versionless_model = None
    versionless_generic = None
    for item in calibrations:
        if item.get("gpu") != gpu or item.get("backend") != backend:
            continue
        item_model = item.get("model")
        model_matches = model is not None and item_model == model
        generic = item_model in (None, "")

        if backend_version and item.get("backend_version") == backend_version:
            if model_matches:
                exact_model = item
                break
            if generic and exact_generic is None:
                exact_generic = item
        elif model_matches and versionless_model is None:
            versionless_model = item
        elif generic and versionless_generic is None:
            versionless_generic = item
    return exact_model or exact_generic or versionless_model or versionless_generic


def clear_calibration_cache() -> None:
    load_serving_calibration.cache_clear()
