# simulator_v2/core/common.py
"""Shared imports for simulator_v2 modules.

Prefer explicit imports:
    from simulator_v2.core.common import Turn, Hardware

Star imports are supported for scripts/notebooks only:
    from simulator_v2.core.common import *
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from simulator_v2.core.types import (
    Cell,
    Cohort,
    GpuConfig,
    Hardware,
    ModelConfig,
    SchedulerSettings,
    Turn,
    TurnPrediction,
)

__all__ = [
    "Any",
    "Cell",
    "Cohort",
    "GpuConfig",
    "Hardware",
    "ModelConfig",
    "Path",
    "Protocol",
    "SchedulerSettings",
    "Turn",
    "TurnPrediction",
    "dataclass",
    "field",
    "runtime_checkable",
]
