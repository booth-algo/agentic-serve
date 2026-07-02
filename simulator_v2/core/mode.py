# simulator_v2/core/mode.py

"""Mode tags (forward / backtest / shared) for code navigation.

A lightweight decorator that stamps `__sim_mode__` on a function or class so the
mode a piece of code serves is visible at its definition and greppable across the
tree (e.g. `rg "@mode\\(Mode.FORWARD\\)"`). Metadata only -- it never changes
behavior. The intended consumer is whoever (human or AI) is navigating the code
and needs to know, in a file that mixes modes, which symbols belong to which path.

  - FORWARD  : forward-prediction path (analytic; no measured run)
  - BACKTEST : backtest path (consumes measured artifacts)
  - SHARED   : used by both modes (mode-agnostic)
"""

from enum import Enum
from typing import Callable, TypeVar


class Mode(str, Enum):
    FORWARD = "forward"
    BACKTEST = "backtest"
    SHARED = "shared"


T = TypeVar("T")


def mode(m: Mode) -> Callable[[T], T]:
    """Tag a function or class with the sim mode it serves (metadata only)."""
    def deco(obj: T) -> T:
        setattr(obj, "__sim_mode__", m)
        return obj
    return deco


def mode_of(obj: object) -> "Mode | None":
    """The Mode tag on a function/class, or None if untagged."""
    return getattr(obj, "__sim_mode__", None)
