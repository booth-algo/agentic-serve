# simulator_v2/kv_wall/saturated_ceiling.py

import json
from dataclasses import dataclass
from pathlib import Path

from simulator_v2.core.mode import Mode, mode


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class SaturatedCeiling:
    """The measured TPOT ceiling: a few (output_tokens, plateau_ms) points.

    When the KV cache is badly overloaded a request's TPOT stops climbing and
    flattens at a plateau. Each anchor is the median measured plateau TPOT (ms)
    for benchmark turns in that overloaded state, bucketed by output length and
    sorted by output. Shorter outputs plateau HIGHER (the recompute cost is spread
    over fewer generated tokens). How "overloaded" was defined when the anchors
    were picked is recorded in the source artifact; this class just holds the
    resulting points and interpolates between them."""
    anchors: tuple[tuple[float, float], ...]

    def ceiling_ms(self, output: float) -> float:
        """Plateau TPOT (ms) for a turn that generates `output` tokens.

        Linear interpolation between the measured anchors; past the measured
        output range it clamps to the nearest anchor (the plateau only falls as
        output grows)."""
        out = max(1.0, float(output))
        a = self.anchors
        if out <= a[0][0]:
            return a[0][1]
        if out >= a[-1][0]:
            return a[-1][1]
        for (o0, p0), (o1, p1) in zip(a, a[1:]):
            if o0 <= out <= o1:
                return p0 + (out - o0) / (o1 - o0) * (p1 - p0)
        return a[-1][1]


@mode(Mode.BACKTEST)
def load_saturated_ceiling(path: Path) -> SaturatedCeiling:
    """Read a ceiling artifact ({"anchors": [{output_tokens, plateau_ms}, ...]})
    into a SaturatedCeiling."""
    data = json.loads(path.read_text())
    anchors = tuple(sorted(
        (float(a["output_tokens"]), float(a["plateau_ms"])) for a in data["anchors"]
    ))
    if not anchors:
        raise RuntimeError(f"no saturated-ceiling anchors in {path}")
    return SaturatedCeiling(anchors=anchors)
