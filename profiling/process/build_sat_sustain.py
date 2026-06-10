#!/usr/bin/env python3
"""Regenerate the SAT_SUSTAIN_LO/HI anchor measurements (audit-v2 items G1/G2).

``simulator/kernel_tpot.py`` gates the saturation weight with a smoothstep over the
turn's output tokens between ``SAT_SUSTAIN_LO = 9.0`` and ``SAT_SUSTAIN_HI = 24.0``.
Until 2026-06-10 those were unpinned analyst read-offs (gate-tuning commit
``41e35f5``): no builder regenerated them, no artifact recorded the population they
were read from, and the test pinned them only as duplicated literals. This builder
makes both anchors regenerable from ground truth and records the population
question EXPLICITLY.

POPULATION PRE-REGISTRATION (the G1 resolution): the predictor consumes TURN-MEDIAN
rows — ``build_simulator_rows.build_turns`` medians one (profile, conc, turn_index)
cell over its successful requests, and those rows are what ``predict_cell_tpot`` /
``predict_turn_tpot`` price (KernelTurnInput.output_tokens IS a turn median). So the
canonical population for any output-anchor read is the turn-median one. The audit's
finding, which this builder reproduces exactly, is that the two populations
DISAGREE:

  * per-request rows  (n=45450 saturated):  p5 output = 9.0   -> the production LO
  * turn-median rows  (n=301  saturated):   p5 output = 24.0  -> equals production HI

i.e. on the population the predictor actually consumes, the [9, 24] band collapses
to a step at 24 — LO=9 is only supported by the per-request population. Production
values DO NOT change this round (2026-06-10 parallel-defit byte-identity contract,
lane L5); the artifact records both readings so the disagreement is a committed,
test-pinned fact instead of a code comment.

The HI anchor (G2): the historical story is "min turn-median plateau output 22
(measured 21.5) + 2", where the +2 'structural offset' had no derivation. The
builder finds the offset is unnecessary as a story: p5 of the turn-median plateau
outputs is 24.0 EXACTLY, so HI is directly derivable as the p5 anchor on the
canonical population (the same quantile that produced LO on the per-request one).
Both readings are recorded.

Saturated/plateau criterion: measured tpot > 100 ms (the same PLATEAU_TPOT_MS the
gate metrics use). Quantiles: empirical lower quantile sorted[ceil(q*n)-1]
(``statistics.quantiles(n=20)`` reproduces the same p5 values on both populations).

Usage:
    python3 -m profiling.process.build_sat_sustain            # write the artifact
    python3 -m profiling.process.build_sat_sustain --dry-run  # print, don't write
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.process.build_simulator_rows import (  # noqa: E402
    BENCH_BASE, CONCURRENCIES, PROFILES, build_turns,
)

BENCH_DIR = "h100_Llama-3.1-8B_tp1_vllm"   # the H100 headline run the anchors were read from
PLATEAU_TPOT_MS = 100.0                     # saturated rows: measured tpot above this
OUT_JSON = REPO_ROOT / "profile_data/kernels/sat_sustain_H100_llama31_8b.json"

# Production values (simulator/kernel_tpot.py). Pinned here AND cross-checked by
# simulator/tests/test_kernel_tpot.py; byte-identity contract — do not retune.
PRODUCTION_LO = 9.0
PRODUCTION_HI = 24.0


def _q_lower(sorted_vals: list[float], frac: float) -> float:
    """Empirical lower quantile: sorted[ceil(frac*n) - 1]."""
    if not sorted_vals:
        raise ValueError("empty population")
    return sorted_vals[max(0, math.ceil(frac * len(sorted_vals)) - 1)]


def collect_populations(bench_root: Path) -> tuple[list[tuple[float, float]],
                                                   list[tuple[float, float]]]:
    """(per_request, turn_median) populations of (output_tokens, measured tpot_ms).

    per_request: every successful request row across the run (success-filtered,
    output clamped at 1 like the predictor's own ``max(1, output)``).
    turn_median: the ``build_turns`` rows — EXACTLY what feeds predict_cell_tpot.
    """
    per_req: list[tuple[float, float]] = []
    turn_med: list[tuple[float, float]] = []
    for prof in PROFILES:
        for conc in CONCURRENCIES:
            f = bench_root / f"{prof}_conc{conc}.json"
            if not f.exists():
                continue
            data = json.loads(f.read_text())
            for r in data.get("per_request") or []:
                if not r.get("success"):
                    continue
                o, t = r.get("output_tokens"), r.get("tpot_ms")
                if isinstance(o, (int, float)) and isinstance(t, (int, float)):
                    per_req.append((max(1.0, float(o)), float(t)))
            turns, _shared_prefix = build_turns(f)
            for tn in turns:
                turn_med.append((float(tn["output_tokens"]), float(tn["tpot_meas"])))
    return per_req, turn_med


def build(bench_root: Path) -> dict[str, Any]:
    per_req, turn_med = collect_populations(bench_root)
    sat_req = sorted(o for o, t in per_req if t > PLATEAU_TPOT_MS)
    sat_turn = sorted(o for o, t in turn_med if t > PLATEAU_TPOT_MS)
    if not sat_req or not sat_turn:
        raise SystemExit(f"no saturated rows under {bench_root} (tpot > {PLATEAU_TPOT_MS} ms)")
    return {
        "gpu": "H100",
        "model": "Llama-3.1-8B",
        "source": str(bench_root),
        "saturated_criterion": f"measured tpot_ms > {PLATEAU_TPOT_MS} (the plateau/saturated regime)",
        "quantile_convention": "empirical lower quantile sorted[ceil(q*n)-1] "
                               "(statistics.quantiles(n=20) p5 agrees on both populations)",
        "population_preregistered": "turn_median",
        "population_note": (
            "turn-median rows (build_simulator_rows.build_turns) are what "
            "predict_cell_tpot consumes — KernelTurnInput.output_tokens IS a turn "
            "median — so they are the canonical population for output anchors; "
            "per-request rows are reported because the production LO was read there."),
        "per_request": {
            "n_rows": len(per_req),
            "n_saturated": len(sat_req),
            "p1_output": _q_lower(sat_req, 0.01),
            "p5_output": _q_lower(sat_req, 0.05),
            "p10_output": _q_lower(sat_req, 0.10),
            "min_output": sat_req[0],
        },
        "turn_median": {
            "n_rows": len(turn_med),
            "n_saturated": len(sat_turn),
            "p1_output": _q_lower(sat_turn, 0.01),
            "p5_output": _q_lower(sat_turn, 0.05),
            "p10_output": _q_lower(sat_turn, 0.10),
            "min_plateau_output": sat_turn[0],
            "smallest_plateau_outputs": sat_turn[:8],
        },
        "production": {
            "SAT_SUSTAIN_LO": PRODUCTION_LO,
            "SAT_SUSTAIN_HI": PRODUCTION_HI,
            "lo_reading": "p5 of saturated PER-REQUEST outputs (9.0; n=45450)",
            "hi_readings": [
                "historical: min turn-median plateau output 21.5 (~22 tok) + 2 "
                "(the +2 was an underived hand margin)",
                "derived equivalent: p5 of saturated TURN-MEDIAN outputs = 24.0 "
                "exactly (same quantile as LO, on the canonical population)",
            ],
        },
        "_notes": (
            "audit-v2 G1/G2 regenerable artifact (2026-06-10, lane L5). The two "
            "populations disagree: per-request p5 = 9.0 (the production LO) vs "
            "turn-median p5 = 24.0 (= the production HI). On the pre-registered "
            "canonical population (turn-medians feed predict_cell_tpot) the [9, 24] "
            "smoothstep band has no measured support below 21.5 — LO = 9 is a "
            "per-request-population read. Production values unchanged this round "
            "(byte-identity contract); any retune must start from the turn-median "
            "numbers recorded here. Regenerate: "
            "python3 -m profiling.process.build_sat_sustain"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    bench_root = BENCH_BASE / BENCH_DIR
    if not bench_root.exists():
        raise SystemExit(f"bench root not mounted: {bench_root}")
    payload = build(bench_root)
    text = json.dumps(payload, indent=2) + "\n"
    if args.dry_run:
        print(text, end="")
        return
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(text)
    pr, tm = payload["per_request"], payload["turn_median"]
    print(f"per-request: n_sat={pr['n_saturated']} p5={pr['p5_output']}")
    print(f"turn-median: n_sat={tm['n_saturated']} p5={tm['p5_output']} "
          f"min_plateau={tm['min_plateau_output']}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
