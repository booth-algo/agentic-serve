# simulator_v2/engine/forward.py
"""Forward predictor: ISL:OSL distribution + roofline hardware -> predictions (no ground truth)."""

import argparse
from pathlib import Path

from simulator_v2.core.types import Cell, TurnPrediction
from simulator_v2.engine.predict import predict
from simulator_v2.getters.hardware import load_roofline_hardware
from simulator_v2.getters.workload import load_distribution
from simulator_v2.core.mode import Mode, mode


@mode(Mode.FORWARD)
def _report(cell: Cell, predictions: list[TurnPrediction]) -> None:
    print(f"profile={cell.profile} concurrency={cell.concurrency} turns={len(cell.turns)}")
    for i, pred in enumerate(predictions):
        print(
            f"turn {i}: ttft={pred.ttft_ms:.2f}ms "
            f"tpot={pred.tpot_ms:.2f}ms e2el={pred.e2el_ms:.2f}ms"
        )


@mode(Mode.FORWARD)
def run(args: argparse.Namespace) -> None:
    """ISL:OSL distribution + roofline hardware -> predict -> report."""
    if not args.distribution:
        raise SystemExit("--distribution is required for forward mode")

    hw = load_roofline_hardware(args.gpu_config, args.model_config, tp=args.tp)
    cell = load_distribution(Path(args.distribution))
    predictions = predict(hw, cell.turns, cell.concurrency)
    _report(cell, predictions)
