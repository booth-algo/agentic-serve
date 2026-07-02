# simulator_v2/engine/backtest.py
"""Backtest predictor: measured benchmark ground truth + hardware -> predictions scored by MAPE."""

import argparse
import statistics
from pathlib import Path

from simulator_v2.core.types import Cell, TurnPrediction
from simulator_v2.engine.predict import predict
from simulator_v2.getters.hardware import load_kernel_composition_hardware
from simulator_v2.getters.workload import load_benchmark
from simulator_v2.core.mode import Mode, mode


@mode(Mode.BACKTEST)
def _ape(pred: float, meas: float) -> float | None:
    if pred <= 0 or meas <= 0:
        return None
    return abs(pred - meas) / meas * 100.0


@mode(Mode.BACKTEST)
def mean_ape(
    predictions: list[TurnPrediction],
    ground_truth: list[TurnPrediction],
    field: str,
) -> float | None:
    apes = []
    for pred, gt in zip(predictions, ground_truth):
        ape = _ape(getattr(pred, field), getattr(gt, field))
        if ape is not None:
            apes.append(ape)
    return statistics.mean(apes) if apes else None


@mode(Mode.BACKTEST)
def _score(cell: Cell, predictions: list[TurnPrediction]) -> None:
    print(f"profile={cell.profile} concurrency={cell.concurrency} turns={len(cell.turns)}")
    if not cell.ground_truth:
        print("no ground truth to score against")
        return

    for field in ("tpot_ms", "ttft_ms", "e2el_ms"):
        value = mean_ape(predictions, cell.ground_truth, field)
        label = field.replace("_ms", "").upper()
        print(f"{label} MAPE: {value:.2f}%" if value is not None else f"{label} MAPE: n/a")


@mode(Mode.BACKTEST)
def run(args: argparse.Namespace) -> None:
    """Benchmark GT + hardware -> predict -> score."""
    if not args.benchmark:
        raise SystemExit("--benchmark is required for backtest mode")

    hw = load_kernel_composition_hardware(args.gpu_config, args.model_config, tp=args.tp)
    cell = load_benchmark(Path(args.benchmark))
    predictions = predict(hw, cell.turns, cell.concurrency, trajectories=cell.trajectories)
    _score(cell, predictions)
