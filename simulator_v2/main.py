# simulator_v2/main.py

import argparse

from simulator_v2.engine import backtest, forward
from simulator_v2.core.mode import Mode, mode


@mode(Mode.BACKTEST)
def run_backtest(args: argparse.Namespace) -> None:
    """Proxy to the backtest predictor."""
    backtest.run(args)


@mode(Mode.FORWARD)
def run_forward(args: argparse.Namespace) -> None:
    """Proxy to the forward predictor."""
    forward.run(args)


MODES = {
    "backtest": run_backtest,
    "forward": run_forward,
}


@mode(Mode.SHARED)
def main() -> None:
    parser = argparse.ArgumentParser(description="simulator_v2")
    parser.add_argument("--mode", choices=MODES.keys(), required=True)
    parser.add_argument(
        "--gpu-config",
        default="simulator_v2/configs/gpu_configs/h100.yaml",
        help="GPU YAML config",
    )
    parser.add_argument(
        "--model-config",
        default="simulator_v2/configs/model_configs/llama3.1-8b.yaml",
        help="Model YAML config",
    )
    parser.add_argument(
        "--benchmark",
        help="Backtest mode - path of inference benchmark's JSON file",
    )
    parser.add_argument(
        "--distribution",
        help="Forward mode - workload ISL:OSL distribution file (includes concurrency)",
    )
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
