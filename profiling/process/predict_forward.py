"""CLI: forward-predict TTFT / TPOT / E2EL for a client's ISL:OSL distribution on a chosen
hardware config — WITHOUT a ground-truth benchmark run. Thin wrapper over simulator.forward.

Usage:
  python -m profiling.process.predict_forward \
      --gpu A100 --tp 4 --engine vllm --model Llama-3.1-70B \
      --concurrency 40 --isl-osl dist.json [--shared-prefix 0] [--json]

--isl-osl is the client's trace as either:
  * a JSON file: a list of [isl, osl] pairs, e.g. [[1800, 210], [2400, 64], ...]; or
  * a CSV/whitespace file with two columns isl, osl per line (a header row is skipped).
Each (isl, osl) is one single-turn request; the cohort reflects the whole distribution.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so the package imports resolve

from simulator.forward import predict_forward  # noqa: E402


def _load_samples(path: str) -> list[tuple[float, float]]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        return [(float(a), float(b)) for a, b in data]
    out: list[tuple[float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            out.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue  # skip a header / malformed row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Forward-predict TTFT/TPOT/E2EL for an ISL:OSL distribution on given hardware.")
    ap.add_argument("--gpu", required=True, help="GPU name (configs/gpus/<gpu>.json), e.g. A100, H100, RTX3090")
    ap.add_argument("--model", required=True, help="model name (configs/models/<model>.json)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel degree (default 1)")
    ap.add_argument("--engine", default="vllm", help="inference engine: vllm | sglang (default vllm)")
    ap.add_argument("--concurrency", type=float, required=True, help="concurrent sessions")
    ap.add_argument("--isl-osl", required=True, help="JSON list of [isl,osl] or a 2-column CSV file")
    ap.add_argument("--shared-prefix", type=float, default=0.0, help="cross-session APC prefix tokens (default 0)")
    ap.add_argument("--json", action="store_true", help="emit a JSON object instead of a table")
    args = ap.parse_args(argv)

    samples = _load_samples(args.isl_osl)
    if not samples:
        print(f"error: no (isl, osl) samples parsed from {args.isl_osl}", file=sys.stderr)
        return 2

    res = predict_forward(
        gpu=args.gpu, model=args.model, tp=args.tp, engine=args.engine,
        concurrency=args.concurrency, isl_osl_samples=samples,
        shared_prefix_tokens=args.shared_prefix,
    )

    if args.json:
        d = {k: v for k, v in dataclasses.asdict(res).items() if k != "per_turn"}
        d["n_samples"] = len(samples)
        print(json.dumps(d, indent=2))
        return 0

    print(f"{args.gpu} tp{args.tp} {args.engine} / {args.model}  @ concurrency {args.concurrency:g}")
    print(f"  workload : {len(samples)} (isl,osl) samples; median isl={res.isl:.0f}  osl={res.osl:.0f}")
    print(f"  TTFT     : {res.ttft_ms:9.1f} ms")
    print(f"  TPOT     : {res.tpot_ms:9.2f} ms/token")
    print(f"  E2EL     : {res.e2el_ms:9.1f} ms")
    status = res.calibration_status.upper()
    print(f"  CONFIDENCE: {status}" + (f"  ({res.calibration_detail})" if res.calibration_detail else ""))
    if res.calibration_status == "extrapolated":
        print("  NOTE: no measured calibration for this hardware — analytic roofline first-cut "
              "(decode/ceiling/utils inherited). Treat as a lower-bound estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
