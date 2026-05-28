"""Emit llm-d-inference-sim CLI config per (profile, hardware) from our predictor.

Downstream of the closed-form TPOT model.  For each profile in the per-request
sidecar, we fit four llm-d-inference-sim closed-form params using our roofline
predictor as the calibration source:

  prefill-overhead          intercept of TTFT(P_fresh) at c=1
  prefill-time-per-token    slope of TTFT(P_fresh) at c=1
  inter-token-latency       predicted TPOT at c=1 with profile-mean context
  time-factor-under-load    predicted TPOT(c=max) / TPOT(c=1)
  max-num-seqs              max concurrency observed in the sweep

These can be passed directly to ``llm-d-inference-sim`` via the
``--config-file`` flag so the mock server reproduces our predictor's
behaviour for downstream router / disaggregation research.

See: https://github.com/llm-d/llm-d-inference-sim
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from simulator.closed_form_tpot import (
    ClosedFormTpotPredictor,
    TurnInput,
)


DEFAULT_PER_REQUEST_JSON = Path(
    "profiling/results/benchmark_per_request_llama31_8b_h100_vllm.json"
)
DEFAULT_ROOFLINE_PARAMS = Path(
    "profiling/data/roofline_params_H100_llama31_8b.json"
)
DEFAULT_OUTPUT = Path(
    "profiling/results/llm_d_inference_sim_configs.json"
)
DEFAULT_HARDWARE_KEY = "H100_Llama-3.1-8B_tp1_vllm"


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _collect_profile_summaries(
    per_turn: dict,
) -> dict[str, dict]:
    """For each profile in the sidecar, summarize across all (c, turn) buckets.

    Returns per-profile aggregate stats needed for fitting:

      concurrencies     sorted unique c values seen
      mean_output       mean output_tokens across all buckets
      mean_new_prefill  mean new_prefill_tokens across all buckets
      mean_cached       mean cached_context_tokens across all buckets
      max_new_prefill   max new_prefill_tokens across all buckets (for slope anchor)
    """
    by_profile: dict[str, dict] = defaultdict(lambda: {
        "concurrencies": set(),
        "output_tokens": [],
        "new_prefill_tokens": [],
        "cached_context_tokens": [],
    })
    for bucket in per_turn.values():
        profile = bucket.get("profile")
        c = bucket.get("concurrency")
        if not profile or not isinstance(c, int):
            continue
        slot = by_profile[profile]
        slot["concurrencies"].add(c)
        for field in ("output_tokens", "new_prefill_tokens", "cached_context_tokens"):
            vals = bucket.get(field) or []
            if isinstance(vals, list):
                slot[field].extend(float(v) for v in vals)
    out: dict[str, dict] = {}
    for profile, slot in by_profile.items():
        if not slot["concurrencies"]:
            continue
        out[profile] = {
            "concurrencies": sorted(slot["concurrencies"]),
            "mean_output": _mean(slot["output_tokens"]),
            "mean_new_prefill": _mean(slot["new_prefill_tokens"]),
            "mean_cached": _mean(slot["cached_context_tokens"]),
            "max_new_prefill": (
                max(slot["new_prefill_tokens"]) if slot["new_prefill_tokens"] else 0.0
            ),
        }
    return out


def fit_profile_config(
    predictor: ClosedFormTpotPredictor,
    summary: dict,
    *,
    profile_name: str,
) -> dict:
    """Run the closed-form predictor at two anchor points and fit four params.

    - TTFT slope/intercept from c=1 with P_fresh ∈ {small, large}.
    - inter_token_latency from c=1, mean_output, mean_new_prefill.
    - time_factor_under_load from TPOT(c=max) / TPOT(c=1).
    """
    cs = summary["concurrencies"]
    c_min, c_max = cs[0], cs[-1]
    mean_output = max(1.0, summary["mean_output"])
    mean_new_prefill = max(1.0, summary["mean_new_prefill"])
    mean_cached = summary["mean_cached"]
    max_new_prefill = max(mean_new_prefill, summary["max_new_prefill"])

    # ITL: TPOT at c=1 with profile-mean context.
    base = predictor.predict(TurnInput(
        profile=profile_name, concurrency=1, turn_index=0,
        output_tokens=mean_output,
        new_prefill_tokens=mean_new_prefill,
        cached_context_tokens=mean_cached,
    ))
    # load factor: TPOT at c=max with same workload (per-request scaling
    # accounted for by closed-form decode roofline).
    peak = predictor.predict(TurnInput(
        profile=profile_name, concurrency=c_max, turn_index=0,
        output_tokens=mean_output,
        new_prefill_tokens=mean_new_prefill,
        cached_context_tokens=mean_cached,
    ))
    tf = peak.tpot_ms / max(base.tpot_ms, 1e-6)

    # Prefill slope: TTFT at c=1, two P_fresh anchors.
    small_pfresh = max(1.0, 0.1 * max_new_prefill)
    large_pfresh = max(small_pfresh + 1.0, max_new_prefill)
    small_ttft = predictor.predict(TurnInput(
        profile=profile_name, concurrency=1, turn_index=0,
        output_tokens=1.0,
        new_prefill_tokens=small_pfresh,
        cached_context_tokens=0.0,
    )).ttft_ms
    large_ttft = predictor.predict(TurnInput(
        profile=profile_name, concurrency=1, turn_index=0,
        output_tokens=1.0,
        new_prefill_tokens=large_pfresh,
        cached_context_tokens=0.0,
    )).ttft_ms
    slope = (large_ttft - small_ttft) / (large_pfresh - small_pfresh)
    intercept = max(0.0, small_ttft - slope * small_pfresh)

    return {
        "prefill-overhead": f"{intercept:.3f}ms",
        "prefill-time-per-token": f"{slope:.6f}ms",
        "inter-token-latency": f"{base.tpot_ms:.3f}ms",
        "time-factor-under-load": round(tf, 3),
        "max-num-seqs": int(c_max),
        # Diagnostic context (consumers can ignore):
        "_anchor_c_min": c_min,
        "_anchor_c_max": c_max,
        "_mean_output_tokens": round(mean_output, 1),
        "_mean_new_prefill_tokens": round(mean_new_prefill, 1),
        "_mean_cached_context_tokens": round(mean_cached, 1),
        "_predicted_tpot_at_c_min_ms": round(base.tpot_ms, 4),
        "_predicted_tpot_at_c_max_ms": round(peak.tpot_ms, 4),
        # KV-pressure diagnostics — flag profiles where the GPU's KV budget is
        # exceeded at c_max (wave_factor > 1 means TPOT scales nonlinearly).
        # llm-d-inference-sim itself does not model KV pressure, so the
        # `time-factor-under-load` we fit already bakes this in.
        "_effective_decode_batch_at_c_max": int(peak.effective_decode_batch),
        "_wave_factor_at_c_max": round(peak.wave_factor, 3),
    }


def emit(
    per_request_json: Path,
    roofline_params_json: Path,
    output_path: Path,
    hardware_key: str,
) -> None:
    payload = json.loads(per_request_json.read_text())
    per_turn = payload.get("per_turn") or {}
    predictor = ClosedFormTpotPredictor.from_json(roofline_params_json)
    summaries = _collect_profile_summaries(per_turn)
    if not summaries:
        raise SystemExit(f"no profiles found in {per_request_json}")
    out: dict = {hardware_key: {}}
    for profile, summary in sorted(summaries.items()):
        out[hardware_key][profile] = fit_profile_config(
            predictor, summary, profile_name=profile
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {output_path} with {len(summaries)} profile fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-request-json", type=Path, default=DEFAULT_PER_REQUEST_JSON,
        help="Sidecar produced by extract_benchmark_per_request.",
    )
    p.add_argument(
        "--roofline-params-json", type=Path, default=DEFAULT_ROOFLINE_PARAMS,
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write the llm-d-inference-sim configs JSON.",
    )
    p.add_argument(
        "--hardware-key", default=DEFAULT_HARDWARE_KEY,
        help="Top-level key in the output JSON; identifies the hardware+model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emit(
        per_request_json=args.per_request_json,
        roofline_params_json=args.roofline_params_json,
        output_path=args.output,
        hardware_key=args.hardware_key,
    )


if __name__ == "__main__":
    main()
