"""Diagnostic: attribute per-turn TPOT amplification to specific vLLM mechanisms.

Per the diagnostic-first hybrid plan, we do NOT fit constants. We emit a per-turn
table with:
  - observed quantities (tpot_meas, engine_total_step_ms, engine_steps, …)
  - physical predictions from RooflineParams + workload (no fits):
      decode_only_step_ms  = (weights + running·ctx·kv_bytes) / (bw·util)
      prefill_intrusion_ms = (engine_total_prefill_tokens / engine_steps) · prefill_per_tok
      predicted_step_ms    = decode_only_step_ms + prefill_intrusion_ms
      predicted_tpot_ms    = predicted_step_ms                    (1 tok/session/step)
  - two residual ratios:
      step_residual = engine_total_step_ms / engine_steps / predicted_step_ms
      tpot_residual = tpot_meas / predicted_tpot_ms
  - a REGIME LABEL drawn from the runtime config + engine evidence:
      no_pressure
      kv_admission_throttle
      single_request_near_limit
      single_request_exceeds_limit
      swap_or_offload     (placeholder; never fires today)

Critical exploration finding (2026-05-27): engine traces show ZERO preemption
events across 2,276 scheduler steps. So the "re-prefill on preempt" story
cannot be the amplifier. The diagnostic exposes what IS.

Inputs:
  - inference-benchmark/dashboard/public/simulator-predictions.json
      (per-turn engine aggregates: engine_total_step_ms, engine_steps,
       engine_capacity_waiting_requests, engine_max_decode_batch,
       engine_total_prefill_tokens, tpot_meas, workload)

Outputs:
  - profiling/results/hybrid_tpot_diagnostic.csv
  - per-(profile, regime) median residual summary in stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.closed_form_tpot import RooflineParams  # noqa: E402


DEFAULT_SIM_PREDICTIONS = Path(
    "inference-benchmark/dashboard/public/simulator-predictions.json"
)
DEFAULT_OUTPUT = Path("profiling/results/hybrid_tpot_diagnostic.csv")

# Confirmed empirically from engine traces (see plan Step 1): vLLM v1 default.
MAX_NUM_BATCHED_TOKENS = 8192


@dataclass(frozen=True)
class TurnPrediction:
    profile: str
    concurrency: int
    turn_index: int

    # Observed
    tpot_meas_ms: float | None
    engine_total_step_ms: float | None
    engine_steps: int | None
    engine_max_decode_batch: float | None
    engine_capacity_waiting_requests: float | None
    engine_total_prefill_tokens: float | None
    engine_mixed_steps: float | None
    cached_context_tokens: float
    new_prefill_tokens: float
    output_tokens: float
    total_context_tokens: float

    # Predicted (physical, no fits)
    ctx_mid: float
    running: float
    per_session_blocks: int
    capacity_batch: int
    decode_only_step_ms: float
    chunked_prefill_intrusion_ms: float
    predicted_step_ms: float
    predicted_tpot_ms: float

    # Diagnostic columns
    engine_total_decode_slots: float | None
    decode_coverage: float | None   # slots / (steps × running) — fraction of running sessions that got a token per step

    # Residuals (None when missing input)
    observed_step_ms_per_step: float | None
    step_residual: float | None
    tpot_residual: float | None

    regime: str


def physical_decode_step_ms(running: float, ctx: float, p: RooflineParams) -> float:
    """Bandwidth-bound decode step time: read weights + KV per running session per step."""
    bw_eff = p.peak_bw_bytes_per_s * p.util_bw
    weights_bytes = p.n_params * p.bytes_per_param
    kv_bytes_step = running * ctx * p.kv_bytes_per_token
    return (weights_bytes + kv_bytes_step) / bw_eff * 1000.0  # ms


def physical_prefill_per_token_ms(p: RooflineParams) -> float:
    """Compute-bound prefill cost per token: 2·n_params FLOPs per token (one MAC per param)."""
    flops_per_token = 2.0 * p.n_params
    flops_eff = p.peak_flops_per_s * p.util_flops
    return flops_per_token / flops_eff * 1000.0  # ms


def label_regime(
    *,
    waiting_max: float,
    per_session_blocks: int,
    capacity_batch: int,
    running: float,
    concurrency: int,
    p: RooflineParams,
) -> str:
    if per_session_blocks > p.available_kv_blocks:
        return "single_request_exceeds_limit"
    if per_session_blocks >= 0.9 * p.available_kv_blocks:
        return "single_request_near_limit"
    if waiting_max >= 3.0:
        return "kv_admission_throttle"
    # No-pressure: running batch is at full concurrency, no waiting, ample capacity
    if running >= 0.95 * concurrency and waiting_max <= 1.0:
        return "no_pressure"
    # Mild pressure (running below c, no big wait queue)
    return "mild_pressure"


def diagnose_turn(
    profile: str,
    concurrency: int,
    turn: dict[str, Any],
    p: RooflineParams,
) -> TurnPrediction:
    cached = float(turn.get("cached_context_tokens") or 0.0)
    new_prefill = float(turn.get("new_prefill_tokens") or 0.0)
    output = float(turn.get("output_tokens") or 0.0)
    total_ctx = float(turn.get("total_context_tokens") or (cached + new_prefill))
    ctx_mid = cached + new_prefill + 0.5 * output

    block_size = max(1, p.cache_block_size)
    per_session_blocks = max(1, math.ceil(ctx_mid / block_size))
    capacity_batch = max(1, p.available_kv_blocks // per_session_blocks)

    running_field = turn.get("engine_max_decode_batch")
    running = float(running_field) if isinstance(running_field, (int, float)) else float(concurrency)

    waiting_field = turn.get("engine_capacity_waiting_requests")
    waiting = float(waiting_field) if isinstance(waiting_field, (int, float)) else 0.0

    decode_step_ms = physical_decode_step_ms(running, ctx_mid, p)
    prefill_per_tok = physical_prefill_per_token_ms(p)

    eng_steps_f = turn.get("engine_steps")
    eng_steps = int(eng_steps_f) if isinstance(eng_steps_f, (int, float)) and eng_steps_f > 0 else None
    eng_total_prefill = turn.get("engine_total_prefill_tokens")
    total_prefill = float(eng_total_prefill) if isinstance(eng_total_prefill, (int, float)) else 0.0
    if eng_steps is not None and eng_steps > 0:
        avg_prefill_tokens_per_step = total_prefill / eng_steps
    else:
        avg_prefill_tokens_per_step = 0.0
    prefill_intrusion_ms = avg_prefill_tokens_per_step * prefill_per_tok
    predicted_step_ms = decode_step_ms + prefill_intrusion_ms
    predicted_tpot_ms = predicted_step_ms  # 1 token/session/step assumption

    tpot_meas = turn.get("tpot_meas")
    eng_total_step_ms = turn.get("engine_total_step_ms")

    observed_step_ms_per_step = None
    if isinstance(eng_total_step_ms, (int, float)) and eng_steps is not None and eng_steps > 0:
        observed_step_ms_per_step = eng_total_step_ms / eng_steps

    step_residual = None
    if observed_step_ms_per_step is not None and predicted_step_ms > 0:
        step_residual = observed_step_ms_per_step / predicted_step_ms

    tpot_residual = None
    if isinstance(tpot_meas, (int, float)) and predicted_tpot_ms > 0:
        tpot_residual = float(tpot_meas) / predicted_tpot_ms

    decode_slots_field = turn.get("engine_total_decode_slots")
    decode_slots = float(decode_slots_field) if isinstance(decode_slots_field, (int, float)) else None
    decode_coverage = None
    if decode_slots is not None and eng_steps is not None and eng_steps > 0 and running > 0:
        decode_coverage = decode_slots / (eng_steps * running)

    regime = label_regime(
        waiting_max=waiting,
        per_session_blocks=per_session_blocks,
        capacity_batch=capacity_batch,
        running=running,
        concurrency=concurrency,
        p=p,
    )

    return TurnPrediction(
        profile=profile,
        concurrency=concurrency,
        turn_index=int(turn.get("turn_index", -1)),
        tpot_meas_ms=float(tpot_meas) if isinstance(tpot_meas, (int, float)) else None,
        engine_total_step_ms=float(eng_total_step_ms) if isinstance(eng_total_step_ms, (int, float)) else None,
        engine_steps=eng_steps,
        engine_max_decode_batch=running,
        engine_capacity_waiting_requests=waiting,
        engine_total_prefill_tokens=total_prefill,
        engine_mixed_steps=float(turn.get("engine_mixed_steps") or 0.0),
        cached_context_tokens=cached,
        new_prefill_tokens=new_prefill,
        output_tokens=output,
        total_context_tokens=total_ctx,
        ctx_mid=ctx_mid,
        running=running,
        per_session_blocks=per_session_blocks,
        capacity_batch=capacity_batch,
        decode_only_step_ms=decode_step_ms,
        chunked_prefill_intrusion_ms=prefill_intrusion_ms,
        predicted_step_ms=predicted_step_ms,
        predicted_tpot_ms=predicted_tpot_ms,
        engine_total_decode_slots=decode_slots,
        decode_coverage=decode_coverage,
        observed_step_ms_per_step=observed_step_ms_per_step,
        step_residual=step_residual,
        tpot_residual=tpot_residual,
        regime=regime,
    )


def load_sim_predictions(path: Path) -> list[TurnPrediction]:
    payload = json.loads(path.read_text())
    rows = payload.get("H100", [])
    params = RooflineParams()
    out: list[TurnPrediction] = []
    for row in rows:
        prof = row.get("profile")
        c = row.get("concurrency")
        turns = row.get("multiturn_turn_predictions") or []
        if not prof or c is None or not turns:
            continue
        for t in turns:
            out.append(diagnose_turn(prof, int(c), t, params))
    return out


def write_csv(rows: list[TurnPrediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile", "concurrency", "turn_index", "regime",
        # Observed
        "tpot_meas_ms",
        "engine_total_step_ms", "engine_steps", "observed_step_ms_per_step",
        "engine_max_decode_batch", "engine_capacity_waiting_requests",
        "engine_total_prefill_tokens", "engine_mixed_steps",
        # Workload
        "cached_context_tokens", "new_prefill_tokens", "output_tokens",
        "total_context_tokens", "ctx_mid",
        # Derived
        "running", "per_session_blocks", "capacity_batch",
        # Physical predictions
        "decode_only_step_ms", "chunked_prefill_intrusion_ms",
        "predicted_step_ms", "predicted_tpot_ms",
        # Diagnostic
        "engine_total_decode_slots", "decode_coverage",
        # Residuals
        "step_residual", "tpot_residual",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = {fn: getattr(r, fn) for fn in fieldnames}
            for k, v in list(d.items()):
                if isinstance(v, float) and not math.isnan(v) and not math.isinf(v):
                    d[k] = round(v, 4)
            w.writerow(d)


def summarize(rows: list[TurnPrediction]) -> str:
    """Per-(profile, regime) median residual. Stdout-friendly markdown table."""
    by_pr: dict[tuple[str, str], list[float]] = {}
    by_pr_step: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.tpot_residual is not None:
            by_pr.setdefault((r.profile, r.regime), []).append(r.tpot_residual)
        if r.step_residual is not None:
            by_pr_step.setdefault((r.profile, r.regime), []).append(r.step_residual)

    lines = ["", "## Per-(profile, regime) residuals", ""]
    lines.append("| profile | regime | n | tpot_residual median (p10–p90) | step_residual median (p10–p90) |")
    lines.append("|---|---|---:|---|---|")

    def pct(xs: list[float], p: float) -> float:
        xs_sorted = sorted(xs)
        i = max(0, min(len(xs_sorted) - 1, int(p * len(xs_sorted))))
        return xs_sorted[i]

    for key in sorted(by_pr):
        prof, reg = key
        amps = by_pr[key]
        steps = by_pr_step.get(key) or []
        amp_med = median(amps)
        amp_p10 = pct(amps, 0.10) if len(amps) >= 3 else min(amps)
        amp_p90 = pct(amps, 0.90) if len(amps) >= 3 else max(amps)
        if steps:
            step_med = median(steps)
            step_p10 = pct(steps, 0.10) if len(steps) >= 3 else min(steps)
            step_p90 = pct(steps, 0.90) if len(steps) >= 3 else max(steps)
            step_str = f"{step_med:.2f} ({step_p10:.2f}–{step_p90:.2f})"
        else:
            step_str = "—"
        lines.append(
            f"| {prof} | {reg} | {len(amps)} | "
            f"{amp_med:.2f} ({amp_p10:.2f}–{amp_p90:.2f}) | {step_str} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulator-predictions-json", type=Path, default=DEFAULT_SIM_PREDICTIONS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    rows = load_sim_predictions(args.simulator_predictions_json)
    write_csv(rows, args.output)
    print(f"wrote {args.output}  ({len(rows)} per-turn records)")
    print(summarize(rows))


if __name__ == "__main__":
    main()
