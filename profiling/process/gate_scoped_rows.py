#!/usr/bin/env python3
"""Scoped Llama-3.1-8B vLLM rebuild + gate metrics for the ramp-knee adoption workflow.

Tooling item 2 of ``profiling/docs/ramp_knee_adoption_plan.md`` (replaces the
thrice-rewritten /tmp harness). Rebuilds the per-turn predictions for ``--gpu-keys``
(default H100,A100,H100x2) EXACTLY like ``profiling.process.build_simulator_rows.main()``
does — same per-config decode-grid / saturated-ceiling swap, same ``build_row`` — with
OPTIONAL adoption-candidate overrides applied as MODULE-ATTRIBUTE patches only, never
source edits:

  * ``--floor-excess "H100=1.2,A100=0.8,H100x2=2.0"``  ->  ``kernel_tpot.decode_step_ms``
    wrapped with ``+D``, swapped per active config (D absent/0 -> the unwrapped original).

(The 2026-06-09 ``--p-lo`` / ``--p-hi-short`` knee-override flags were REMOVED with the
2026-06-10 ramp restructure: ``kernel_tpot`` no longer has tuned ramp-knee attributes —
the saturation onset/width are computed per cell by ``_overflow_weight``.)

Patch-binding facts (the decode wrapper is self-verified at startup by
``_verify_decode_patch_binds``):

  * ``kernel_tpot.predict_turn_tpot`` resolves ``decode_step_ms`` through the kernel_tpot
    MODULE GLOBALS at call time (``from simulator.kernel_step_cost import decode_step_ms``
    creates a kernel_tpot-global binding), so patching ``kernel_tpot.decode_step_ms``
    binds the wrapper for all TPOT pricing.
  * ``simulator/ttft_queue_sim.py`` imports ``decode_step_ms`` DIRECTLY from
    ``kernel_step_cost`` (its own module global, line 73) — the wrapper does NOT leak into
    the queue sim's mixed-step decode pricing (``_mixed_step``). Residual leak path:
    ``ttft_queue_sim._fallback_ttft`` (and ``predict_e2el_qsim`` when ``tpot_preds`` is
    None) call ``kernel_tpot.predict_cell_tpot``, which DOES see the wrapper — so the
    static-formula FALLBACK TTFT used only for turns no simulated session reached includes
    +D in its tpot argument. Documented and accepted: it is off the main TTFT path
    (``build_row`` composes e2el inline, so the e2el path never hits it).

Metrics JSON shape::

    {gpu: {tpot_cell, ttft_cell, e2el_cell, tpot_turn_overall,
           tpot_profile: {chat, swebench, osworld, terminalbench},
           tpot_plateau_profile: {same keys}}}

  * cell metrics = mean over rows of the row's ``{tpot,ttft,e2el}_err`` (cell MAPE);
  * turn metrics = mean APE over turns with a measured value;
  * plateau      = turns with ``tpot_meas > 100`` ms;
  * profile keys = bench profile name minus ``-multiturn-synth``.

Usage:
    python3 -m profiling.process.gate_scoped_rows \
        --out /tmp/ramp_adopt/baseline.predictions.json \
        --metrics-out /tmp/ramp_adopt/baseline.metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulator.kernel_step_cost as kernel_step_cost  # noqa: E402
import simulator.kernel_tpot as kernel_tpot  # noqa: E402
from simulator.kernel_step_cost import load_grid  # noqa: E402
from simulator.kernel_tpot import KernelTurnInput  # noqa: E402
from profiling.process.build_simulator_rows import (  # noqa: E402
    BENCH_BASE, CONCURRENCIES, PROFILES, build_row,
)
from configs.loader import all_deployments  # noqa: E402

MODEL = "Llama-3.1-8B"
ENGINE = "vllm"
DEFAULT_GPU_KEYS = "H100,A100,H100x2"
PLATEAU_TPOT_MS = 100.0  # plateau turns: tpot_meas above this (the saturated regime)
# Metrics-JSON profile key order per the plan (bench profile minus '-multiturn-synth').
METRIC_PROFILE_KEYS = ["chat", "swebench", "osworld", "terminalbench"]


def _wrap_decode(orig, d: float):
    """``decode_step_ms`` + a constant pressure-independent floor excess D (ms)."""
    def wrapped(batch: float, context_tokens: float, params=None,
                _orig=orig, _d=float(d)) -> float:
        return _orig(batch, context_tokens, params) + _d
    return wrapped


def _verify_decode_patch_binds() -> None:
    """Prove predict_turn_tpot resolves decode_step_ms through kernel_tpot module globals.

    A low-pressure probe (weight = 0) must move by EXACTLY the wrapper's +D; if
    predict_turn_tpot had bound the function any other way the patch would be silent.
    """
    probe = KernelTurnInput(cached_context_tokens=200.0, new_prefill_tokens=100.0,
                            output_tokens=80.0, scheduled_requests=1.0)
    base = kernel_tpot.predict_turn_tpot(probe)
    orig = kernel_tpot.decode_step_ms
    try:
        kernel_tpot.decode_step_ms = _wrap_decode(orig, 5.0)
        patched = kernel_tpot.predict_turn_tpot(probe)
    finally:
        kernel_tpot.decode_step_ms = orig
    if abs((patched - base) - 5.0) > 1e-9:
        raise SystemExit(
            "decode_step_ms wrapper did NOT bind through kernel_tpot module globals "
            f"(probe moved {patched - base:.6f} ms, expected 5.0) — module-attribute "
            "patching is broken; aborting before producing misleading rows")


def _parse_floor_excess(spec: str | None) -> dict[str, float]:
    """Parse ``"H100=1.2,A100=0.8,H100x2=2.0"`` into {gpu_key: D_ms}."""
    out: dict[str, float] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition("=")
        if not sep:
            raise SystemExit(f"bad --floor-excess entry {part!r} (want GPU=ms)")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"bad --floor-excess value in {part!r} (want a float)") from None
    return out


def _gpu_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate metric block for one GPU's rows (see module docstring for definitions)."""
    def cell(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(st.mean(vals), 4) if vals else None

    def mean4(vals: list[float]) -> float | None:
        return round(st.mean(vals), 4) if vals else None

    turn_apes: list[float] = []
    by_profile: dict[str, list[float]] = {k: [] for k in METRIC_PROFILE_KEYS}
    plateau_by_profile: dict[str, list[float]] = {k: [] for k in METRIC_PROFILE_KEYS}
    for r in rows:
        pkey = str(r.get("profile", "")).replace("-multiturn-synth", "")
        for t in r.get("multiturn_turn_predictions") or []:
            ape = t.get("tpot_err")  # per-turn APE, None when no measured value
            if not isinstance(ape, (int, float)):
                continue
            turn_apes.append(float(ape))
            if pkey in by_profile:
                by_profile[pkey].append(float(ape))
                meas = t.get("tpot_meas")
                if isinstance(meas, (int, float)) and meas > PLATEAU_TPOT_MS:
                    plateau_by_profile[pkey].append(float(ape))
    return {
        "tpot_cell": cell("tpot_err"),
        "ttft_cell": cell("ttft_err"),
        "e2el_cell": cell("e2el_err"),
        "tpot_turn_overall": mean4(turn_apes),
        "tpot_profile": {k: mean4(by_profile[k]) for k in METRIC_PROFILE_KEYS},
        "tpot_plateau_profile": {k: mean4(plateau_by_profile[k]) for k in METRIC_PROFILE_KEYS},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-keys", default=DEFAULT_GPU_KEYS, metavar="K1,K2",
                    help=f"comma-separated deployment gpu_keys (default: {DEFAULT_GPU_KEYS})")
    ap.add_argument("--floor-excess", default=None, metavar="GPU=MS,...",
                    help='per-gpu decode floor excess D, e.g. "H100=1.2,A100=0.8,H100x2=2.0" '
                         "(wraps kernel_tpot.decode_step_ms with +D per active config)")
    ap.add_argument("--out", required=True, help="predictions JSON ({gpu: rows})")
    ap.add_argument("--metrics-out", required=True, help="gate metrics JSON")
    args = ap.parse_args(argv)

    gpu_keys = [k.strip() for k in args.gpu_keys.split(",") if k.strip()]
    floor_excess = _parse_floor_excess(args.floor_excess)
    unknown = sorted(set(floor_excess) - set(gpu_keys))
    if unknown:
        raise SystemExit(f"--floor-excess keys not in --gpu-keys: {unknown}")

    _verify_decode_patch_binds()

    configs = [c for c in all_deployments()
               if c.gpu_key in gpu_keys and c.model == MODEL and c.engine == ENGINE]
    missing = [k for k in gpu_keys if k not in {c.gpu_key for c in configs}]
    if missing:
        raise SystemExit(f"no {MODEL} {ENGINE} deployment for gpu keys: {missing}")

    # The per-GPU realized pool files are GITIGNORED (~100MB) — a fresh worktree silently lacks
    # them, the TTFT cohort falls back to the pooled forward mode, and TTFT/E2EL gates then
    # measure a NON-PRODUCTION configuration (H100 ttft_cell ~33% instead of ~18%). This exact
    # failure flipped a host-split adoption verdict on 2026-06-09/10 — see the De-fit log.
    if not list(Path("inference-benchmark/data/distributions").glob("*_realized_*.json")):
        print("WARNING: no per-GPU realized distribution files found under "
              "inference-benchmark/data/distributions/ — TTFT trajectory REPLAY IS OFF "
              "(pooled cohort, non-production). TTFT/E2EL gates are NOT production-faithful; "
              "the pools are COMMITTED since 2026-06-10 (lane L2) — a missing file means a stale/"
              "corrupted checkout: git checkout -- inference-benchmark/data/distributions/ or "
              "regenerate via build_realized_session_distributions.", flush=True)

    orig_grid = kernel_step_cost._default_grid
    orig_ceiling = kernel_tpot._active_ceiling_json
    orig_decode = kernel_tpot.decode_step_ms
    payload: dict[str, list[dict[str, Any]]] = {}
    try:
        for cfg in configs:
            bench_root = BENCH_BASE / cfg.bench_dir
            if not bench_root.exists():
                print(f"SKIP {cfg.gpu_key}: bench root not found ({bench_root})")
                continue
            # Mirror build_simulator_rows.main(): per-config decode grid + ceiling swap.
            if cfg.decode_grid is not None and cfg.decode_grid.exists():
                grid = load_grid(cfg.decode_grid)
                kernel_step_cost._default_grid = lambda grid=grid: grid
                grid_desc = f"measured grid {cfg.decode_grid.name}"
            else:
                agrid = kernel_step_cost.analytic_grid()
                kernel_step_cost._default_grid = lambda agrid=agrid: agrid
                grid_desc = "analytic decode roofline"
            kernel_tpot._active_ceiling_json = (
                cfg.saturated_ceiling
                if cfg.saturated_ceiling is not None and cfg.saturated_ceiling.exists()
                else orig_ceiling)
            # Per-config floor-excess wrapper (module-attribute patch, never a source edit).
            d = float(floor_excess.get(cfg.gpu_key, 0.0))
            kernel_tpot.decode_step_ms = _wrap_decode(orig_decode, d) if d else orig_decode

            rows: list[dict[str, Any]] = []
            for profile in PROFILES:
                for conc in CONCURRENCIES:
                    row = build_row(profile, conc, cfg.roofline, cfg, bench_root)
                    if row:
                        rows.append(row)
            payload.setdefault(cfg.gpu_key, []).extend(rows)
            print(f"{cfg.gpu_key}: {len(rows)} rows ({grid_desc}; floor_excess={d} ms; "
                  f"max_num_batched_tokens={cfg.roofline.max_num_batched_tokens})")
    finally:
        kernel_step_cost._default_grid = orig_grid
        kernel_tpot._active_ceiling_json = orig_ceiling
        kernel_tpot.decode_step_ms = orig_decode

    if not payload:
        raise SystemExit("no rows produced")
    metrics = {gpu: _gpu_metrics(rows) for gpu, rows in payload.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    mout = Path(args.metrics_out)
    mout.parent.mkdir(parents=True, exist_ok=True)
    mout.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {sum(len(v) for v in payload.values())} rows -> {out}")
    print(f"wrote metrics -> {mout}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
