"""Forward latency predictor: a client's ISL:OSL distribution + a hardware config
``(gpu, tp, engine, model)`` -> TTFT / TPOT / E2EL, WITHOUT a measured ground-truth run.

Why this exists
---------------
The batch generator ``profiling/process/build_simulator_rows.build_row`` is a *backtester*: it
reads the per-turn ``(cached, new, output)`` token trajectory out of a measured benchmark file, so
it can only "predict" a cell that was already run. The two underlying predictors, however, are pure
forward functions of ``(cohort, RooflineParams, swappable hardware artifacts)``:

  * ``kernel_tpot.predict_cell_tpot(turns, params)``           -> per-turn TPOT
  * ``ttft_queue_sim.predict_cell_ttft_qsim(turns, ..., cohort=)`` -> per-turn TTFT (queue sim)

This module is the thin forward driver: it turns a client workload into the cohort + turn specs the
predictors consume, assembles the hardware model for any ``(gpu, tp, engine, model)`` (reusing a
calibrated deployment's measured artifacts when one exists, else composing an analytic roofline from
the spec sheet), runs both predictors, and reports the metrics with a ``calibration_status`` flag.

Hardware artifacts are module globals the predictors read (``kernel_step_cost._default_grid``,
``kernel_tpot._active_ceiling_json``); ``active_hardware`` swaps them for the duration of a call
exactly as ``build_simulator_rows.main()`` does, then restores them. NOT thread-safe (single-call).
"""

from __future__ import annotations

import contextlib
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import simulator.kernel_step_cost as kernel_step_cost
import simulator.kernel_tpot as kernel_tpot
from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import analytic_grid, load_grid
from simulator.kernel_tpot import KernelTurnInput, predict_cell_tpot
from simulator.ttft_queue_sim import (
    QSimSchedConfig,
    Session,
    _cohort_from_pool,
    _prefill_floor_for,
    predict_cell_ttft_qsim,
)
from configs.loader import all_deployments, compose_roofline
import configs.kv_pool as kv_pool
from configs.loader import CONFIGS_DIR, _read


# --------------------------------------------------------------- hardware artifact swap


@contextlib.contextmanager
def active_hardware(decode_grid: Path | None, saturated_ceiling: Path | None):
    """Swap the module-global decode grid + saturated-ITL ceiling for the duration of a prediction,
    then restore — the same swap ``build_simulator_rows.main()`` does per config, made reusable and
    exception-safe. ``decode_grid=None`` -> the analytic decode roofline (scales with this config's
    own weight bytes / bandwidth / KV); ``saturated_ceiling=None`` -> leave the in-code default
    (H100) anchors, matching main()'s inherit branch."""
    orig_grid = kernel_step_cost._default_grid
    orig_ceiling = kernel_tpot._active_ceiling_json
    try:
        if decode_grid is not None and Path(decode_grid).exists():
            grid = load_grid(Path(decode_grid))
            kernel_step_cost._default_grid = lambda grid=grid: grid
        else:
            agrid = analytic_grid()
            kernel_step_cost._default_grid = lambda agrid=agrid: agrid
        if saturated_ceiling is not None and Path(saturated_ceiling).exists():
            kernel_tpot._active_ceiling_json = Path(saturated_ceiling)
        # saturated_ceiling None -> keep orig_ceiling (the H100 default anchors).
        yield
    finally:
        kernel_step_cost._default_grid = orig_grid
        kernel_tpot._active_ceiling_json = orig_ceiling


# --------------------------------------------------------------- hardware resolution


@dataclass
class HardwareResolved:
    params: RooflineParams
    decode_grid: Path | None
    saturated_ceiling: Path | None
    prefill_floor_ms: float | None
    sched: QSimSchedConfig | None
    calibration_status: str
    gpu_key: str
    # Llama-only trajectory-replay/floor gate the backtester uses (None for non-Llama-8B); exposed
    # so the self-consistency gate can reproduce build_row's exact arg derivation.
    pool_gpu_key: str | None
    calibration_detail: str = ""


def _derive_pool(gpu: str, model: str, tp: int) -> int:
    """Analytic KV-pool blocks for an uncalibrated config (configs/kv_pool.available_kv_blocks)."""
    g = _read(CONFIGS_DIR / "gpus" / f"{gpu}.json")
    m = _read(CONFIGS_DIR / "models" / f"{model}.json")
    total_mem = g.get("total_memory_bytes")
    if not total_mem:
        return int(RooflineParams().available_kv_blocks)  # last-resort: H100 default
    return kv_pool.available_kv_blocks(
        total_memory_bytes=float(total_mem),
        gpu_mem_util=float(g.get("gpu_mem_util", 0.90)),
        weight_bytes=float(m["n_params"]) * float(m["bytes_per_param"]),
        tp=int(tp),
        kv_bytes_per_token=float(m["kv_bytes_per_token"]),
        kv_heads=int(m["kv_heads"]),
        block_size=int(m.get("cache_block_size", 16)),
    )


def _sched_for(max_model_len, max_num_seqs, params: RooflineParams) -> QSimSchedConfig | None:
    """Per-config vLLM scheduler truth for the queue sim's admission arithmetic — built ONLY when
    the deployment pins it (mirrors build_row); unpinned -> None -> the sim's H100 constants."""
    if max_model_len is None and max_num_seqs is None:
        return None
    return QSimSchedConfig(
        max_num_batched_tokens=int(params.max_num_batched_tokens),
        long_prefill_token_threshold=(int(max_model_len * 0.04) if max_model_len is not None else None),
        max_num_seqs=max_num_seqs,
    )


def resolve_hardware(gpu: str, model: str, tp: int, engine: str = "vllm") -> HardwareResolved:
    """Assemble the hardware model for ``(gpu, model, tp, engine)``. If a calibrated deployment JSON
    exists, reuse its measured RooflineParams + owned decode grid / saturated ceiling + measured
    prefill floor (calibration_status="measured"). Otherwise compose an analytic roofline from the
    GPU + model spec sheets with an analytically-derived KV pool (calibration_status="extrapolated":
    analytic decode roofline, H100-default ceiling, H100-borrowed utils)."""
    for cfg in all_deployments():
        if cfg.gpu == gpu and cfg.model == model and int(cfg.tp) == int(tp) and cfg.engine == engine:
            # Coarse, honest status from ACTUAL measured-artifact presence (the deployment's own
            # calibration_status string can be stale): both decode grid + saturated ceiling owned
            # -> "measured"; a deployment exists but inherits one -> "partial".
            owned = (cfg.decode_grid is not None) + (cfg.saturated_ceiling is not None)
            return HardwareResolved(
                params=cfg.roofline,
                decode_grid=cfg.decode_grid,
                saturated_ceiling=cfg.saturated_ceiling,
                prefill_floor_ms=_prefill_floor_for(cfg.gpu_key),
                sched=_sched_for(cfg.max_model_len, cfg.max_num_seqs, cfg.roofline),
                calibration_status="measured" if owned == 2 else "partial",
                gpu_key=cfg.gpu_key,
                pool_gpu_key=(cfg.gpu_key if cfg.model == "Llama-3.1-8B" else None),
                calibration_detail=cfg.calibration_status or "",
            )
    # Uncalibrated -> spec-sheet roofline + analytic pool.
    pool = _derive_pool(gpu, model, int(tp))
    params = compose_roofline(gpu, model, int(tp), pool)
    gpu_key = f"{gpu}x{tp}" if int(tp) > 1 else gpu
    return HardwareResolved(
        params=params, decode_grid=None, saturated_ceiling=None,
        prefill_floor_ms=_prefill_floor_for(gpu_key), sched=None,
        calibration_status="extrapolated", gpu_key=gpu_key, pool_gpu_key=None,
        calibration_detail="spec-sheet roofline (no deployment): analytic decode roofline + "
                           "analytic KV pool, H100-borrowed utils, inherited H100 ceiling",
    )


# --------------------------------------------------------------- prediction core


def predict_turns(
    turns: list[dict],
    params: RooflineParams,
    concurrency: float,
    *,
    decode_grid: Path | None = None,
    saturated_ceiling: Path | None = None,
    qbar: float = 1.0,
    shared_prefix_tokens: float = 0.0,
    sched: QSimSchedConfig | None = None,
    prefill_floor_ms: float | None = None,
    cohort: list[Session] | None = None,
    gpu_key: str | None = None,
    profile: str = "forward",
) -> tuple[list[float], list[float], list[float]]:
    """The shared prediction primitive — IDENTICAL wiring to build_row's headline path
    (KernelTurnInput per turn + qbar -> predict_cell_tpot; turns + cohort -> predict_cell_ttft_qsim;
    E2EL = ttft + output*tpot), but artifact-swapped via ``active_hardware`` and accepting a
    caller-supplied ``cohort`` (the forward trace) instead of a GT-derived one. Returns
    ``(tpot, ttft, e2el)`` per-turn lists aligned to ``turns``."""
    kin = [
        KernelTurnInput(
            t["cached_context_tokens"], t["new_prefill_tokens"], t["output_tokens"],
            t["scheduled_requests"], cohort_scale_mean=qbar,
        )
        for t in turns
    ]
    with active_hardware(decode_grid, saturated_ceiling):
        tpot = predict_cell_tpot(kin, params)
        ttft = predict_cell_ttft_qsim(
            turns, profile, float(concurrency), params,
            shared_prefix_tokens=shared_prefix_tokens, gpu_key=gpu_key,
            prefill_floor_ms=prefill_floor_ms, sched=sched, cohort=cohort,
        )
    e2el = [float(tf) + float(t["output_tokens"]) * float(tp) for t, tp, tf in zip(turns, tpot, ttft)]
    return tpot, ttft, e2el


# --------------------------------------------------------------- workload -> cohort


def _single_turn_cohort(
    samples: Sequence[tuple[float, float]], concurrency: float, shared_prefix: float = 0.0,
) -> tuple[list[Session], list[dict], float]:
    """Map a single-turn ISL:OSL sample set into (cohort, per-turn specs, qbar). Each ``(isl, osl)``
    sample is one session with a single turn ``[cached=shared_prefix, new=isl, output=osl]``; the
    cohort is ``concurrency`` sessions cycled deterministically from the samples (``_cohort_from_pool``).
    ``qbar`` = mean per-session (total_context / median total_context) — the cohort's KV-demand
    spread the TPOT overflow model integrates."""
    C = max(1, int(round(float(concurrency))))
    isls = [float(i) for i, _ in samples]
    osls = [max(1.0, float(o)) for _, o in samples]
    pool = [[[float(shared_prefix), isl, osl]] for isl, osl in zip(isls, osls)]
    cohort = _cohort_from_pool(pool, C)
    med_isl = st.median(isls)
    med_osl = st.median(osls)
    total_ctx = [shared_prefix + i for i in isls]
    med_ctx = st.median(total_ctx)
    qbar = st.mean([c / med_ctx for c in total_ctx]) if med_ctx > 0 else 1.0
    turns = [{
        "turn_index": 0,
        "cached_context_tokens": float(shared_prefix),
        "new_prefill_tokens": float(med_isl),
        "output_tokens": float(med_osl),
        "total_context_tokens": float(shared_prefix + med_isl),
        "scheduled_requests": float(C),
    }]
    return cohort, turns, qbar


# --------------------------------------------------------------- public API


@dataclass
class ForwardResult:
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float
    calibration_status: str
    gpu_key: str
    concurrency: float
    isl: float
    osl: float
    calibration_detail: str = ""
    per_turn: dict = field(default_factory=dict)


def predict_forward(
    *,
    gpu: str,
    model: str,
    tp: int,
    concurrency: float,
    isl_osl_samples: Sequence[tuple[float, float]],
    engine: str = "vllm",
    shared_prefix_tokens: float = 0.0,
) -> ForwardResult:
    """Forward prediction for a client's single-turn ISL:OSL sample set on ``(gpu, tp, engine, model)``
    at a given concurrency. Returns TTFT/TPOT/E2EL (ms) + a ``calibration_status`` ("measured" when a
    calibrated deployment backs this hardware, "extrapolated" when composed from the spec sheet)."""
    if not isl_osl_samples:
        raise ValueError("isl_osl_samples is empty")
    hw = resolve_hardware(gpu, model, int(tp), engine)
    cohort, turns, qbar = _single_turn_cohort(isl_osl_samples, concurrency, shared_prefix_tokens)
    tpot, ttft, e2el = predict_turns(
        turns, hw.params, concurrency,
        decode_grid=hw.decode_grid, saturated_ceiling=hw.saturated_ceiling,
        qbar=qbar, shared_prefix_tokens=shared_prefix_tokens, sched=hw.sched,
        prefill_floor_ms=hw.prefill_floor_ms, cohort=cohort, gpu_key=None,
    )
    return ForwardResult(
        ttft_ms=round(ttft[0], 4), tpot_ms=round(tpot[0], 4), e2el_ms=round(e2el[0], 4),
        calibration_status=hw.calibration_status, gpu_key=hw.gpu_key,
        concurrency=float(concurrency),
        isl=turns[0]["total_context_tokens"], osl=turns[0]["output_tokens"],
        calibration_detail=hw.calibration_detail,
        per_turn={"tpot": tpot, "ttft": ttft, "e2el": e2el},
    )
