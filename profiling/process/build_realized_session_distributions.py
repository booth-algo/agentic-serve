#!/usr/bin/env python3
"""Build the REALIZED per-profile session distributions the TTFT queue sim's cohort uses.

The committed ``*_multiturn*.json`` dist files carry the GENERATOR's *intended* session-length
histogram (and stale source-trace contexts). But the synthetic benchmark that actually ran has a
DIFFERENT realized distribution — sessions truncate/fail, so fewer survive (spec osworld S(18)=0.30
vs realized 0.26), and each session runs systematically larger/smaller contexts than the per-turn
median. Near the sharp KV-pool eviction cliff (measured on H100, ~8.5x at the boundary) those gaps
decide whether a cohort sits over or under the cliff — the osworld saturate-RECOVER vs a false spike.

This writes ``<spec>_realized.json`` for each profile (pointed to by ``ramp_tpot.PROFILE_DIST``) with:
  * ``histograms.turn_count`` — realized survival: per-session ``max(success turn_index)+1``,
    pooled over all concurrencies (survival fraction is ~concurrency-independent for c>=80).
  * ``context_scale_quantiles`` — p0..p100 of each session's MEDIAN ratio of its
    ``total_context_tokens`` to the per-(conc,turn) median. The cohort applies a session's quantile
    scale to the median trajectory so the KV working set has the real SPREAD (small sessions stay
    resident=hits, the large minority is preempted), keeping the MEDIAN session a hit near the cliff.

Both are measured WORKLOAD properties (inputs), not TTFT fits — same category as the kernel grids.

Usage:
    python3 -m profiling.process.build_realized_session_distributions \
        [--bench-root <dir>] [--nq 101]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.ramp_tpot import DIST_DIR, PROFILE_DIST, _gpu_slug  # noqa: E402

BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
DEFAULT_BENCH_ROOT = BENCH_BASE / "h100_Llama-3.1-8B_tp1_vllm"
# spec file each realized file is derived from / falls back to for non-turn_count metadata.
SPEC_OF = {
    "swebench-multiturn-synth": "swebench_multiturn_short_tracereplay_filtered-mse.json",
    "terminalbench-multiturn-synth": "terminalbench_multiturn_short_tracereplay_filtered-mse.json",
    "osworld-multiturn-synth": "osworld_multiturn.json",
    "chat-multiturn-synth": "chat_multiturn.json",
}


def _quantiles(sorted_vals: list[float], nq: int) -> list[float]:
    n = len(sorted_vals)
    if n == 0:
        return [1.0] * nq
    return [sorted_vals[min(n - 1, int(round(q / (nq - 1) * (n - 1))))] for q in range(nq)]


def _accumulate(profile: str, bench_root: Path) -> tuple[dict[str, dict[int, int]], dict[str, list[float]]]:
    """Raw per-concurrency accumulators from a GPU's benchmark per_request (success-filtered):
    ``(tc_by_conc[conc][turn_count]=count, ratios_by_conc[conc]=[per-session scale ratio])``.
    Per-session scale = median over the session's turns of ``total_context / per-(conc,turn)-median``.
    ONE definition shared by the pooled build, the per-conc build, and the LOCO check (no drift)."""
    files = [f for f in glob.glob(str(bench_root / f"{profile}_conc*.json")) if "_per_turn" not in f]
    tc_by_conc: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ratios_by_conc: dict[str, list[float]] = defaultdict(list)
    for fn in files:
        conc = str(int(fn.split("conc")[-1].split(".")[0]))
        data = json.loads(Path(fn).read_text())
        max_turn: dict[Any, int] = defaultdict(int)
        per_session: dict[Any, dict[int, float]] = defaultdict(dict)
        medians: dict[int, list[float]] = defaultdict(list)
        for r in data.get("per_request") or []:
            if not r.get("success"):
                continue
            sid = r.get("session_id")
            if sid is None:
                continue
            ti = int(r.get("turn_index") or 0)
            ctx = float(r.get("total_context_tokens") or 0.0)
            max_turn[sid] = max(max_turn[sid], ti)
            per_session[sid][ti] = ctx
            medians[ti].append(ctx)
        med = {ti: st.median(v) for ti, v in medians.items()}
        for mt in max_turn.values():
            tc_by_conc[conc][mt + 1] += 1
        for turns in per_session.values():
            rs = [turns[ti] / med[ti] for ti in turns if med.get(ti)]
            if rs:
                ratios_by_conc[conc].append(st.median(rs))
    return tc_by_conc, ratios_by_conc


def _block(tc_hist: dict[int, int], ratios: list[float], nq: int) -> dict[str, Any]:
    """One realized block: turn_count histogram + context-scale quantiles (p0..p100) + n_sessions."""
    return {
        "turn_count": {str(k): int(v) for k, v in sorted(tc_hist.items())},
        "context_scale_quantiles": [round(x, 4) for x in _quantiles(sorted(ratios), nq)],
        "n_sessions": sum(tc_hist.values()),
    }


def build_per_conc(profile: str, bench_root: Path, nq: int) -> dict[str, dict[str, Any]]:
    """Per-concurrency realized blocks ``{conc: {turn_count, context_scale_quantiles, n_sessions}}``.
    Reused by the generator (``by_concurrency``) and the LOCO generalization check."""
    tc_by_conc, ratios_by_conc = _accumulate(profile, bench_root)
    return {c: _block(tc_by_conc[c], ratios_by_conc.get(c, []), nq) for c in tc_by_conc}


def build_trajectory_pools(profile: str, bench_root: Path, cap: int = 320) -> dict[str, list[list[list[int]]]]:
    """Per-CONCURRENCY real session trajectory pools for concurrency-MATCHED cohort replay (the
    tournament winner 2026-06-04): ``{conc: [[ [cached, new, output], ...turns... ], ...]}``. Each cell
    replays trajectories from its own / nearest measured concurrency — osworld's trajectory SHAPES are
    concurrency-dependent, so conc-matching reaches the oracle floor (vs a pooled-over-conc pool, which
    helps osworld OR the other profiles but not both). Success-filtered, turn-ordered; per-conc capped
    to bound file size (the cohort builder cycles the pool to the target size)."""
    files = sorted(f for f in glob.glob(str(bench_root / f"{profile}_conc*.json")) if "_per_turn" not in f)
    pools: dict[str, list[list[list[int]]]] = {}
    for fn in files:
        conc = str(int(fn.split("conc")[-1].split(".")[0]))
        data = json.loads(Path(fn).read_text())
        by_sid: dict[Any, dict[int, list[int]]] = defaultdict(dict)
        for r in data.get("per_request") or []:
            if not r.get("success"):
                continue
            sid = r.get("session_id")
            if sid is None:
                continue
            ti = int(r.get("turn_index") or 0)
            by_sid[sid][ti] = [
                round(float(r.get("cached_context_tokens") or 0.0)),
                round(float(r.get("new_prefill_tokens") or 0.0)),
                max(1, round(float(r.get("output_tokens") or 1.0))),
            ]
        pool = [[turns[ti] for ti in sorted(turns)] for turns in by_sid.values() if turns]
        if cap and len(pool) > cap:
            step = len(pool) / cap
            pool = [pool[int(i * step)] for i in range(cap)]
        if pool:
            pools[conc] = pool
    return pools


def build_profile(profile: str, bench_root: Path, nq: int, per_conc: bool = False) -> dict[str, Any] | None:
    """Pooled realized payload (turn_count + context-scale quantiles, byte-identical VALUES to the
    legacy pooled output). With ``per_conc=True`` also attach ``by_concurrency`` +
    ``measured_concurrencies`` — used ONLY for the per-GPU files; the pooled ``*_realized.json`` stay
    pooled-only so ``gpu_key=None`` consumers (ramp TPOT column, ttft_predict) are byte-identical."""
    tc_by_conc, ratios_by_conc = _accumulate(profile, bench_root)
    if not tc_by_conc:
        return None
    tc_pooled: dict[int, int] = defaultdict(int)
    for hist in tc_by_conc.values():
        for k, v in hist.items():
            tc_pooled[k] += v
    ratios_pooled: list[float] = [r for rs in ratios_by_conc.values() for r in rs]
    payload: dict[str, Any] = {
        "name": f"{profile}_realized",
        "source": "realized success-filtered max-turn from bench per_request, per-concurrency + pooled",
        "histograms": {"turn_count": {str(k): int(v) for k, v in sorted(tc_pooled.items())}},
        "context_scale_quantiles": [round(x, 4) for x in _quantiles(sorted(ratios_pooled), nq)],
        "context_scale_source": "per-session median(total_context/per-(conc,turn)-median), pooled, success-filtered",
        "n_sessions": sum(tc_pooled.values()),
    }
    if per_conc:
        bc = {c: _block(tc_by_conc[c], ratios_by_conc.get(c, []), nq) for c in tc_by_conc}
        pools = build_trajectory_pools(profile, bench_root)
        for c, blk in bc.items():
            if pools.get(c):
                blk["trajectory_pool"] = pools[c]   # concurrency-MATCHED replay cohort (per-conc)
        payload["by_concurrency"] = bc
        payload["measured_concurrencies"] = sorted(int(c) for c in bc)
        payload["trajectory_pool_source"] = (
            "per-concurrency success-filtered real session trajectories [cached,new,output] per turn, "
            "concurrency-matched (nearest-conc) replay, per-conc-capped"
        )
    return payload


def _write(profile: str, spec: str, payload: dict[str, Any], slug: str | None) -> None:
    suffix = f"_realized_{slug}.json" if slug else "_realized.json"
    out = DIST_DIR / spec.replace(".json", suffix)
    out.write_text(json.dumps(payload, indent=2))
    q = payload["context_scale_quantiles"]
    bc = f" by_conc={sorted(payload.get('measured_concurrencies', []))}" if payload.get("by_concurrency") else ""
    print(f"  wrote {out.name}  N={payload['n_sessions']}  scale p10/p50/p90={q[10]}/{q[50]}/{q[90]}{bc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, default=None,
                    help="single-root mode: build from one bench dir (needs --gpu-key for a per-GPU file).")
    ap.add_argument("--gpu-key", default=None, help="single-root mode: GPU key naming the per-GPU output file.")
    ap.add_argument("--nq", type=int, default=101, help="number of context-scale quantiles (p0..p100)")
    ap.add_argument("--refresh-pooled", action="store_true",
                    help="ALSO regenerate the legacy pooled *_realized.json from the H100 root "
                         "(pooled-only; the gpu_key=None fallback). Off by default to preserve committed pooled curves.")
    a = ap.parse_args()

    if a.bench_root is not None:
        # explicit single-root mode (ad-hoc). With --gpu-key -> per-GPU per-conc file; else legacy pooled.
        if not a.bench_root.exists():
            raise SystemExit(f"bench root not found: {a.bench_root}")
        slug = _gpu_slug(a.gpu_key) if a.gpu_key else None
        for profile, spec in SPEC_OF.items():
            payload = build_profile(profile, a.bench_root, a.nq, per_conc=bool(a.gpu_key))
            if payload is None:
                print(f"  SKIP {profile}: no benchmark files"); continue
            if a.gpu_key:
                payload["gpu_key"] = a.gpu_key
            _write(profile, spec, payload, slug)
        return

    # default: enumerate every Llama-3.1-8B deployment -> one per-GPU per-conc realized file each
    # (ADDITIVE; the committed pooled *_realized.json are left untouched so gpu_key=None is byte-identical).
    from configs.loader import all_deployments
    seen: set[str] = set()
    for dep in all_deployments():
        if dep.model != "Llama-3.1-8B":
            continue
        slug = _gpu_slug(dep.gpu_key)
        if not slug or slug in seen:
            continue
        root = BENCH_BASE / dep.bench_dir
        if not root.exists():
            continue
        wrote = 0
        for profile, spec in SPEC_OF.items():
            payload = build_profile(profile, root, a.nq, per_conc=True)
            if payload is None:
                continue
            payload["gpu_key"] = dep.gpu_key
            _write(profile, spec, payload, slug)
            wrote += 1
        if wrote:
            seen.add(slug)
            print(f"--- {dep.gpu_key} ({slug}): {wrote} files from {dep.bench_dir} ---")

    if a.refresh_pooled:
        if not DEFAULT_BENCH_ROOT.exists():
            raise SystemExit(f"H100 root not found for pooled refresh: {DEFAULT_BENCH_ROOT}")
        print("refreshing legacy pooled *_realized.json (H100 root, pooled-only):")
        for profile, spec in SPEC_OF.items():
            payload = build_profile(profile, DEFAULT_BENCH_ROOT, a.nq, per_conc=False)
            if payload is not None:
                _write(profile, spec, payload, None)


if __name__ == "__main__":
    main()
