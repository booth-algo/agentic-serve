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

from simulator.ramp_tpot import DIST_DIR, PROFILE_DIST  # noqa: E402

DEFAULT_BENCH_ROOT = Path(
    "/mnt/100g/agent-bench/results/synthetic_distributional/h100_Llama-3.1-8B_tp1_vllm"
)
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


def build_profile(profile: str, bench_root: Path, nq: int) -> dict[str, Any] | None:
    files = [f for f in glob.glob(str(bench_root / f"{profile}_conc*.json")) if "_per_turn" not in f]
    if not files:
        return None
    tc_counter: dict[int, int] = defaultdict(int)        # turn_count -> #sessions
    per_session: dict[tuple, dict[int, float]] = defaultdict(dict)  # (conc,sid) -> {turn: ctx}
    medians: dict[tuple, list[float]] = defaultdict(list)          # (conc,turn) -> [ctx]
    for fn in files:
        conc = fn.split("conc")[-1].split(".")[0]
        data = json.loads(Path(fn).read_text())
        max_turn: dict[Any, int] = defaultdict(int)
        for r in data.get("per_request") or []:
            if not r.get("success"):
                continue
            sid = r.get("session_id")
            ti = int(r.get("turn_index") or 0)
            ctx = float(r.get("total_context_tokens") or 0.0)
            if sid is None:
                continue
            max_turn[sid] = max(max_turn[sid], ti)
            per_session[(conc, sid)][ti] = ctx
            medians[(conc, ti)].append(ctx)
        for sid, mt in max_turn.items():
            tc_counter[mt + 1] += 1
    if not tc_counter:
        return None
    med = {k: st.median(v) for k, v in medians.items()}
    ratios: list[float] = []
    for (conc, sid), turns in per_session.items():
        rs = [turns[ti] / med[(conc, ti)] for ti in turns if med.get((conc, ti))]
        if rs:
            ratios.append(st.median(rs))
    ratios.sort()
    return {
        "name": f"{profile}_realized",
        "source": "realized success-filtered max-turn from bench per_request, pooled over concs",
        "histograms": {"turn_count": {str(k): int(v) for k, v in sorted(tc_counter.items())}},
        "context_scale_quantiles": [round(x, 4) for x in _quantiles(ratios, nq)],
        "context_scale_source": "per-session median(total_context/per-(conc,turn)-median), pooled, success-filtered",
        "n_sessions": sum(tc_counter.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--nq", type=int, default=101, help="number of context-scale quantiles (p0..p100)")
    a = ap.parse_args()
    if not a.bench_root.exists():
        raise SystemExit(f"bench root not found: {a.bench_root}")
    for profile, spec in SPEC_OF.items():
        payload = build_profile(profile, a.bench_root, a.nq)
        if payload is None:
            print(f"  SKIP {profile}: no benchmark files")
            continue
        out = DIST_DIR / spec.replace(".json", "_realized.json")
        out.write_text(json.dumps(payload, indent=2))
        q = payload["context_scale_quantiles"]
        print(f"  wrote {out.name}  N={payload['n_sessions']}  scale p10/p50/p90={q[10]}/{q[50]}/{q[90]}")


if __name__ == "__main__":
    main()
