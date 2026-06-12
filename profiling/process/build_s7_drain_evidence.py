#!/usr/bin/env python3
"""L13 S7: build the engine-truth drain evidence table from the 3090 GT-protocol replay.

Inputs (pulled from 3090:/home/kevinlau/m3090_run/l13/s7/, committed under
``profile_data/results/s7_replay/``):
  s7_metrics_tp{N}.jsonl   0.2s samples of the vLLM /metrics counters
                           (vllm:gpu_prefix_cache_queries/_hits are TOKEN counters,
                           vllm:prompt_tokens_total, num_requests_running/waiting, ...)
  s7_cells_tp{N}.log       CELL_START/CELL_END wall-clock markers per replayed GT cell
  s7_chat_conc{C}_tp{N}.json  harness result per cell (same runner + flags as GT)

For every replayed cell this prints (and writes as CSV) the per-TURN engine truth:

  * computed  = delta(queries - hits)  — tokens the GPU actually (re)prefilled that turn
                (the volume ``ttft_queue_sim``'s barrier drain must price)
  * bench_new = the benchmark rows' client-side ``new_prefill_tokens`` view of the same turn
  * drain_s   = wall duration of the prefill burst (cluster of rising ``queries``)
  * rate      = drain_s / computed (ms per computed token — the BATCHED drain rate)
  * ttft_med  = the replay's measured median TTFT vs the original GT cell's

Turn clusters are detected on the metrics timeline itself (a turn's barrier drain is a
contiguous burst of rising ``gpu_prefix_cache_queries``; between turns the herd is
decode-only and the counter is flat). Pure measurement reduction — nothing in the
simulator consumes this file; the derived pins carry their own provenance notes.

Usage:
    python3 -m profiling.process.build_s7_drain_evidence \
        --run-dir profile_data/results/s7_replay --tp 4 --concs 1,5,10,20,40,80,120 \
        [--gt-base /mnt/100g/agent-bench/results/synthetic_distributional] \
        [--out-csv profile_data/results/s7_replay/s7_drain_evidence_tp4.csv]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics as st
from pathlib import Path


def _read_text(path: Path) -> str:
    """Read ``path`` or its committed ``.gz`` twin (raws are stored gzipped)."""
    if path.exists():
        return path.read_text()
    gz = path.with_name(path.name + ".gz")
    with gzip.open(gz, "rt") as f:
        return f.read()


def _exists(path: Path) -> bool:
    return path.exists() or path.with_name(path.name + ".gz").exists()


def _col(sample: dict, frag: str) -> float:
    for k, v in sample.items():
        if frag in k:
            return float(v)
    return 0.0


def load_samples(path: Path) -> list[dict]:
    out = []
    for line in _read_text(path).splitlines():
        d = json.loads(line)
        if "error" not in d:
            out.append(d)
    out.sort(key=lambda d: d["ts"])
    return out


def load_cells(path: Path) -> list[tuple[int, float, float]]:
    cells, starts = [], {}
    for line in _read_text(path).splitlines():
        p = line.split()
        if p[0] == "CELL_START":
            starts[p[3]] = float(p[4])
        elif p[0] == "CELL_END":
            cells.append((int(p[3]), starts[p[3]], float(p[4])))
    return cells


def clusters_in(win: list[dict], gap_s: float = 1.5) -> list[tuple[dict, dict]]:
    """Contiguous bursts of rising gpu_prefix_cache_queries (turn barrier drains)."""
    out, cur = [], None
    for a, b in zip(win, win[1:]):
        if _col(b, "prefix_cache_queries") - _col(a, "prefix_cache_queries") > 0:
            if cur is None:
                cur = [a, b]
            elif b["ts"] - cur[1]["ts"] > gap_s:
                out.append(tuple(cur))
                cur = [a, b]
            else:
                cur[1] = b
        elif cur is not None and b["ts"] - cur[1]["ts"] > gap_s:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def ttft_medians(result_json: Path) -> dict[int, float]:
    if not _exists(result_json):
        return {}
    byt: dict[int, list[float]] = {}
    for r in json.loads(_read_text(result_json)).get("per_request", []):
        if r.get("success"):
            byt.setdefault(int(r["turn_index"]), []).append(float(r["ttft_ms"]))
    return {ti: st.median(v) for ti, v in byt.items()}


def bench_new_medians(result_json: Path) -> dict[int, float]:
    if not _exists(result_json):
        return {}
    byt: dict[int, list[float]] = {}
    for r in json.loads(_read_text(result_json)).get("per_request", []):
        if r.get("success"):
            byt.setdefault(int(r["turn_index"]), []).append(float(r["new_prefill_tokens"]))
    return {ti: st.median(v) for ti, v in byt.items()}


def turn_medians(result_json: Path, key: str) -> dict[int, float]:
    if not _exists(result_json):
        return {}
    byt: dict[int, list[float]] = {}
    for r in json.loads(_read_text(result_json)).get("per_request", []):
        if r.get("success"):
            byt.setdefault(int(r["turn_index"]), []).append(float(r[key]))
    return {ti: st.median(v) for ti, v in byt.items()}


def cell_volume_rows(run_dir: Path, prefix: str, profile: str, tp: int
                     ) -> list[dict]:
    """CELL-level engine-truth volumes for one replayed profile ladder: per cell, the
    monotonic /metrics counter deltas over the whole CELL window (immune to the
    admission-burst cluster-window problem — the counters move at SCHEDULING time, so
    intra-cell cluster durations are NOT compute durations at mid concurrency) vs the
    benchmark-view ``new_prefill_tokens`` sum, plus the cohort-final context median for
    the no-eviction criterion."""
    short = profile.split("-")[0]
    samples = load_samples(run_dir / f"{prefix}_metrics_tp{tp}.jsonl")
    cells = load_cells(run_dir / f"{prefix}_cells_tp{tp}.log")
    out = []
    for conc, t0, t1 in cells:
        win = [s for s in samples if t0 <= s["ts"] <= t1]
        res = run_dir / f"{prefix}_{short}_conc{conc}_tp{tp}.json"
        if len(win) < 2 or not _exists(res):
            continue
        dq = _col(win[-1], "prefix_cache_queries") - _col(win[0], "prefix_cache_queries")
        dh = _col(win[-1], "prefix_cache_hits") - _col(win[0], "prefix_cache_hits")
        reqs = [r for r in json.loads(_read_text(res)).get("per_request", [])
                if r.get("success")]
        if not reqs:
            continue
        bench_new = sum(float(r["new_prefill_tokens"]) for r in reqs)
        last_turn = max(int(r["turn_index"]) for r in reqs)
        final_ctx = st.median([float(r["total_context_tokens"]) for r in reqs
                               if int(r["turn_index"]) == last_turn])
        out.append({"profile": profile, "conc": conc, "computed": dq - dh,
                    "bench_new": bench_new, "final_ctx_med": final_ctx})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--concs", default="1,5,10,20,40,80,120")
    ap.add_argument("--gt-base", type=Path,
                    default=Path("/mnt/100g/agent-bench/results/synthetic_distributional"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--emit-pin", type=Path, default=None,
                    help="write the qsim_response_resident_fraction pin artifact JSON")
    # L13 round 3 (S8): the same reducer over a NON-chat profile ladder replay. Defaults
    # reproduce the S7 file layout byte-identically (prefix s7, chat profile).
    ap.add_argument("--prefix", default="s7",
                    help="file-name prefix of the run dir's artifacts (s7|s8; default s7)")
    ap.add_argument("--profile", default="chat-multiturn-synth",
                    help="GT profile replayed (default chat-multiturn-synth)")
    ap.add_argument("--emit-comm-sat-pin", type=Path, default=None,
                    help="write the prefill_tp_comm_saturated_ms_per_token pin artifact JSON "
                         "(PRE-REGISTERED L13 round-3 estimator: r_sat = median rate over "
                         "clusters with computed >= 20000; comm_sat = r_sat - analytic "
                         "gemm(util_sat) - dispatch, both module constants)")
    ap.add_argument("--pool-csv", type=Path, action="append", default=[],
                    help="additional reduced evidence CSV(s) whose same-tp clusters pool into "
                         "the --emit-comm-sat-pin estimator (e.g. the other S8 profile leg)")
    ap.add_argument("--emit-dup-pin", type=Path, default=None,
                    help="write the qsim_duplicate_session_fraction pin artifact JSON "
                         "(L13 round-3 estimator: f = 1 - median over CELLS with conc >= 10 "
                         "AND conc*median_final_ctx <= pool_tokens of "
                         "cell_computed/cell_bench_new — CELL-level monotonic counter deltas, "
                         "immune to the admission-burst cluster-window problem)")
    ap.add_argument("--pool-tokens", type=int, default=None,
                    help="GT-protocol KV pool size in tokens for the --emit-dup-pin "
                         "no-eviction cell criterion (e.g. 477856 tp4 / 175808 tp2)")
    ap.add_argument("--dup-cells", action="append", default=[], metavar="DIR:PROFILE",
                    help="additional run-dir:profile pairs whose same-tp CELLS pool into the "
                         "--emit-dup-pin estimator (other S8 profile legs; each dir uses the "
                         "same --prefix file layout)")
    args = ap.parse_args()
    short = args.profile.split("-")[0]

    samples = load_samples(args.run_dir / f"{args.prefix}_metrics_tp{args.tp}.jsonl")
    cells = load_cells(args.run_dir / f"{args.prefix}_cells_tp{args.tp}.log")
    want = {int(c) for c in args.concs.split(",")}
    rows = []
    for conc, t0, t1 in cells:
        if conc not in want:
            continue
        win = [s for s in samples if t0 <= s["ts"] <= t1]
        res = args.run_dir / f"{args.prefix}_{short}_conc{conc}_tp{args.tp}.json"
        probe_ttft = ttft_medians(res)
        bench_new = bench_new_medians(res)
        prompt_med = turn_medians(res, "cached_context_tokens")  # client prompt-basis estimate
        out_med = turn_medians(res, "output_tokens")
        # per-turn request counts
        counts: dict[int, int] = {}
        if _exists(res):
            for r in json.loads(_read_text(res)).get("per_request", []):
                if r.get("success"):
                    counts[int(r["turn_index"])] = counts.get(int(r["turn_index"]), 0) + 1
        gt_ttft = ttft_medians(
            args.gt_base / f"3090_Llama-3.1-8B_tp{args.tp}_vllm/{args.profile}_conc{conc}.json")
        for i, (a, b) in enumerate(clusters_in(win)):
            dq = _col(b, "prefix_cache_queries") - _col(a, "prefix_cache_queries")
            dh = _col(b, "prefix_cache_hits") - _col(a, "prefix_cache_hits")
            comp = dq - dh
            dur = b["ts"] - a["ts"] + 0.2  # one sampling interval of edge slack
            # response-resident fraction rho (the qsim_response_resident_fraction pin basis):
            # aggregate hits/req = prev-prompt(block-aligned) + rho * prev-output.
            rho = ""
            n = counts.get(i, 0)
            if i >= 1 and n > 0 and out_med.get(i - 1):
                prompt_blocks = (int(prompt_med.get(i, 0.0)) // 16) * 16
                rho = round((dh / n - prompt_blocks) / out_med[i - 1], 4)
            rows.append({
                "tp": args.tp, "conc": conc, "cluster": i,
                "queries": int(dq), "hits": int(dh), "computed": int(comp),
                "drain_s": round(dur, 2),
                "rate_ms_per_tok": round(dur * 1000.0 / comp, 4) if comp > 0 else "",
                "bench_new_med": bench_new.get(i, ""),
                "rho_response_resident": rho,
                "ttft_probe_med_ms": round(probe_ttft.get(i, float("nan")), 1),
                "ttft_gt_med_ms": round(gt_ttft.get(i, float("nan")), 1),
            })
            r = rows[-1]
            print(f"tp{args.tp} c{conc:>3} cl{i:>2}: computed={r['computed']:>7} "
                  f"(queries {r['queries']:>7} hits {r['hits']:>7}) drain={r['drain_s']:>6}s "
                  f"rate={r['rate_ms_per_tok']!s:>8} ms/tok bench_new={r['bench_new_med']!s:>6} "
                  f"ttft probe/gt={r['ttft_probe_med_ms']}/{r['ttft_gt_med_ms']}")
    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.out_csv}")

    if args.emit_pin:
        # qsim_response_resident_fraction pin: median rho over the WITHIN-CELL-CLEAN clusters —
        # turns 1-4 (cross-cell prompt reuse only contaminates turn 0; later turns of deep cells
        # add eviction/retention effects that are the sim cache's job, not the match fraction) at
        # 10 <= conc <= 80 (c1/c5 have 1-5 requests/cluster; c120's pool sits at the edge).
        pin_vals = sorted(
            float(r["rho_response_resident"]) for r in rows
            if r["rho_response_resident"] != "" and 10 <= int(r["conc"]) <= 80
            and 1 <= int(r["cluster"]) <= 4
        )
        pin = st.median(pin_vals)
        art = {
            "constants": {"qsim_response_resident_fraction": pin},
            "n_clusters": len(pin_vals),
            "cluster_rho_values": pin_vals,
            "filter": f"turns 1-4, 10<=conc<=80, {args.profile} GT-protocol replay",
            "source": (
                f"S7 engine-side /metrics prefix-cache counters ({args.prefix}_metrics_tp{args.tp}.jsonl), "
                "3090 host, GT protocol (sweep_multiturn_profiles.sh class: one server, ascending "
                "ladder, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, vllm 0.19.0); rho per turn "
                "cluster = (hits/req - blockfloor16(prompt_med)) / prev_output_med"
            ),
            "date": "2026-06-12",
        }
        args.emit_pin.write_text(json.dumps(art, indent=1) + "\n")
        print(f"rho pin {pin} (n={len(pin_vals)}) -> {args.emit_pin}")

    if args.emit_comm_sat_pin:
        # PRE-REGISTERED (L13 round 3, BEFORE the S8 results): the engine's BULK barrier-drain
        # rate saturates well below the c1 stage-split composed rate (S7 tp2: 0.89 ms/tok small
        # bursts -> 0.179 median at computed>=50k). Estimator: r_sat = median rate_ms_per_tok
        # over clusters with computed >= 20,000 tokens (>= ~10 full engine steps; tb/swe windows
        # are prefill-dominated, outputs ~10-50 tok, so no chat-style decode contamination).
        # Pin identification (matches _prefill_gemm_per_tok_loaded at frac=1):
        #   comm_sat = r_sat - gemm(util=PREFILL_GEMM_UTIL_SAT) - dispatch_residual
        # so that the sim's SATURATED prefill total equals the measured bulk rate; the c1
        # anchors (prefill_tp_comm_ms_per_token) are untouched at small chunks.
        import sys as _sys
        _repo = Path(__file__).resolve().parents[2]
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from simulator.ttft_queue_sim import (  # noqa: E402
            PREFILL_GEMM_UTIL_SAT, PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN)
        from configs.loader import all_deployments  # noqa: E402
        gpu_key = f"RTX3090x{args.tp}"
        cfg = next(c for c in all_deployments()
                   if c.gpu_key == gpu_key and c.model == "Llama-3.1-8B" and c.engine == "vllm")
        p = cfg.roofline
        gemm_sat = 2.0 * (float(p.n_params) / args.tp) / (p.peak_flops_per_s * PREFILL_GEMM_UTIL_SAT) * 1e3
        pin_rows: list[dict] = list(rows)
        for extra in args.pool_csv:
            with open(extra, newline="") as f:
                pin_rows.extend(r for r in csv.DictReader(f) if int(r["tp"]) == args.tp)
        sat_rates = sorted(
            float(r["rate_ms_per_tok"]) for r in pin_rows
            if r["rate_ms_per_tok"] != "" and int(float(r["computed"])) >= 20000)
        if not sat_rates:
            raise SystemExit("no clusters with computed >= 20000 — cannot emit comm_sat pin")
        r_sat = st.median(sat_rates)
        comm_sat = r_sat - gemm_sat - PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN
        art = {
            "constants": {"prefill_tp_comm_saturated_ms_per_token": comm_sat},
            "r_sat_ms_per_tok": r_sat,
            "n_clusters": len(sat_rates),
            "cluster_rates": sat_rates,
            "gemm_sat_ms_per_tok": gemm_sat,
            "dispatch_ms_per_tok": PREFILL_NEW_DISPATCH_RESIDUAL_MS_PER_TOKEN,
            "filter": (f"clusters with computed>=20000, {args.profile} GT-protocol replay, tp{args.tp}"
                       + (f", pooled with {[p.name for p in args.pool_csv]}" if args.pool_csv else "")),
            "estimator": ("r_sat = median(rate_ms_per_tok | computed>=20000); comm_sat = r_sat - "
                          "gemm(util_sat=1.0, analytic) - PREFILL_NEW_DISPATCH_RESIDUAL (pre-registered "
                          "L13 round 3 BEFORE the S8 measurement)"),
            "source": (
                f"engine-side /metrics prefix-cache counters ({args.prefix}_metrics_tp{args.tp}.jsonl), "
                "3090 host, GT protocol replay (one server, ascending ladder, "
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, vllm 0.19.0)"),
            "date": "2026-06-12",
        }
        args.emit_comm_sat_pin.write_text(json.dumps(art, indent=1) + "\n")
        print(f"comm_sat pin {comm_sat:.6f} (r_sat {r_sat:.4f}, n={len(sat_rates)}) -> {args.emit_comm_sat_pin}")

    if args.emit_dup_pin:
        # L13 round-3 duplicate-session fraction (see the RooflineParams field comment):
        # f = 1 - median(cell_computed / cell_bench_new) over CELLS with conc >= 10 (a
        # duplicate needs a twin; c1/c5 cohorts are too small for the asymptote) AND
        # conc * median_final_ctx <= pool_tokens (no-eviction cells only — under eviction
        # computed includes full-context re-prefills and the ratio measures eviction, not
        # dedup). CELL-level monotonic counter deltas (admission-burst windows are not
        # compute windows; cell deltas are exact regardless).
        if args.pool_tokens is None:
            raise SystemExit("--emit-dup-pin needs --pool-tokens (GT-protocol pool size)")
        vol_rows = cell_volume_rows(args.run_dir, args.prefix, args.profile, args.tp)
        for spec in args.dup_cells:
            dir_s, _, prof = spec.rpartition(":")
            vol_rows += cell_volume_rows(Path(dir_s), args.prefix, prof, args.tp)
        kept = [r for r in vol_rows
                if r["conc"] >= 10 and r["conc"] * r["final_ctx_med"] <= args.pool_tokens
                and r["bench_new"] > 0]
        if not kept:
            raise SystemExit("no qualifying no-eviction cells (conc>=10) for the dup pin")
        ratios = sorted(r["computed"] / r["bench_new"] for r in kept)
        f_dup = 1.0 - st.median(ratios)
        art = {
            "constants": {"qsim_duplicate_session_fraction": f_dup},
            "n_cells": len(kept),
            "cells": [{k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in kept],
            "cell_ratios": [round(x, 4) for x in ratios],
            "pool_tokens": args.pool_tokens,
            "estimator": ("f = 1 - median(cell_computed/cell_bench_new) over cells with "
                          "conc>=10 and conc*median_final_ctx <= pool_tokens (no-eviction); "
                          "cell-level monotonic /metrics counter deltas"),
            "source": (
                f"S8 engine-side /metrics prefix-cache counters, 3090 host, GT protocol "
                f"replay (one server per profile ladder, ascending concs, "
                f"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, vllm 0.19.0), tp{args.tp}; "
                f"profiles: {[args.profile] + [s.rpartition(':')[2] for s in args.dup_cells]}"),
            "date": "2026-06-12",
        }
        args.emit_dup_pin.write_text(json.dumps(art, indent=1) + "\n")
        print(f"dup pin {f_dup:.4f} (n={len(kept)} cells, ratios {ratios}) -> {args.emit_dup_pin}")


if __name__ == "__main__":
    main()
