# simulator_v2/scripts/score_llmservingsim.py

"""Score LLMServingSim per-request output against measured vLLM GT.

Cell-MAPE, matching our sim's aggregation: per-turn MEDIAN over successful
requests -> mean over turn-indices. One MAPE per (profile, conc, metric).

Mapping: LLMServingSim assigns request ids in trace file-line order, sequential
across each session's sub_requests. So id -> turn_index is reconstructed by
replaying the trace (turn_index == sub_request position, since we emit turns in
order). GT carries turn_index directly.

Metrics: TTFT, TPOT, E2EL. Sim CSV cols (ns): TTFT, TPOT, latency(=E2EL).
GT per_request (ms): ttft_ms, tpot_ms, e2el_ms. TPOT turns with <=0 either side
are dropped (undefined for 1-token outputs).
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

GT_BASE = "/mnt/100g/agent-bench/results/synthetic_distributional"
PROFILES = ["chat", "osworld", "swebench", "terminalbench"]
CONCS = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]


def _median_by_turn(pairs):
    d = defaultdict(list)
    for t, v in pairs:
        if v is not None:
            d[t].append(v)
    return {t: statistics.median(v) for t, v in d.items() if v}


def sim_per_turn(sim_csv, trace_jsonl):
    id2turn, nid = {}, 0
    with open(trace_jsonl) as f:
        for line in f:
            for i in range(len(json.loads(line)["sub_requests"])):
                id2turn[nid] = i
                nid += 1
    ttft, tpot, e2e = [], [], []
    with open(sim_csv) as fh:
        for r in csv.DictReader(fh):
            t = id2turn.get(int(r["request id"]))
            if t is None:
                continue
            ttft.append((t, float(r["TTFT"]) / 1e6))
            tpot.append((t, float(r["TPOT"]) / 1e6))
            e2e.append((t, float(r["latency"]) / 1e6))
    return _median_by_turn(ttft), _median_by_turn(tpot), _median_by_turn(e2e)


def gt_per_turn(gt_json):
    data = json.load(open(gt_json))
    ttft, tpot, e2e = [], [], []
    for r in data.get("per_request", []):
        if r.get("success") is False:
            continue
        t = r.get("turn_index")
        if t is None:
            continue
        t = int(t)
        ttft.append((t, r.get("ttft_ms")))
        tpot.append((t, r.get("tpot_ms")))
        e2e.append((t, r.get("e2el_ms")))
    return _median_by_turn(ttft), _median_by_turn(tpot), _median_by_turn(e2e)


def cell_mape(sim_med, gt_med, drop_nonpos=False):
    """mean over shared turns of |sim-gt|/gt (%). Returns (mape, n_turns)."""
    apes = []
    for t in sorted(set(sim_med) & set(gt_med)):
        g, s = gt_med[t], sim_med[t]
        if g <= 0 or (drop_nonpos and s <= 0):
            continue
        apes.append(abs(s - g) / g)
    return (100 * statistics.mean(apes), len(apes)) if apes else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-dir", required=True, help="dir of <profile>_conc<N>.csv from LLMServingSim")
    ap.add_argument("--trace-dir", required=True, help="dir of our <profile>_conc<N>.jsonl")
    ap.add_argument("--config", required=True, help="GT bench_dir, e.g. h100_Llama-3.1-8B_tp1_vllm")
    ap.add_argument("--gt-base", default=GT_BASE)
    ap.add_argument("--out", default="", help="optional CSV path for the sheet")
    a = ap.parse_args()

    rows = []
    hdr = f"{'profile':14} {'conc':>4} | {'TTFT%':>7} {'TPOT%':>7} {'E2EL%':>7} | turns"
    print(hdr)
    print("-" * len(hdr))
    for pf in PROFILES:
        for c in CONCS:
            sim_csv = Path(a.sim_dir) / f"{pf}_conc{c}.csv"
            trace = Path(a.trace_dir) / f"{pf}_conc{c}.jsonl"
            gt = Path(a.gt_base) / a.config / f"{pf}-multiturn-synth_conc{c}.json"
            if not (sim_csv.exists() and trace.exists() and gt.exists()):
                continue
            s_ttft, s_tpot, s_e2e = sim_per_turn(sim_csv, trace)
            g_ttft, g_tpot, g_e2e = gt_per_turn(gt)
            m_ttft, n = cell_mape(s_ttft, g_ttft)
            m_tpot, _ = cell_mape(s_tpot, g_tpot, drop_nonpos=True)
            m_e2e, _ = cell_mape(s_e2e, g_e2e)
            def f(x):
                return f"{x:7.1f}" if x is not None else "    n/a"
            print(f"{pf:14} {c:>4} | {f(m_ttft)} {f(m_tpot)} {f(m_e2e)} | {n}")
            rows.append({"config": a.config, "profile": pf, "conc": c,
                         "TTFT_mape": m_ttft, "TPOT_mape": m_tpot, "E2EL_mape": m_e2e, "turns": n})
    # profile-level + overall means
    def avg(key, subset=None):
        vs = [r[key] for r in rows if r[key] is not None and (subset is None or r["profile"] == subset)]
        return statistics.mean(vs) if vs else None
    print("-" * len(hdr))
    for pf in PROFILES:
        def f(x):
            return f"{x:7.1f}" if x is not None else "    n/a"
        print(f"{pf+' MEAN':14} {'':>4} | {f(avg('TTFT_mape',pf))} {f(avg('TPOT_mape',pf))} {f(avg('E2EL_mape',pf))} |")
    print(f"{'OVERALL':14} {'':>4} | "
          f"{(avg('TTFT_mape') or 0):7.1f} {(avg('TPOT_mape') or 0):7.1f} {(avg('E2EL_mape') or 0):7.1f} |")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", "profile", "conc", "TTFT_mape", "TPOT_mape", "E2EL_mape", "turns"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
