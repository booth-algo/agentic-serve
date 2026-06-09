#!/usr/bin/env python3
"""Analyze serving_stage_split_H100.csv -> the SERVING-LEVEL COST BREAKDOWN of c1 prefill TTFT.

Reads the CSV emitted by ``serving_stage_split.py`` and, for every per-stage column, regresses

    stage_ms ~ FLOOR + (ms/1k-new) * new + (ms/1k-cached) * cached

at concurrency 1, then prints a cost-breakdown table that says which stage OWNS:
  * the NEW dispatch residual (~6 ms/1k, expected in framework-dispatch / prefill_span above roofline)
  * the CACHED host residual (~3.7 ms/1k, expected in frontend_residual = HTTP-parse + chat-template + IPC)

It also runs the de-fit reconciliation gates from prefill_stage_split_results.md:
  * wall ttft  ~ new 29.4 / cached 5.89 ms/1k   (banked live ground truth)
  * prefill_span.new should approach the GEMM roofline ~25 ms/1k (LANE A wall still includes dispatch)
  * frontend_residual.cached should approach the ~3.7 ms/1k serving-stack host residual

If a LANE B device CSV is supplied (--device-csv, columns: new,cached,device_kernel_ms), the script
ALSO computes the framework-dispatch split:
    framework_dispatch_ms = prefill_span_ms - device_kernel_ms
and reports device.new (expect ~25 ms/1k GEMM) vs dispatch.new (expect ~6 ms/1k).

Pure stdlib + numpy (numpy optional -- falls back to a hand-rolled normal-equation solve).

Usage:
    python3 profiling/gpu_profiling/vllm/analyze_serving_stage_split.py \
      --csv profile_data/results/serving_stage_split_H100.csv \
      [--device-csv profile_data/results/serving_stage_device_H100.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Stage columns we regress, in lifecycle order, with a human label and the de-fit interpretation.
# (column name in CSV, pretty label, what its (new,cached) slopes should mean)
_STAGE_COLS = [
    ("ttft_ms",              "wall TTFT (client)",      "GROUND TRUTH; partition target (~29.4 new / 5.89 cached)"),
    ("client_connect_ms",    "1. HTTP recv/connect",    "loopback ~const; tiny size term"),
    ("frontend_residual_ms", "frontend+IPC residual",   "stages 1-5 + 9-11 lumped; owns CACHED host residual ~3.7"),
    ("queue_span_ms",        "6. scheduler-admit/queue", "~0 at c1 (no contention)"),
    ("prefill_span_ms",      "7+8 model-forward WALL",   "device GEMM + dispatch; .new ~ 25 + dispatch residual"),
    ("e2e_span_ms",          "e2e (Prom, cross-check)", "should ~ wall TTFT at max_tokens=1"),
    ("ttft_prom_ms",         "ttft (Prom, cross-check)", "engine TTFT; ~ wall TTFT"),
]


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _col(rows, name):
    """Return (X_design, y) over rows where the column has a numeric value."""
    X, y = [], []
    for r in rows:
        v = r.get(name, "")
        if v is None or v == "":
            continue
        try:
            yv = float(v)
            new = float(r["new"])
            cached = float(r["cached"])
        except (ValueError, KeyError):
            continue
        X.append([1.0, new, cached])
        y.append(yv)
    return X, y


def _ols(X, y):
    """Least squares beta for X (list of [1,new,cached]) and y. numpy if present, else normal eqs."""
    if len(y) < 3:
        return None
    try:
        import numpy as np
        b, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        return [float(c) for c in b]
    except ImportError:
        pass
    # Hand-rolled 3x3 normal equations (X^T X) b = X^T y, Cramer/Gauss elimination.
    n = 3
    XtX = [[0.0] * n for _ in range(n)]
    Xty = [0.0] * n
    for xi, yi in zip(X, y):
        for a in range(n):
            Xty[a] += xi[a] * yi
            for b_ in range(n):
                XtX[a][b_] += xi[a] * xi[b_]
    # Gaussian elimination with partial pivoting on the augmented matrix.
    M = [XtX[i] + [Xty[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        M[col] = [v / pivval for v in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [v - factor * mc for v, mc in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def _fmt_beta(beta):
    if beta is None:
        return ("    n/a", "    n/a", "    n/a")
    floor, per_new, per_cached = beta
    return (f"{floor:7.2f}", f"{per_new * 1000:7.3f}", f"{per_cached * 1000:7.3f}")


def print_breakdown(rows):
    print("\n" + "=" * 96)
    print("SERVING-LEVEL COST BREAKDOWN  (c1)   stage ~ FLOOR + (ms/1k-new)*new + (ms/1k-cached)*cached")
    print("=" * 96)
    print(f"{'stage':<28}{'FLOOR ms':>10}{'new ms/1k':>12}{'cached ms/1k':>14}   interpretation")
    print("-" * 96)
    betas = {}
    for name, label, interp in _STAGE_COLS:
        X, y = _col(rows, name)
        beta = _ols(X, y)
        betas[name] = beta
        floor, pn, pc = _fmt_beta(beta)
        npts = len(y)
        note = interp if npts >= 3 else f"(only {npts} pts -- not fit)"
        print(f"{label:<28}{floor:>10}{pn:>12}{pc:>14}   {note}")
    print("-" * 96)
    return betas


def print_reconciliation(betas):
    print("\n" + "=" * 96)
    print("DE-FIT RECONCILIATION (vs banked truth)")
    print("=" * 96)

    def slopes(name):
        b = betas.get(name)
        return (None, None) if b is None else (b[1] * 1000, b[2] * 1000)

    wall_new, wall_cached = slopes("ttft_ms")
    pre_new, pre_cached = slopes("prefill_span_ms")
    fr_new, fr_cached = slopes("frontend_residual_ms")
    q_new, q_cached = slopes("queue_span_ms")

    def line(msg, val, target, tol_pct=20.0, abs_tol=0.0):
        # abs_tol is an additive floor on the tolerance band so a ~0 target (e.g. queue at c1)
        # doesn't false-flag on sub-ms noise (relative tol of 0 is degenerate).
        if val is None:
            print(f"  [skip] {msg}: not measured (column blank)")
            return
        band = abs(target) * tol_pct / 100.0 + abs_tol
        ok = abs(val - target) <= band
        flag = "OK " if ok else "!! "
        print(f"  [{flag}] {msg}: {val:7.3f} ms/1k  (target ~{target}, band +-{band:.3f})")

    print("Wall TTFT (the partition target):")
    line("    wall.new   reproduces live 29.4", wall_new, 29.4)
    line("    wall.cached reproduces live 5.89", wall_cached, 5.89)

    print("LANE A 3-way partition (wall = frontend_residual + queue + prefill_span):")
    line("    prefill_span.new ~ GEMM roofline 25 + dispatch", pre_new, 31.0, tol_pct=25.0)
    line("    queue.new   ~0 at c1", q_new, 0.0, abs_tol=1.0)
    line("    queue.cached ~0 at c1", q_cached, 0.0, abs_tol=1.0)
    line("    frontend_residual.cached owns ~3.7 host residual", fr_cached, 3.7, tol_pct=40.0)

    # Additivity check: do the three spans sum back to the wall slopes?
    if None not in (wall_new, pre_new, fr_new, q_new):
        sum_new = pre_new + fr_new + q_new
        sum_cached = (pre_cached or 0) + (fr_cached or 0) + (q_cached or 0)
        print("Additivity (frontend_residual + queue + prefill should == wall):")
        print(f"    new   : spans sum {sum_new:7.3f} vs wall {wall_new:7.3f} ms/1k")
        print(f"    cached: spans sum {sum_cached:7.3f} vs wall {wall_cached:7.3f} ms/1k")
        print("    (frontend_residual is DEFINED as wall-queue-prefill, so this is ~exact by"
              " construction -- a non-zero gap flags missing/blank rows.)")


def print_device_split(rows, device_csv: Path):
    """If a LANE B device CSV is present, compute framework_dispatch = prefill_span - device_kernel."""
    print("\n" + "=" * 96)
    print("LANE B DEVICE SPLIT (framework_dispatch = prefill_span - device_kernel)")
    print("=" * 96)
    if not device_csv.exists():
        print(f"  device CSV not found: {device_csv}  -- run the nsys-wrapped LANE B server first")
        print("  (serving_stage_split.py --emit-nsys-cmd prints the exact command).")
        return
    # device CSV: rows keyed by (new,cached) with column device_kernel_ms (Sigma end-start in NVTX window).
    dev = {}
    with device_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (int(float(r["new"])), int(float(r["cached"])))
                dev[key] = float(r["device_kernel_ms"])
            except (KeyError, ValueError):
                continue
    # join to prefill_span from the host CSV; framework_dispatch = prefill_span - device.
    X, y_dev, y_disp = [], [], []
    for r in rows:
        try:
            key = (int(float(r["new"])), int(float(r["cached"])))
            prefill = float(r["prefill_span_ms"]) if r.get("prefill_span_ms") not in (None, "") else None
        except (KeyError, ValueError):
            continue
        if key not in dev or prefill is None:
            continue
        X.append([1.0, float(key[0]), float(key[1])])
        y_dev.append(dev[key])
        y_disp.append(prefill - dev[key])
    if len(y_dev) < 3:
        print("  not enough joined (new,cached) points to regress the device split.")
        return
    bd = _ols(X, y_dev)
    bp = _ols(X, y_disp)
    f1, n1, c1 = _fmt_beta(bd)
    f2, n2, c2 = _fmt_beta(bp)
    print(f"  device_kernel    FLOOR={f1} ms | new={n1} ms/1k | cached={c1} ms/1k   (expect new~25 GEMM, cached~1.5 paged-attn)")
    print(f"  framework_dispatch FLOOR={f2} ms | new={n2} ms/1k | cached={c2} ms/1k   (expect new~6 dispatch residual)")
    if bd and bp:
        print(f"\n  => NEW prefill_span {bd[1]*1000 + bp[1]*1000:7.3f} ms/1k splits as "
              f"DEVICE {bd[1]*1000:.2f} + DISPATCH {bp[1]*1000:.2f} ms/1k.")
        print("  This is the device-vs-host split the offline torch.profiler attempt could NOT do.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="profile_data/results/serving_stage_split_H100.csv")
    ap.add_argument("--device-csv", default="profile_data/results/serving_stage_device_H100.csv",
                    help="optional LANE B device CSV (new,cached,device_kernel_ms) from the nsys run")
    a = ap.parse_args()

    path = Path(a.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun serving_stage_split.py first.")
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f"CSV empty: {path}")
    print(f"loaded {len(rows)} (new,cached) rows from {path}")

    betas = print_breakdown(rows)
    print_reconciliation(betas)
    print_device_split(rows, Path(a.device_csv))

    print("\n" + "=" * 96)
    print("FEED-BACK (how these slopes de-fit the constants):")
    print("  * PREFILL_NEW_DISPATCH_RESIDUAL  <- framework_dispatch.new (LANE B), else")
    print("    prefill_span.new - 25 (roofline) as an upper bound.")
    print("  * PREFILL_HOST_SHARED / PERREQ    <- frontend_residual.cached, split via the live")
    print("    concurrency B-sweep (live_split_probe.py -> 50/50). c1 gives only the SUM here.")
    print("  * tokenize 1.33 / IPC 0.7 / GEMM 25 / paged-attn 1.5 are already de-fitted constants;")
    print("    they live INSIDE frontend_residual (tokenize, IPC) and prefill_span (GEMM, paged-attn).")
    print("=" * 96)


if __name__ == "__main__":
    main()
