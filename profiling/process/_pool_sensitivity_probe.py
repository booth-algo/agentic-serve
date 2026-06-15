"""Offline KV-pool sensitivity probe for one cell (no GPU).

Varies available_kv_blocks for a single deployment and reports overall E2EL/TTFT/TPOT
cell-MAPE vs ground truth. Used to find the pool magnitude that minimizes a cell's error
and to test whether the pool is the dominant lever (vs needing a measured decode grid).

Usage: python3 -m profiling.process._pool_sensitivity_probe <bench_dir> [pool1 pool2 ...]
"""
from __future__ import annotations

import dataclasses
import os
import statistics as st
import sys
from pathlib import Path

import profiling.process.build_simulator_rows as B
from configs.loader import all_deployments
from simulator import kernel_step_cost, kernel_tpot


def main() -> None:
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "3090_Qwen3.5-9B_tp1_vllm"
    pools = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [
        198, 256, 512, 1000, 1500, 2000, 2560, 3168, 3643, 5000, 8000, 12000]

    cell = next(c for c in all_deployments() if c.bench_dir == bench_dir)

    # Replicate main()'s per-config grid/ceiling setup so the probe is faithful.
    # DECODE_GRID_OVERRIDE lets us swap in a freshly-measured grid CSV before it's wired into a
    # deployment manifest (validation). CEILING_OVERRIDE similarly swaps the saturated-ITL ceiling.
    grid_override = os.environ.get("DECODE_GRID_OVERRIDE")
    if grid_override:
        grid = B.load_grid(Path(grid_override))
        kernel_step_cost._default_grid = lambda grid=grid: grid
        gridnote = f"OVERRIDE {Path(grid_override).name} ({len(grid.cells)} cells)"
    elif cell.decode_grid is not None and cell.decode_grid.exists():
        grid = B.load_grid(cell.decode_grid)
        kernel_step_cost._default_grid = lambda grid=grid: grid
        gridnote = f"measured {cell.decode_grid.name}"
    else:
        agrid = kernel_step_cost.analytic_grid()
        kernel_step_cost._default_grid = lambda agrid=agrid: agrid
        gridnote = "analytic roofline"
    if cell.saturated_ceiling is not None and cell.saturated_ceiling.exists():
        kernel_tpot._active_ceiling_json = cell.saturated_ceiling
        ceilnote = cell.saturated_ceiling.name
    else:
        ceilnote = "default (H100-inherited)"

    bench_root = B.BENCH_BASE / cell.bench_dir
    print(f"cell: {bench_dir}  model={cell.model}  gpu_key={cell.gpu_key}  tp={cell.tp}")
    print(f"  decode_grid: {gridnote}   ceiling: {ceilnote}")
    print(f"  current available_kv_blocks: {cell.roofline.available_kv_blocks}")
    print(f"  kv_bytes_per_token={cell.roofline.kv_bytes_per_token} kv_heads={cell.roofline.kv_heads}")
    print()

    def eval_pool(pool: int):
        params = dataclasses.replace(cell.roofline, available_kv_blocks=pool)
        rows = []
        for prof in B.PROFILES:
            for conc in B.CONCURRENCIES:
                r = B.build_row(prof, conc, params, cell, bench_root)
                if r:
                    rows.append(r)

        def ov(k: str):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            return round(st.mean(vals), 2) if vals else None

        return ov("e2el_err"), ov("ttft_err"), ov("tpot_err"), len(rows)

    print(f"{'pool':>8} {'e2el%':>8} {'ttft%':>8} {'tpot%':>8} {'n':>4}")
    best = None
    for pool in pools:
        e, t, tp, n = eval_pool(pool)
        mark = ""
        if e is not None and (best is None or e < best[1]):
            best = (pool, e)
            mark = "  <-"
        print(f"{pool:>8} {e!s:>8} {t!s:>8} {tp!s:>8} {n:>4}{mark}")
    if best:
        print(f"\nbest E2EL: pool={best[0]} -> {best[1]}%")


if __name__ == "__main__":
    main()
