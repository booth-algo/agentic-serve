"""Deterministic-side tests for the serving-context decode grid tool (L11).

Covers the PRE-REGISTERED arithmetic that must not drift: the 26-cell tp4 lattice and its
prompt/osl fixed point + pool cap, the steady-window / per-request-p50 summary, the
validation flags, and the builder's raw->CSV merge (latest run wins per cell).
The live-SSE side runs only on the GPU host (aiohttp lazily imported there).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from profiling.process.build_serving_decode_grid import (  # noqa: E402
    OUT_FIELDS, _client, build,
)

solve_cell = _client.solve_cell
lattice_cells = _client.lattice_cells
summarize_cell = _client.summarize_cell
LATTICE_TP4 = _client.LATTICE_TP4
KV_POOL = _client.KV_POOL_TOKENS_TP4


def test_lattice_is_the_26_preregistered_cells() -> None:
    cells = lattice_cells()
    assert len(cells) == 26
    keys = {(c["batch_size"], c["nominal_T"]) for c in cells}
    assert {(b, t) for b in (1, 8, 32, 80) for t in (512, 2048, 8192, 16384)} <= keys
    assert {(160, t) for t in (512, 2048, 8192, 12288)} <= keys
    assert {(256, 512), (256, 2048), (256, 6144)} <= keys
    assert {(320, 512), (320, 2048), (320, 4096)} <= keys


def test_lattice_cap_rule() -> None:
    # every cell within 0.95*pool except the directive-named (160, 12288) at 95.2%
    for c in lattice_cells():
        if (c["batch_size"], c["nominal_T"]) == (160, 12288):
            assert c["exceeds_cap"] and c["kv_frac_of_pool"] < 0.96
            assert c["kv_final_tokens"] < KV_POOL  # still inside the LIVE pool: no preemption
        else:
            assert not c["exceeds_cap"], c


def test_solve_cell_fixed_point() -> None:
    for b in LATTICE_TP4:
        for t in LATTICE_TP4[b]:
            prompt, osl = solve_cell(b, t)
            assert prompt == t - osl // 2
            assert osl == 384 + math.ceil(b * prompt / 8192)
    prompt, osl = solve_cell(1, 512)
    assert osl == 385 and prompt == 512 - 385 // 2


def _mk_record(req: int, first_wall: float, n_events: int, itl_ms: float,
               prompt: int = 1820, osl: int = 456, nominal_t: int = 2048,
               lag: float = 0.1) -> dict:
    return {"req": req, "shard": 0, "prompt_tokens": prompt, "osl": osl,
            "nominal_T": nominal_t, "t_first_wall": first_wall,
            "deltas_ms": [itl_ms] * (n_events - 1), "n_events": n_events,
            "lag_p99_ms": lag}


def test_summarize_cell_constant_itl() -> None:
    # 4 requests, 7 ms ITL, staggered ramp: request i emits its first token at i*0.1 s.
    recs = [_mk_record(i, first_wall=100.0 + 0.1 * i, n_events=456, itl_ms=7.0)
            for i in range(4)]
    row = summarize_cell(recs)
    assert row["batch_size"] == 4
    assert row["decode_step_ms"] == 7.0
    assert row["validation_status"] == "ok"
    assert row["n_samples"] >= 64
    # effective context = prompt + median in-window progress, near nominal by construction
    assert abs(row["context_len"] - (1820 + row["median_inwindow_progress"])) <= 1
    assert row["nominal_T"] == 2048 and row["osl"] == 456


def test_summarize_cell_flags() -> None:
    # too few in-window samples -> check
    short = [_mk_record(i, 100.0, n_events=40, itl_ms=7.0) for i in range(2)]
    assert summarize_cell(short)["validation_status"] == "check"
    # loop lag p99 over 2 ms -> check
    lag = [_mk_record(i, 100.0, n_events=456, itl_ms=7.0, lag=3.5) for i in range(2)]
    assert summarize_cell(lag)["validation_status"] == "check"
    # disjoint steady window (one request ends before another starts) -> check
    disjoint = [_mk_record(0, 100.0, n_events=100, itl_ms=1.0),
                _mk_record(1, 300.0, n_events=100, itl_ms=1.0)]
    assert summarize_cell(disjoint)["validation_status"] == "check"


def test_builder_merge_latest_wins(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.jsonl.gz"
    run2 = tmp_path / "run2.jsonl.gz"
    cell = [4, 2048]
    with gzip.open(run1, "wt") as f:
        f.write(json.dumps({"_meta": True, "tool": "t"}) + "\n")
        for i in range(4):
            f.write(json.dumps({"cell": cell, **_mk_record(i, 100.0 + 0.1 * i, 456, 9.0)}) + "\n")
    with gzip.open(run2, "wt") as f:
        for i in range(4):
            f.write(json.dumps({"cell": cell, **_mk_record(i, 100.0 + 0.1 * i, 456, 7.0)}) + "\n")
    rows = build([run1, run2])
    assert len(rows) == 1
    assert rows[0]["decode_step_ms"] == 7.0          # run2 (latest) wins
    assert rows[0]["source_file"] == "run2.jsonl.gz"


def test_builder_csv_is_load_grid_compatible(tmp_path: Path) -> None:
    run = tmp_path / "run.jsonl.gz"
    with gzip.open(run, "wt") as f:
        for b, t, itl in ((1, 512, 3.0), (8, 512, 3.4)):
            prompt, osl = solve_cell(b, t)
            for i in range(b):
                f.write(json.dumps({"cell": [b, t], **_mk_record(
                    i, 100.0 + 0.05 * i, osl, itl, prompt=prompt, osl=osl, nominal_t=t)}) + "\n")
    out = tmp_path / "grid.csv"
    subprocess.run([sys.executable, "-m", "profiling.process.build_serving_decode_grid",
                    "--inputs", str(run), "--out", str(out)],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["batch_size"] for r in rows] == ["1", "8"]
    assert all(set(OUT_FIELDS) <= set(r.keys()) for r in rows)

    from simulator.kernel_step_cost import load_grid
    g = load_grid(out)
    assert len(g.cells) == 2 and g.fixed_floor_ms == 3.0
