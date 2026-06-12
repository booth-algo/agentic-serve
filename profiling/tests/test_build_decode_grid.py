"""Tests for the decode-grid merge builder (newest-run-wins union of raw run CSVs)."""

from __future__ import annotations

import csv
from pathlib import Path

from profiling.process.build_decode_grid import MERGE_FIELDS, main, merge
from simulator.kernel_step_cost import load_grid


def _write_run(path: Path, rows: list[tuple[int, int, float]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "gpu", "batch_size", "context_len", "observed_context_len",
            "total_kv_tokens", "decode_step_ms", "generated_tokens",
            "decode_intervals", "gpu_ms",
        ])
        w.writeheader()
        for b, t, ms in rows:
            w.writerow({
                "gpu": "H100x2", "batch_size": b, "context_len": t,
                "observed_context_len": t + 1, "total_kv_tokens": b * (t + 128),
                "decode_step_ms": ms, "generated_tokens": 128 * b,
                "decode_intervals": 127 * b, "gpu_ms": round(ms * 127, 4),
            })


def test_newest_run_wins_and_union(tmp_path: Path) -> None:
    old = tmp_path / "run_2026-06-01.csv"
    new = tmp_path / "run_2026-06-10.csv"
    # (1,512) re-measured (warm-up outlier superseded); (4,2048) only in old; (8,1024) only in new
    _write_run(old, [(1, 512, 9.10), (4, 2048, 4.80)])
    _write_run(new, [(1, 512, 4.58), (8, 1024, 4.92)])

    rows = merge([old, new])
    by_key = {(int(r["batch_size"]), int(r["context_len"])): r for r in rows}
    assert set(by_key) == {(1, 512), (4, 2048), (8, 1024)}

    re_measured = by_key[(1, 512)]
    assert float(str(re_measured["decode_step_ms"])) == 4.58       # newest wins
    assert re_measured["source_file"] == new.name
    assert float(str(re_measured["superseded_ms"])) == 9.10        # cross-check kept visible
    assert float(str(re_measured["drift_pct"])) == -49.67

    only_old = by_key[(4, 2048)]
    assert only_old["source_file"] == old.name
    assert only_old["superseded_ms"] == "" and only_old["drift_pct"] == ""


def test_output_sorted_deterministic_and_load_grid_compatible(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_run(a, [(16, 2048, 5.5), (1, 512, 4.6)])
    _write_run(b, [(4, 1024, 4.7), (1, 512, 4.5)])
    out = tmp_path / "merged.csv"
    main(["--inputs", str(a), str(b), "--out", str(out)])

    with out.open() as f:
        r = csv.DictReader(f)
        assert r.fieldnames == MERGE_FIELDS
        keys = [(int(row["batch_size"]), int(row["context_len"])) for row in r]
    assert keys == sorted(keys)  # deterministic cell order

    # the merged artifact is directly consumable by the simulator's grid loader
    g = load_grid(out)
    assert len(g.cells) == 3
    assert g.cells[(1.0, 512.0)] == 4.5  # newest value, not the superseded 4.6
    assert g.fixed_floor_ms == 4.5


def test_missing_input_fails_loudly(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(SystemExit, match="missing raw run CSV"):
        main(["--inputs", str(tmp_path / "nope.csv"),
              "--out", str(tmp_path / "merged.csv")])
