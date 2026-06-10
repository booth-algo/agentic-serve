"""Tests for the measured decode-step kernel grid lookup."""

from __future__ import annotations

import statistics as st
from dataclasses import replace
from pathlib import Path

from simulator.closed_form_tpot import RooflineParams
from simulator.kernel_step_cost import (
    A100_GRID_CSV,
    A100_ROOFLINE_JSON,
    H100X2_GRID_CSV,
    _default_grid,
    _launch_floor_classes,
    analytic_grid,
    decode_step_ms,
    default_launch_floor_ms,
    launch_floor_for,
    load_grid,
)

TP2_OLD_CSV = Path("profile_data/results/decode_profile_H100x2_2026-06-01.csv")


def test_grid_loads_with_expected_axes() -> None:
    g = _default_grid()
    assert g.b_axis[0] == 1.0
    assert g.t_axis[0] == 512.0
    assert 256.0 in g.b_axis
    assert 16384.0 in g.t_axis
    # measured small-batch floor is ~6.5 ms
    assert 6.0 < g.fixed_floor_ms < 7.0


def test_fixed_floor_is_robust_to_warmup_outlier(tmp_path: Path) -> None:
    # fixed_floor anchors the analytic decode roofline for cells beyond the measured grid; it must be
    # the MIN over the smallest-batch row, not the single (B_min, T_min) cell, so a one-off warm-up
    # overhead in that cell does not inflate every analytic-fill cell (the H100x2 B=1,T=512=9.1 vs the
    # row's true 4.7 floor was the dominant tp2 high-conc TPOT over-prediction).
    csv = tmp_path / "grid.csv"
    csv.write_text(
        "batch_size,context_len,decode_step_ms,validation_status\n"
        "1,512,9.10,ok\n"      # warm-up outlier at (B_min, T_min)
        "1,2048,4.68,ok\n"     # the real small-batch floor
        "1,8192,5.49,ok\n"
        "256,512,10.5,ok\n"
        "256,2048,16.25,ok\n"
    )
    g = load_grid(csv)
    assert g.fixed_floor_ms == 4.68          # row minimum, NOT the 9.10 corner
    # well-behaved grid (corner already the row min) is unaffected
    csv2 = tmp_path / "grid2.csv"
    csv2.write_text(
        "batch_size,context_len,decode_step_ms,validation_status\n"
        "1,512,6.55,ok\n1,2048,6.60,ok\n256,512,10.0,ok\n"
    )
    assert load_grid(csv2).fixed_floor_ms == 6.55


def test_lookup_matches_grid_at_measured_points() -> None:
    g = _default_grid()
    # exact grid points should round-trip (interp weight collapses to the node)
    assert abs(decode_step_ms(1, 512) - g.cells[(1.0, 512.0)]) < 1e-6
    assert abs(decode_step_ms(128, 2048) - g.cells[(128.0, 2048.0)]) < 1e-6


def test_step_time_rises_with_batch_and_context() -> None:
    # bandwidth-bound region: more batch / context => longer step
    assert decode_step_ms(128, 2048) > decode_step_ms(1, 512)
    assert decode_step_ms(64, 8192) > decode_step_ms(64, 1024)


def test_uncovered_corner_fill_is_monotone() -> None:
    """High-B × high-T cells are absent from the tp1 grid (skipped by the
    sweep's 500k KV-token cap — NOT OOM; audit-v2 S12); the beyond-hull
    measured-scaling fill must keep the surface monotone and finite.
    """
    v = decode_step_ms(256, 16384)  # far beyond the measured hull
    assert v > decode_step_ms(256, 1024)
    assert v > decode_step_ms(1, 16384)
    assert v < 1000.0  # sane magnitude, not a blow-up


def test_floor_holds_at_tiny_workload() -> None:
    # batch=1, short ctx => essentially the fixed floor, never below it
    g = _default_grid()
    assert decode_step_ms(1, 512) >= g.fixed_floor_ms - 1e-6


def test_clamps_below_axis_minimum() -> None:
    # sub-1 batch / sub-min context clamp rather than extrapolate downward
    assert decode_step_ms(0.1, 1.0) == decode_step_ms(1, 512)


def test_analytic_fill_continuous_at_boundary() -> None:
    """The analytic fill should be close to the measured value at the last
    covered cell, so the measured→analytic handoff has no large step.
    """
    g = _default_grid()
    p = RooflineParams()
    measured = g.cells[(128.0, 2048.0)]
    analytic = g._analytic(128.0, 2048.0, p)
    assert abs(measured - analytic) / measured < 0.20


def test_load_grid_skips_capped_and_check_cells() -> None:
    g = load_grid()
    # (256, 4096) was skipped by the sweep's KV-footprint cap (B·(T+128) ≤ 500k)
    # and must be absent (it is then beyond-hull filled by lookup)
    assert (256.0, 4096.0) not in g.cells
    # but a low-corner cell is present
    assert (1.0, 512.0) in g.cells


# ------------------------------------------------- tp2 merged grid (de-fit 2026-06-10, L3)


def test_h100x2_merged_grid_loads() -> None:
    """The 54-cell merged tp2 grid: dense B×T rectangle + T=24576 tail up to the
    real 998,656-token KV pool; floor 4.40 ms (warm-up-outlier-robust row min).
    """
    g = load_grid(H100X2_GRID_CSV)
    assert len(g.cells) == 54
    assert g.b_axis == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    assert 24576.0 in g.t_axis
    assert abs(g.fixed_floor_ms - 4.4045) < 1e-6
    # the re-measured (1, 512) superseded the 9.10 ms warm-up outlier
    assert abs(g.cells[(1.0, 512.0)] - 4.5843) < 1e-6


def test_beyond_hull_fill_is_sublinear_vs_linear_roofline() -> None:
    """Beyond the measured hull the fill must price BELOW (or at) the retired
    linear-in-b·ctx analytic fill — the 2026-06-10 measurement showed the real
    kernel is sub-linear there (linear fill over-priced 1.10–1.24× at B·T≥200k).
    """
    g = load_grid(H100X2_GRID_CSV)
    p2 = replace(RooflineParams(), tensor_parallel=2)
    for b, t in [(64.0, 16384.0), (128.0, 8192.0), (256.0, 4096.0)]:
        assert (b, t) not in g.cells  # beyond the KV-pool hull
        fill = g._beyond_hull(b, t, p2)
        assert fill is not None
        assert fill <= g._analytic(b, t, p2) + 1e-9
        # monotone: never below the measured edges it extrapolates from
        row_edge = max(tt for bb, tt in g.cells if bb == b)
        assert fill >= g.cells[(b, row_edge)] - 1e-9


def test_interior_holes_keep_analytic_fill(tmp_path: Path) -> None:
    """Interior holes (bracketed by measurements — the S12 dropped 'check'
    rows) keep the S12-validated analytic fill; _beyond_hull must decline them.
    """
    csv = tmp_path / "grid.csv"
    csv.write_text(
        "batch_size,context_len,decode_step_ms,validation_status\n"
        "1,512,6.55,ok\n1,2048,6.80,ok\n1,8192,7.50,ok\n"
        "4,512,6.90,ok\n4,8192,9.50,ok\n"     # (4, 2048) is an interior hole
        "16,512,7.50,ok\n16,2048,9.00,ok\n"   # (16, 8192) is beyond the hull
    )
    g = load_grid(csv)
    p = RooflineParams()
    assert g._beyond_hull(4.0, 2048.0, p) is None          # interior -> analytic
    assert g._beyond_hull(16.0, 8192.0, p) is not None     # beyond both edges


def test_extrapolation_capped_at_linear_roofline_increment(tmp_path: Path) -> None:
    """A measured near-linear slope must NOT scale the whole step time
    proportionally (that over-shoots the roofline by the fixed-floor share):
    the extrapolant is capped at v_edge + linear KV increment.
    """
    csv = tmp_path / "grid.csv"
    p = RooflineParams()
    kv_ms = p.kv_bytes_per_token / (p.peak_bw_bytes_per_s * p.util_bw) * 1e3
    # T=8192 column doubles with B (proportional incl. the floor share = a
    # log-log slope of exactly 1); the (32, 8192) node is beyond the hull in
    # both axes (row 32 ends at T=1024, column 8192 ends at B=16).
    rows = ["batch_size,context_len,decode_step_ms,validation_status"]
    for b in (1, 2, 4, 8, 16):
        rows.append(f"{b},512,{6.5 + 0.1 * b:.6f},ok")
        rows.append(f"{b},8192,{3.0 * b:.6f},ok")
    rows.append("32,512,9.70,ok")
    rows.append("32,1024,9.80,ok")
    csv.write_text("\n".join(rows) + "\n")
    g = load_grid(csv)
    v = g._beyond_hull(32.0, 8192.0, p)
    assert v is not None
    # proportional scaling would price 2 * 48 = 96 ms — over-shooting the
    # roofline by the non-KV share; the cap pins the increment to the linear
    # KV-read term instead.
    cap = g.cells[(16.0, 8192.0)] + (32.0 - 16.0) * 8192.0 * kv_ms
    assert abs(v - cap) < 1e-6
    assert v < 2.0 * g.cells[(16.0, 8192.0)]


def test_beyond_axis_extrapolation_is_continuous_at_edge() -> None:
    g = load_grid(H100X2_GRID_CSV)
    p2 = replace(RooflineParams(), tensor_parallel=2)
    edge = g.lookup(8.0, 24576.0, p2)
    just_past = g.lookup(8.0, 24600.0, p2)
    assert abs(just_past - edge) / edge < 0.01
    assert just_past >= edge - 1e-9  # monotone past the axis


def test_sublinear_fill_beats_linear_on_heldout_tp2_cells() -> None:
    """Regression pin of the de-fit validation: the 19-cell 2026-06-01 grid +
    the measured-scaling extrapolation, evaluated at the 35 newly measured
    2026-06-10 cells, must over-price LESS than the retired linear fill
    (documented: linear median 1.084 / B·T≥200k median 1.098, worst 1.241;
    sub-linear 1.067 / 1.049, worst 1.111) and introduce no under-pricing.
    """
    g_old = load_grid(TP2_OLD_CSV)
    g_new = load_grid(H100X2_GRID_CSV)
    p2 = replace(RooflineParams(), tensor_parallel=2)
    ratios, big = [], []
    for (b, t), meas in g_new.cells.items():
        if (b, t) in g_old.cells:
            continue
        r = g_old.lookup(b, t, p2) / meas
        ratios.append(r)
        if b * t >= 200_000:
            big.append(r)
    assert len(ratios) == 35
    assert st.median(ratios) < 1.08    # linear scored 1.084
    assert st.median(big) < 1.07       # linear scored 1.098
    assert max(big) < 1.15             # linear's worst cell was 1.241
    assert min(ratios) > 0.90          # no under-pricing introduced (linear: 0.930)


# ------------------------------------------------- per-config launch floors (audit-v2 G5)


def test_launch_floor_classes_match_measured_grids() -> None:
    """One residual per measured grid: H100 tp1 ~1.37, H100x2 tp2 ~1.82 (from
    the re-measured 4.40 ms floor), A100 tp1 ~2.06 — 'config-independent' is
    contradicted by measurement. Ranges are loose where inputs may be
    re-anchored (the A100 utils are calibrated placeholders).
    """
    classes = {c.name: c for c in _launch_floor_classes()}
    assert set(classes) == {"H100", "A100", "H100x2"}
    assert abs(classes["H100"].residual_ms - default_launch_floor_ms()) < 1e-12
    assert 1.30 < classes["H100"].residual_ms < 1.45
    assert 1.70 < classes["H100x2"].residual_ms < 1.95
    assert 1.0 < classes["A100"].residual_ms < 3.0
    # the tp2 collective launch makes the floor strictly larger than tp1 H100
    assert classes["H100x2"].residual_ms > classes["H100"].residual_ms


def test_launch_floor_resolution_picks_nearest_measured_class() -> None:
    classes = {c.name: c.residual_ms for c in _launch_floor_classes()}
    pa = RooflineParams.from_json(A100_ROOFLINE_JSON)
    p2 = replace(RooflineParams(), tensor_parallel=2)
    assert launch_floor_for(RooflineParams()) == classes["H100"]
    assert launch_floor_for(p2) == classes["H100x2"]
    assert launch_floor_for(pa) == classes["A100"]
    # tp regime is matched first: any tp>=2 config takes the measured tp2 class
    assert launch_floor_for(replace(pa, tensor_parallel=4)) == classes["H100x2"]
    # an unseen single-GPU bandwidth resolves to the nearest measured class
    slow = replace(RooflineParams(), peak_bw_bytes_per_s=0.9e12)
    assert launch_floor_for(slow) == classes["A100"]


def test_analytic_grid_resolves_launch_floor_per_params() -> None:
    """analytic_grid() (no explicit floor) prices an analytic config with the
    nearest measured class's residual; an explicit floor pins it (back-compat).
    """
    pa = RooflineParams.from_json(A100_ROOFLINE_JSON)
    classes = {c.name: c.residual_ms for c in _launch_floor_classes()}
    dynamic = analytic_grid().lookup(1, 512, pa)
    pinned_a100 = analytic_grid(classes["A100"]).lookup(1, 512, pa)
    pinned_h100 = analytic_grid(classes["H100"]).lookup(1, 512, pa)
    assert abs(dynamic - pinned_a100) < 1e-9
    assert abs(dynamic - pinned_h100 - (classes["A100"] - classes["H100"])) < 1e-9


def test_grid_csvs_exist() -> None:
    # the class derivation + manifest wiring depend on these artifacts
    assert H100X2_GRID_CSV.exists()
    assert A100_GRID_CSV.exists()
    assert A100_ROOFLINE_JSON.exists()
    assert TP2_OLD_CSV.exists()
