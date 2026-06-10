# L3 — tp2 decode sub-linearity (lane-local de-fit entries)

Merged into `prediction_construction.md`'s De-fit log by the integration phase. Lane branch
`tp2-sublinearity`, worktree `/root/agentic-serve-tp2`. Pre-registered measurement plan executed
2026-06-10 on h100 GPUs 6+7 (run dir `/data48/kevinlau/tp2grid_run/`; GPUs left clean).

- **2026-06-10 — tp2 decode grid: 19 sparse cells → 54-cell dense rectangle + long-ctx tail
  (MEASURED; the linear-in-`b·ctx` analytic fill no longer prices any reachable H100x2 decode
  state)** (`kernel_step_cost` decode grids; campaign L3, the deferred tp2 fix).
  THE PROBLEM (three independent evidence lines, pre-registered): beyond the sparse 2026-06-01
  tp2 grid edge the analytic decode fill prices `b·ctx` LINEARLY while the real kernel is
  SUB-linear; the floor-excess measurement found the grid OVER-predicting at low batch (raw
  median −0.73 ms, negative batch trend); H100x2 TPOT cell-MAPE ~28.7 vs ~14.5 tp1.
  **RUN 1 (tp2, GPUs 6+7, `decode_steps.py --tensor-parallel-size 2 --gpu-memory-utilization
  0.90 --max-model-len 25600 --max-total-kv-tokens 998656`):** 54 cells —
  B∈{1..256}×T∈{512..16384} dense rectangle + T=24576 tail, feasibility cap = the REAL H100x2
  KV pool (62,416 blocks × 16 = 998,656 tokens), so every reachable serving state is now inside
  the measured hull → `decode_profile_H100x2_2026-06-10_main.csv` (raw, append-only).
  **Session-drift cross-check** (all 19 old cells re-measured, same script): median −0.85%,
  mean −1.26%, max |13.8|% — the two sessions agree; EXCEPT the known (B=1,T=512) warm-up
  outlier 9.10→4.58 ms (−49.6%), now superseded by re-measurement. That outlier was the
  documented driver of the low-batch floor-excess over-prediction (the −0.73 ms raw median):
  it inflated every interpolation touching the (1,512) corner.
  **Measured sub-linearity vs the old pipeline surface** (old 19-cell grid + linear analytic
  fill, evaluated at the 35 newly-covered cells with the H100x2 RooflineParams): median
  over-price **1.084×**, mean 1.110×; in the big-`B·T` region the 2.29× KV pool actually
  lives in (B·T ≥ 200k): median **1.098×**, worst **1.241×** (B=32,T=16384: 18.40 pred vs
  14.83 ms meas; B=128,T=4096: 1.235×) — matching the pre-registered ~1.25–1.30× estimate.
  Marginal KV-read cost measured vs the fill's constant 21.0 ms per 1M `b·ctx` tokens
  (kv/shard 65,536 B ÷ 3.116e12 B/s): ~23 ms/1M at T=2048 falling to **~17–22 (T=8192) and
  ~11–17 (T=16384)** — the kernel is genuinely SUB-linear in `b·ctx`, ~25–40% below linear at
  long context; small-batch increments are even negative (B=1→2 at T≥8192 — single-sequence
  FA partitioning inefficiency at B=1). No constant was tuned: the fix is measurement.
  **Merge artifact:** new deterministic builder `profiling/process/build_decode_grid.py`
  (union of dated raw run CSVs, NEWEST RUN WINS per cell, per-cell `source_file` +
  `superseded_ms`/`drift_pct` provenance; tests `profiling/tests/test_build_decode_grid.py`)
  → `decode_profile_H100x2_merged_2026-06-10.csv` (54 cells, floor 4.40 ms). Raw CSVs stay
  append-only.
  **WIRED IN (Round 1; the orchestrator allowed the H100x2 `decode_grid` manifest entry):**
  `configs/deployments/h100x2_llama31-8b_tp2.json` `data.decode_grid.path` →
  `profile_data/results/decode_profile_H100x2_merged_2026-06-10.csv` (that one entry only;
  source/date provenance added).
  **Gate (replay-ON; lane baseline captured pre-edit; candidate = merged grid via
  module-attribute patch, no source/config edits): PASS, success target EXCEEDED** —
  H100x2 TPOT cell-MAPE **28.7486 → 23.1829 (−5.57 pt**; target ≥3 pt from ~28.7**)**,
  turn-overall 29.25→24.34, E2EL cell 21.83→**19.24**, TTFT cell unchanged (29.0163);
  per-profile TPOT: chat 17.24→**9.72**, swebench 31.41→27.99, osworld 25.16→21.47,
  terminalbench 42.03→35.42 (all improve; plateau-osworld +1.0 is the known S3 ×z
  over-fire, advisory). **H100 and A100 predictions verified BYTE-IDENTICAL** (JSON
  compare of the full per-turn payloads) — the binding H100/A100 ≤ +0.3 gates pass
  trivially; 189 tests + 12 subtests green.
- **2026-06-10 — S12 closed: tp1 'check'-row drop RE-MEASURED and RETAINED; the false "OOM"
  docstring fixed** (`kernel_step_cost.load_grid`; audit-v2 S12; tp1 behavior byte-identical).
  The tp1 grid docstring claimed absent high-B×high-T cells "OOM on a single H100" — false:
  they were skipped by the sweep's 500k KV-token cap. And `load_grid` silently drops 7
  measured interior cells flagged `validation_status='check'` by an off-repo bucketing
  threshold. **RUN 2 (tp1, GPU 6 alone, sequential after RUN 1):** event-timed wall
  re-measurement of the 7 cells + 15 'ok' neighbours (`decode_steps.py`, 22 cells →
  `decode_profile_H100_2026-06-10_s12recheck.csv`). Findings: (a) the old 'check' values sat
  in interior dips BELOW their own shorter-context neighbours (e.g. B=16 row: 6.77 @T=2048 vs
  6.47 @T=1024 then 10.12 @T=4096); the re-measured surface is monotone in T — the dips do
  not reproduce; (b) re-measured walls sit +12% to +35% above the dropped values, vs only
  +8–16% on neighbouring 'ok' floor cells (a methodology offset: `decode_steps` walls include
  host gaps; the 2026-05-17 numbers came from the trace tooling) — i.e. the 'check' cells
  moved ~10pt MORE than their neighbours: they were under-measured; (c) the analytic fill that
  replaced them (+6.5–22.9% above the dropped values, audit S12) lands on the re-measured
  monotone surface modulo that offset. **Decision (honest stop-point): keep the drop** —
  restoring the dropped values would re-introduce non-physical dips; replacing the whole tp1
  grid with `decode_steps` walls is a methodology change out of L3 scope (tp1 contract:
  byte-identical-or-better). Docstrings fixed (module + `load_grid`); no behavior change,
  tp1 predictions byte-identical by construction.
- **2026-06-10 — G5 ADOPTED (Round 1, per the orchestrated implementation instruction —
  supersedes the same-day "evidence logged, no adoption" stop-point): per-config launch-floor
  residuals from the measured grids.** With the merged tp2 grid the H100x2 measured floor is
  **4.40 ms** (was 4.68; the warm-up-outlier-robust row-min). Same derivation as
  `default_launch_floor_ms` (floor − weight-read/tp − min-KV-read, each grid evaluated with
  the params of the config it was measured on; A100 params read from
  `roofline_params_A100_llama31_8b.json` at runtime so a future util re-anchor flows
  through): **H100 tp1 1.3697 / H100x2 tp2 1.8163 / A100 tp1 2.0643 ms** — audit G5's
  "config-independent" single residual stays contradicted (the audit's ~2.09 H100x2 estimate
  came from the old 4.68 floor; ~2.06 A100 confirmed exactly). `kernel_step_cost` now carries
  the class table (`_launch_floor_classes`) and `analytic_grid()` (no explicit floor)
  resolves PER QUERY to the nearest measured class (`launch_floor_for`): tp regime first
  (the tp collective launch is part of the floor), then nearest log peak-HBM-bandwidth (GPU
  generation). Affects ONLY analytic-only (uncalibrated, advisory) configs — the three gated
  configs all use measured grids and never touch the analytic launch floor, so the gate is
  structurally unmoved by this item; explicit `analytic_grid(floor)` stays available and
  byte-identical for pinning.
- **2026-06-10 — Round 1 IMPLEMENTED in `kernel_step_cost.py`: merged tp2 grid wired in +
  beyond-hull analytic fill made SUB-linear (measured-scaling extrapolation, no free
  parameters). Gate PASS, H100x2 TPOT cell-MAPE 28.7486 → 21.5336 (−7.22 pt; target ≥3).**
  Mechanism: absent grid cells are now split. INTERIOR holes (bracketed by measurements in
  either axis — the S12 dropped 'check' rows and sweep gaps) keep the linear analytic fill,
  which the S12 re-measurement validated. Cells BEYOND the measured hull (past the row's
  last T *and* the column's last B) and queries past the outer axes use the grid's OWN
  measured scaling: per-axis power law with exponent = log-log slope of the last two
  measured cells, clamped to [0,1] (monotonicity), CAPPED at the linear roofline increment
  above the measured edge (`v_edge + Δ(B·T·kv/bw)`; the power law scales the whole step
  time, but the fixed launch/weight share must not grow with B·T), max of the row/col
  extrapolants. Every number is a measured-cell ratio or the roofline — nothing tuned.
  **Pre-registered hold-out validation** (old 19-cell grid extrapolated to the 35 newly
  measured cells, H100x2 params): linear fill median **1.084×** / B·T≥200k median 1.098× /
  worst 1.241× → sub-linear median **1.067×** / 1.049× / worst **1.111×**, min 0.925 (linear
  0.930 — no new under-pricing). The max combination is the only variant improving every
  statistic without under-pricing (geo/min: 0.85/0.77 worst); the roofline cap bounds
  reachable tp1-state movement (KV pool 436k tokens) to ≲11%.
  **Gate (replay-ON, baseline re-captured pre-edit in this worktree and matching the
  measure-phase numbers):** H100x2 TPOT cell **28.7486 → 21.5336** (grid swap alone gave
  23.18 — the sub-linear fill adds another −1.65 pt), e2el_cell 21.83 → **18.55**,
  turn-overall 29.25 → 22.35, TTFT unchanged (29.0163); per-profile TPOT chat 17.24 → 9.80,
  swebench 31.41 → 26.24, osworld 25.16 → 17.21, terminalbench 42.03 → 34.41 (plateau
  osworld +2.3 / terminalbench +1.3 are the known S3 ×z over-fire region, advisory).
  Binding H100/A100 gates all PASS: H100 tpot_cell 14.3182 → 14.4697 (+0.15 ≤ +0.3; its
  beyond-hull corners are re-filled by the same measured-scaling rule — NOT byte-identical;
  honest note: chat −0.04, ttft −0.002, swe-plateau −0.007 improve, e2el +0.02), A100
  tpot_cell 14.3417 → 14.3728 (+0.03), ttft +0.0001, e2el +0.002, chat +0.05. S12
  leftovers fixed in the same pass (the dataclass "(B=1, T=min)" floor comment, the
  remaining "OOM" wording in `_analytic`/`_decode_roofline_full`/`lookup`/test names).
  Tests: 199 passed + 12 subtests (new pins: merged-grid load, sub-linear ≤ linear
  beyond-hull, interior-holes-keep-analytic, roofline-increment cap, hold-out regression,
  launch-floor classes/resolution/analytic-grid dynamics).
