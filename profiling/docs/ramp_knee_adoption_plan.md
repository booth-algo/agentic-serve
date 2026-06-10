# Ramp-knee adoption plan: corrected floor → re-measured band → gated adoption

**Goal:** make the `kernel_tpot` ramp knees (`P_LO`/`P_HI_SHORT`) adoptable from the
`build_ramp_knees` measurement WITHOUT regressing, by first fixing the misattribution the
2026-06-09 de-fit attempt exposed. Every rule below is **pre-registered** — fixed before any
result is looked at; the executor applies them mechanically. Every phase has an honest
stop-point: "stop, keep tuned values, document what the measurement showed" is a SUCCESS
outcome, not a failure.

**Context** (see the 2026-06-09 entry in `prediction_construction.md` De-fit log): the measured
ramp band ([0.45, 1.69] vs tuned [0.88, 1.22]) was gate-rejected because the implied weight
`w = (tpot_meas − kernel_step)/(t_upper − kernel_step)` attributes ANY excess over the decode
kernel grid to KV pressure. At low pressure that excess is not pressure-caused (host serving
stack per decode step, scheduling jitter) — the same class of host cost the TTFT stage-split
measured for prefill (`frontend.new = 5.745 ms/1k`). The tuned late/steep band compensates for
that misattribution. Plan: measure the pressure-independent excess **D**, correct the floor,
re-measure the band, and only then test adoption.

---

## Phase 0 — measure the decode floor-excess `D` (per GPU, from existing GT; no GPU needed)

**Rule (pre-registered):** over turns with `pressure < 0.30` (saturation impossible),
`output ≥ 24` (sustain-clean), `tpot_meas > 0`:
`D_raw = median(tpot_meas − kernel_step)`; `D = max(0, D_raw)` (a serving overhead cannot be
negative). Report n, IQR, and Spearman(excess, scheduled) — a strong batch trend (|ρ| ≥ 0.5)
is reported and documented but D stays a constant this round (no new fitted shape).
One D per deployment artifact (H100, H100x2, A100).

**Production form (if adopted):** `floor' = decode_step_ms(b_eff, ctx) + D`;
`t_upper = max(floor', saturated_ceiling_ms(...))`; `TPOT = floor' + weight·(t_upper − floor')`.
At weight=1 the prediction still lands exactly on the measured ceiling — D corrects the FLOOR
only. D is read from the ramp-knees artifact (swappable per config exactly like
`_active_ceiling_json`; missing artifact → D = 0).

**Stop-point:** if `D < 0.5 ms` on every GPU, the floor-misattribution hypothesis is
unsupported at the floor level → stop after documenting (knees remain tuned; the De-fit log
gains the negative result).

## Phase 1 — re-measure the band against the corrected floor

Re-run the band detection (identical pre-registered rule as 2026-06-09: sustain-clean,
conditioning `t_upper − floor' ≥ max(10ms, 0.5·floor')`, ≥8 turns, both crossings interior,
smoothstep inversion, median aggregation, jackknife) with `floor'` in place of `kernel_step`.
The artifact keeps the ORIGINAL uncorrected `knees` block untouched (the existing pin test
stays green) and adds `floor_excess_ms` + `knees_corrected`.

**Cross-GPU convergence gate (the physics test):** pressure is already pool-normalized, so a
real physical band must agree across GPUs. Among GPUs with ≥3 usable cells for the estimate:
- `P_LO` adoptable only if ≥2 GPUs qualify AND `max(onset) − min(onset) ≤ 0.20`.
- `P_HI_SHORT` adoptable only if ≥2 GPUs qualify AND spread ≤ 0.30.
- `P_HI_LONG`: NOT adoptable this round regardless (data-starved: 2 cells, 1 profile).
The uncorrected measurement fails this today (onsets 0.45 / 0.85 / 0.60) — if the corrected
one still fails, that is definitive evidence no global pressure constant exists and the
physical fix is the audit-item-6 ramp restructure. Stop + document.

**LOCO generalization gate (knee candidates only):** for each of the 4 profiles, re-derive the
corrected band excluding that profile (`--exclude-profile`), rebuild predictions with those
LOO knees, and require the held-out profile's all-turns TPOT MAPE ≤ baseline + 0.5pt (H100).

## Phase 2 — adoption matrix, sliced gates

Candidates (each = one scoped rebuild + metrics): `D-only`, `D+P_LO`, `D+P_HI_SHORT`,
`D+both` (knee candidates only run if their convergence gate passed). Candidate values come
from the H100 artifact's `knees_corrected`.

**Binding gates** (vs the pre-change baseline captured in Phase 0, all in MAPE points):

| metric | gate |
|---|---|
| H100 tpot_cell, ttft_cell, e2el_cell | ≤ baseline + 0.3 |
| H100 chat all-turns TPOT | ≤ baseline + 0.3 (where adoption died last time) |
| H100 swebench plateau TPOT | ≤ baseline + 0.3 (the spec's named gate) |
| A100 tpot_cell, ttft_cell, e2el_cell, chat | ≤ baseline + 0.3 |
| LOCO (knee candidates) | each held-out profile ≤ baseline + 0.5 |

**Advisory (recorded, non-binding):** H100x2 — moving the knees shifts the saturation
transition into the region where tp2's analytic decode fill is known-wrong (the deferred
sub-linearity spec); its numbers are documented, not gating. Neutral results are acceptable
for adoption (a measured value at no accuracy cost beats a tuned one).

**Adoption priority:** largest passing candidate wins: `D+both` > `D+P_HI_SHORT` > `D+P_LO` >
`D-only` > none. The outcome (including "none") is wired honestly: literals/pins/comments
updated to exactly what passed; everything else keeps the tuned-knob labels.

## Tooling (built in Phase 0, kept as repo utilities — no /tmp throwaways)

1. `profiling/process/build_ramp_knees.py` v2: adds the D measurement, `knees_corrected`,
   and `--exclude-profile <name>` (LOCO). Original `knees` block byte-compatible.
2. `profiling/process/gate_scoped_rows.py` (new, replaces the thrice-rewritten /tmp harness):
   scoped Llama-vLLM rebuild for `--gpu-keys H100,A100,H100x2` with override flags
   `--p-lo`, `--p-hi-short`, `--floor-excess H100=...,A100=...` (module-attribute patches —
   `kernel_tpot.P_LO` etc. and a wrapper over `kernel_tpot.decode_step_ms` — never source
   edits), writing predictions JSON + a metrics JSON:
   `{gpu: {tpot_cell, ttft_cell, e2el_cell, tpot_turn_overall, tpot_profile: {chat..,swebench..,osworld..,terminalbench..}, tpot_plateau_profile: {...}}}`
   (profile keys shortened to chat/swebench/osworld/terminalbench; plateau = turns with
   tpot_meas > 100ms).

## Verification

- Baseline metrics captured BEFORE any production edit; final wiring re-verified against them.
- If outcome = none or D-only-rejected: production predictions byte-identical (diff the JSONs).
- `pytest simulator/tests/` green (124+; pin tests updated to match the adopted outcome).
- Artifacts regenerate deterministically (run builder twice, byte-identical).
- Docs: De-fit log entry (outcome + numbers), debt table, spec rows if knees adopted,
  this file's Execution record filled in.

---

## Execution record (filled by the workflow run)

**Run date:** 2026-06-09. **Outcome: ADOPTED = none** — the pre-registered **Phase-0
stop-point triggered**: `D < 0.5 ms` on every GPU, so the floor-misattribution hypothesis is
unsupported at the floor level. Knees stay tuned; no production code change; D is NOT wired.

### Phase 0 — decode floor-excess D (per GPU)

| GPU | D (ms) | raw median (ms) | IQR | n | Spearman(excess, scheduled) |
|---|---|---|---|---|---|
| H100 | **0.0** | −0.0382 | 0.5401 | 334 | −0.0909 |
| A100 | **0.1246** | 0.1246 | 0.9569 | 178 | +0.4665 |
| H100x2 | **0.0** | −0.7251 | 0.9292 | 486 | −0.4593 |

No GPU reaches the 0.5 ms adoption floor. Spearman stays below the |ρ| ≥ 0.5 documentation
threshold on all three, but H100x2 (−0.46) and A100 (+0.47) are near it. The H100x2 raw median
is strongly **negative** (grid *over*predicts at low batch, with a negative batch trend) —
consistent with the known tp2 analytic-fill over-pricing (deferred sub-linearity spec), not
with a host floor-excess.

**decode_step_ms leak check (binding audit):** `simulator/ttft_queue_sim.py:73` imports
`decode_step_ms` directly from `simulator.kernel_step_cost`, so a +D wrapper patched onto
`kernel_tpot.decode_step_ms` does NOT leak into the queue sim's mixed-step decode pricing.
Residual leak path: `kernel_tpot.predict_cell_tpot` used by `_fallback_ttft` (off the main
TTFT path) would include +D — documented in `gate_scoped_rows`' docstring, accepted. Binding
verified two ways (module-global resolution + a startup probe proving a +5 ms wrapper moves a
low-pressure prediction by exactly +5 ms).

### Phase 1 — corrected band (floor' = kernel_step + D)

| GPU | onset (P_LO) | n | hi_short (P_HI_SHORT) | n | hi_long |
|---|---|---|---|---|---|
| H100 | 0.4456 | 8 | 1.6866 | 6 | 2.3089 (n=2) |
| A100 | 0.6048 | 2 | 1.7842 | 1 | — |
| H100x2 | 0.8540 | 8 | 1.4073 | 8 | — |

Because D = 0 on H100/H100x2 the corrected band is **identical** to the uncorrected one there;
A100's corrected band matches the uncorrected at 4 dp. The artifacts keep the original `knees`
block untouched (pin test unchanged on that block) and add `floor_excess_ms`,
`knees_corrected`, `corrected_rule`, `cells_corrected`, `excluded_profile`.

**Cross-GPU convergence verdicts (≥3 usable cells to qualify):**
- `P_LO`: qualifying GPUs = H100 (0.4456, n=8) and H100x2 (0.8540, n=8); A100 disqualified
  (n=2). Spread 0.4084 > 0.20 → **FAIL** (`pLoOK = false`).
- `P_HI_SHORT`: qualifying GPUs = H100 (1.6866, n=6) and H100x2 (1.4073, n=8); A100
  disqualified (n=1). Spread 0.2793 ≤ 0.30 → **PASS** (`pHiOK = true`) — moot under the
  Phase-0 stop, recorded for the audit-item-6 ramp restructure.
- `P_HI_LONG`: not adoptable this round by pre-registration (H100 point est. 2.3089, n=2).

The corrected onsets (0.45 / 0.60 / 0.85) still disagree across GPUs → no global pressure
onset constant exists even after the floor correction; the physical fix remains the
audit-item-6 ramp restructure (converges with the deferred tp2 sub-linearity spec).

### Phase 2 — candidate gates

**Not run** (`candidates = {}`): the Phase-0 stop-point pre-empts the adoption matrix — with
D negligible everywhere, `D-only` is a no-op and the knee candidates are the same values the
2026-06-09 de-fit already gate-rejected. **H100x2 advisory numbers** (recorded, non-binding):
D raw median −0.7251 ms / Spearman −0.4593 — the tp2 grid over-prices low-batch decode, so any
floor-excess wiring would have moved H100x2 the wrong way.

**LOCO:** not run as a gate (`loco = {}`, knee candidates never reached the gate). Mechanics
smoke-tested (`--exclude-profile swebench-multiturn-synth` → `/tmp/ramp_adopt/loco_swebench`):
D stays full-sample, band cells drop the held-out profile, and the builder guard refuses
`--exclude-profile` with the canonical out-dir.

### Baseline (pre-change, ZERO overrides; P_LO=0.88, P_HI_SHORT=1.22, floor_excess=0)

| GPU | tpot_cell | ttft_cell | e2el_cell | tpot all-turns overall | chat | swebench | osworld | terminalbench | swe-plateau |
|---|---|---|---|---|---|---|---|---|---|
| H100 | 14.5356 | 18.1962 | 11.3344 | 15.4065 | 5.5586 | 16.1346 | 16.2147 | 21.7062 | 8.6521 |
| A100 | 15.3672 | 21.9376 | 15.8354 | 14.8834 | 19.9851 | 14.5593 | 9.4757 | 18.4113 | 7.0242 |
| H100x2 | 28.7348 | 34.7114 | 25.9191 | 29.1740 | 17.2432 | 32.6385 | 23.1278 | 42.6084 | 6.7283 |

H100 all-turns overall 15.4065 matches the documented 15.4% pre-change reference. chat plateau
is null on H100/H100x2 (no chat turns with tpot_meas > 100 ms; the chat gate is all-turns).

### Wiring (outcome = none)

- **No production simulator module changed.** Final verification: re-ran
  `python3 -m profiling.process.gate_scoped_rows` with no overrides and diffed against
  `/tmp/ramp_adopt/baseline.predictions.json` → **byte-identical**.
- Pin test `test_ramp_knees_tuned_values_and_measured_band_both_pinned` extended to the final
  state: tuned literals (0.88 / 1.22 / 2.0), the artifact's uncorrected band (0.4456 / 1.6866 /
  long-not-adoptable), per-GPU `floor_excess_ms` pinned at {H100: 0.0, A100: 0.1246,
  H100x2: 0.0} and `< 0.5` (the stop-point rule), and H100 `knees_corrected` == `knees`.
- Artifacts regenerated deterministically (builder run twice, byte-identical; the v1 `knees`
  block values unchanged from the pre-existing artifacts).
- Tooling kept as repo utilities: `profiling/process/build_ramp_knees.py` (v2: D measurement,
  `knees_corrected`, `--exclude-profile`), `profiling/process/gate_scoped_rows.py` (new).
- Docs: De-fit log entry (2026-06-09 corrected-floor follow-up) in
  `prediction_construction.md`; memory status line appended.

## Restructure outcome (2026-06-10) — the audit-item-6 restructure ADOPTED under these Phase-2 gates

The "no global pressure constant exists" conclusion above was resolved by the ramp RESTRUCTURE
(distribution-overflow eviction-drain weight, `kernel_tpot._overflow_weight`; 3 rounds, see the
`prediction_construction.md` De-fit log): the tuned band `P_LO`/`P_HI_SHORT`/`P_HI_LONG` +
`OUT_KNEE` interpolation was ELIMINATED, onset/width computed per cell. The pre-registered
Phase-2 binding gates of THIS plan were the arbiter each round (vs the reproduced baseline
captured here; the TPOT baselines reproduce exactly — recorded TTFT/e2el baselines were
re-captured in-environment, `/tmp/restructure/baseline.metrics.json`):

| round | result |
|---|---|
| 1 (overflow duty, z-only onset) | 7/10 FAIL→ kept on branch (H100 tpot_cell +1.31, chat +0.3025, A100 e2el +1.25) |
| 2 (P0b pool-full gate + chunk-quantized drain + rotation + fresh-crossing damping) | 8/9, FAIL H100 swe-plateau 8.652→9.674 |
| 3 (firing-gate hysteresis: arm at pressure≥1∧z>1, hold while z>1, release at z≤1) | **9/9 PASS** — H100 swe-plateau **8.511**, tpot_cell 14.556, e2el 21.304, chat 5.469; A100/H100x2 rows bit-identical to round 2 |

The measured `ramp_knees_*` artifacts stay the valid measured record of the old band (pinned as
history in `test_kernel_tpot`); in z-units (`z = p_low·qbar`, qbar from the measured
`context_scale_quantiles`) the per-GPU onset medians collapse 0.964/1.188/0.963 — the
cross-GPU disagreement documented above was the cohort context-spread.

**Adopted wiring (this branch):** `kernel_tpot._overflow_weight` + the cell-path armed state in
`predict_cell_tpot` (`predict_turn_tpot` gains kw-only `armed: bool = False`; cell signature
unchanged), `simulator/cohort_scale.py` (trapezoid mean of the measured quantiles),
`KernelTurnInput.cohort_scale_mean` set per cell by `build_simulator_rows`, and the
`max_num_batched_tokens` ENGINE-CONFIG key in the deployment JSONs (vLLM device-rule default:
8192 H100-class / 2048 A100). The tuned literals `P_LO`/`P_HI_SHORT`/`P_HI_LONG`/`OUT_KNEE` are
deleted; zero tuned numeric constants remain in the ramp. Determinism verified (gate rebuild run
twice, byte-identical); the protected twin A100 term@20 never arms (pressure peak 0.965).
Honest residuals carried forward (documented in the De-fit log round-3 entry): first-fire
overshoot magnitude (needs a re-derivation, not a knob), shallow-z armed duty undershoot /
sub-pool-full saturation (runtime effective pool < traced `available_kv_blocks`), and the
non-binding H100x2 osworld plateau advisory regression. Full suite green: 357 passed,
2 skipped (2026-06-10).
