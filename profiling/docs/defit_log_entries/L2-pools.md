# L2 (pin-replay-pools) — De-fit log entries

_Lane-local entries per the parallel de-fit campaign protocol; merged into
`prediction_construction.md` at integration. Lane scope: S13 + D1–D4._

## S13 — replay-pool footgun closed: per-GPU realized pools COMMITTED (minified)

**Problem.** The per-GPU realized files
(`inference-benchmark/data/distributions/*_realized_<slug>.json`), which carry the
TTFT trajectory-replay pools, were gitignored (~105MB). A fresh checkout silently
fell back to the pooled cohort, which flipped two gate verdicts on 2026-06-09/10
(replay-OFF gates are invalid for TTFT/E2EL — cross-lane fact).

**Pre-registered decision rule.** Measure the real footprint of the files production
needs (the Llama-3.1-8B ground-truth gpu_keys only; consider stripping unread
`by_concurrency` content and minifying). If a slimmed committed set ≤ 25MB total →
COMMIT it; otherwise implement a loud failure path.

**Measurement (2026-06-10).**
- 78 per-GPU files exist (4 profiles × the Llama ground-truth gpu_key slugs).
- Raw (pretty-printed builder output): **105.0MB** — the size that justified
  gitignoring was whitespace, not data.
- Minified (`json` separators `(',', ':')`): **19.9MB**.
- Minified + stripped to only the fields `ramp_tpot` readers consume
  (`histograms.turn_count`, `context_scale_quantiles`, `by_concurrency.{turn_count,
  context_scale_quantiles, trajectory_pool}`, top-level `trajectory_pool`): **19.8MB**
  — stripping buys only 0.1MB because the weight IS the `trajectory_pool` blocks
  production reads, so nothing was stripped (provenance metadata kept).
- Gate-only subset (`gate_scoped_rows` DEFAULT_GPU_KEYS H100/A100/H100x2 = 12 files):
  3.8MB — not used, because `build_simulator_rows` resolves the per-GPU cohort for
  **every** Llama-3.1-8B deployment gpu_key, not just the gate's three.
- Largest single file: 0.51MB (no GitHub large-file concerns).

**Outcome: COMMIT branch taken (19.9MB ≤ 25MB).** All 78 files committed minified,
content-identical to the builder output (parsed-equality verified file-by-file)
plus a top-level `_committed_note` regeneration note in each. `.gitignore` pattern
removed (replaced by a comment documenting the decision). Worktree-setup symlinks
replaced by the real files.

**Defense-in-depth (residual risk: future gpu_keys / deleted files).**
`ramp_tpot._resolve_dist_path` no longer falls back silently: a requested-but-absent
per-GPU file emits an unmissable once-per-(gpu_key, profile) banner to stderr, and
`RAMP_TPOT_REQUIRE_POOLS=1` (for gate runs) escalates every occurrence to a hard
`FileNotFoundError`. Zero behavior change on the resolved path (same files, same
parsed content, same fallback target). Note: gpu_keys that never had ground truth for
a profile (e.g. `rtx2080tix4` osworld) will now warn once per build instead of
falling back silently — intended.

**Pinned by test** (`simulator/tests/test_ramp_tpot.py`): committed-pool presence +
read-field integrity for the gate gpu_keys; regeneration note in every file;
per-GPU resolution preferred; warn-once semantics; pooled-fallback equality;
hard-error escalation under the env flag.

## D1–D4 — ramp_tpot false provenance comments fixed (comment/docstring only)

**D1–D3 (`DEF_LO=-0.12`, `DEF_HI=0.22`, `DEF_SAT=1.0`).** The comments claimed the
knees were "read off the measured jump-pressure cluster" at pressure ≈ 0.88–1.22
("NOT a MAPE fit"). The reproducible jump-band measurement
(`profiling.process.build_ramp_knees` → `profile_data/kernels/ramp_knees_h100_llama31_8b.json`)
disproved this: measured H100 onset P_LO = **0.4456** (defcap −0.55), short-output
knee P_HI_SHORT = **1.6866** (defcap +0.69), long-output knee P_HI_LONG = 2.3089
(defcap +1.31; n=2, `adoptable: false`) — no cluster exists at 0.88–1.22. The
kernel-side twins were honest-relabeled on 2026-06-09 (commit 2515007) and later
restructured away (`kernel_tpot._overflow_weight`); this diagnostic side-by-side
column now mirrors that honesty: all three knees relabeled **TUNED KNOBS** citing the
artifact, values unchanged. The module docstring's "(DEF_LO, pool ~88% committed)"
watermark claim fixed the same way.

**D4 (stale import docstring).** The comment claimed "reused verbatim from
kernel_tpot (SAT_SUSTAIN_LO/HI = 10/24 tok; OUT_KNEE_LO/HI = 40/80 tok)" while the
imports resolve **9/24** and **28/86**. Fixed to the real values with their honest
labels (measured saturated-turn p5/plateau anchors; measured output-cluster labels).

**Behavior change: none** outside the new S13 warning path — D1–D4 are
comment/docstring edits only; all knee literals byte-identical.
