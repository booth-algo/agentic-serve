# Parallel de-fit campaign (2026-06-10): 6 workflow lanes + dashboard rebuild

**Base:** main `5f06393` (PR #74 merged). Each lane runs in its OWN git worktree/branch, commits
locally, and NEVER pushes or opens a PR — integration is a separate sequential phase (merge lanes
one at a time into an integration branch, re-gate after each, then ONE PR). Lanes must not touch
files owned by another lane (matrix below). De-fit log entries go to lane-local files
`profiling/docs/defit_log_entries/<lane>.md` — NOBODY edits `prediction_construction.md` (the
integration phase merges entries into its De-fit log).

**Shared gate protocol** (every lane that can move predictions): capture the lane's OWN replay-ON
baseline in its worktree BEFORE any edit (`python3 -m profiling.process.gate_scoped_rows`; the
worktree setup symlinked the realized pools — the tool warns if replay is OFF, treat the warning
as FATAL). Binding: H100/A100 `ttft_cell`/`e2el_cell`/`tpot_cell`/`chat` ≤ baseline + 0.3 and
H100 swe-plateau ≤ +0.3, unless the lane's own contract says byte-identical or names H100x2 as
its target. Honest stop-points everywhere: "keep current behavior + document the measured
disagreement" is a SUCCESS outcome (knee/util-cap precedent).

**GPU rules** (hard): follow `profiling/docs/h100_setup.md` env (TMPDIR/XDG_CACHE_HOME on
/data48, fresh run dirs under `/data48/kevinlau/<lane>_run/` — never reuse, never `rm` inside an
ssh command: the deny rule `ssh * rm *` blocks it). Check YOUR assigned GPUs are free
(`nvidia-smi --query-compute-apps`) before every launch; other GPUs may be in use by sibling
lanes — do not touch them. NEVER download model weights (Rule #1): if a needed model/env is
missing on a host, STOP and report. Leave GPUs clean (no compute apps) when done.

| Lane | Branch / worktree | Scope (audit-v2 refs) | Owns (nobody else touches) | GPU |
|---|---|---|---|---|
| L1 | — (main checkout) | Dashboard rebuild: full `build_simulator_rows` → publish merged predictions | gitignored `simulator-predictions.json` only | none |
| L2 | `pin-replay-pools` / `agentic-serve-pools` | S13: end the silent replay-OFF footgun (pin/commit capped pools OR hard-fail loudly) + D1–D4 (ramp_tpot false "measured cluster" comments, stale 10/24-40/80 docstring) | `simulator/ramp_tpot.py`, `simulator/tests/test_ramp_tpot.py`, `.gitignore`, `inference-benchmark/data/distributions/*` | none |
| L3 | `tp2-sublinearity` / `agentic-serve-tp2` | The deferred tp2 decode fix: analytic fill linear in `b·ctx` vs sub-linear real kernel (~1.25–1.3× over-price past the sparse tp2 grid edge). Also G5 (per-config launch-floor residuals) and S12 (the 7 dropped 'check' grid rows + the false "OOM" docstring) — same file/domain. Success target: **H100x2 TPOT cell-MAPE improves ≥3pt** (from ~28.7), tp1 byte-identical-or-better | `simulator/kernel_step_cost.py`, `simulator/tests/test_kernel_step_cost.py`, tp2/tp1 decode-grid CSVs + their builders, `profiling/profile/vllm/cuda_events/decode_steps.py` usage | h100 GPUs **6+7** |
| L4 | `queue-eviction-rederive` / `agentic-serve-queue` | S7–S9: trace-validate + re-derive the sim's eviction cluster against real vLLM v1 semantics (tier-2 MRU whole-session vs LRU-oldest-block; barrier-frozen hit/miss vs live; herd protection) using engine traces as oracles (NOT predictor inputs). S10 floor-split only if free. PRIZE: after the re-derivation, re-gate the measured util curve (`prefill_gemm_util_H100.json`, already pinned) — if it then passes, the last RED compensating fit retires | `simulator/ttft_queue_sim.py` INTERNALS (queue/eviction/cohort code; NOT the pricing constants at the top — those are settled), `simulator/tests/test_ttft_queue_sim.py` sim-behavior tests (NOT the constant pins), `profiling/profile/vllm/engine_trace/*` | h100 GPU **5** |
| L5 | `gray-defit-batch` / `agentic-serve-gray` | G1/G2 (SAT_SUSTAIN 9/24 → regenerable builder + pin, resolve the per-request-vs-turn-median population choice), G6 (RESERVE_BYTES 3.5e9 → stated rule or per-GPU derivation), G8 (sglang `chunked_prefill_size` memory-tier rule → generator + JSONs + parity test), G9 (ceiling-cut sensitivity documented in the artifact) | `simulator/kernel_tpot.py` (SAT_SUSTAIN block only), `simulator/tests/test_kernel_tpot.py`, `configs/kv_pool.py`, `configs/generate_deployments.py`, `configs/deployments/*_sglang.json`, `simulator/tests/test_deployment_configs.py`, `profiling/process/build_saturated_ceiling.py`, new builders | none |
| L6 | `a100-roofline-utils` / `agentic-serve-a100` | G7+G4: own the roofline-utils RECIPE (a deterministic builder over serving wall traces) — re-derive H100 `util_bw=0.93` with the pinned recipe AND measure A100 `util_flops`/`util_bw`/`scheduler_overhead` (replacing the H100 placeholders). Update gpu JSONs + artifacts; gate (A100 binding; H100 within ±0.3 if its util_bw moves) | `configs/gpus/*.json`, `simulator/closed_form_tpot.py` (util defaults/comments), roofline-params artifacts, new `profiling/process/build_roofline_utils.py` | **gpu-4** (A100, pick 1 free GPU; verify vllm env + local weights exist, STOP if missing — no downloads) |
| L7 | `deadpath-cleanup` / `agentic-serve-cleanup` | D5–D9 + residual stale-provenance comments not owned above: `session_regime_classifier.py` (deleted-P_LO references, unflagged 0.4 cap/+2 pull), `kernel_tpot_hint.py` (stale column claims), `_legacy` caller retirement where safe, `fitted_constants_audit.md` label fixes (S7 VLLM-CONFIG misfile). Predictions must be BYTE-IDENTICAL (cleanup lane) | `simulator/session_regime_classifier.py`, `simulator/kernel_tpot_hint.py`, `simulator/_legacy/*`, `profiling/docs/fitted_constants_audit.md` | none |

**Cross-lane facts** (so lanes don't re-litigate): the measured per-step util curve plateaus at
0.754 (`prefill_gemm_util_H100.json`); the H100x2 decode grid OVER-predicts at low batch (floor
excess raw median −0.73 ms); pressure-based onset constants don't exist (knees retired); replay-OFF
gates are invalid for TTFT/E2EL.

**Integration (after all lanes report):** sequential merges into `defit-campaign-integration`
ordered L7 → L2 → L5 → L6 → L3 → L4 (cheap/byte-identical first, prediction-moving last),
re-gate after each merge, merge `defit_log_entries/*` into `prediction_construction.md`, full
pytest + replay-ON gate + dashboard re-rebuild, ONE PR.

## Execution record

_pending — filled at integration_
