# MSE Validation Sampling Plan

Date: 2026-05-05

## Goal

Improve the trace-fidelity validation used by the NeurIPS paper without turning
it back into a full benchmark sweep. The paper-facing claim should remain
scoped: distributional replay is used only when it passes validation against
real-trace replay under matched model, GPU, backend, context cap, concurrency,
and session count.

## Decision

Use no-replacement source-session sampling as the default distributional
generator behavior. Keep source-locked replay as a validation ablation, not as
the primary generator.

## Rationale

The current short-profile validation showed strong aggregate fidelity, but raw
per-turn MAPE is inflated by session-population variance. For example, the
SWE-Bench C=5, S=40 MSE run sampled only 29 unique source sessions while the
REAL run used 40 unique sessions. The synthetic replay duplicated some source
trajectories and skipped others, which is avoidable sampling noise.

No-replacement sampling is the honest generator behavior when the requested
session count is less than or equal to the available source-session count. It
still represents a sampled distributional workload, but avoids unnecessary
duplicates.

Source-locked replay answers a narrower diagnostic question: if the synthetic
and REAL runs use the exact same source-session IDs, how much residual error is
caused by synthetic text content rather than population mismatch? This is useful
for an ablation, but it should not be the main generator evidence.

## Implementation Plan

1. Make distributional sampling use source sessions without replacement when
   enough source sessions are available.
2. Add a source-locking hook that accepts a source-session-ID file for validation
   runs.
3. Add regression tests for no-replacement and source-locked sampling.
4. Rerun the paper validation on H100:
   - SWE-Bench, C=5, S=40
   - TerminalBench, C=5, S=40
   - Optional: SWE-Bench, C=20, S=40
5. Report the main Option 2 fidelity table:
   - aggregate median TPOT/E2EL
   - binned per-turn TPOT/E2EL
   - TTFT as secondary/noisy
6. If time permits, run Option 3 as a small ablation:
   - SWE-Bench, C=5, S=40, source-locked
   - Use it to show how much residual per-turn error comes from session
     population variance rather than synthetic content.

## Acceptance Criteria

- For `num_sessions <= available source sessions`, distributional replay uses
  unique `source_session_id` values.
- Source-locked replay emits synthetic sessions in the exact requested source-ID
  order and fails clearly if an ID is unavailable.
- Existing distributional profiles without `source_session_id` remain
  compatible.
- Unit tests cover both new modes.
- H100 rerun commands are short, explicit, and avoid the old full MSE sweep.

## Paper Reporting Guidance

Main table: Option 2 no-replacement sampling. Report aggregate median TPOT and
E2EL errors, plus binned per-turn TPOT/E2EL over turn ranges such as 0-4, 5-9,
10-19, and 20-29. TTFT should be reported as a noisier secondary metric and
bounded against REAL-vs-REAL variation.

Source-locked table: optional ablation only. Frame it as isolating synthetic
content effects after controlling for session-population variance.
