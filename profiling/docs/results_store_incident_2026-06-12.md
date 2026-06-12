# Shared results store deletion + recovery (2026-06-12)

## What happened

The local shared results store `profile_data/results/` (gitignored, regenerable measured
grids + prediction outputs) was **deleted by `git pull` of the PR #75 merge** and replaced
with a self-pointing symlink. The predictions rebuild then crashed with
`OSError: [Errno 40] Too many levels of symbolic links: 'profile_data/results/cached_prefill_v3_H100.csv'`.

Mechanism (three independent footguns composing):

1. During the 6-lane parallel de-fit campaign, lane worktrees accessed the shared store via
   a symlink `profile_data/results -> /root/agentic-serve/profile_data/results` (absolute
   path to the main checkout). Correct *inside a lane worktree*.
2. `.gitignore` had only the trailing-slash pattern `profile_data/results/`, which matches a
   **directory** at that path — a **symlink** named `profile_data/results` does not match,
   so the integration lane committed it without `-f` and without any warning
   (commit `afa1cb8`, merged to main in PR #75).
3. On `git pull` in the main checkout, git considers ignored paths expendable: it deleted
   the real ignored `profile_data/results/` directory to make room for the tracked symlink.
   In the main checkout the symlink's absolute target is **itself** → ELOOP.

No git history held the data (never committed), no R2 prefix existed for it (the R2-sync of
grid CSVs was a known-pending item), and the lane worktrees had already been removed.

## Fix in this PR

- `git rm profile_data/results` (the tracked symlink).
- `.gitignore`: add the no-slash forms `profile_data/results` and `profile_data/_archive`
  so a symlink (or regular file) at these paths can never be silently committed again.
- This document.

Rule going forward: **never create the lane symlink at a tracked-tree path**. If a worktree
needs the shared store, set an env var / config override, or symlink *individual files*
outside the repo tree.

## Recovery manifest

All runtime-critical files (everything `build_simulator_rows` + the deployment manifests
reference) were recovered from the GPU hosts' original run dirs and re-verified against the
md5 pins recorded in `profiling/docs/defit_log_entries/L*.md`:

| File | Source | md5 | Pin match |
|---|---|---|---|
| `cached_prefill_v3_H100.csv` | h100 `/data48/kevinlau/agentic-serve/profiling/results/` | `3559d50a5ad68633fe4d156e7ca4a84c` | no pin recorded (canonical host copy) |
| `decode_kernel_trace_H100_large_2026-05-17_wide_summary.csv` | h100 (same dir) | `e143b6f92487e5776898821cd29069e1` | no pin recorded (canonical host copy) |
| `decode_profile_H100_2026-06-10_s12recheck.csv` | h100 `/data48/kevinlau/tp2grid_run/results/` | `3c227ff8bc2c3fe9ce127dd57a105e3a` | ✅ L-entry pin |
| `decode_profile_H100x2_2026-06-10_main.csv` | h100 (same dir) | `38933d281fdd34bbda00d0e24e5e4e0e` | ✅ L-entry pin |
| `decode_profile_H100x2_2026-06-01.csv` | h100 `/data48/kevinlau/tmp/cpbatch_run/` | `d191debfeef94d39617d07a656a01022` | merge input (output pin verifies it) |
| `decode_profile_H100x2_merged_2026-06-10.csv` | regenerated: `python3 -m profiling.process.build_decode_grid` | `089aca9074604e2aa7895e7f0615d614` | ✅ byte-identical to pin |
| `serving_decode_grid_RTX3090x2_2026-06-11{,_pass2,_pass3}{.jsonl.gz,_summary.csv}` | 3090 `~/m3090_run/l13/` | summaries `e0eb6693…`/`f8fe1797…`/`c562fbf6…` | ✅ all three pins |
| `serving_decode_grid_RTX3090x4_2026-06-11{,_pass2}{.jsonl.gz,_summary.csv}` | 3090 (same dir) | summaries `a8c15470…`/`017de38a…` | ✅ both pins |
| `serving_decode_grid_RTX3090x2_merged_2026-06-11.csv` | regenerated: `build_serving_decode_grid --inputs <3 raws> --lag-upgrade` | `264c1dde73ac488be1269d6ca78af261` | ✅ byte-identical to pin |
| `serving_decode_grid_RTX3090x4_merged_2026-06-11.csv` | regenerated: `build_serving_decode_grid --inputs <2 raws> --lag-upgrade` | `fb51185961ada7984709778eb51ada9c` | ✅ byte-identical to pin |
| `decode_profile_A100_2026-06-02.csv` | a100 `/data/kevinlau/tmp/a100_profile/` | `801d68bcba811a418e43101883c39a57` | no pin recorded (canonical host copy) |

A copy of the recovered store is kept outside the repo at
`/root/profile-data-results-backup-20260612/` on the Hetzner box.

## Still missing locally (non-blocking, recoverable)

Builder/gate *evidence* files that lived in the store but are not runtime inputs: the
serving stage-split CSVs (`serving_stage_split_*.csv`), `s7_replay/` + `s8_replay/` evidence
trees (gzipped raws; md5s pinned in L13), `prefill_live_split_H100.csv`,
`prefill_util_sweep_H100.csv`, and the older H100 prediction-output CSVs. All have md5 pins
or live on the GPU hosts' run dirs. Re-pull on demand.

## Lesson → action

The store survived only because the GPU hosts still had the raw run dirs (the
`ssh * rm *` deny rule meant lane agents never deleted remote evidence) and because every
L-entry pinned md5s. The pending **R2-sync of the shared store** should now actually happen
— it is the only durable home for files whose hosts get reimaged.
