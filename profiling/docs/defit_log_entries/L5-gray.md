# L5 (gray-defit-batch) — de-fit log entries

Lane L5 of the 2026-06-10 parallel de-fit campaign (`parallel_defit_campaign.md`): the four
GRAY batch items G1/G2, G6, G8, G9 from `fitted_constants_audit_v2.md`. Contract: production
predictions for the vLLM Llama gate configs (H100/A100/H100x2) are **byte-identical** — these
items pin/document/derive, they do not retune. sglang-config rows may move (G8). Entries below
are formatted for merge into `prediction_construction.md`'s De-fit log at integration.

- **2026-06-10 — `SAT_SUSTAIN_LO`/`HI` 9/24: regenerable builder + population question resolved — audit-v2 G1/G2 PINNED (values unchanged)** (`kernel_tpot`; lane L5).
  The anchors were unpinned analyst read-offs from gate-tuning commit `41e35f5` (no builder, no
  artifact, test pinned duplicated literals). New `profiling/process/build_sat_sustain.py`
  regenerates BOTH from GT (H100 headline run, saturated = measured tpot > 100 ms) →
  `profile_data/kernels/sat_sustain_H100_llama31_8b.json`, pinned by
  `test_sat_sustain_anchors_pinned_to_builder_artifact` + a bench-gated regeneration-parity test.
  **Population question (G1), resolved explicitly:** pre-registered canonical population =
  TURN-MEDIAN rows — `build_simulator_rows.build_turns` medians are what `predict_cell_tpot`
  consumes (the smoothstep's `out` argument IS a turn median). The builder reproduces the audit's
  disagreement exactly: per-request rows (n=45450 saturated) p5 = **9.0** (the production LO);
  turn-median rows (n=301 saturated) p5 = **24.0** (= the production HI). I.e. on the canonical
  population the [9, 24] band has no measured support below 21.5 — LO=9 stands only on the
  per-request read. **HI's "+2 offset" (G2):** the historical story (min turn-median plateau
  output 21.5 ≈ 22 tok + an underived +2 hand margin) is superseded by an exact derivation the
  builder found — 24.0 IS the p5 of turn-median plateau outputs, the same quantile as LO on the
  canonical population; both readings land on 24, so no underived offset is needed. The
  disagreement and both readings are committed in the artifact and stated in the SAT_SUSTAIN
  code comment. **Values unchanged** (byte-identity contract): H100/A100/H100x2 gate predictions
  byte-identical (verified vs the lane's replay-ON baseline). `sustain_mid = 16.5` (development
  clock) documented as derived from LO/HI. Honest stop-point: a retune (e.g. collapsing the band
  to the turn-median anchors) is a future gated change, starting from this artifact.

- **2026-06-10 — `RESERVE_BYTES = 3.5e9`: rule STATED + executable test — audit-v2 G6 CLOSED (no value change)** (`configs/kv_pool.py`; lane L5).
  The reserve was honest but rule-less (a hand round inside the 2.71–4.10 GB back-solved
  envelope). The rule is now stated in the module docstring with the reproduction arithmetic:
  **RESERVE = mean of the back-solved reserves of the 3 exactly-known pools, rounded to 0.1 GB**
  — `reserve_i = total·util − weights/tp − pool_blocks·bytes_per_block` gives 4.102 (H100 80GiB
  pool 27250) / 2.710 (A100 40GiB pool 8458) / 3.831 GB (H100x2 pool 62416), mean **3.548 → 3.5**.
  `test_reserve_rule_reproduces_known_pools` executes the rule: the single 3.5 GB reserve
  reproduces the 3 known pools within the documented 5% (+1.05% / −4.46% / +0.51%) and the
  back-solved mean rounds to the production constant. Small-pool amplification flagged in the
  docstring (the ±0.6–0.8 GB envelope is ±30–60% of small derived pools, e.g. RTX3090: 1117
  blocks). The previous docstring's "3.81 GB" H100x2 back-solve corrected to the exact 3.831.
  No value change; derived pools byte-identical.

- **2026-06-10 — sglang `chunked_prefill_size`: real memory-tier rule emitted per deployment — audit-v2 G8 CLOSED (sglang rows sanctioned to move)** (`configs/generate_deployments.py` + 39 `*_sglang.json`; lane L5).
  All sglang deployments used to inherit the loader's vLLM 8192 default (key absent — a 4×
  budget error on 24GiB devices). Implemented the real engine rule, read from sglang source:
  `python/sglang/srt/server_args.py` `ServerArgs._handle_gpu_memory_settings` (upstream main @
  `255843d45462`, fetched 2026-06-10; per-DEVICE memory, MiB tiers): <20 GiB → 2048 (T4/4080),
  <35 GiB → 2048 (A10/4090; RTX3090 24GiB), <60 GiB → 4096 (A100-40G/L40), <90 GiB → 8192
  (H100/A100-80G), <160 GiB → 8192 (H20/H200), else 16384 (B200/MI300). NOT vLLM's ≥70GiB
  non-A100 device rule. `generate_deployments.py` now emits `max_num_batched_tokens` for
  engine==sglang (the loader key the simulator prices as the per-step chunked-prefill budget);
  all 39 sglang JSONs regenerated (+1 key each: 3090-class → **2048** (11 files), A100 →
  **4096** (17), H100 → **8192** (11)); regeneration verified to leave every vLLM JSON
  byte-identical. Parity tests extended to sglang
  (`test_sglang_memory_tier_rule_values` pins the cited tiers;
  `test_sglang_chunked_prefill_size_matches_engine_rule` is the S14-style key-dropping guard).
  **Gate configs unaffected** (vLLM-only; H100/A100/H100x2 byte-identical). **sglang
  Llama-3.1-8B rows before→after** (full build_row capture, all profiles × concs;
  metrics = cell-MAPE %):

  | gpu_key | budget | tpot_cell | ttft_cell | e2el_cell |
  |---|---|---|---|---|
  | A100 (sglang) | 8192→4096 | 15.95→16.33 | 26.97→26.97 | 16.17→16.06 |
  | A100x2 (sglang) | 8192→4096 | 47.04→46.99 | 51.61→51.61 | 47.92→47.84 |
  | A100x4 (sglang) | 8192→4096 | 50.32→51.21 | 50.00→49.99 | 50.47→50.42 |
  | H100 (sglang) | 8192→8192 | rows IDENTICAL | — | — |
  | H100x2 (sglang) | 8192→8192 | rows IDENTICAL | — | — |
  | RTX3090 (sglang) | 8192→2048 | 50.30→68.12 | 150.00→174.02 | 97.12→117.89 |
  | RTX3090x2 (sglang) | 8192→2048 | 62.72→61.31 | 42.21→42.21 | 55.17→53.78 |
  | RTX3090x4 (sglang) | 8192→2048 | 80.86→80.63 | 49.99→49.97 | 73.40→73.11 |

  Honest read: these are advisory, analytic-first-cut configs (no measured grid, derived pools,
  H100-inherited ceiling/floor) — the engine constant is adopted for CORRECTNESS, not accuracy.
  A100-tier moves are noise-level (±0.9pt); the 8192-tier H100 sglang rows are bit-identical;
  RTX3090 tp1 gets WORSE (tpot 50.3→68.1, e2el 97.1→117.9) — the wrong 4×-too-large budget had
  been masking the tiny derived pool's (1117 blocks, the G6 amplification case) overflow
  pricing. Keeping the honest engine value and recording the regression is the
  knee/util-cap-precedent stop-point; the fix is a measured 3090 pool/grid, not a budget fudge.
  Raw row payloads: `/tmp/l5_sglang.{before,after}.json` (full build_row capture, all
  profiles × concs; capture script preserved in the entry's commit message context).
  Non-Llama sglang configs (gpt-oss/Qwen/70B etc.) carry the same corrected budgets; their rows
  move at the next dashboard rebuild (L1).

- **2026-06-10 — saturated-ceiling cuts: sensitivity measured + embedded in the artifacts — audit-v2 G9 CLOSED (cuts unchanged)** (`profiling/process/build_saturated_ceiling.py`; lane L5).
  `PRESSURE_THRESHOLD = 2.5` is a one-time curve read-off ("saturates by ~2.5") with no
  derivation; the builder now recomputes each artifact's anchors at thresholds 2.0/3.0 and
  embeds the ACTUAL movements in the artifact `_notes` (audit's 0.2–1.8% claim confirmed):
  H100 out=28: 243.1 → 238.8 @2.0 (−1.77%) / 244.2 @3.0 (+0.45%), out=86: 134.9 → 134.2
  (−0.52%) / 137.3 (+1.78%); A100 out=27: 175.4 → 174.8 (−0.34%) / 176.1 (+0.40%), out=87:
  125.8 → 123.9 (−1.51%) / 126.1 (+0.24%); H100x2 out=28: 143.5 → 142.6 (−0.63%) / 144.2
  (+0.49%). `CLUSTER_SPLIT_OUTPUT = 50` checked at 40/60: anchors identical everywhere except
  the audit's known A100 split=40 wrinkle (short anchor 175.5 vs 175.4 — the ≤0.1 ms not-quite-
  empty gap, now recorded in the artifact itself). All three ceiling artifacts regenerated and
  verified **byte-identical except `_notes`**; production cuts and anchors unchanged; gate
  predictions byte-identical.

## Lane gate record

- Baseline (replay ON — realized `*_realized_*.json` pools present, warning-glob non-empty;
  captured before any edit): `/tmp/l5_base.{predictions,metrics}.json`.
- Post-change rerun (all four items in): `/tmp/l5_after.{predictions,metrics}.json` —
  **predictions AND metrics byte-identical** (`cmp` clean on both files). Contract met:
  H100/A100/H100x2 vLLM Llama gate rows unmoved.
- Full `pytest simulator/tests/ profiling/tests/ -q`: **191 passed, 1 skipped
  (pre-existing: dashboard JSON unavailable — gitignored L1 artifact), 12 subtests passed.**
