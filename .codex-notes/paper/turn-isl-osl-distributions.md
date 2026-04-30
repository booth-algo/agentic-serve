# Turn, ISL, And OSL Distributions

Generated from `inference-benchmark/data/distributions/*.json` on 2026-04-30.

`ISL` below means full prompt/context tokens at that turn (`total_context_tokens`). `New prefill` is the incremental prompt-token delta used for prefix-cache-aware synthetic traces. `OSL` means output tokens for the turn.

## Overall Distribution

| Workload | Sessions | Turn samples | Turns p50/p90/max | ISL p50/p90/p95/max | New prefill p50/p90 | OSL p50/p90/p95/max | Cache hit p50/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat-multiturn | 423 | 4,619 | 10/18/18 | 1,291/1,739/1,762/1,979 | 159/307 | 169/297/308/331 | 0.85/1.00 |
| osworld-multiturn | 60 | 788 | 8/30/30 | 1,399/4,660/9,249/12,454 | 4/1,220 | 85/111/122/462 | 1.00/1.00 |
| swebench-multiturn | 165 | 15,509 | 85/152/320 | 9,995/27,980/36,719/114,294 | 56/694 | 32/86/175/2,184 | 0.99/1.00 |
| terminalbench-multiturn | 267 | 20,042 | 61/130/876 | 7,811/29,536/45,666/232,823 | 60/575 | 31/101/195/5,344 | 0.99/1.00 |

## Turn Count Shape

| Workload | Turn-count distribution summary |
|---|---|
| chat-multiturn | Exactly 5, 10, or 18 turns from legacy short/medium/long buckets: 143 rows at 5 turns, 142 at 10 turns, 138 at 18 turns. |
| osworld-multiturn | Broad but capped at 30 turns: p50=8, p90=30, max=30. 17 of 60 sessions hit 30 turns. |
| swebench-multiturn | Long agent traces: p50=85, p90=152, max=320. |
| terminalbench-multiturn | Longest tail: p50=61, p90=130, max=876. |

## Per-Turn Medians

Only the first 12 turns are shown here so the table stays readable. The distribution JSONs contain the full per-turn series.

### chat-multiturn

| Turn | n | ISL p50 | New prefill p50 | OSL p50 | Cache hit p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 423 | 90 | 90 | 224 | 0.00 |
| 2 | 423 | 377 | 277 | 250 | 0.27 |
| 3 | 423 | 676 | 297 | 248 | 0.56 |
| 4 | 423 | 980 | 290 | 253 | 0.73 |
| 5 | 423 | 1,276 | 208 | 276 | 0.77 |
| 6 | 280 | 1,342 | 173 | 217 | 0.87 |
| 7 | 280 | 1,377 | 159 | 216 | 0.90 |
| 8 | 280 | 1,442 | 86 | 190 | 0.94 |
| 9 | 280 | 1,587 | 58 | 190 | 0.96 |
| 10 | 280 | 1,708 | 120 | 175 | 0.93 |
| 11 | 138 | 1,626 | 1 | 163 | 1.00 |
| 12 | 138 | 1,641 | 12 | 149 | 0.99 |

### osworld-multiturn

| Turn | n | ISL p50 | New prefill p50 | OSL p50 | Cache hit p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 1,241 | 1,241 | 84 | 0.00 |
| 2 | 52 | 1,258 | 13 | 74 | 0.99 |
| 3 | 49 | 1,263 | 3 | 81 | 1.00 |
| 4 | 44 | 1,316 | 40 | 80 | 0.99 |
| 5 | 43 | 1,331 | 9 | 84 | 0.99 |
| 6 | 40 | 1,498 | 7 | 82 | 0.99 |
| 7 | 36 | 1,564 | 10 | 86 | 1.00 |
| 8 | 32 | 1,418 | 3 | 92 | 1.00 |
| 9 | 28 | 1,352 | 5 | 87 | 1.00 |
| 10 | 27 | 1,374 | 3 | 86 | 1.00 |
| 11 | 25 | 1,365 | 3 | 89 | 1.00 |
| 12 | 24 | 1,364 | 3 | 93 | 1.00 |

### swebench-multiturn

| Turn | n | ISL p50 | New prefill p50 | OSL p50 | Cache hit p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 165 | 1,032 | 1,032 | 36 | 0.00 |
| 2 | 165 | 1,092 | 57 | 9 | 0.95 |
| 3 | 165 | 1,200 | 58 | 18 | 0.95 |
| 4 | 165 | 1,336 | 61 | 20 | 0.96 |
| 5 | 165 | 1,518 | 60 | 24 | 0.96 |
| 6 | 165 | 1,796 | 61 | 28 | 0.96 |
| 7 | 165 | 1,989 | 66 | 25 | 0.96 |
| 8 | 165 | 2,164 | 63 | 22 | 0.97 |
| 9 | 165 | 2,525 | 62 | 24 | 0.97 |
| 10 | 165 | 2,791 | 63 | 28 | 0.97 |
| 11 | 165 | 2,937 | 66 | 24 | 0.97 |
| 12 | 165 | 3,211 | 65 | 25 | 0.98 |

### terminalbench-multiturn

| Turn | n | ISL p50 | New prefill p50 | OSL p50 | Cache hit p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 267 | 980 | 980 | 44 | 0.00 |
| 2 | 267 | 1,048 | 59 | 10 | 0.95 |
| 3 | 266 | 1,156 | 78 | 21 | 0.93 |
| 4 | 265 | 1,275 | 57 | 27 | 0.95 |
| 5 | 264 | 1,395 | 57 | 31 | 0.96 |
| 6 | 262 | 1,555 | 57 | 34 | 0.96 |
| 7 | 260 | 1,756 | 61 | 31 | 0.96 |
| 8 | 257 | 1,904 | 61 | 31 | 0.97 |
| 9 | 253 | 2,052 | 62 | 29 | 0.97 |
| 10 | 253 | 2,207 | 65 | 32 | 0.97 |
| 11 | 252 | 2,325 | 61 | 32 | 0.97 |
| 12 | 251 | 2,463 | 59 | 32 | 0.97 |

## Notes For Paper Framing

- The first turn is not generally the whole story for agentic workloads. SWE-bench and TerminalBench have modest first-turn medians around 1K tokens, then accumulate long context over many turns.
- New-prefill tokens are much smaller than full ISL after turn 1, which is why prefix-cache-aware benchmarking is required.
- Chat multi-turn is structurally different from the agentic traces: it has fixed bucketed turn counts and much shorter total context.
- OSWorld has a high cache-hit estimate after turn 1, but its source includes non-growth/decrease behavior in some turns, so treat the exact delta distribution as approximate unless future runner telemetry is available.
