# Workload Profile Audit — 2026-04-29

Audited `inference-benchmark/src/workloads/profiles.py`,
`inference-benchmark/src/workloads/dataset.py`, and current
`inference-benchmark/dashboard/public/data.json`.

## Summary

The single-turn ShareGPT profiles are not length buckets. `chat-short`,
`chat-medium`, and `chat-long` only set upper bounds for ShareGPT filtering,
and the loader does not enforce a minimum ISL. All three therefore sample
mostly short first-turn ShareGPT prompts.

ShareGPT multi-turn profiles have real turn-count separation, but still weak
token-length separation. Agent multi-turn profiles are stronger: they bucket
by step count and produce substantially longer contexts, although they are
turn buckets rather than exact token buckets.

Synthetic random profiles are the only profiles that try to target exact
ISL/OSL lengths.

## Actual Current Data

Zero-token legacy/broken rows excluded.

| Profile | Dataset | Declared ISL/OSL | n | Actual ISL median [min,max] | Actual OSL median [min,max] | Verdict |
|---|---|---:|---:|---:|---:|---|
| chat-short | sharegpt | 500/300 | 390 | 129 [71,180] | 169 [50,183] | Overlapping short ShareGPT |
| chat-medium | sharegpt | 2000/1000 | 340 | 157 [97,211] | 286 [215,327] | Overlapping short ShareGPT |
| chat-long | sharegpt | 8000/2000 | 337 | 187 [118,258] | 299 [226,321] | Overlapping short ShareGPT |
| chat-multiturn-short | sharegpt-multi-turn | 8192/1000 | 143 | 673 [591,741] | 298 [288,312] | Weak length separation |
| chat-multiturn-medium | sharegpt-multi-turn | 16384/1500 | 142 | 835 [729,914] | 246 [238,254] | Weak length separation |
| chat-multiturn-long | sharegpt-multi-turn | 32768/2000 | 138 | 937 [844,1006] | 149 [146,156] | Weak length separation |
| coding-agent | jsonl | 17000/800 | 150 | 6298 [6036,7376] | 282 [123,351] | Real file distribution |
| prefill-heavy | random | 8192/256 | 191 | 9223 [7891,16751] | 250 [108,256] | Controlled long prefill |
| decode-heavy | random | 256/4096 | 149 | 300 [280,537] | 353 [214,2747] | Intended decode stress, but output often ends early |
| random-1k | random | 1024/1024 | 231 | 1160 [1017,2118] | 301 [115,1024] | Intended 1k/1k, but output often ends early |
| swebench-multiturn-short | swebench-multi-turn | 32768/2000 | 70 | 8002 [6428,8716] | 37 [32,39] | Strong long-context agent profile |
| swebench-multiturn-medium | swebench-multi-turn | 65536/2000 | 62 | 13355 [8317,13719] | 41 [36,45] | Strong long-context agent profile |
| terminalbench-multiturn-short | terminalbench-multi-turn | 32768/2000 | 93 | 5045 [2128,5450] | 47 [31,56] | Strong-ish agent profile |
| terminalbench-multiturn-medium | terminalbench-multi-turn | 65536/2000 | 67 | 10472 [6752,10726] | 60 [47,68] | Strong long-context agent profile |
| osworld-multiturn-short | osworld-multi-turn | 32768/500 | 79 | 5059 [2715,5946] | 81 [38,83] | Agent turn bucket, not token bucket |
| osworld-multiturn-medium | osworld-multi-turn | 65536/500 | 85 | 4666 [2867,5695] | 83 [39,86] | Agent turn bucket; not monotonic in current rows |

## Implementation Notes

- `WorkloadProfile.isl_tokens` is documented as a max bound for ShareGPT:
  `profiles.py`, `WorkloadProfile.isl_tokens`.
- `ShareGPTDataset` only rejects `isl_est > max_isl_tokens`; it has no
  `min_isl_tokens`.
- `ShareGPTDataset` extracts only the first human+assistant pair, so
  `chat-long` does not use full conversation history.
- `ShareGPTMultiTurnDataset` filters by min/max turns, not target token
  lengths. It does build growing history, but the selected ShareGPT sessions
  remain short in current data.
- `TrajectoryMultiTurnDataset` also filters by turn count and max token caps,
  not exact token buckets.

## Profiles With No Current Rows

- `swebench-multiturn-long`
- `swebench-multiturn-xl`
- `terminalbench-multiturn-long`
- `terminalbench-multiturn-xl`
- `osworld-multiturn-long`
- `chat-multiturn-xl`
- `fixed-seq128`
- `test`

## Recommendation

For paper figures and predictor validation:

- Use `chat-short/medium/long` only as natural short ShareGPT chat.
- Do not use current single-turn ShareGPT profiles as short/medium/long
  prefill buckets.
- Use `prefill-heavy`, `coding-agent`, `swebench-multiturn-*`,
  `terminalbench-multiturn-*`, and `osworld-multiturn-*` for long-context
  claims.
- If real single-turn chat length buckets are needed, add `min_isl_tokens`
  or percentile/top-k selection to `ShareGPTDataset`, then rebuild results.
