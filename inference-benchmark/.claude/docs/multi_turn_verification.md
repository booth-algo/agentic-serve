# Multi-Turn Benchmark Verification Report

**Date:** 2026-03-27  
**Investigator:** Scientist agent  
**Server:** SGLang, Llama-3.1-8B-Instruct, localhost:8000 (no prefix caching)  
**Dataset:** Aeala/ShareGPT_Vicuna_unfiltered (HuggingFace)

---

## Objective

Verify that the multi-turn benchmark implementation correctly:
1. Sends growing conversation histories with proper message structure
2. Increases input token count at each turn
3. Preserves prefix sharing across turns
4. Uses interleaved round-robin scheduling across sessions
5. Uses pre-recorded ShareGPT assistant replies (not model responses)
6. Tags per-turn metrics correctly in result files

---

## Verification 1: Message Structure

**Method:** Instantiated `ShareGPTMultiTurnDataset(min_turns=3, max_turns=5, num_sessions=5)` and inspected the `BenchmarkRequest.messages` list for each turn of each session.

**Expected:**
- Turn 0: 2 messages `[system, user]`
- Turn 1: 4 messages `[system, user, assistant, user]`
- Turn 2: 6 messages `[system, user, assistant, user, assistant, user]`
- Turn N: `2 + 2*N` messages, always ending with `user`

**Evidence (Session 8, 5 turns):**
```
Turn 0: 2 msgs | roles=['system', 'user']                → expected 2  ✓
Turn 1: 4 msgs | roles=['system', 'user', 'assistant', 'user']     → expected 4  ✓
Turn 2: 6 msgs | roles=['system', 'user', 'assistant', 'user', 'assistant', 'user'] → expected 6  ✓
Turn 3: 8 msgs | roles=[system, user, assistant, user, assistant, user, assistant, user] → expected 8  ✓
Turn 4: 10 msgs | roles=[...] → expected 10  ✓
```

All 3 inspected sessions (8, 13, 7) matched the expected pattern exactly.

**Result: PASS**

---

## Verification 2: Growing Context

**Method:** Measured total content character length across all messages for each turn within a session. Verified strictly increasing.

**Evidence (Session 8):**
```
Turn 0:  123 chars
Turn 1:  788 chars  (growing=True)
Turn 2: 1415 chars  (growing=True)
Turn 3: 3273 chars  (growing=True)
Turn 4: 4829 chars  (growing=True)
```

**Live server validation:** Sent one session's turns to localhost:8000 and captured `prompt_tokens` from the server's usage response:
```
Turn 0:   63 input tokens
Turn 1:  842 input tokens
Turn 2: 1344 input tokens
Turn 3: 1422 input tokens
```
Server-reported input tokens are strictly increasing, confirming the growing context reaches the server.

**Smoke test aggregate data** (200 sessions, per-turn file):
```
Turn 0: avg_isl =  105.5 tokens
Turn 1: avg_isl =  484.6 tokens
Turn 2: avg_isl =  860.3 tokens
Turn 3: avg_isl = 1149.1 tokens
Turn 4: avg_isl = 1356.8 tokens
```
Monotonically increasing across all 5 turns.

**Result: PASS**

---

## Verification 3: Prefix Sharing

**Method:** For each turn N (N > 0), verified that the first `2 + 2*(N-1)` messages are identical (same role and content) to all messages in turn N-1. This confirms history is growing by appending, not replacing.

**Evidence (Session 8):**
```
Turn 1: first 2 msgs match Turn 0's 2 msgs  ✓
Turn 2: first 4 msgs match Turn 1's 4 msgs  ✓
Turn 3: first 6 msgs match Turn 2's 6 msgs  ✓
Turn 4: first 8 msgs match Turn 3's 8 msgs  ✓
```

Verified across all 3 inspected sessions. The code achieves this through `list(messages_so_far)` snapshot copies in `_load()` (line 542 of dataset.py), which creates a new list at each turn while the underlying `messages_so_far` list continues to grow.

**Result: PASS**

---

## Verification 4: Interleaved Round-Robin Scheduling

**Method:** Examined both the `_build_flat_requests()` output and the `run_multi_turn_benchmark()` code path.

**Dataset-level interleaving** (`_build_flat_requests`):
```
Flat request order:
Idx  Session  Turn
  0        8     0    ← All sessions' turn 0 first
  1       13     0
  2        7     0
  3        6     0
  4       14     0
  5        8     1    ← Then all sessions' turn 1
  6       13     1
  7        7     1
  ...
 19        8     4    ← Fewer sessions at later turns
 20        6     4
 21       14     4
```

Turn indices are non-decreasing (all turn 0 before turn 1, etc.), and within each turn round, all eligible sessions are present.

**Runner-level scheduling** (`run_multi_turn_benchmark`, lines 165-193 of runner.py):
```python
for turn_idx in range(max_turns):          # outer: by turn
    for conv_session in sessions:           # inner: by session
        if turn_idx < len(conv_session.turns):
            turn_requests.append(...)
    tasks = [dispatch(...) for ... in turn_requests]
    completed = await asyncio.gather(*tasks)  # all turn N concurrent
```

This is **batched round-robin**: all sessions' turn N are dispatched concurrently via `asyncio.gather`, then the runner awaits completion before moving to turn N+1. This is stricter than pure interleaving (it guarantees all turn-N requests complete before turn N+1 starts), which maximizes KV cache eviction pressure between turns.

**Result: PASS**

---

## Verification 5: Pre-Recorded Assistant Replies

**Method:** Verified that assistant messages in turn N's `BenchmarkRequest.messages` are:
1. Present at dataset construction time (before any HTTP calls)
2. Sourced from ShareGPT `gpt` turns, not from model responses
3. Consistent across turns (assistant[i] in turn N == assistant[i] in turn N+1)

**Evidence (Session 8):**
```
Turn 0: no assistant msgs (correct)
Turn 1: 1 assistant msg: "It's possible to use algebraic topology to solve Minesweeper..."
Turn 2: 2 assistant msgs: same msg[0] as Turn 1 + new msg[1]
Turn 3: 3 assistant msgs: same msg[0:2] as Turn 2 + new msg[2]
Turn 4: 4 assistant msgs: same msg[0:3] as Turn 3 + new msg[3]
```

The code path in `_load()` (dataset.py lines 534-547):
```python
for human_msg, assistant_msg, osl_est in pairs:
    messages_so_far.append({"role": "user", "content": human_msg})
    turns.append(BenchmarkRequest(messages=list(messages_so_far), ...))
    messages_so_far.append({"role": "assistant", "content": assistant_msg})
    # ^ assistant_msg comes from ShareGPT: turn.get("value", "")
```

The model's actual response is never captured or injected. The `run_multi_turn_benchmark` function uses the pre-built `session.turns[turn_idx]` requests directly, never modifying them with model output.

**Result: PASS**

---

## Verification 6: Per-Turn Metrics

**Method:** Analyzed `results/multi_turn_smoke_test.json` and `results/multi_turn_smoke_test_per_turn.json`.

**6a: Per-turn file structure**
```
Turn indices sequential [0..4]: True
5 turn entries, each with: turn_index, num_requests, successful, mean_ttft_ms,
median_ttft_ms, p90_ttft_ms, p99_ttft_ms, mean_tpot_ms, median_tpot_ms,
mean_e2el_ms, median_e2el_ms, avg_input_tokens, avg_output_tokens
```

**6b: turn_index tagging in main results file**
```
Total per-request entries: 808
Entries with turn_index field: 808/808 (100%)
```

**6c: Distribution consistency**
```
Turn 0: 200 requests in per-request == 200 in per-turn file  ✓
Turn 1: 200 requests in per-request == 200 in per-turn file  ✓
Turn 2: 200 requests in per-request == 200 in per-turn file  ✓
Turn 3: 136 requests in per-request == 136 in per-turn file  ✓
Turn 4:  72 requests in per-request ==  72 in per-turn file  ✓
```

**6d: Metrics trend validation**

| Turn | Avg ISL | Median TTFT | p90 TTFT | Median TPOT | N |
|------|---------|-------------|----------|-------------|---|
| 1    | 105.5   | 27.1 ms     | 60.3 ms  | 8.9 ms      | 200 |
| 2    | 484.6   | 33.5 ms     | 48.2 ms  | 7.4 ms      | 200 |
| 3    | 860.3   | 37.7 ms     | 58.5 ms  | 7.9 ms      | 200 |
| 4    | 1149.1  | 42.7 ms     | 73.8 ms  | 9.1 ms      | 136 |
| 5    | 1356.8  | 46.1 ms     | 167.9 ms | 10.3 ms     | 72  |

Median TTFT increases monotonically with context size (27.1 → 33.5 → 37.7 → 42.7 → 46.1 ms), which is expected: larger prefill = longer time to first token. Request counts decrease at later turns because some sessions have fewer turns (max_turns=5 profile, but not all ShareGPT conversations have 5 valid pairs).

**Result: PASS**

---

## Summary

| Check | Description | Result |
|-------|-------------|--------|
| 1 | Message structure (correct count and roles per turn) | **PASS** |
| 2 | Growing context (input tokens strictly increasing) | **PASS** |
| 3 | Prefix sharing (turn N is prefix of turn N+1) | **PASS** |
| 4 | Interleaved round-robin scheduling | **PASS** |
| 5 | Pre-recorded assistant replies from ShareGPT | **PASS** |
| 6 | Per-turn metrics and turn_index tagging | **PASS** |

**Overall: All 6 verification checks PASS.**

---

## Observations

1. **Batched scheduling is stricter than design doc.** The design doc describes interleaving as `[A1,B1,C1,A2,B2,C2,...]`, implying fine-grained interleaving. The implementation uses `asyncio.gather` per turn round, meaning all turn-N requests must complete before turn N+1 starts. This is actually better for cache eviction testing (all sessions complete turn N, their KV cache entries age, then turn N+1 tests reuse).

2. **TTFT trend confirms functional correctness.** Median TTFT increases with context size (27ms at 106 tokens to 46ms at 1357 tokens), which is the expected behavior for growing-context prefill without prefix caching enabled.

3. **Request count drop-off.** 200 sessions at turn 0, but only 72 at turn 4, because ShareGPT conversations vary in length and the `max_osl_tokens`/`min_osl_tokens` filters reject turns with extreme reply lengths.

## Limitations

- Live verification used only 1 session (4 turns) due to time constraints. Statistical confidence comes from the 200-session smoke test results.
- No prefix caching was enabled on the test server, so we cannot verify prefix cache *hit rate* improvements. The TTFT trend shows the expected no-cache behavior (monotonically increasing with context size).
- The verification does not test edge cases like sessions with exactly `min_turns` or conversations where all pairs are filtered out.

---

*Visualization saved to: `.omc/scientist/figures/multi_turn_verification.png`*
