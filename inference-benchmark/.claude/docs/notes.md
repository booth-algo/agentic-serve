# llm-bench Code Walkthrough Notes (Legacy Reference)

> **Note:** This documents the OLD InferenceX benchmark tool that inference-benchmark was built to replace.
> Kept as reference for understanding design decisions and gaps we fixed.

## benchmark_dataset.py — Data layer

Three classes, all thread-safe via `threading.Lock`.

**`FileDataset`** — loads a single static prompt from a `.txt` file. Every request gets the same
prompt. Used for `output-short` and `output-long` profiles (fixed prefill/decode scenarios).

**`TestDataset`** — hardcoded "Hello, are you there?" string. Smoke testing only.

**`ShareGPTDataset`** — downloads `Aeala/ShareGPT_Vicuna_unfiltered` from HuggingFace, shuffles
with a fixed seed, takes the first N entries, extracts the first user message from each
conversation. Cycles back to the start when prompts run out. Gives varied ISL per request —
much more realistic than the static files.

Key design: `get_next_prompt()` is the only interface all code uses. Thread-safe via lock.

---

## throughput_benchmark.py — Main benchmark

**Concurrency model:** sliding window of threads. Launches threads one by one; when
`len(threads) >= concurrency`, joins the oldest thread before starting the next. At most
`concurrency` threads alive at once. Not truly async — if one request is slow it blocks the
queue from advancing past it.

**What it measures:** each thread puts `total_tokens` (prompt + completion) into a
`queue.Queue`. After all threads finish, sums tokens / wall-clock time = **tokens/second**.
That's the only metric reported.

**Profiles and max_tokens (hardcoded):**
- `output-short` → `long_input_short_output.txt` + `max_tokens=128` (prefill-heavy, ~1200 token input)
- `output-long` → `short_input_long_output.txt` + `max_tokens=1024` (decode-heavy, ~180 token input)
- `sharegpt` → ShareGPTDataset(500 prompts) + `max_tokens=180`

**Warmup:** sends `--warmup-requests` (default 5) requests, discards results, then runs the
actual benchmark with exactly `--concurrency` requests.

**Failure handling:** on any exception, puts `0` into the queue. Failed requests silently count
as zero tokens — throughput looks better than reality when errors occur.

---

## tftt_benchmark.py — TTFT benchmark

Uses `stream=True` in the payload, then iterates over raw response bytes. First byte arrival =
TTFT (`time.time() - start_time`).

**Approximation caveat:** iterates `for message in response:` which iterates HTTP chunks, not
SSE tokens. "First token" time is actually the first HTTP chunk, which may contain multiple
tokens. A proper implementation needs to parse SSE `data:` lines.

**All requests launched at once:** no sliding window — starts all `concurrency` threads
simultaneously. Measures TTFT under burst load, not steady-state.

**Output:** prints average TTFT only. No per-request data, no percentiles.

---

## Prompt files

- `prompts/long_input_short_output.txt` — ~1200 token input, used with `max_tokens=128`.
  Stresses the prefill phase.
- `prompts/short_input_long_output.txt` — ~180 token input, used with `max_tokens=1024`.
  Stresses the decode phase.

---

## Key gaps to fix when building the new client

1. **Thread-based sliding window** → replace with `asyncio` + `aiohttp` for accurate
   per-request timing and better concurrency at scale.
2. **TTFT approximation** → parse SSE `data:` lines properly to get true first-token timing.
3. **No per-request metrics** → need `RequestResult` dataclass capturing TTFT, per-token
   timestamps, total tokens, success/failure per request.
4. **No percentiles** → p50/p90/p99 for TTFT, TPOT, E2E latency.
5. **Silent failure** → track failed requests separately; include in throughput denominator.
6. **`max_tokens` hardcoded** → move into configurable profile definitions.
7. **No JSON output** → results should be saved to file for later analysis and charting.

---

## RunPod setup notes

- Use official image: `vllm/vllm-openai:v0.7.3` (pin a specific tag, not `latest` — avoids
  reproducibility issues like InferenceX #393).
- Port exposure: RunPod proxies ports, URL format is
  `https://<pod-id>-8000.proxy.runpod.net` not a raw IP.
- Recommended startup command:
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --api-key test
    # --enable-prefix-caching  <- add/remove per run, don't bake in
  ```
- Set env var: `HUGGING_FACE_HUB_TOKEN=hf_xxxx`
- `--max-model-len 32768` is more realistic than InferenceX's `ISL+OSL+margin` approach.
- Test both with and without `--enable-prefix-caching` — it's a major real-world optimization
  that InferenceX deliberately skips.
