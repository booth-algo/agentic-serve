# Multi-Turn Benchmark Design

## Design Decision: How to build growing conversation history

### Option A — Use actual server replies per turn

Send turn 1 → get model's reply → append to history → send turn 2 → ...

**Pros:**
- Tests actual growing context from the model being benchmarked
- Most realistic end-to-end simulation

**Cons:**
- Sequential within each session (can't pre-build requests)
- Non-deterministic — different models produce different replies, changing context length
- Reply quality affects subsequent turns
- Harder to reproduce across runs

### Option B — Use ShareGPT's pre-recorded replies to build growing history (CHOSEN)

Pre-build all requests using ShareGPT's existing assistant replies as history filler:

- Turn 1: `[system, human1]` → max_tokens=osl1
- Turn 2: `[system, human1, sharegpt_assistant1, human2]` → max_tokens=osl2
- Turn 3: `[system, human1, sharegpt_assistant1, human2, sharegpt_assistant2, human3]` → max_tokens=osl3

**Pros:**
- Deterministic and reproducible
- All requests pre-built → works with existing semaphore dispatch
- Enables interleaved round-robin scheduling [A1,B1,C1,A2,B2,C2,...] to stress KV cache eviction
- What we're measuring is prefix cache reuse on growing context, not reply quality

**Cons:**
- Assistant messages in history are from GPT-3.5 (ShareGPT source), not the model under test
- Slightly less realistic than live replies

**Why Option B is better for benchmarking:**
The goal is to measure how inference engines handle growing KV cache and prefix cache reuse
under interleaved multi-session load. Deterministic pre-built requests isolate the engine's
behavior from model output variance. The assistant messages are just "realistic context filler" —
their origin doesn't affect the metrics we care about (TTFT scaling, prefix cache hit rate, throughput under memory pressure).
