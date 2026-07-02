# LLMServingSim 2.0 ↔ vLLM GT — cross-validation handoff

> For George. Goal: run LLMServingSim 2.0 on our workloads and check whether its TTFT/TPOT/E2EL
> predictions match our **measured vLLM ground truth** — i.e. whether LLMServingSim is genuinely
> accurate. Set up + verified 2026-07-01/02.

## 0. The verdict question

One table, MAPE vs the **same** measured vLLM GT, three predictor columns side by side:
**LLMServingSim 2.0 · our roofline/forward · our sim v2**. If LLMServingSim's MAPE is in the
ballpark of ours it's real; if it's 2–5× worse it isn't. **Every number is MAPE vs GT — never
sim-vs-sim.**

## 1. Config matrix — models × GT coverage × trace set

GT engine = **vLLM 0.19.0**. Bench_dir = `<gpu>_<model>_tp<N>_vllm` under the central store.
Profiles: `chat / osworld / swebench / terminalbench` (`-multiturn-synth`). Concurrencies
`1,5,10,20,40,80,120,160,200,256,320` (ragged on smaller GPUs / bigger models — use whatever the
bench_dir actually has).

| model | R2 trace dir | vLLM GT (GPU × TP) | LLMServingSim |
| --- | --- | --- | --- |
| Llama-3.1-8B | `llama-3.1/` | H100, A100, 3090 · tp1/2/4 | ✅ runs |
| Llama-3.1-70B | `llama-3.1/` | H100 tp4, A100 tp4 | ✅ runs |
| Mixtral-8x7B | `mixtral/` | H100 tp2/tp4, A100 tp4 | ✅ runs (`mixtral` arch) |
| gpt-oss-20b | `gpt-oss/` | 3090 tp1/2/4, A100 tp1/2/4, H100 tp1 | ⚠️ traces ready; needs a `gpt_oss` arch YAML to RUN |
| gpt-oss-120b | `gpt-oss/` | 3090 tp4, A100 tp4 | ⚠️ same — needs `gpt_oss` arch YAML |

Notes:

- **Traces are hardware-agnostic *within a tokenizer*** → one trace dir feeds every GPU/TP of its
  models. Llama-3.1-8B & 70B share `llama-3.1/`; gpt-oss-20b & 120b share `gpt-oss/`. So A100/3090
  need **no new traces** — just profile that GPU and score vs its GT.
- **Run only (GPU, TP) that have a bench_dir** — infeasible VRAM combos simply weren't benchmarked
  (e.g. 70B / gpt-oss-120b only exist at high TP).
- **LLMServingSim arch support** = `llama, qwen3, qwen3_moe, mixtral, phimoe`. **gpt-oss**
  (`model_type=gpt_oss`) needs a new arch YAML first — LLMServingSim's own
  `docs/profiler/adding-model-architecture.md` uses `gpt_oss` as *the* worked example. (Our GT also
  has `Qwen3.5`=`qwen3_5` and `Qwen2.5` models — also unsupported archs, not part of this set.)

## 2. Workloads — already built, on R2 (pull these)

**Hardware-agnostic but tokenizer-keyed** — one 44-cell set per **tokenizer family** (the workload is
identical across GPU/TP/engine, but token-IDs depend on the tokenizer; see the §1 notes). Pick the dir
matching your model's tokenizer — it feeds every GPU/TP for that family:

```text
https://pub-38e30ed030784867856634f1625c7130.r2.dev/data/llmservingsim_traces/<tokenizer>/<profile>_conc<N>.jsonl

# tokenizer ∈ { llama-3.1 (Llama-3.1-8B/70B) , mixtral (Mixtral-8x7B) , gpt-oss (gpt-oss-20b/120b) }
# profile   ∈ { chat, osworld, swebench, terminalbench }
# N (conc)  ∈ { 1,5,10,20,40,80,120,160,200,256,320 }
# each <tokenizer> dir: 44 jsonl + MANIFEST.csv (LLMServingSim agentic format)
```

**`pub-*.r2.dev` serves objects, not directories** — a folder URL (`.../llmservingsim_traces/` or
`.../<tokenizer>/`) 404s by design. Open a `<tokenizer>/MANIFEST.csv` for the file list, or list via
the S3 API:

```bash
aws s3 ls s3://agent-bench/data/llmservingsim_traces/ --recursive \
  --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com --profile r2
```

**Provenance (how they were built — do NOT reconstruct from GT `per_request` output).** They come from
inference-benchmark's *own* generator, so LLMServingSim sees exactly the input vLLM saw and
independently decides its cache/eviction/latency:

- `src/workloads/dataset.py:make_dataset(profile, seed=42, num_sessions=max(profile.num_sessions,conc),
  tokenizer_name=<the model's tokenizer>, max_context=32768, context_safety_margin=256)`
- env (from `scripts/bench_jobs.json`): `DISTRIBUTIONAL_SYNTHETIC_STYLE=code
  DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN=3.8 DISTRIBUTIONAL_PREFIX_AWARE=1
  DISTRIBUTIONAL_SHARED_PREFIX_TOKENS=1024`
- each turn's chat-templated prompt is tokenized → `input_tok_ids` (preserves the **1024-token
  block-aligned shared prefix** → cross-session cache hits, exactly as vLLM had them)
- `output_toks` = sampler-prescribed `max_tokens`; `arrival_time_ns=0`, `tool_duration_ns=0`
  (closed loop: `semaphore == concurrency == num_sessions`)

Regenerate with:

```bash
python3 simulator_v2/scripts/build_llmservingsim_traces.py --tokenizer <tok_dir> --label <name>
```

Verified vs GT: **cached/new split and session counts are exact**; `input_toks` is exact for
`llama-3.1`/`mixtral` and **+4 const/turn for `gpt-oss`** (harmony generation-prompt suffix — the
per-turn split is still exact, so cache behaviour is unaffected).

## 3. LLMServingSim setup

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
# build ASTRA-sim — easiest via their container (protobuf/abseil/boost prebuilt):
docker run -d --name lss -v $PWD/LLMServingSim:/app/LLMServingSim -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 sleep infinity
docker exec lss bash -c 'pip3 install -q pyyaml pyinstrument transformers datasets msgspec \
  scikit-learn xgboost==3.1.2 pandas numpy && ./scripts/compile.sh'   # ~10 min
```

**Per-hardware latency profile (the GPU step).** LLMServingSim ships no H100/A100/3090 data — run
their vLLM-layerwise profiler once per (GPU, model, tp) in a vLLM-0.19 env (dummy weights, single
layer — no real weights or HF token needed):

```bash
python3 -m profiler profile meta-llama/Llama-3.1-8B --hardware H100 --tp 1 \
  --max-num-batched-tokens 2048 --max-num-seqs 256      # add --skip-skew for a fast first pass
# output -> profiler/perf/H100/meta-llama/Llama-3.1-8B/bf16/tp1/
```

**Run a cell** (sim runs on CPU; put a trace at `_ourtrace/<cell>.jsonl`):

```bash
python3 -m serving --cluster-config configs/cluster/<gpu_model_tpN>.json \
  --dtype bfloat16 --block-size 16 --enable-prefix-caching --enable-chunked-prefill \
  --max-num-seqs 256 --max-num-batched-tokens 2048 \
  --dataset _ourtrace/swebench_conc80.jsonl --output outputs/swebench_conc80.csv
```

**Example** cluster configs (single node/instance, H100): `configs/cluster/h100_llama8b_tp1.json`,
`h100_llama8b_tp2.json`, `h100_llama70b_tp4.json`. For any other (GPU, model, TP), copy one and set
`model_name`, `tp_size` + `num_npus`, and `hardware` (`H100`/`A100`/`3090` — **must match the profiled
`perf/<hardware>/...` dir**) with `npu_mem` = that GPU's HBM (H100/A100 ≈ 80 GB @ 3350/2039 GB/s,
3090 = 24 GB @ 936 GB/s).

**Two gotchas:** sanity-check the sim's reported KV blocks land near GT's measured **27,250 blocks**
(8B tp1) — nudge `npu_mem` if far off; and **do not leave `max_num_seqs` unlimited** — that KV-OOMs the
fat cells.

## 4. ⚠️ Known blocker — profiler hangs in the `attention` phase

On this setup (vLLM 0.19.0) the profiler completes `dense` + `per_sequence` then **deadlocks entering
`attention`** — GPU 0%, futex wait, no progress. Reproduced in **both** multiprocessing and
single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) modes. This blocks the latency profile, hence the
real MAPE. Untried options: a different vLLM point-release, a coarser attention grid
(`--attention-chunk-factor`/`--attention-kv-factor` larger, smaller `--attention-max-kv`), or an
upstream bug report. Everything downstream (sim + scoring) works once a profile exists — confirmed by
running our traces against the shipped Qwen profile (clean run, 93.7% prefix-hit, correct per-turn CSV).

## 5. Scoring — `simulator_v2/scripts/score_llmservingsim.py`

Reduces LLMServingSim's per-request CSV **exactly like our pipeline**: per-turn **median** → **mean
over turn-indices** = cell MAPE vs GT. `request id` maps to `(session, turn)` by replaying the trace in
order (deterministic). Compare on **TTFT, TPOT, E2EL** (`latency` = E2EL; all ns).

```bash
python3 simulator_v2/scripts/score_llmservingsim.py \
  --sim-dir outputs/h100_tp1 --trace-dir <traces> --config h100_Llama-3.1-8B_tp1_vllm --out sheet.csv
```

## 6. Google Sheet — one row per (config, profile, conc, metric)

| config | profile | conc | metric | GT | LLMServingSim2.0 | Ours (roofline) | Ours (sim v2) | MAPE_llmsim | MAPE_ours | pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

**George** fills `LLMServingSim2.0` (+ `MAPE_llmsim`). **Kevin** fills `GT`, our columns, `pressure`.

## 7. Verdict rules / caveats

1. **Judge on TPOT and on TTFT where `pressure > 1`.** Neither LLMServingSim nor our sim models the
   vLLM API-server frontend (HTTP/tokenize/IPC), so **both** under-predict TTFT in the sub-saturation
   band for the same reason — it doesn't discriminate engine quality. (If LLMServingSim nails that band,
   flag it — interesting.)
2. Same GT only; never sim-vs-sim.
3. Both sims are *told* the output length (they predict latency for a request shape).
4. Ignore LLMServingSim's "sub-3% end-to-end" headline — that's their validation setup, not our cells.
5. Match the run config to §3 (dtype / block-size / max_model_len 32768 / prefix-cache + chunked-prefill
   on) or MAPEs won't be comparable.
