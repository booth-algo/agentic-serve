# simulator_v2 — knobs reference

Every tunable in one place. Knobs live in **two YAMLs** (`configs/gpu_configs/`,
`configs/model_configs/`) plus a handful of **code constants**. This doc maps all of
them: value (H100 / Llama-3.1-8B), where to change it, provenance, and what it drives.

**Provenance legend:** `spec` = datasheet · `arch` = model architecture · `measured`
= NCU/benchmark microbench · `derived` = computed from others · `config` = real
deployment setting · `fit` = tuned/compensating · `struct` = law with no free
parameter · `⚠ handwavy` = placeholder/probe, the accuracy levers to re-measure.

---

## 1. GPU — `configs/gpu_configs/h100.yaml` → `GpuConfig`

| knob | value | prov | controls |
|---|---|---|---|
| `compute.peak_flops_per_s` | 9.89e14 | spec | GEMM/prefill compute roofline |
| `compute.util_flops` | 0.65 | ⚠ handwavy | prefill FLOP utilization |
| `memory.peak_bw_bytes_per_s` | 3.35e12 | spec | decode/KV bandwidth roofline |
| `memory.util_bw` | 0.93 | ⚠ handwavy | decode BW utilization |
| `memory.total_bytes` | 85.9e9 (~80 GiB) | spec | KV-pool sizing |
| `memory.gpu_mem_util` | 0.90 | config | vLLM `--gpu-memory-utilization` |
| `scheduler.overhead_ms_per_step` | 5.7 | ⚠ handwavy | per-step scheduler cost (Roofline only) |
| `scheduler.max_model_len` | 32768 | config | sets chunked-prefill cap (×0.04) |
| `scheduler.request_overhead_ms` | 0.0 (RETIRED) | measured | superseded 2026-07-03: the client-referenced frontend floor carries the send+return path |
| `prefill_host.*` | 0.0 (RETIRED) | — | superseded 2026-07-03 by `frontend:` (was the same tokenize cost measured at the c1 stage split; keeping both double-counts — V1 A/B broke every c1 cell) |
| `frontend.floor_ms` | 9.8 | measured | client-referenced per-request frontend floor (send+parse+return path) |
| `frontend.new_ms_per_token` | 6.0e-3 | measured | frontend slope on new tokens (tokenize the re-sent prompt) |
| `frontend.cached_ms_per_token` | 6.1e-3 | measured | frontend slope on cached tokens (same re-tokenize; APC can't skip it) |
| `frontend.mult_curve` | (0,1.0)…(40,1.45) | measured | streaming-load inflation of f, evaluated at herd/2 (decoy sweep D=2..160; D≥40 mean anchors the tail) |
| `frontend.lanes_curve` | (1,1.0)…(160,2.25) | measured | effective frontend parallelism vs herd size (client-side drain rate) |
| `compute.cross_attn_ms_per_token_pair` | 7.29e-7 | measured | chunk-vs-resident cross-attention slope (rate·U·P per prefill chunk); drives the saturated-tail TTFT slope. Constant from the FA3-cached grid fit (grid CSV not in profile_data — re-profile to upgrade to interp). 0 = off |
| `memory.kv_pool_blocks` | 27250 | measured | KV pool; overrides analytic estimate (0 = analytic) |

## 2. Model — `configs/model_configs/llama3.1-8b.yaml` → `ModelConfig`

| knob | value | prov | controls |
|---|---|---|---|
| `n_params` | 8.03e9 | arch | weight bytes, compute roofline |
| `kv_bytes_per_token` | 131072 | arch | KV-pool tokens, decode BW |
| `kv_heads` | 8 | arch | KV sharding, attention |
| `bytes_per_param` | 2.0 | arch | weight bytes (bf16) |
| `cache_block_size` | 16 | config | blocks/session, pressure |
| `n_layers / hidden_dim / intermediate_size` | 32 / 4096 / 14336 | arch | kernel-floor GEMM shapes |
| `n_heads / head_dim / vocab_size` | 32 / 128 / 128256 | arch | attention + lm_head shapes |

## 3. Scheduler — `SchedulerSettings` (built in `getters/hardware.py.__post_init__`)

| knob | value | prov | controls | ⚠ |
|---|---|---|---|---|
| `max_num_batched_tokens` | 8192 | config | per-step prefill token budget | hardcoded default in `hardware.py` — should come from deployment manifest |
| `long_prefill_token_threshold` | `max_model_len×0.04` = 1310 | config | per-request chunk cap | the `0.04` is vLLM's rule, hardcoded in `hardware.py` |
| `max_num_seqs` | 256 (`_DEFAULT_MAX_SEQS`) | config | max running/prefilling reqs | hardcoded in `queue_sim.py` |

## 4. Kernel floor (backtest) — `kernel_floor/*.py`

| knob | value | where | prov | controls |
|---|---|---|---|---|
| `dtype_bytes` | 2 | `sum_kernels.py`, kernel tables | config | bf16 byte width in roofline fallbacks |
| gemm `reduce` | `"min"` | `gemm.py load_gemm_table` | fit | duplicate-run reduction (`"mean"` biases decode floor +14%) |
| kernel artifact paths | `profile_data/kernels/**/{gpu}.*` | `getters/hardware.py _kernel_artifact_paths` | — | which measured tables load |

## 5. KV pool — `configs/kv_pool.py`

| knob | value | prov | controls |
|---|---|---|---|
| `RESERVE_BYTES` | 3.5e9 | ⚠ derived | non-torch/activation carve-out; only estimated constant in the pool formula (mean of 3 back-solved pools) |

## 6. TPOT amplifier — `engine/amplifier.py`

| knob | value | prov | controls |
|---|---|---|---|
| `SAT_SUSTAIN_LO / HI` | 9.0 / 24.0 | fit | output-length sustain gate (short turns can't reach the ceiling) |
| `context_spread` (arg) | 1.0 default | measured* | `z = pressure·context_spread`; *wired to `cohort_context_spread` but defaults to no-spread |
| `_overflow_weight` law | — | struct | eviction-recompute duty cycle; no free params (uses pressure, z, budget, ctx) |

## 7. TTFT queue sim — `engine/queue_sim.py`

Consumes §1 (`request_overhead_ms`, `prefill_host`), §3 (scheduler), §5 (pool). Its own constants:

| knob | value | prov | controls |
|---|---|---|---|
| `_DEFAULT_MAX_BATCHED` | 8192 | config | budget fallback when `sched` unset |
| `_DEFAULT_MAX_SEQS` | 256 | config | running-set fallback |
| `_TURNS_PER_SESSION` | 4096 | struct | rid encoding headroom (not tunable) |
| `_EVENT_GUARD` | 5e6 | struct | runaway-event safety cap (not tunable) |
| step price | `max(decode + prefill + cross, host)` | struct | additive GPU (fused pass, 2026-07-02; was max piggyback); host pipelines via max |
| shared-prefix pool dedup | — | struct | reservation + decode-growth net of `shared_prefix_tokens` (APC stores the span once); no knob — falls out of the measured shared-prefix input |

## 8. Roofline (forward) — `getters/hardware.py`

| knob | value | prov | controls |
|---|---|---|---|
| `saturated_step_ms` | 200.0 | ⚠ placeholder | forward-mode ceiling (unsolved — no measured anchors) |
| `tp` | 1 | config | tensor-parallel degree (KV sharding, weight split) |

## 9. Dashboard generator — `inference-benchmark/scripts/build_simulator_v2_predictions.py`

| knob | value | controls |
|---|---|---|
| `PROFILES` | 4 multi-turn (matches v1) | which cells are emitted (excludes chat-singleturn) |
| `GPU_KEY` / `BENCH_DIR` | `H100` / `h100_Llama-3.1-8B_tp1_vllm` | which config's cells |
| headline error | `_cell_mape` (mean per-turn APE) | matches v1's metric |

---

## The accuracy levers (⚠ handwavy, ranked by leverage)

1. **`prefill_host.*`** (§1) — probe-measured; raising/re-probing on real chat prompts is the top TTFT lever for the sub-saturation (pressure<1) band.
2. **`util_flops` / `util_bw`** (§1) — flagged handwavy in the YAML; set the kernel-floor roofline fallbacks and decode/prefill cost.
3. **`request_overhead_ms`** (§1) — flat 25 ms host floor; re-measure.
4. **`saturated_step_ms`** (§8) — forward ceiling placeholder; needs measured anchors.
5. **`RESERVE_BYTES`** (§5) — ±0.6 GB envelope; fine on 80 GB, ±30–60% on small GPUs.
6. **`SAT_SUSTAIN_LO/HI`, gemm `reduce`** — fit/compensating; revisit only with data.

> Not yet a single editable config: §3 (scheduler), §5 (RESERVE), §6–7 constants are
> hardcoded in code, not the YAML. Lifting the *tunable* ones (`max_num_batched_tokens`,
> `max_num_seqs`, the `0.04` chunk factor, `RESERVE_BYTES`, `SAT_SUSTAIN_*`) into the GPU
> YAML would make this a true single source of truth — see the open follow-up.

---

## v1 knobs: ported vs not

v2 is a deliberate **subset** of v1 — it takes the *physical* mechanisms and skips v1's
compensating fits. Status (from `simulator/closed_form_tpot.py` `RooflineParams` +
`ttft_queue_sim.py` constants + `kernel_tpot.py`):

| v1 knob | what it does | in v2? |
|---|---|---|
| `shared_prefix_tokens` (cross-session APC dedup) | one session pays a profile-constant shared prefix (osworld/swebench 1024, terminal 976, chat 48), the rest hit | ✅ ported (measured input) — biggest TTFT lever; osworld now beats v1 |
| measured `available_kv_blocks` (27250) | pins the observed pool vs analytic estimate | ✅ ported (`memory.kv_pool_blocks`) |
| response-resident (`ρ`) | prev-turn response KV is resident → attend, don't re-GEMM | ❌ **removed** — the ground truth already re-prefills the response, so any credit double-counts (see finding below) |
| `qsim_duplicate_session_fraction` | tracereplay cohorts repeat traces; twin's KV is a cross-session hit | ❌ workload artifact (finite trace pool), not production physics |
| hit/miss **freeze** at barrier | decide hit/miss on the barrier snapshot, not live | ❌ compensating fit; v2 decides live (correct) |
| `PREFILL_GEMM_UTIL_SAT` (util ramp 0.65→1.0) | prefill GEMM util ramps to 1.0 | ❌ measured plateau is 0.754; v2 uses the flat measured floor |
| `PREFILL_FA3_MS_PER_TOKEN2` (cached-prefix re-encode) | FA3 attention over the resident prefix | ✅ landed 2026-07-02 in recompute-tail form: `cross_attn_ms_per_token_pair` (rate·U·P per chunk; the earlier c1-band attempt stays reverted) |
| `PREFILL_TP_COMM` / `EP_ALLTOALL` | tp>1 all-reduce / MoE all-to-all | ❌ N/A at tp1-dense (→0) |
| eviction `policy='tail'` | MRU-first trim | ❌ trace-falsified, retired in v1 too |
| TPOT cell-state (**B2**) | fixes TPOT onset *timing* per cell | ❌ deferred (v2 amplifier is stateless B1) |
| MoE `n_active_params` / `ep_size` | active-param prefill / EP | ❌ N/A (dense) |

### Finding: response-resident (ρ) — removed as a structural double-count
An earlier take added `response_resident_fraction` (ρ) and set `ρ=1` on *first-principles*
grounds (the prev response's KV is resident, so attend it, don't re-GEMM), noting it
*worsened* backtest MAPE (26.9→28.6%) and reading that as an exposed sub-saturation
contention gap. **A per-step decomposition of chat conc120 (`engine/step_trace.py`)
disproved that premise.** The ground truth accounts prefill with
`cache_estimate_source = "previous_prompt_tokens"`: per request `new_prefill =
total_context − cached_context` *exactly*, and `cached_context[t] = total_context[t−1]`
(the previous **prompt**). That excludes the previous **response** (`asst[t−1]`,
~230 tok/turn) — so the ground-truth `new_prefill[t]` we feed the sim **already includes
the re-prefilled response**. Any ρ>0 removes it a *second* time.

That makes ρ a structural double-count for this ground truth, not a tunable: it is correct
only at 0 (a no-op), and wrong at every other value. So the knob was **removed** entirely
(`GpuConfig`, the `Hardware` protocol, `queue_sim` `_Req.prev_output` / `_ServerState` /
`_schedule` credit, and the YAML) — the sim now takes `new_prefill` at face value.

Evidence (chat conc120, TTFT): turn-0 measured median 362 ms ≈ the sim's own throughput on
the full 15 000-token herd (394 ms), but ρ=1 predicted 240 ms because it credited ~5 700 tok
away. Dropping the credit: **chat conc120 39.5→27.1%**; **aggregate 28.6→26.9%**, chat
39.4→32.6%, no material regression (osworld −0.5, swebench +0.2, terminal ≈flat).

ρ would only be a *real* mechanism (not a double-count) if a serving stack genuinely cached
the response AND the harness recorded that in `cached_context` — in which case `new_prefill`
would already exclude it and the correction belongs in the **workload loader**, not a
queue-sim multiplier. TODO: if an S7-style `/metrics` prefix-cache probe ever shows the
harness over-counts vs true GPU work, fix `new_prefill` at the source.
