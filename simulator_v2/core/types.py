# simulator_v2/core/types.py

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from simulator_v2.core.mode import Mode, mode

### Configs for hardware and model ###

@mode(Mode.SHARED)
@dataclass(frozen=True)
class GpuConfig:
    name: str
    peak_flops_per_s: float
    util_flops: float
    peak_bw_bytes_per_s: float
    util_bw: float
    total_bytes: int
    gpu_mem_util: float
    scheduler_overhead_ms_per_step: float
    max_model_len: int = 32768       # vLLM --max-model-len; sets the chunked-prefill per-request cap
    request_overhead_ms: float = 25.0  # per-request host overhead (tokenize/dispatch/detok/return)
    # Measured KV-pool blocks for THIS deployment (gpu+model+tp) -- the observed free
    # blocks from vLLM engine traces. 0 -> fall back to the analytic estimate (kv_pool.py).
    kv_pool_blocks: int = 0
    # Measured serving-stack host costs (ms/token) for re-sending the prompt each turn
    # (parse + tokenize + IPC). Layered onto the prefill step, so they dominate TTFT at
    # low concurrency (cache hits: tiny GPU prefill, but the full context is re-tokenized).
    # 0 = off (GPU-only step, byte-identical).
    prefill_host_shared_ms_per_token: float = 0.0   # re-tokenize cached, amortized once/step
    prefill_host_perreq_ms_per_token: float = 0.0   # re-tokenize cached, per request (frac-spread)
    prefill_host_new_ms_per_token: float = 0.0       # new-token dispatch (on the chunk)
    # Measured cross-context attention slope (ms per chunk-token x resident-token pair,
    # full model): a prefill chunk of U tokens attending P already-resident tokens costs
    # +rate*U*P on top of the full-causal chunk grid. 0 = off (byte-identical).
    cross_attn_ms_per_token_pair: float = 0.0


@mode(Mode.SHARED)
@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_params: int
    kv_bytes_per_token: float
    kv_heads: int
    bytes_per_param: float
    cache_block_size: int
    # Architectural dims — needed by kernel_floor composition (not the Roofline
    # model). Optional so macro-only configs still load; compose validates them.
    n_layers: int | None = None
    hidden_dim: int | None = None
    intermediate_size: int | None = None
    n_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None

### Types for the simulator ###

@mode(Mode.SHARED)
@dataclass
class Turn:
    isl_tokens: float
    osl_tokens: float
    cache_hit_tokens: float
    new_prefill_tokens: float
    scheduled_requests: float

@mode(Mode.SHARED)
@dataclass
class TurnPrediction:
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float

@mode(Mode.SHARED)
@dataclass
class SchedulerSettings:
    """vLLM scheduler settings"""
    max_num_batched_tokens: int | None = None
    long_prefill_token_threshold: int | None = None
    max_num_seqs: int | None = None

@mode(Mode.SHARED)
@dataclass
class Cohort:
    """Cohort-level workload aggregates the amplifier needs beyond per-turn medians.

    `context_spread` is the cohort's context-size spread: the mean
    per-session `total_context / median_context`. The amplifier uses
    `z = pressure · context_spread` so a cohort with a few oversized sessions
    overflows the pool before median-session pressure hits 1. Default 1.0 = no
    spread (onset exactly at pool-full).

    SHARED: computable in both modes (the measured realized distribution in
    backtest; the input ISL distribution in forward) -- see
    getters/workload.cohort_context_spread."""
    context_spread: float = 1.0

@mode(Mode.SHARED)
@dataclass
class Cell:
    """One (profile, concurrency) benchmark cell: turn history/behaviour, plus optional ground truth."""
    turns: list[Turn]
    concurrency: int
    profile: str = ""
    ground_truth: list[TurnPrediction] | None = None
    # Per-session turn trajectories (varied sizes/turn counts); the queue sim
    # replays these. None -> consumers fall back to replicating `turns`.
    trajectories: list[list[Turn]] | None = None
    # Cross-session APC prefix (tokens): a profile-constant prefix every session
    # shares (system prompt / tool schemas), which vLLM dedups -- one session
    # prefills it, the rest hit. Measured from request_metadata. 0 = none.
    shared_prefix_tokens: float = 0.0

### Protocols for the simulator ###

@runtime_checkable
class Hardware(Protocol):
    kv_pool_blocks: int
    cache_block_size: int  # KV block size; the amplifier needs it for per-session blocks
    request_overhead_ms: float  # per-request host overhead (ms); the queue sim adds it at first token
    saturated_step_ms: float
    sched: SchedulerSettings
    cross_attn_ms_per_token_pair: float  # measured chunk-vs-resident attention slope (ms/token-pair)

    @property
    def prefill_host_rates(self) -> tuple[float, float, float]:
        """(shared, perreq, new) measured host re-tokenize/dispatch rates, ms/token --
        the serving-stack cost the queue sim layers onto the prefill step."""
        ...

    @mode(Mode.SHARED)
    def decode_step_ms(self, batch: int, ctx_tokens: float) -> float: ...
    @mode(Mode.SHARED)
    def prefill_ms(self, new: float, cached: float, batch: int) -> float: ...
    @mode(Mode.SHARED)
    def fused_step_ms(self, prefill_tokens: int, decode_batch: int, decode_ctx: float) -> float: ...
    @mode(Mode.BACKTEST)
    def saturated_ceiling_ms(self, output: float) -> float: ...
