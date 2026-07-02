# simulator_v2/kernel_floor/sum_kernels.py

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from simulator_v2.core.types import GpuConfig, ModelConfig
from simulator_v2.kernel_floor.elementwise import ElementwiseTable, load_elementwise_table
from simulator_v2.kernel_floor.flash_attn import (
    DecodeAttnGrid,
    PrefillAttnGrid,
    load_decode_attn_grid,
    load_prefill_attn_grid,
)
from simulator_v2.kernel_floor.gemm import GemmTable, load_gemm_table
from simulator_v2.core.mode import Mode, mode

_REQUIRED_ARCH = ("n_layers", "hidden_dim", "intermediate_size", "n_heads", "head_dim", "vocab_size")

@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class KernelFloor:
    """The four measured kernel-cost tables (gemm / decode-attn / prefill-attn /
    elementwise) bound to one model/GPU, composed into per-step costs."""
    model: ModelConfig
    gpu: GpuConfig
    gemm: GemmTable
    decode_attn: DecodeAttnGrid
    prefill_attn: PrefillAttnGrid
    elementwise: ElementwiseTable
    dtype_bytes: int = 2

    def _pricers(self) -> tuple[Callable[[int, int, int], float], Callable[[str, int], float]]:
        """(gemm, elem) closures bound to this floor's gpu/dtype."""
        db = self.dtype_bytes
        gemm = lambda mm, nn, kk: self.gemm.gemm_us(mm, nn, kk, self.gpu, dtype_bytes=db)
        elem = lambda kernel, elems: self.elementwise.elementwise_us(
            kernel, elems, self.gpu, dtype_bytes=db
        )
        return gemm, elem

    def _block_linear_us(self, m_tokens: int, gemm, elem) -> float:
        """Cost (us) of the NON-attention part of one transformer block for
        `m_tokens` tokens: its 4 (fused) GEMMs + 6 elementwise ops (norms,
        rotary, activation, residuals). Same whether the tokens are decode rows or
        a prefill chunk; the caller multiplies by n_layers for the full stack."""
        m = self.model
        hidden, inter = m.hidden_dim, m.intermediate_size
        n_q, n_kv, hd = m.n_heads, m.kv_heads, m.head_dim
        n_qkv = (n_q + 2 * n_kv) * hd
        n_gate_up = 2 * inter
        return (
            elem("rmsnorm", m_tokens * hidden)
            + gemm(m_tokens, n_qkv, hidden)
            + elem("rotary_emb", m_tokens * (n_q + n_kv) * hd)
            + gemm(m_tokens, hidden, hidden)
            + elem("residual_add", m_tokens * hidden)
            + elem("rmsnorm", m_tokens * hidden)
            + gemm(m_tokens, n_gate_up, hidden)
            + elem("silu_mul", m_tokens * inter)
            + gemm(m_tokens, hidden, inter)
            + elem("residual_add", m_tokens * hidden)
        )

    def _decode_only_us(self, batch: int, ctx: float, gemm, elem) -> float:
        """Cost (us) of one decode step of `batch` sequences at context `ctx`,
        summed from the per-kernel tables: the non-attention block (× n_layers) +
        decode attention (added once) + the tail (kv-cache write, lm_head, and
        sampling, for `batch` tokens). Returns 0 when batch <= 0."""
        if batch <= 0:
            return 0.0
        m = self.model
        n_q, n_kv, hd = m.n_heads, m.kv_heads, m.head_dim
        dbytes = self.dtype_bytes
        total_us = self._block_linear_us(batch, gemm, elem) * m.n_layers
        total_us += self.decode_attn.decode_us(
            int(ctx), batch, self.gpu,
            n_heads=n_q, n_kv_heads=n_kv, head_dim=hd, dtype_bytes=dbytes,
        )
        total_us += elem("kv_cache_write", batch * n_kv * hd * 2)
        total_us += gemm(batch, m.vocab_size, m.hidden_dim)  # lm_head
        total_us += elem("sampling", batch * m.vocab_size)
        return total_us

    def _prefill_only_us(
        self, prefill_tokens: int, gemm, elem, *, completes_prefill: bool = True
    ) -> float:
        """Cost (us) of one prefill chunk of `prefill_tokens` tokens, summed from
        the per-kernel tables: the non-attention block (× n_layers) + causal
        prefill attention (added once) + the tail (kv-cache write, plus lm_head +
        sampling for the ONE first token when the chunk finishes the prompt).
        Returns 0 when prefill_tokens <= 0. Models a full chunk with no cached
        prefix."""
        if prefill_tokens <= 0:
            return 0.0
        m = self.model
        n_q, n_kv, hd = m.n_heads, m.kv_heads, m.head_dim
        dbytes = self.dtype_bytes
        total_us = self._block_linear_us(prefill_tokens, gemm, elem) * m.n_layers
        total_us += self.prefill_attn.prefill_us(
            prefill_tokens, self.gpu,
            n_heads=n_q, n_kv_heads=n_kv, head_dim=hd, dtype_bytes=dbytes,
        )
        total_us += elem("kv_cache_write", prefill_tokens * n_kv * hd * 2)
        if completes_prefill:
            total_us += gemm(1, m.vocab_size, m.hidden_dim)  # lm_head, one token
            total_us += elem("sampling", m.vocab_size)
        return total_us

    def fused_step_ms(
        self, prefill_tokens: int, decode_batch: int, decode_ctx: float,
        *, completes_prefill: bool = True,
    ) -> float:
        """Step floor (ms): a prefill chunk of `prefill_tokens` co-scheduled with
        `decode_batch` decode seqs at context `decode_ctx`.

        Prices the two populations independently (`_decode_only_us` /
        `_prefill_only_us`) and takes the max(): they share one forward pass, so
        the cheaper phase rides free under the costlier one rather than adding.
        With one population the max() collapses to it -- the decode_step_ms /
        prefill_step_ms corners. Full-causal chunk only; cached-prefix attention
        is a separate (future) grid.
        """
        _require_arch(self.model)
        gemm, elem = self._pricers()
        decode_us = self._decode_only_us(int(decode_batch), decode_ctx, gemm, elem)
        prefill_us = self._prefill_only_us(
            int(prefill_tokens), gemm, elem, completes_prefill=completes_prefill
        )
        return max(decode_us, prefill_us) / 1000.0

    def decode_step_ms(self, batch: int, ctx_tokens: float) -> float:
        """Step floor (ms) for `batch` decode seqs at context `ctx_tokens` -- the
        prefill_tokens=0 corner."""
        return self.fused_step_ms(0, int(batch), ctx_tokens)

    def prefill_step_ms(self, prefill_tokens: int) -> float:
        """Step floor (ms) for one sequence's prefill chunk of `prefill_tokens` --
        the decode_batch=0 corner (the chunk completes, so it emits one token)."""
        return self.fused_step_ms(int(prefill_tokens), 0, 0)


@mode(Mode.BACKTEST)
def _require_arch(model: ModelConfig) -> None:
    missing = [f for f in _REQUIRED_ARCH if getattr(model, f) is None]
    if missing:
        raise ValueError(
            f"{model.name} is missing arch dims required for kernel composition: "
            f"{missing}. Add them to the model config YAML."
        )


@mode(Mode.BACKTEST)
def load_kernel_floor(
    model: ModelConfig,
    gpu: GpuConfig,
    gemm_path: Path,
    flash_attn_path: Path,
    prefill_attn_path: Path,
    elementwise_path: Path,
    *,
    dtype_bytes: int = 2,
) -> KernelFloor:
    """Load the four measured kernel-cost tables and bind them to a model/GPU.

    Each table prices ONE kernel family from NCU data (microseconds; analytic
    roofline only as a fallback) and knows nothing about a transformer step:

        gemm         (M, N, K)          -> us   gemm.py
        decode_attn  (kv_len, batch)    -> us   flash_attn.py
        prefill_attn (seq_len, causal)  -> us   flash_attn.py
        elementwise  (kernel, elements) -> us   elementwise.py

    KernelFloor composes them: it walks the transformer block in this module and
    prices each kernel launch off the matching table. flash_attn_path is the
    decode grid (q_len=1); prefill_attn_path the causal prefill grid (q_len=kv_len).
    """
    return KernelFloor(
        model=model,
        gpu=gpu,
        gemm=load_gemm_table(gemm_path),
        decode_attn=load_decode_attn_grid(flash_attn_path),
        prefill_attn=load_prefill_attn_grid(prefill_attn_path),
        elementwise=load_elementwise_table(elementwise_path),
        dtype_bytes=dtype_bytes,
    )
