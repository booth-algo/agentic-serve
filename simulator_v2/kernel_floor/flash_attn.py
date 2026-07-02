# simulator_v2/kernel_floor/flash_attn.py

"""Flash-attention kernel cost (us) from the measured NCU tables -- one grid class
per phase (decode q_len=1; prefill causal), since each phase is a different
computation, not just a different size. Separable linear interp/extrap over the
measured grid (lookup order in each grid's `*_us`). Native unit us."""

import csv
from dataclasses import dataclass
from pathlib import Path

from simulator_v2.core.types import GpuConfig
from simulator_v2.core.mode import Mode, mode


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class _AttnGrid:
    """Shared by all phase grids: the head config the grid was measured at, plus
    a head-config match check. The analytic roofline (`_roofline_us`) is a free
    function, not a method, because the mismatch fallback must price with the
    *query's* head config, not the grid's.
    """
    n_heads: int
    n_kv_heads: int
    head_dim: int

    def _matches(self, n_heads: int, n_kv_heads: int, head_dim: int) -> bool:
        return (n_heads, n_kv_heads, head_dim) == (self.n_heads, self.n_kv_heads, self.head_dim)


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class DecodeAttnGrid(_AttnGrid):
    """Measured decode flash-attention latencies (us) over (kv_len, batch) at a
    fixed head config (q_len=1).

    Duplicate (kv_len, batch) rows are averaged on load (same neutral choice as
    the GEMM table).
    """
    kv_axis: tuple[int, ...]
    batch_axis: tuple[int, ...]
    cells: dict[tuple[int, int], float]
    batches_by_kv: dict[int, tuple[int, ...]]  # ascending measured batches per kv column

    def decode_us(
        self, kv_len: int, batch: int, gpu: GpuConfig,
        *, n_heads: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2,
    ) -> float:
        """Cost (us) for one decode attention call (q_len=1), in order of preference:

          1. exact (kv_len, batch) measured  -> return the measured cell
          2. otherwise                       -> separable linear estimate:
             interp/extrapolate in batch within each bracketing kv column, then
             interp/extrapolate across kv
          3. head mismatch / empty grid      -> analytic roofline

        Case 2 stays on the measured trend (never roofline), so the staircase
        holes at high kv*batch extrapolate linearly instead of collapsing.
        """
        if not self._matches(n_heads, n_kv_heads, head_dim) or not self.cells:
            return _roofline_us(1, kv_len, batch, gpu,
                                n_heads, n_kv_heads, head_dim, dtype_bytes)

        exact = self.cells.get((int(kv_len), int(batch)))
        if exact is not None:
            return exact
        return self._interp(float(kv_len), float(batch))

    def _interp(self, kv: float, batch: float) -> float:
        i = _bracket(self.kv_axis, kv)
        kv0, kv1 = self.kv_axis[i], self.kv_axis[i + 1]
        c0 = self._column_at_batch(kv0, batch)
        c1 = self._column_at_batch(kv1, batch)
        if kv1 == kv0:
            return c0
        return c0 + (c1 - c0) * (kv - kv0) / (kv1 - kv0)

    def _column_at_batch(self, kv: int, batch: float) -> float:
        """Price the kv column at `batch`: linear interp, or linear
        extrapolation past the column's measured batch range."""
        bs = self.batches_by_kv[kv]
        if len(bs) == 1:
            return self.cells[(kv, bs[0])]
        j = _bracket(bs, batch)
        b0, b1 = bs[j], bs[j + 1]
        y0, y1 = self.cells[(kv, b0)], self.cells[(kv, b1)]
        return y0 + (y1 - y0) * (batch - b0) / (b1 - b0)


@mode(Mode.BACKTEST)
@dataclass(frozen=True)
class PrefillAttnGrid(_AttnGrid):
    """Measured full-causal prefill flash-attention (us) over seq_len
    (q_len = kv_len = N, causal, batch=1), all-layers per-step.

    Source is the per-layer FA3 median x n_layers (flash_full_model_ms), so --
    like the decode grid -- it's already an all-layers per-step number and is
    added ONCE in the recipe, not multiplied by n_layers. Cost is launch-floored
    at small N and ~quadratic at large N (causal N^2), so we interpolate
    piecewise-linearly between the (dense) measured seq_lens.
    """
    seq_axis: tuple[int, ...]
    cells: dict[int, float]

    def prefill_us(
        self, seq_len: int, gpu: GpuConfig,
        *, n_heads: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2,
    ) -> float:
        """Cost (us) for one sequence's prefill attention (q_len=kv_len=seq_len,
        causal), in order of preference:

          1. exact seq_len measured     -> return the measured cell
          2. otherwise                  -> piecewise-linear interp in seq_len
             (linear extrapolation past the measured range; underestimates the
             quadratic tail beyond the grid, but vLLM caps chunk size)
          3. head mismatch / empty grid -> analytic causal roofline
        """
        if not self._matches(n_heads, n_kv_heads, head_dim) or not self.cells:
            return _roofline_us(seq_len, seq_len, 1, gpu,
                                n_heads, n_kv_heads, head_dim, dtype_bytes, causal=True)
        s = int(seq_len)
        exact = self.cells.get(s)
        if exact is not None:
            return exact
        i = _bracket(self.seq_axis, float(s))
        s0, s1 = self.seq_axis[i], self.seq_axis[i + 1]
        y0, y1 = self.cells[s0], self.cells[s1]
        if s1 == s0:
            return y0
        return y0 + (y1 - y0) * (s - s0) / (s1 - s0)


@mode(Mode.BACKTEST)
def _bracket(axis: tuple[int, ...], value: float) -> int:
    """Lower index of the bracketing pair (i, i+1) for value in a sorted axis.
    Clamps to the end pairs so out-of-range values extrapolate linearly."""
    last = len(axis) - 2
    if value <= axis[0]:
        return 0
    if value >= axis[-1]:
        return last
    for i in range(last + 1):
        if axis[i] <= value <= axis[i + 1]:
            return i
    return last


@mode(Mode.BACKTEST)
def _roofline_us(
    q_len: int, kv_len: int, batch: int, gpu: GpuConfig,
    n_heads: int, n_kv_heads: int, head_dim: int, dtype_bytes: int,
    *, causal: bool = False,
) -> float:
    """Analytic flash-attention roofline in us: max(compute-bound, memory-bound).

    `causal` halves the FLOPs (the causal triangle) -- decode (q_len=1) leaves it
    False; prefill (q_len=kv_len=N) sets it True.
    """
    flops = 4.0 * batch * q_len * kv_len * n_heads * head_dim
    if causal:
        flops *= 0.5
    q_bytes = batch * q_len * n_heads * head_dim * dtype_bytes
    kv_bytes = 2.0 * batch * kv_len * n_kv_heads * head_dim * dtype_bytes
    o_bytes = batch * q_len * n_heads * head_dim * dtype_bytes
    bytes_moved = q_bytes + kv_bytes + o_bytes
    compute_us = flops / (gpu.peak_flops_per_s * gpu.util_flops) * 1e6
    memory_us = bytes_moved / (gpu.peak_bw_bytes_per_s * gpu.util_bw) * 1e6
    return max(compute_us, memory_us)


@mode(Mode.BACKTEST)
def load_decode_attn_grid(path: Path) -> DecodeAttnGrid:
    """FA CSV (cols: q_len, kv_len, n_heads, n_kv_heads, head_dim, batch, causal,
    phase, latency_us) -> DecodeAttnGrid built from the decode rows (q_len=1).

    The decode rows must share one head config (n_heads, n_kv_heads, head_dim);
    a mixed-config file raises, since the grid is keyed by (kv_len, batch) only.
    """
    sums: dict[tuple[int, int], list[float]] = {}
    head_cfgs: set[tuple[int, int, int]] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "decode":
                continue
            head_cfgs.add((int(row["n_heads"]), int(row["n_kv_heads"]), int(row["head_dim"])))
            key = (int(row["kv_len"]), int(row["batch"]))
            sums.setdefault(key, []).append(float(row["latency_us"]))
    if not sums:
        raise RuntimeError(f"no decode flash-attention rows in {path}")
    if len(head_cfgs) != 1:
        raise RuntimeError(f"expected one decode head config in {path}, got {head_cfgs}")
    n_heads, n_kv_heads, head_dim = head_cfgs.pop()
    cells = {key: sum(vals) / len(vals) for key, vals in sums.items()}
    kv_axis = tuple(sorted({kv for kv, _ in cells}))
    batch_axis = tuple(sorted({b for _, b in cells}))
    batches_by_kv = {
        kv: tuple(sorted(b for (k, b) in cells if k == kv)) for kv in kv_axis
    }
    return DecodeAttnGrid(
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        kv_axis=kv_axis, batch_axis=batch_axis, cells=cells,
        batches_by_kv=batches_by_kv,
    )


@mode(Mode.BACKTEST)
def load_prefill_attn_grid(path: Path) -> PrefillAttnGrid:
    """FA3 prefill CSV (cols: prefill_tokens, q_len, kv_len, n_heads, n_kv_heads,
    head_dim, flash_full_model_ms, ...) -> PrefillAttnGrid.

    Keyed by prefill_tokens (= q_len = kv_len). flash_full_model_ms is the
    all-layers per-step value in MILLISECONDS; we store microseconds to match the
    other leaves. Rows must share one head config.
    """
    cells: dict[int, float] = {}
    head_cfgs: set[tuple[int, int, int]] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            head_cfgs.add((int(row["n_heads"]), int(row["n_kv_heads"]), int(row["head_dim"])))
            cells[int(row["prefill_tokens"])] = float(row["flash_full_model_ms"]) * 1000.0
    if not cells:
        raise RuntimeError(f"no prefill flash-attention rows in {path}")
    if len(head_cfgs) != 1:
        raise RuntimeError(f"expected one prefill head config in {path}, got {head_cfgs}")
    n_heads, n_kv_heads, head_dim = head_cfgs.pop()
    seq_axis = tuple(sorted(cells))
    return PrefillAttnGrid(
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        seq_axis=seq_axis, cells=cells,
    )
