"""Block-level KV cache pool mirroring vLLM v1's BlockPool semantics.

This is a thin simulator-side stand-in for ``vllm/v1/core/kv_cache_manager.py``
plus ``vllm/v1/core/block_pool.py``.  It models:

- A fixed-size pool of KV blocks (``total_blocks`` × ``block_size`` tokens).
- A block-aligned prefix cache keyed by ``BlockHash`` — siblings see another
  request's blocks as cached only after ``cache_blocks()`` is called for them.
- Per-request ownership: each request consumes some blocks; freeing the
  request returns them to the free pool.
- Allocation failure when the free count is insufficient.

Eviction is not modeled here — when the pool fills, callers (the scheduler
simulator) decide whether to wait, preempt, or stop admitting.  This mirrors
the v1 scheduler's structure where ``allocate_slots()`` returns ``None`` and
the scheduler reacts by preempting the tail running request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BlockHash:
    """Identifies one KV block's content for cache lookup.

    For the simulator we don't need the full vLLM block-hash chain — we just
    need a stable identifier per (prefix_key, block_index).  Requests in a
    shared-prefix batch all carry the same ``prefix_key`` for the blocks they
    share; the unique tail uses ``prefix_key=(request_id, "tail")`` so
    siblings cannot accidentally collide.
    """

    prefix_key: object
    block_index: int


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / divisor))


class BlockPool:
    """Free-list + cache map.  See module docstring for semantics."""

    def __init__(self, total_blocks: int, block_size: int = 16) -> None:
        if total_blocks <= 0:
            raise ValueError("BlockPool requires total_blocks > 0")
        if block_size <= 0:
            raise ValueError("BlockPool requires block_size > 0")
        self._total_blocks = int(total_blocks)
        self._block_size = int(block_size)
        self._free_blocks = int(total_blocks)
        # request_id -> number of blocks currently owned by the request.
        self._owned_by_request: dict[int, int] = {}
        # hash -> sentinel (presence means the block is cached and visible to
        # siblings).  Eviction would remove entries; for now we don't evict.
        self._cached_hashes: set[BlockHash] = set()

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def get_num_free_blocks(self) -> int:
        return self._free_blocks

    def num_blocks_owned(self, request_id: int) -> int:
        return self._owned_by_request.get(request_id, 0)

    def num_cached_blocks(self) -> int:
        return len(self._cached_hashes)

    def find_longest_cache_hit(
        self, block_hashes: Sequence[BlockHash], max_hit_blocks: int | None = None
    ) -> int:
        """Walk ``block_hashes`` and return the count of leading cached blocks.

        Stops at the first miss (block-aligned only).  ``max_hit_blocks``
        caps the search; vLLM uses ``num_tokens - 1`` so the last partial
        block is never considered "cached" for the requesting query.
        """
        hit = 0
        limit = len(block_hashes) if max_hit_blocks is None else min(
            len(block_hashes), max_hit_blocks
        )
        for i in range(limit):
            if block_hashes[i] in self._cached_hashes:
                hit += 1
            else:
                break
        return hit

    def allocate(self, request_id: int, num_blocks: int) -> bool:
        """Reserve ``num_blocks`` for ``request_id``.

        Returns True on success, False when the pool can't fit.  Allocations
        accumulate per-request — calling allocate twice for the same request
        adds to its owned count.
        """
        num_blocks = int(num_blocks)
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")
        if num_blocks == 0:
            return True
        if num_blocks > self._free_blocks:
            return False
        self._free_blocks -= num_blocks
        self._owned_by_request[request_id] = (
            self._owned_by_request.get(request_id, 0) + num_blocks
        )
        return True

    def cache_blocks(self, block_hashes: Sequence[BlockHash]) -> None:
        """Mark the listed block hashes as cached (visible to siblings).

        Idempotent — duplicate hashes are silently ignored.
        """
        for h in block_hashes:
            self._cached_hashes.add(h)

    def free_request(self, request_id: int) -> int:
        """Return the request's blocks to the free pool.

        Returns the number of blocks freed.  The cached hashes are NOT
        evicted here (matching vLLM's behavior where freed blocks stay
        cache-resident in the LRU free list until a real eviction).
        """
        owned = self._owned_by_request.pop(request_id, 0)
        self._free_blocks += owned
        return owned

    def free_partial(self, request_id: int, num_blocks: int) -> int:
        """Return ``num_blocks`` of the request's blocks to the free pool (the TAIL of its
        KV under LRU — recent blocks evict before the shared prefix). Caps at what's owned;
        drops the request when fully freed. Returns the number actually freed.

        Mirrors vLLM's block-granular LRU eviction: a partially-evicted request keeps its
        surviving (prefix) blocks cache-resident, so its next prefill is a partial hit.
        """
        owned = self._owned_by_request.get(request_id, 0)
        n = min(owned, max(0, int(num_blocks)))
        if n <= 0:
            return 0
        remaining = owned - n
        if remaining > 0:
            self._owned_by_request[request_id] = remaining
        else:
            self._owned_by_request.pop(request_id, None)
        self._free_blocks += n
        return n

    def tokens_to_blocks(self, num_tokens: int) -> int:
        """Helper: how many blocks does ``num_tokens`` require?"""
        return _ceil_div(num_tokens, self._block_size)
