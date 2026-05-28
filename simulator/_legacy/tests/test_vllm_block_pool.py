from __future__ import annotations

import pytest

from simulator._legacy.vllm_block_pool import BlockHash, BlockPool


def _hashes(prefix_key: object, count: int) -> list[BlockHash]:
    return [BlockHash(prefix_key, i) for i in range(count)]


def test_allocate_and_free_returns_blocks_to_pool() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    assert pool.get_num_free_blocks() == 10
    assert pool.allocate(request_id=0, num_blocks=4)
    assert pool.get_num_free_blocks() == 6
    assert pool.num_blocks_owned(0) == 4

    pool.free_request(request_id=0)
    assert pool.get_num_free_blocks() == 10
    assert pool.num_blocks_owned(0) == 0


def test_allocate_fails_when_pool_too_small() -> None:
    pool = BlockPool(total_blocks=5, block_size=16)
    assert pool.allocate(0, 3)
    assert not pool.allocate(1, 3)  # only 2 free, asking 3
    assert pool.get_num_free_blocks() == 2
    assert pool.num_blocks_owned(1) == 0  # failure does not mutate ownership


def test_cache_hit_propagates_after_cache_blocks_call() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    prefix = "turn-7-c40"
    hashes = _hashes(prefix, 5)

    # Sibling sees no cache before the anchor caches.
    assert pool.find_longest_cache_hit(hashes) == 0

    # Anchor allocates and caches its blocks.
    assert pool.allocate(0, 5)
    pool.cache_blocks(hashes)

    # Sibling now sees all 5 blocks cached.
    assert pool.find_longest_cache_hit(hashes) == 5


def test_cache_hit_stops_at_first_miss() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    prefix = "shared"
    pool.allocate(0, 3)
    pool.cache_blocks(_hashes(prefix, 3))

    # New request with 5 blocks: first 3 shared (cached), last 2 unique tail.
    new_request_hashes = _hashes(prefix, 3) + [
        BlockHash(("tail", 1), 0),
        BlockHash(("tail", 1), 1),
    ]
    assert pool.find_longest_cache_hit(new_request_hashes) == 3


def test_max_hit_blocks_caps_lookup() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    pool.allocate(0, 4)
    pool.cache_blocks(_hashes("p", 4))

    # Caller asks "what's cached but never claim the last block".
    assert pool.find_longest_cache_hit(_hashes("p", 4), max_hit_blocks=3) == 3
    assert pool.find_longest_cache_hit(_hashes("p", 4), max_hit_blocks=10) == 4


def test_free_keeps_cache_resident() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    hashes = _hashes("p", 2)
    pool.allocate(0, 2)
    pool.cache_blocks(hashes)
    pool.free_request(0)

    # Blocks back in the free pool, but the cache_blocks call still grants
    # the hit (matches vLLM where freed blocks remain in LRU as cached
    # until evicted).
    assert pool.get_num_free_blocks() == 10
    assert pool.find_longest_cache_hit(hashes) == 2


def test_tokens_to_blocks_ceil_division() -> None:
    pool = BlockPool(total_blocks=100, block_size=16)
    assert pool.tokens_to_blocks(0) == 0
    assert pool.tokens_to_blocks(1) == 1
    assert pool.tokens_to_blocks(16) == 1
    assert pool.tokens_to_blocks(17) == 2
    assert pool.tokens_to_blocks(7188) == 450  # ceil(7188/16)


def test_repeated_allocate_for_same_request_accumulates() -> None:
    pool = BlockPool(total_blocks=10, block_size=16)
    assert pool.allocate(0, 3)
    assert pool.allocate(0, 2)
    assert pool.num_blocks_owned(0) == 5
    assert pool.get_num_free_blocks() == 5


def test_constructor_rejects_nonpositive_sizes() -> None:
    with pytest.raises(ValueError):
        BlockPool(total_blocks=0)
    with pytest.raises(ValueError):
        BlockPool(total_blocks=10, block_size=0)


def test_allocate_zero_is_noop_success() -> None:
    pool = BlockPool(total_blocks=5, block_size=16)
    assert pool.allocate(0, 0)
    assert pool.get_num_free_blocks() == 5
    assert pool.num_blocks_owned(0) == 0
