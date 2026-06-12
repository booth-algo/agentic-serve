"""Regeneration-parity guards for configs/deployments/*.json (audit-v2 S14 + G8 + G6).

The chunked-prefill budget ``max_num_batched_tokens`` is ENGINE CONFIG (the vLLM
OpenAI-server default rule), but the 2048 pins were originally hand-added in the
2026-06-10 ramp-restructure commit while the deployment generator omitted the key —
so a regeneration silently reverted 29 configs to the loader's 8192 default
(quadrupling the overflow budget the restructured TPOT weight prices). These tests
lock the invariant both ways: every committed vLLM deployment's EFFECTIVE budget
equals the engine rule, and any config whose rule differs from the loader default
must carry the key explicitly (so a key-dropping regeneration fails loudly here).

2026-06-10 (G8): the same parity now covers the sglang deployments — their budget is
sglang's PER-DEVICE memory-tier ``chunked_prefill_size`` rule (24GiB-class -> 2048,
40GiB -> 4096, 80GiB -> 8192; server_args.py), not vLLM's device rule the key's
absence used to imply. G6 adds the executable RESERVE_BYTES rule check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOYMENTS = sorted((REPO / "configs/deployments").glob("*.json"))
LOADER_DEFAULT = 8192  # configs/loader.py fallback when the key is absent


def _rule(total_memory_bytes: float, gpu_name: str) -> int:
    from configs.generate_deployments import vllm_max_num_batched_tokens

    return vllm_max_num_batched_tokens(total_memory_bytes, gpu_name)


def test_deployments_exist() -> None:
    assert len(DEPLOYMENTS) > 30


def test_vllm_max_num_batched_tokens_matches_engine_rule() -> None:
    """Effective budget == the vLLM device rule for EVERY committed vLLM deployment."""
    for f in DEPLOYMENTS:
        d = json.loads(f.read_text())
        if d.get("engine", "vllm") != "vllm":
            continue
        gpu = json.loads((REPO / f"configs/gpus/{d['gpu']}.json").read_text())
        rule = _rule(float(gpu["total_memory_bytes"]), d["gpu"])
        effective = int(d.get("max_num_batched_tokens", LOADER_DEFAULT))
        assert effective == rule, (
            f"{f.name}: effective max_num_batched_tokens {effective} != engine rule {rule}"
        )


def test_sglang_memory_tier_rule_values() -> None:
    """Pin the sglang tier function to the cited source (sglang
    python/sglang/srt/server_args.py ServerArgs._handle_gpu_memory_settings,
    upstream main @ 255843d45462, fetched 2026-06-10; thresholds in MiB)."""
    from configs.generate_deployments import sglang_chunked_prefill_size as rule

    gib = 1024**3
    assert rule(11 * gib) == 2048    # RTX2080Ti (< 20 GiB tier: T4/4080)
    assert rule(24 * gib) == 2048    # RTX3090 (< 35 GiB tier: A10/4090/5090)
    assert rule(40 * gib) == 4096    # A100 40GB (< 60 GiB tier: A100-40G/L40)
    assert rule(80 * gib) == 8192    # H100 / A100 80GB (< 90 GiB tier)
    assert rule(141 * gib) == 8192   # H20/H200 (< 160 GiB tier)
    assert rule(192 * gib) == 16384  # B200/MI300


def test_sglang_chunked_prefill_size_matches_engine_rule() -> None:
    """audit-v2 G8: every committed sglang deployment's EFFECTIVE budget equals the
    sglang memory-tier rule — they may no longer silently inherit the loader's
    vLLM 8192 default (a 4x error on 24GiB devices). Same regeneration-parity
    guard as the vLLM test: a non-default rule value must be pinned in the JSON."""
    from configs.generate_deployments import sglang_chunked_prefill_size

    seen = 0
    for f in DEPLOYMENTS:
        d = json.loads(f.read_text())
        if d.get("engine") != "sglang":
            continue
        seen += 1
        gpu = json.loads((REPO / f"configs/gpus/{d['gpu']}.json").read_text())
        rule = sglang_chunked_prefill_size(float(gpu["total_memory_bytes"]))
        effective = int(d.get("max_num_batched_tokens", LOADER_DEFAULT))
        assert effective == rule, (
            f"{f.name}: effective max_num_batched_tokens {effective} != sglang "
            f"memory-tier rule {rule}")
        if rule != LOADER_DEFAULT:
            assert "max_num_batched_tokens" in d, (
                f"{f.name}: sglang rule is {rule} (≠ loader default {LOADER_DEFAULT}) "
                f"but the key is absent — a regeneration dropped it")
    assert seen > 10  # the sglang half of the deployment matrix is covered


def test_reserve_rule_reproduces_known_pools() -> None:
    """audit-v2 G6: RESERVE_BYTES = 3.5e9 follows a STATED rule — the mean of the
    back-solved reserves of the 3 configs with exactly known pools, rounded to
    0.1 GB ((4.102 + 2.710 + 3.831)/3 = 3.548 -> 3.5). This test executes the rule:
    the single reserve must reproduce all 3 known pools within the documented 5%
    (measured: +1.05% / -4.46% / +0.51%), and the back-solved mean must round to
    the production constant. No value change (byte-identity contract)."""
    from configs.kv_pool import RESERVE_BYTES, available_kv_blocks

    mdl = json.loads((REPO / "configs/models/Llama-3.1-8B.json").read_text())
    weights = float(mdl["n_params"]) * float(mdl["bytes_per_param"])
    kv_bpt = float(mdl["kv_bytes_per_token"])
    kv_heads = int(mdl["kv_heads"])

    known = [  # (gpu json, gpu_mem_util, tp, measured pool blocks, documented signed err %)
        ("H100", 0.90, 1, 27250, +1.05),
        ("A100", 0.85, 1, 8458, -4.46),
        ("H100", 0.90, 2, 62416, +0.51),  # H100x2
    ]
    reserves = []
    for gpu, util, tp, measured, doc_err in known:
        total = float(json.loads((REPO / f"configs/gpus/{gpu}.json").read_text())
                      ["total_memory_bytes"])
        pred = available_kv_blocks(total, util, weights, tp, kv_bpt, kv_heads)
        err_pct = (pred - measured) / measured * 100.0
        assert abs(err_pct) < 5.0, (gpu, tp, pred, measured, err_pct)
        assert abs(err_pct - doc_err) < 0.05, (gpu, tp, err_pct, doc_err)
        bytes_per_block = 16 * kv_bpt / min(tp, kv_heads)
        reserves.append(total * util - weights / tp - measured * bytes_per_block)
    mean_gb = sum(reserves) / len(reserves) / 1e9
    assert round(mean_gb, 1) == RESERVE_BYTES / 1e9 == 3.5
    # the documented per-config envelope (small-pool amplification caveat)
    assert min(reserves) / 1e9 == pytest.approx(2.710, abs=0.005)
    assert max(reserves) / 1e9 == pytest.approx(4.102, abs=0.005)


def test_non_default_budgets_are_pinned_explicitly() -> None:
    """A config whose rule differs from the loader default MUST carry the key — this is
    the regeneration-parity guard: dropping the key (e.g. regenerating with a generator
    that does not emit it) flips the effective value to 8192 and fails HERE."""
    for f in DEPLOYMENTS:
        d = json.loads(f.read_text())
        if d.get("engine", "vllm") != "vllm":
            continue
        gpu = json.loads((REPO / f"configs/gpus/{d['gpu']}.json").read_text())
        rule = _rule(float(gpu["total_memory_bytes"]), d["gpu"])
        if rule != LOADER_DEFAULT:
            assert "max_num_batched_tokens" in d, (
                f"{f.name}: rule is {rule} (≠ loader default {LOADER_DEFAULT}) but the key "
                f"is absent — a regeneration dropped it; the generator must emit it"
            )
