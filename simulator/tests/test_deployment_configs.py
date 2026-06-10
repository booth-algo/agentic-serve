"""Regeneration-parity guards for configs/deployments/*.json (audit-v2 item S14).

The chunked-prefill budget ``max_num_batched_tokens`` is ENGINE CONFIG (the vLLM
OpenAI-server default rule), but the 2048 pins were originally hand-added in the
2026-06-10 ramp-restructure commit while the deployment generator omitted the key —
so a regeneration silently reverted 29 configs to the loader's 8192 default
(quadrupling the overflow budget the restructured TPOT weight prices). These tests
lock the invariant both ways: every committed vLLM deployment's EFFECTIVE budget
equals the engine rule, and any config whose rule differs from the loader default
must carry the key explicitly (so a key-dropping regeneration fails loudly here).
"""

from __future__ import annotations

import json
from pathlib import Path

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
