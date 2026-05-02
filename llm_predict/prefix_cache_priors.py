"""Prefix-cache priors derived from prompt structure for single-turn trace workloads.

These priors estimate per-profile cache features from the underlying prompt data
(common system-prompt prefix vs varying user-message suffix). They are
physics-based: the prior comes from prompt structure, not measured latency.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

PRIOR_PATH = os.path.join(os.path.dirname(__file__), "data", "prefix_cache_priors.json")

_CHARS_PER_TOKEN = 3.5  # English code-heavy text heuristic


def _chars_to_tokens(chars: int) -> int:
    return max(1, round(chars / _CHARS_PER_TOKEN))


@dataclass(frozen=True)
class PrefixCachePrior:
    profile: str
    cached_context_tokens: int
    new_prefill_tokens: int
    total_context_tokens: int
    cache_hit_rate: float
    source: str


def _load_priors() -> dict[str, Any]:
    if not os.path.exists(PRIOR_PATH):
        return {}
    with open(PRIOR_PATH) as f:
        return json.load(f)


def get_prefix_cache_prior(
    profile: str,
    model: str | None = None,
    gpu: str | None = None,
) -> PrefixCachePrior | None:
    """Return a structure-derived cache prior for a profile, or None.

    The prior is model-agnostic (character-count based with a fixed
    chars-to-tokens heuristic). Future versions may add model-keyed
    tokenizer tables.
    """
    priors = _load_priors()
    entry = priors.get(profile)
    if not entry:
        return None

    system_chars = entry.get("system_prompt_chars", 0)
    user_chars = entry.get("user_message_chars", {}).get("p50", 0)
    if system_chars <= 0 or user_chars <= 0:
        return None

    cached_tokens = _chars_to_tokens(system_chars)
    new_tokens = _chars_to_tokens(user_chars)
    total = cached_tokens + new_tokens
    hit_rate = cached_tokens / max(1, total)

    return PrefixCachePrior(
        profile=profile,
        cached_context_tokens=cached_tokens,
        new_prefill_tokens=new_tokens,
        total_context_tokens=total,
        cache_hit_rate=hit_rate,
        source=entry.get("source", ""),
    )


def build_coding_singleturn_prior() -> None:
    """Build the coding-singleturn prefix-cache prior from the prompt JSONL.

    Reads coding_agent_prompts.jsonl, computes system prompt character count
    and per-user-message character count percentiles, then writes the
    model-agnostic prior to PRIOR_PATH.
    """
    jsonl_path = os.path.join(
        os.path.dirname(__file__), "..",
        "inference-benchmark", "data", "coding_agent_prompts.jsonl",
    )
    if not os.path.exists(jsonl_path):
        print(f"JSONL not found: {jsonl_path}")
        return

    prompts: list[dict[str, str]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj)

    if not prompts:
        print("No prompts loaded")
        return

    system_chars = len(prompts[0]["system"])
    all_systems = {p["system"] for p in prompts}
    if len(all_systems) > 1:
        print(f"WARNING: {len(all_systems)} distinct system prompts found (expected 1). Using first.")

    user_chars = sorted(len(p["user"]) for p in prompts)
    n = len(user_chars)
    percentiles = {
        "min": user_chars[0],
        "p25": user_chars[n // 4],
        "p50": user_chars[n // 2],
        "p75": user_chars[3 * n // 4],
        "p90": user_chars[9 * n // 10],
        "max": user_chars[-1],
    }
    system_est = _chars_to_tokens(system_chars)
    user_est = _chars_to_tokens(percentiles["p50"])
    total_est = system_est + user_est
    hit_rate = round(system_est / max(1, total_est), 4)

    prior = {
        "coding-singleturn": {
            "system_prompt_chars": system_chars,
            "estimated_system_tokens": system_est,
            "num_prompts": len(prompts),
            "distinct_system_prompts": len(all_systems),
            "user_message_chars": percentiles,
            "estimated_median_user_tokens": user_est,
            "estimated_median_total_tokens": total_est,
            "estimated_median_cache_hit_rate": hit_rate,
            "source": "coding_agent_prompts.jsonl: all 500 prompts share identical system prompt; "
                      "with prefix caching only the unique user-message suffix needs prefill "
                      "after the first concurrent request",
        },
    }

    os.makedirs(os.path.dirname(PRIOR_PATH), exist_ok=True)
    with open(PRIOR_PATH, "w") as f:
        json.dump(prior, f, indent=2)
        f.write("\n")
    print(f"Wrote {PRIOR_PATH}")
    print(f"  System: {system_chars} chars (~{system_est} tokens)")
    print(f"  User messages: p50={percentiles['p50']} chars (~{user_est} tokens)")
    print(f"  Cache hit rate: {hit_rate}")


if __name__ == "__main__":
    build_coding_singleturn_prior()
