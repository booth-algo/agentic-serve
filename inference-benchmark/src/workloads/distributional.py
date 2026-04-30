"""Distributional synthetic multi-turn workload sampling.

This module backs the canonical distributional multi-turn profiles. It turns
compact trace distributions into synthetic growing-history sessions whose
prompt deltas can be audited and recorded by the benchmark runner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .dataset import BenchmarkRequest
from .trace_distributions import TraceDistribution, TraceTurnSample


TOKEN_WORD_RATIO = 1.35
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 256


@dataclass(frozen=True)
class SyntheticTurnSpec:
    turn_index: int
    sampled_new_prefill_tokens: int
    actual_new_prefill_tokens: int
    cached_context_tokens: int
    total_context_tokens: int
    new_user_tokens: int
    output_tokens: int
    cache_hit_rate: float
    context_window_tokens: int | None = None
    context_safety_margin_tokens: int = 0
    prompt_token_budget: int | None = None
    planned_total_with_output_tokens: int | None = None
    truncated_by_context_limit: bool = False


@dataclass
class SyntheticSession:
    session_id: int
    turns: list[BenchmarkRequest]
    specs: list[SyntheticTurnSpec]


class DistributionalSampler:
    """Sample synthetic sessions from empirical trace distributions."""

    def __init__(
        self,
        distribution: TraceDistribution,
        *,
        seed: int = 42,
        max_context_tokens: int | None = None,
        context_safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
        system_prompt: str = "",
    ):
        if not distribution.turn_counts:
            raise ValueError("Distribution has no turn-count samples")
        if not distribution.turns:
            raise ValueError("Distribution has no turn samples")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive when provided")
        if context_safety_margin_tokens < 0:
            raise ValueError("context_safety_margin_tokens must be non-negative")
        if (
            max_context_tokens is not None
            and context_safety_margin_tokens >= max_context_tokens
        ):
            raise ValueError("context_safety_margin_tokens must be smaller than max_context_tokens")

        self.distribution = distribution
        self.rng = random.Random(seed)
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin_tokens = context_safety_margin_tokens
        self.system_prompt = system_prompt
        self._turns_by_index = distribution.turns_by_index

    def sample_session(self, session_id: int = 0) -> SyntheticSession:
        turn_count = self.rng.choice(self.distribution.turn_counts)
        return self.sample_session_with_turn_count(session_id=session_id, turn_count=turn_count)

    def sample_sessions(self, num_sessions: int) -> list[SyntheticSession]:
        if num_sessions <= 0:
            raise ValueError("num_sessions must be positive")
        return [self.sample_session(session_id=i) for i in range(num_sessions)]

    def sample_session_with_turn_count(self, *, session_id: int, turn_count: int) -> SyntheticSession:
        if turn_count <= 0:
            raise ValueError("turn_count must be positive")

        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        turns: list[BenchmarkRequest] = []
        specs: list[SyntheticTurnSpec] = []
        previous_prompt_context = estimate_message_tokens(messages)
        previous_output_tokens = 0

        for turn_index in range(turn_count):
            sample = self._sample_turn(turn_index)
            output_tokens = max(1, sample.output_tokens)
            new_user_tokens = max(1, sample.new_prefill_tokens - previous_output_tokens)
            context_before_user = previous_prompt_context + previous_output_tokens
            desired_total_context = context_before_user + new_user_tokens
            prompt_token_budget = self._prompt_token_budget(output_tokens)
            truncated = False

            if prompt_token_budget is not None and desired_total_context > prompt_token_budget:
                remaining = prompt_token_budget - context_before_user
                if remaining <= 0:
                    break
                new_user_tokens = max(1, remaining)
                desired_total_context = context_before_user + new_user_tokens
                truncated = True

            user_text = synthetic_text(f"s{session_id}_t{turn_index}_user", new_user_tokens)
            messages.append({"role": "user", "content": user_text})

            actual_total_context = context_before_user + new_user_tokens
            cached_context = previous_prompt_context
            actual_new_prefill = actual_total_context - cached_context
            cache_hit_rate = cached_context / actual_total_context if actual_total_context > 0 else 0.0

            turns.append(
                BenchmarkRequest(
                    messages=list(messages),
                    max_tokens=sample.output_tokens,
                    metadata={
                        "synthetic_session_id": session_id,
                        "synthetic_turn_index": turn_index,
                        "sampled_new_prefill_tokens": sample.new_prefill_tokens,
                        "planned_new_prefill_tokens": actual_new_prefill,
                        "planned_cached_context_tokens": cached_context,
                        "planned_total_context_tokens": actual_total_context,
                        "planned_cache_hit_rate": round(cache_hit_rate, 6),
                        "planned_new_user_tokens": new_user_tokens,
                        "planned_output_tokens": output_tokens,
                        "planned_total_with_output_tokens": actual_total_context + output_tokens,
                        "context_window_tokens": self.max_context_tokens,
                        "context_safety_margin_tokens": self.context_safety_margin_tokens,
                        "prompt_token_budget": prompt_token_budget,
                        "truncated_by_context_limit": truncated,
                    },
                )
            )
            specs.append(
                SyntheticTurnSpec(
                    turn_index=turn_index,
                    sampled_new_prefill_tokens=sample.new_prefill_tokens,
                    actual_new_prefill_tokens=actual_new_prefill,
                    cached_context_tokens=cached_context,
                    total_context_tokens=actual_total_context,
                    new_user_tokens=new_user_tokens,
                    output_tokens=output_tokens,
                    cache_hit_rate=cache_hit_rate,
                    context_window_tokens=self.max_context_tokens,
                    context_safety_margin_tokens=self.context_safety_margin_tokens,
                    prompt_token_budget=prompt_token_budget,
                    planned_total_with_output_tokens=actual_total_context + output_tokens,
                    truncated_by_context_limit=truncated,
                )
            )

            assistant_text = synthetic_text(
                f"s{session_id}_t{turn_index}_assistant",
                output_tokens,
            )
            messages.append({"role": "assistant", "content": assistant_text})
            previous_prompt_context = actual_total_context
            previous_output_tokens = output_tokens

            if truncated:
                break

        return SyntheticSession(session_id=session_id, turns=turns, specs=specs)

    def _prompt_token_budget(self, output_tokens: int) -> int | None:
        """Return max prompt tokens after reserving output and tokenizer headroom."""
        if self.max_context_tokens is None:
            return None
        return self.max_context_tokens - output_tokens - self.context_safety_margin_tokens

    def _sample_turn(self, turn_index: int) -> TraceTurnSample:
        candidates = self._turns_by_index.get(turn_index)
        if not candidates:
            candidates = self.distribution.turns
        return self.rng.choice(candidates)


def synthetic_text(label: str, target_tokens: int) -> str:
    """Return deterministic placeholder text with roughly target token count."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    word_count = max(1, math.ceil(target_tokens / TOKEN_WORD_RATIO))
    return " ".join(f"{label}_{i}" for i in range(word_count))


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += estimate_text_tokens(str(msg.get("content", "")))
    return total


def estimate_text_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * TOKEN_WORD_RATIO))
