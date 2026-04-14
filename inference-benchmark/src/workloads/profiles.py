"""
Workload profile definitions.

Profiles are organized into groups:
  Group 1: Real agent data (SWEBench PLLM, SWEBench trajectories, TerminalBench trajectories)
  Group 2: Chat — single-turn and multi-turn (ShareGPT)
  Group 3: Synthetic stress tests (random tokens, file-based)

Each profile defines the data source, ISL/OSL bounds, and metadata tags.
"""

from dataclasses import dataclass
from typing import Optional


# Valid tag values
AGENT_TYPES = ["chat", "coding", "terminal", "computer-use"]
TURN_STYLES = ["single-turn", "multi-turn"]
SERVING_STYLES = ["disaggregated", "not-disaggregated"]
DATA_SOURCES = ["sharegpt", "swebench", "terminalbench", "osworld", "file", "random", "test"]


@dataclass
class WorkloadProfile:
    name: str
    isl_tokens: int   # for random: exact target ISL; for sharegpt: max ISL filter bound
    osl_tokens: int   # for random: exact target OSL; for sharegpt: max OSL filter bound (also max_tokens)
    isl_stddev: float        # stddev as fraction of isl (for Gaussian sampling)
    description: str
    dataset: str             # "sharegpt", "file", "test", "random", "jsonl", "sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn"
    file_path: str = ""      # used when dataset="file" or "jsonl"
    system_prompt: str = "You are a helpful assistant."
    tokenizer_name: str = "" # used when dataset="random"
    mode: str = "single-turn"           # "stress-test" | "single-turn" | "multi-turn"
    prefix_caching_required: bool = False  # True = server must be launched with --enable-prefix-caching
    min_turns: int = 1                   # multi-turn: minimum turns per session
    max_turns: int = 1                   # multi-turn: maximum turns per session
    num_sessions: int = 200              # multi-turn: number of concurrent sessions
    agent_type: str = ""           # "chat" | "coding" | "terminal"
    turn_style: str = "single-turn"  # "single-turn" | "multi-turn"
    serving_style: str = "not-disaggregated"  # "disaggregated" | "not-disaggregated"
    data_source: str = ""          # "sharegpt" | "swebench" | "terminalbench" | "file" | "random" | "test"


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

PROFILES: dict[str, WorkloadProfile] = {

    # ===================================================================
    # Group 1: Real Agent Data
    # ===================================================================

    # Single-turn PLLM planning call — real SWEBench prompts (~6K ISL)
    "coding-agent": WorkloadProfile(
        name="coding-agent",
        isl_tokens=17000,
        osl_tokens=800,
        isl_stddev=0.0,
        description="Real coding-agent prompts from Sequrity SWEBench runs (PLLM planning calls, ~17K ISL, ~800 OSL)",
        dataset="jsonl",
        file_path="data/coding_agent_prompts.jsonl",
        system_prompt="",  # system prompt is embedded in the JSONL
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="coding",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),

    # Multi-turn SWEBench coding agent — real trajectories from harbor/jobs/
    # Note: "turns" here are agent steps (tool calls), not logical conversation rounds.
    # SWEBench sessions have min=13, median=85, max=320 steps.
    # Data uses compressed trajectory.json (summary messages ~100 chars each).
    # TODO: Extract full rollout JSONL for realistic ISL/OSL per step.
    "swebench-multiturn-short": WorkloadProfile(
        name="swebench-multiturn-short",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 13-30 step sessions (shortest available)",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=13,
        max_turns=30,
        num_sessions=100,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),
    "swebench-multiturn-medium": WorkloadProfile(
        name="swebench-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 30-80 step sessions",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=30,
        max_turns=80,
        num_sessions=100,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),
    "swebench-multiturn-long": WorkloadProfile(
        name="swebench-multiturn-long",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 80-150 step sessions",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=80,
        max_turns=150,
        num_sessions=50,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),
    "swebench-multiturn-xl": WorkloadProfile(
        name="swebench-multiturn-xl",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real SWEBench coding agent: 150+ step sessions (longest available)",
        dataset="swebench-multi-turn",
        file_path="data/swebench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=150,
        max_turns=400,
        num_sessions=30,
        agent_type="coding",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="swebench",
    ),

    # Multi-turn TerminalBench CLI agent — real trajectories from harbor/jobs/
    # Note: "turns" here are agent steps (tool calls), not logical conversation rounds.
    # TerminalBench sessions have min=2, median=61, max=876 steps.
    # Data uses compressed trajectory.json (summary messages ~100 chars each).
    # TODO: Extract full rollout JSONL for realistic ISL/OSL per step.
    "terminalbench-multiturn-short": WorkloadProfile(
        name="terminalbench-multiturn-short",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 2-20 step sessions (shortest available)",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=2,
        max_turns=20,
        num_sessions=100,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
    ),
    "terminalbench-multiturn-medium": WorkloadProfile(
        name="terminalbench-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 20-60 step sessions",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=20,
        max_turns=60,
        num_sessions=100,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
    ),
    "terminalbench-multiturn-long": WorkloadProfile(
        name="terminalbench-multiturn-long",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 60-150 step sessions",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=60,
        max_turns=150,
        num_sessions=50,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
    ),
    "terminalbench-multiturn-xl": WorkloadProfile(
        name="terminalbench-multiturn-xl",
        isl_tokens=131072,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="Real TerminalBench CLI agent: 150+ step sessions (longest available)",
        dataset="terminalbench-multi-turn",
        file_path="data/terminalbench_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=150,
        max_turns=1000,
        num_sessions=30,
        agent_type="terminal",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="terminalbench",
    ),

    # Multi-turn OSWorld computer-use agent — real WebArena trajectories
    # Note: "turns" here are agent steps (browser actions).
    # OSWorld sessions have min=1, median=8, max=30 steps.
    # ISL/OSL ratio ~120:1 (massive DOM context, tiny action output).
    "osworld-multiturn-short": WorkloadProfile(
        name="osworld-multiturn-short",
        isl_tokens=32768,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 2-10 step sessions (short browsing tasks)",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=2,
        max_turns=10,
        num_sessions=50,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
    ),
    "osworld-multiturn-medium": WorkloadProfile(
        name="osworld-multiturn-medium",
        isl_tokens=65536,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 10-20 step sessions",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=10,
        max_turns=20,
        num_sessions=30,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
    ),
    "osworld-multiturn-long": WorkloadProfile(
        name="osworld-multiturn-long",
        isl_tokens=131072,
        osl_tokens=500,
        isl_stddev=0.0,
        description="Real OSWorld computer-use agent: 20-30 step sessions (longest available)",
        dataset="osworld-multi-turn",
        file_path="data/osworld_trajectories.jsonl",
        system_prompt="",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=20,
        max_turns=30,
        num_sessions=20,
        agent_type="computer-use",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="osworld",
    ),

    # ===================================================================
    # Group 2: Chat — ShareGPT (single-turn and multi-turn)
    # ===================================================================

    # --- Single-turn ---

    "chat-short": WorkloadProfile(
        name="chat-short",
        isl_tokens=500,
        osl_tokens=300,
        isl_stddev=0.15,
        description="Short Q&A chat — most common pattern (ShareGPT, ISL≤500, OSL≤300)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-medium": WorkloadProfile(
        name="chat-medium",
        isl_tokens=2000,
        osl_tokens=1000,
        isl_stddev=0.15,
        description="Medium chat — longer conversations and detailed answers (ShareGPT, ISL≤2000, OSL≤1000)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-long": WorkloadProfile(
        name="chat-long",
        isl_tokens=8000,
        osl_tokens=2000,
        isl_stddev=0.15,
        description="Long chat — longest natural ShareGPT conversations (ISL≤8000, OSL≤2000)",
        dataset="sharegpt",
        mode="single-turn",
        prefix_caching_required=True,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),

    # --- Multi-turn ---

    "chat-multiturn-short": WorkloadProfile(
        name="chat-multiturn-short",
        isl_tokens=8192,
        osl_tokens=1000,
        isl_stddev=0.0,
        description="ShareGPT multi-turn chat: 3-5 turns, moderate growing context",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=3,
        max_turns=5,
        num_sessions=200,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-multiturn-medium": WorkloadProfile(
        name="chat-multiturn-medium",
        isl_tokens=16384,
        osl_tokens=1500,
        isl_stddev=0.0,
        description="ShareGPT multi-turn chat: 5-10 turns, large growing context",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=5,
        max_turns=10,
        num_sessions=100,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-multiturn-long": WorkloadProfile(
        name="chat-multiturn-long",
        isl_tokens=32768,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="ShareGPT multi-turn chat: 10-20 turns, deep KV cache stress",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=10,
        max_turns=20,
        num_sessions=50,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),
    "chat-multiturn-xl": WorkloadProfile(
        name="chat-multiturn-xl",
        isl_tokens=65536,
        osl_tokens=2000,
        isl_stddev=0.0,
        description="ShareGPT multi-turn chat: 20-30 turns, extreme context length stress",
        dataset="sharegpt-multi-turn",
        mode="multi-turn",
        prefix_caching_required=True,
        min_turns=20,
        max_turns=30,
        num_sessions=30,
        agent_type="chat",
        turn_style="multi-turn",
        serving_style="not-disaggregated",
        data_source="sharegpt",
    ),

    # ===================================================================
    # Group 3: Synthetic Stress Tests
    # ===================================================================

    "prefill-heavy": WorkloadProfile(
        name="prefill-heavy",
        isl_tokens=8192,
        osl_tokens=256,
        isl_stddev=0.0,
        description="Synthetic prefill stress: long input, short output (ISL=8192, OSL=256)",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
    ),
    "decode-heavy": WorkloadProfile(
        name="decode-heavy",
        isl_tokens=256,
        osl_tokens=4096,
        isl_stddev=0.0,
        description="Synthetic decode stress: short input, long output (ISL=256, OSL=4096)",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
    ),
    "random-1k": WorkloadProfile(
        name="random-1k",
        isl_tokens=1024,
        osl_tokens=1024,
        isl_stddev=0.0,
        description="InferenceX cross-validation: random tokens ISL=1024 OSL=1024",
        dataset="random",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="stress-test",
        prefix_caching_required=False,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="random",
    ),

    # ===================================================================
    # Group 4: Utility
    # ===================================================================

    "test": WorkloadProfile(
        name="test",
        isl_tokens=10,
        osl_tokens=20,
        isl_stddev=0.0,
        description="Quick smoke test",
        dataset="test",
        mode="single-turn",
        prefix_caching_required=False,
        agent_type="chat",
        turn_style="single-turn",
        serving_style="not-disaggregated",
        data_source="test",
    ),
}


# ---------------------------------------------------------------------------
# Old → new profile name mapping (for backward compat with existing results)
# ---------------------------------------------------------------------------

PROFILE_ALIASES: dict[str, str] = {
    # Old ShareGPT profiles → chat-short (they were all ~200 ISL anyway)
    "chatbot-short": "chat-short",
    "chatbot-multi-turn": "chat-medium",
    "rag-retrieval": "chat-medium",
    "rag-heavy": "chat-medium",
    "coding-assist": "chat-medium",
    "coding-heavy": "chat-medium",
    "summarization": "chat-medium",
    "agentic-tool-use": "chat-medium",
    "computer-use-basic": "chat-short",
    "customer-support-basic": "chat-short",
    # Old synthetic profiles
    "output-short": "prefill-heavy",
    "output-long": "decode-heavy",
    "random-inferencex": "random-1k",
    "random-inferencex-legacy": "random-1k",
    "random-inferencex-doublewrap": "random-1k",
    # Old multi-turn
    "multi-turn-short": "chat-multiturn-short",
    "multi-turn-medium": "chat-multiturn-medium",
    "multi-turn-long": "chat-multiturn-long",
}


# ---------------------------------------------------------------------------
# Filtering and lookup
# ---------------------------------------------------------------------------

def filter_profiles(
    agent_type: Optional[str] = None,
    turn_style: Optional[str] = None,
    serving_style: Optional[str] = None,
    data_source: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Filter profiles by tag values. None means 'any'."""
    result = {}
    for name, p in PROFILES.items():
        if agent_type is not None and p.agent_type != agent_type:
            continue
        if turn_style is not None and p.turn_style != turn_style:
            continue
        if serving_style is not None and p.serving_style != serving_style:
            continue
        if data_source is not None and p.data_source != data_source:
            continue
        if mode is not None and p.mode != mode:
            continue
        result[name] = p
    return result


# Convenience filters
STRESS_TEST_PROFILES = filter_profiles(mode="stress-test")
SINGLE_TURN_PROFILES = filter_profiles(turn_style="single-turn")
MULTI_TURN_PROFILES = filter_profiles(turn_style="multi-turn")
REAL_DATA_PROFILES = {
    k: v for k, v in PROFILES.items()
    if v.data_source in ("swebench", "terminalbench")
}


def get_profile(name: str) -> WorkloadProfile:
    """Look up a profile by name, resolving aliases for old names."""
    if name in PROFILES:
        return PROFILES[name]
    if name in PROFILE_ALIASES:
        resolved = PROFILE_ALIASES[name]
        return PROFILES[resolved]
    raise ValueError(
        f"Unknown profile '{name}'. Available: {sorted(PROFILES.keys())}"
    )


def resolve_profile_name(name: str) -> str:
    """Return the canonical profile name, resolving aliases."""
    if name in PROFILES:
        return name
    if name in PROFILE_ALIASES:
        return PROFILE_ALIASES[name]
    return name
