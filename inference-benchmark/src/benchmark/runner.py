"""
Benchmark runner — orchestrates a full benchmark run.

Usage:
    python -m src.benchmark.runner \
        --url http://localhost:8000/v1/chat/completions \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend vllm \
        --profile output-short \
        --concurrency 10 \
        --num-requests 100 \
        --api-key test \
        --output results/run_001.json

    # TRT-LLM (point URL at /generate_stream):
    python -m src.benchmark.runner \
        --url http://localhost:8000/generate_stream \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend trtllm \
        --profile output-short \
        --concurrency 10 \
        --num-requests 100
"""

import asyncio
import argparse
import json
import sys
import time
import os
from pathlib import Path

import aiohttp

from .metrics import aggregate, aggregate_per_turn, print_summary, print_multi_turn_summary
from ..workloads.profiles import get_profile
from ..workloads.dataset import make_dataset
from ..workloads.arrival import make_arrival_times
from ..engines import get_backend, SUPPORTED_BACKENDS


async def run_benchmark(
    url: str,
    model: str,
    profile_name: str,
    concurrency: int,
    num_requests: int,
    backend_name: str = "vllm",
    api_key: str = "test",
    arrival_pattern: str = "steady",
    target_rate: float = 10.0,
    warmup_requests: int = 3,
    seed: int = 42,
    timeout: int = 120,
    ignore_eos: bool = False,
):
    """
    Run a benchmark and return (results, duration).
    """
    backend = get_backend(backend_name)
    profile = get_profile(profile_name)
    dataset = make_dataset(profile)
    arrival_times = make_arrival_times(
        pattern=arrival_pattern,
        num_requests=num_requests,
        concurrency=concurrency,
        target_rate=target_rate,
        seed=seed,
    )

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        # Warmup
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} requests...")
            await backend.run_warmup(url, model, api_key, warmup_requests, timeout)
            print("Warmup done.")

        # Schedule requests
        semaphore = asyncio.Semaphore(concurrency)
        results = [None] * num_requests
        benchmark_start = time.perf_counter()

        async def dispatch(i: int, dispatch_time: float):
            now = time.perf_counter() - benchmark_start
            delay = dispatch_time - now
            if delay > 0:
                await asyncio.sleep(delay)

            request = dataset.get_next_request()
            async with semaphore:
                result = await backend.send_request(
                    session=session,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos,
                )
            results[i] = result

        tasks = [dispatch(i, t) for i, t in enumerate(arrival_times)]
        await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start
    return results, benchmark_duration


async def run_multi_turn_benchmark(
    url: str,
    model: str,
    profile_name: str,
    concurrency: int,
    backend_name: str = "vllm",
    api_key: str = "test",
    warmup_requests: int = 3,
    timeout: int = 120,
    ignore_eos: bool = False,
):
    """
    Run a multi-turn benchmark with interleaved round-robin scheduling.

    Scheduling: [A1, B1, C1, A2, B2, C2, ...] where A1 = session A turn 1.
    This forces KV cache eviction between turns of the same session,
    testing prefix cache reuse under realistic memory pressure.

    Returns (results_by_turn, duration) where results_by_turn is a dict
    mapping turn_index → list[RequestResult].
    """
    from ..workloads.dataset import ShareGPTMultiTurnDataset, TrajectoryMultiTurnDataset

    backend = get_backend(backend_name)
    profile = get_profile(profile_name)
    dataset = make_dataset(profile)

    if not isinstance(dataset, (ShareGPTMultiTurnDataset, TrajectoryMultiTurnDataset)):
        raise ValueError(f"Profile '{profile_name}' does not use a multi-turn dataset")

    sessions = dataset.sessions
    if not sessions:
        raise ValueError("No multi-turn sessions loaded — check ShareGPT dataset and filter bounds")

    max_turns = max(len(s.turns) for s in sessions)
    print(f"Loaded {len(sessions)} sessions, max {max_turns} turns per session")

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session_http:
        # Warmup
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} requests...")
            await backend.run_warmup(url, model, api_key, warmup_requests, timeout)
            print("Warmup done.")

        semaphore = asyncio.Semaphore(concurrency)
        # results_by_turn[turn_idx] = list of RequestResult
        results_by_turn: dict[int, list] = {i: [] for i in range(max_turns)}
        benchmark_start = time.perf_counter()

        async def dispatch(session_id: int, request, t_idx: int):
            async with semaphore:
                result = await backend.send_request(
                    session=session_http,
                    url=url,
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    api_key=api_key,
                    ignore_eos=ignore_eos,
                )
            return t_idx, result

        # Interleaved round-robin: process all sessions' turn N before turn N+1
        for turn_idx in range(max_turns):
            turn_requests = []
            for conv_session in sessions:
                if turn_idx < len(conv_session.turns):
                    turn_requests.append((conv_session.session_id, conv_session.turns[turn_idx]))

            if not turn_requests:
                continue

            print(f"  Turn {turn_idx + 1}/{max_turns}: dispatching {len(turn_requests)} requests...")

            tasks = [dispatch(sid, req, turn_idx) for sid, req in turn_requests]
            completed = await asyncio.gather(*tasks)

            for t_idx, result in completed:
                results_by_turn[t_idx].append(result)

    benchmark_duration = time.perf_counter() - benchmark_start

    # Flatten results, tagging each with turn_index
    all_results = []
    for turn_idx in sorted(results_by_turn.keys()):
        for r in results_by_turn[turn_idx]:
            r.turn_index = turn_idx
            all_results.append(r)

    return all_results, results_by_turn, benchmark_duration


def save_results(summary, results, output_path: str, config: dict):
    """Save summary + per-request data to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": config,
        "summary": summary.to_dict(),
        "per_request": [
            {
                "success": r.success,
                "ttft_ms": round(r.ttft * 1000, 2) if r.ttft else None,
                "tpot_ms": round(r.tpot * 1000, 2) if r.tpot else None,
                "itl_ms": [round(t * 1000, 2) for t in r.itl] if r.itl else [],
                "e2el_ms": round(r.e2el * 1000, 2) if r.e2el else None,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "error": r.error,
                **({"turn_index": r.turn_index} if r.turn_index is not None else {}),
            }
            for r in results if r is not None
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description="inference-benchmark runner")
    parser.add_argument("--url", required=False, help="Server endpoint URL")
    parser.add_argument("--model", required=False)
    parser.add_argument("--backend", default="vllm", choices=SUPPORTED_BACKENDS,
                        help="Backend type (vllm/sglang/openai → /v1/chat/completions, trtllm → /generate_stream)")
    parser.add_argument("--profile", default="output-short", help="Workload profile name")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--arrival", default="steady", choices=["steady", "poisson", "ramp"])
    parser.add_argument("--target-rate", type=float, default=10.0, help="req/s for poisson/ramp")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="results/latest.json")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Pass ignore_eos=true to vLLM (needed for FP8 models with random token workloads)")
    parser.add_argument("--mode", choices=["stress-test", "single-turn", "multi-turn"],
                        help="Benchmark mode (sets profile defaults and required flags). "
                             "Use --profile for a specific profile within a mode.")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--agent-type", type=str, default=None, help="Filter profiles by agent type")
    parser.add_argument("--turn-style", type=str, default=None, help="Filter profiles by turn style")
    parser.add_argument("--serving-style", type=str, default=None, help="Filter profiles by serving style")
    parser.add_argument("--data-source", type=str, default=None, help="Filter profiles by data source")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.list_profiles:
        from ..workloads.profiles import filter_profiles, PROFILES, AGENT_TYPES, TURN_STYLES, SERVING_STYLES, DATA_SOURCES
        filtered = filter_profiles(
            agent_type=args.agent_type,
            turn_style=args.turn_style,
            serving_style=args.serving_style,
            data_source=args.data_source,
        )
        print(f"\n{'Name':<30} {'Agent Type':<18} {'Turn Style':<14} {'Serving':<20} {'Data Source':<12} {'ISL':<6} {'OSL':<6}")
        print("-" * 110)
        for name, p in sorted(filtered.items()):
            print(f"{name:<30} {p.agent_type:<18} {p.turn_style:<14} {p.serving_style:<20} {p.data_source:<12} {p.isl_tokens:<6} {p.osl_tokens:<6}")
        print(f"\n{len(filtered)} profiles shown (of {len(PROFILES)} total)")
        if any([args.agent_type, args.turn_style, args.serving_style, args.data_source]):
            active = []
            if args.agent_type: active.append(f"agent_type={args.agent_type}")
            if args.turn_style: active.append(f"turn_style={args.turn_style}")
            if args.serving_style: active.append(f"serving_style={args.serving_style}")
            if args.data_source: active.append(f"data_source={args.data_source}")
            print(f"Filters: {', '.join(active)}")
        sys.exit(0)

    # --url and --model are required for actual benchmark runs
    if not args.url or not args.model:
        print("Error: --url and --model are required for benchmark runs.")
        print("Use --list-profiles to browse profiles without a server.")
        sys.exit(1)

    if args.mode:
        if args.mode == "multi-turn":
            print("NOTE: multi-turn mode requires server launched with --enable-prefix-caching (vLLM)")
            if args.profile == "output-short":  # default — override for multi-turn
                args.profile = "chat-multiturn-short"
        if args.mode == "stress-test":
            if not args.ignore_eos:
                print("NOTE: stress-test mode auto-enables --ignore-eos (required for FP8 models)")
                args.ignore_eos = True
            if args.profile == "output-short":  # default — override for stress-test
                args.profile = "random-inferencex"
        if args.mode == "single-turn":
            print("NOTE: single-turn mode requires server launched with --enable-prefix-caching (vLLM)")
            print("      or radix cache (SGLang default). See scripts/launch_server.sh")

    config = {**vars(args), "mode": args.mode}
    profile = get_profile(args.profile)

    if profile.mode == "multi-turn":
        all_results, results_by_turn, duration = asyncio.run(run_multi_turn_benchmark(
            url=args.url,
            model=args.model,
            profile_name=args.profile,
            concurrency=args.concurrency,
            backend_name=args.backend,
            api_key=args.api_key,
            warmup_requests=args.warmup,
            timeout=args.timeout,
            ignore_eos=args.ignore_eos,
        ))

        summary = aggregate(
            results=[r for r in all_results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=args.profile,
            concurrency=args.concurrency,
        )

        turn_summaries = aggregate_per_turn(results_by_turn)
        print_multi_turn_summary(turn_summaries, summary)
        save_results(summary, all_results, args.output, config)

        # Also save per-turn breakdown
        turn_output = args.output.replace(".json", "_per_turn.json")
        import json as json_mod
        from pathlib import Path as PathMod
        PathMod(turn_output).parent.mkdir(parents=True, exist_ok=True)
        with open(turn_output, "w") as f:
            json_mod.dump({
                "config": config,
                "per_turn": [ts.to_dict() for ts in turn_summaries],
            }, f, indent=2)
        print(f"Per-turn results saved to: {turn_output}")

    else:
        results, duration = asyncio.run(run_benchmark(
            url=args.url,
            model=args.model,
            profile_name=args.profile,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            backend_name=args.backend,
            api_key=args.api_key,
            arrival_pattern=args.arrival,
            target_rate=args.target_rate,
            warmup_requests=args.warmup,
            seed=args.seed,
            timeout=args.timeout,
            ignore_eos=args.ignore_eos,
        ))

        summary = aggregate(
            results=[r for r in results if r is not None],
            duration_s=duration,
            model=args.model,
            profile=args.profile,
            concurrency=args.concurrency,
        )

        print_summary(summary)
        save_results(summary, results, args.output, config)
