#!/usr/bin/env python3
"""L13 S7 sidecar: sample a vLLM /metrics endpoint at a fixed interval and append
one JSON line per sample with the prefix-cache / token counters.

Purpose (L13-3090multi round 2, S7 probe): the GT multi-turn benchmark's
``cached_context_tokens`` is a CLIENT-side estimate (``cache_estimate_source =
'previous_prompt_tokens'``) that does not see engine truth. The engine's own
``vllm:gpu_prefix_cache_queries`` / ``vllm:gpu_prefix_cache_hits`` token counters
(v1 metrics, vllm 0.19) measure exactly how many prompt tokens each scheduling
window re-computed vs hit, so sampling them across a replayed GT cell yields the
TRUE per-turn GPU re-prefill volume (the quantity ttft_queue_sim drains).

Stdlib only; counters are summed across label sets per metric name. Pure
measurement tooling - no simulator behaviour depends on this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

KEEP = (
    "prefix_cache",          # vllm:gpu_prefix_cache_queries / _hits (token counters)
    "prompt_tokens",         # vllm:prompt_tokens_total
    "generation_tokens",     # vllm:generation_tokens_total
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage",
    "num_preemptions",
)


def sample(url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with urllib.request.urlopen(url, timeout=5) as r:
        for raw in r.read().decode("utf-8", "replace").splitlines():
            if not raw or raw.startswith("#"):
                continue
            if not any(k in raw for k in KEEP):
                continue
            try:
                name_labels, val = raw.rsplit(" ", 1)
                name = name_labels.split("{", 1)[0]
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()
    url = f"http://localhost:{args.port}/metrics"
    with open(args.out, "a") as f:
        while True:
            t0 = time.time()
            try:
                m = sample(url)
                m["ts"] = t0
                f.write(json.dumps(m) + "\n")
                f.flush()
            except Exception as e:  # server restarting/teardown: record and continue
                f.write(json.dumps({"ts": t0, "error": str(e)[:200]}) + "\n")
                f.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
