#!/usr/bin/env python3
"""Measure serving TTFT of B concurrent cache-HIT prefills vs (batch B, cached P).

Resolves the qsim's open question: the per-request cached-prefill cost (~6.1 ms/1k
cached, measured at concurrency=1) — does it SUM across B concurrently-prefilling
requests (per-request KV/host cost) or AMORTIZE (shared host/launch overhead)?

Method (matches the benchmark's serving path so host tokenize+prefix-hash is included):
  * Launch the real vLLM OpenAI server (same config as the multi-turn benchmark:
    prefix caching + chunked prefill, max-model-len 32768, gpu-mem 0.80, resolved
    defaults max_num_batched_tokens=8192 / max_num_seqs=1024).
  * For each (B, P): build B DISTINCT random P-token prompts (distinct contexts, like
    distinct sessions). WARM each once (caches its prefix). Then fire all B together,
    each = its cached prefix + a tiny UNIQUE new suffix (cache HIT on P, small new),
    streaming; record TTFT per request. Repeat `trials` times; report median.
  * B x P stays under the KV pool so there is NO eviction (isolates batch scaling).

Output CSV: B, P_tokens, n_ok, ttft_p50_ms, ttft_mean_ms, ttft_min_ms, ttft_max_ms.
Interpretation: ttft_p50(B,P) / ttft_p50(1,P) is the batch-scaling factor — ~1 means
full overlap, ~B means full sum, in-between is the partial-overlap the qsim needs.
"""
from __future__ import annotations
import argparse, asyncio, json, random, subprocess, sys, time, os
from pathlib import Path

import aiohttp


# Common English words that are each a SINGLE Llama token (so word count ~= token count).
_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but "
          "from or have an they which one you were her all she there would their we him been "
          "has when who will more no if out so up said what its about into than them can only "
          "other new some could time these two may then do first any my now such like our over "
          "man me even most made after also did many before must through back years where much").split()


def build_prompt(n_tokens: int, seed: int) -> str:
    # Distinct random sequence of 1-token common words → distinct context per seed (no shared
    # prefix across requests), ~1 token/word. Under-provision slightly (BOS/template overhead);
    # the server reports the true prompt_tokens, which we record.
    rng = random.Random(seed)
    n_words = max(1, int(n_tokens * 0.96))
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


async def stream_ttft(session, url, model, prompt, api_key, max_tokens):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = None
    async with session.post(url, json=body, headers=headers) as resp:
        async for raw in resp.content:
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if ttft is None and choices and (choices[0].get("text")):
                ttft = (time.perf_counter() - t0) * 1000.0
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens")
    return ttft, prompt_tokens


async def warm(session, url, model, prompt, api_key):
    # send the bare prefix once (max_tokens=1) so its blocks are prefix-cached
    await stream_ttft(session, url, model, prompt, api_key, max_tokens=1)


async def run_cell(session, url, model, api_key, B, P, trials):
    base = [build_prompt(P, seed=1000 * P + i) for i in range(B)]
    # warm all prefixes (sequential to avoid first-time contention)
    for pr in base:
        await warm(session, url, model, pr, api_key)
    ttfts = []
    ptoks = []
    for t in range(trials):
        # unique tiny suffix per (trial, req) -> cached prefix HIT + small new
        reqs = [f"{base[i]} q{t}_{i} answer:" for i in range(B)]
        results = await asyncio.gather(*[
            stream_ttft(session, url, model, r, api_key, max_tokens=4) for r in reqs
        ])
        for ttft, pt in results:
            if ttft is not None:
                ttfts.append(ttft)
                if pt:
                    ptoks.append(pt)
    ttfts.sort()
    n = len(ttfts)
    p50 = ttfts[n // 2] if n else float("nan")
    mean = sum(ttfts) / n if n else float("nan")
    medptok = sorted(ptoks)[len(ptoks) // 2] if ptoks else P
    return dict(B=B, P=medptok, n_ok=n, p50=p50, mean=mean,
                tmin=(ttfts[0] if n else float("nan")), tmax=(ttfts[-1] if n else float("nan")))


def wait_health(port, timeout=420):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


async def sweep(port, model, api_key, Bs, Ps, trials, out):
    url = f"http://127.0.0.1:{port}/v1/completions"
    rows = []
    timeout = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=max(Bs) + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        for P in Ps:
            for B in Bs:
                r = await run_cell(s, url, model, api_key, B, P, trials)
                rows.append(r)
                print(f"  B={B:>3} P={r['P']:>6}  ttft_p50={r['p50']:8.1f}ms  mean={r['mean']:8.1f}  "
                      f"n={r['n_ok']}  (per-req-vs-B1: see analysis)", flush=True)
    with open(out, "w") as f:
        f.write("B,P_tokens,n_ok,ttft_p50_ms,ttft_mean_ms,ttft_min_ms,ttft_max_ms\n")
        for r in rows:
            f.write(f"{r['B']},{r['P']},{r['n_ok']},{r['p50']:.3f},{r['mean']:.3f},{r['tmin']:.3f},{r['tmax']:.3f}\n")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", default="0.80")
    ap.add_argument("--max-model-len", default="32768")
    ap.add_argument("--out", default="cached_prefill_batch_ttft_H100.csv")
    ap.add_argument("--bs", default="1,2,4,8,16")
    ap.add_argument("--ps", default="2048,8192,16384")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--no-launch", action="store_true", help="server already running")
    a = ap.parse_args()
    Bs = [int(x) for x in a.bs.split(",")]
    Ps = [int(x) for x in a.ps.split(",")]

    proc = None
    if not a.no_launch:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port),
               "--dtype", "bfloat16", "--gpu-memory-utilization", a.gpu_mem,
               "--max-model-len", a.max_model_len, "--enable-prefix-caching",
               "--enable-chunked-prefill", "--api-key", a.api_key,
               "--no-enable-log-requests"]
        print("launching:", " ".join(cmd), flush=True)
        logf = open("vllm_server.log", "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_health(a.port):
            print("SERVER DID NOT BECOME HEALTHY — see vllm_server.log", flush=True)
            sys.exit(1)
        print("server healthy; starting sweep", flush=True)
        asyncio.run(sweep(a.port, a.model, a.api_key, Bs, Ps, a.trials, a.out))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
