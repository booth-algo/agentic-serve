#!/usr/bin/env python3
"""Herd-scaling probe: how c1 prefill TTFT and its per-stage spans grow when N
identical requests arrive as a SYNCHRONIZED barrier (asyncio.gather).

Companion to serving_stage_split.py (c1-only). Every request in a burst shares the SAME
primed cached prefix (APC hit -> GPU only prefills each request's small fresh `new` tail),
so GPU prefill work stays ~flat across the burst and the TTFT growth with concurrency
isolates FRONTEND + scheduler SERIALIZATION -- the mechanism the v2 queue sim under-
predicts in the sub-saturation band.

TWO JOBS:
  1. VERIFY server-side. The client wall TTFT can be inflated by THIS single-process asyncio
     client serializing N concurrent SSE reads. So we ALSO scrape vLLM's own
     `time_to_first_token_seconds` histogram (delta/conc = mean SERVER TTFT, immune to the
     client event loop). server_frontend = server_ttft - queue - prefill. If THAT grows with
     conc, the serialization is genuinely in the server frontend, not the client.
  2. CHARACTERIZE. Sweep (new, cached) x conc so the per-request frontend service F and its
     token-dependence + sub-linear GPU-overlap can be fit for a serving-frontend term.

Per (new, cached, conc): prime the prefix, fire `conc` requests concurrently (each a fresh
`new` tail), scrape /metrics _sum before/after the burst -> mean per-request spans
(delta/conc), record client wall TTFT median/max.

Run (GPU 7, self-launches the server):
  CUDA_VISIBLE_DEVICES=7 TMPDIR=/data48/kevinlau/tmp XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
    ~/miniconda3/envs/vllm/bin/python \
    profiling/gpu_profiling/vllm/serving_herd_scaling.py \
    --news 128,2048 --cacheds 0,8000 --concs 1,5,10,20 \
    --out profile_data/results/serving_herd_scaling_H100.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:
    sys.path.remove(_p)

import aiohttp  # noqa: E402

# --- prompt construction (verbatim from serving_stage_split.py: ~1 Llama token/word) ---
_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[: int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    return " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))


_PROM_METRICS = {
    "queue_span_s":   "vllm:request_queue_time_seconds",
    "prefill_span_s": "vllm:request_prefill_time_seconds",
    "e2e_s":          "vllm:e2e_request_latency_seconds",
    "ttft_s":         "vllm:time_to_first_token_seconds",  # SERVER-side TTFT (client-loop immune)
}


def _parse_prom_sums(text: str) -> dict:
    wanted = {f"{base}_sum": key for key, base in _PROM_METRICS.items()}
    out = {key: None for key in _PROM_METRICS}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        sp = line.rsplit(" ", 1)
        if len(sp) != 2:
            continue
        name_and_labels, val = sp
        key = wanted.get(name_and_labels.split("{", 1)[0])
        if key is not None:
            try:
                out[key] = float(val)
            except ValueError:
                pass
    return out


async def scrape(session, base_url):
    try:
        async with session.get(base_url + "/metrics") as resp:
            if resp.status != 200:
                return {k: None for k in _PROM_METRICS}
            return _parse_prom_sums(await resp.text())
    except aiohttp.ClientError:
        return {k: None for k in _PROM_METRICS}


def _sub(a, b):
    return None if (a is None or b is None) else a - b


async def ttft_once(session, url, model, content):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    headers = {"Authorization": "Bearer test", "Content-Type": "application/json"}
    t_send = time.perf_counter()
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        ttft_ms = None
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ds = line[len("data:"):].strip()
            if ds == "[DONE]":
                break
            ch = json.loads(ds)
            cc = ch.get("choices", [])
            if ttft_ms is None and cc and cc[0].get("delta", {}).get("content") is not None:
                ttft_ms = (time.perf_counter() - t_send) * 1000.0
        return ttft_ms


async def burst(session, base_url, model, cached, new, conc, trials):
    """Fire `conc` concurrent requests (barrier), `trials` times. Returns mean per-request
    spans (delta/conc) for the SERVER metrics + the client wall TTFT distribution."""
    chat_url = base_url + "/v1/chat/completions"
    prefix = cached_prefix(cached)
    if cached > 0:
        await ttft_once(session, chat_url, model, prefix)  # prime -> APC hit for the burst

    c_med, c_max, q_ms, p_ms, sttft_ms = [], [], [], [], []
    for t in range(trials + 1):  # +1 warmup
        contents = [(prefix + " " + fresh_tail(new)) if cached > 0 else fresh_tail(new)
                    for _ in range(conc)]
        before = await scrape(session, base_url)
        res = await asyncio.gather(*[ttft_once(session, chat_url, model, c) for c in contents])
        after = await scrape(session, base_url)
        if t == 0:
            continue  # discard warmup
        tt = sorted(x for x in res if x is not None)
        if not tt:
            continue
        c_med.append(tt[len(tt) // 2])
        c_max.append(tt[-1])
        dq = _sub(after["queue_span_s"], before["queue_span_s"])
        dp = _sub(after["prefill_span_s"], before["prefill_span_s"])
        dt = _sub(after["ttft_s"], before["ttft_s"])
        if dq is not None:
            q_ms.append(dq * 1000.0 / conc)
        if dp is not None:
            p_ms.append(dp * 1000.0 / conc)
        if dt is not None:
            sttft_ms.append(dt * 1000.0 / conc)

    def med(xs):
        return st.median(xs) if xs else None

    cm, q, p, sttft = med(c_med), med(q_ms), med(p_ms), med(sttft_ms)
    # SERVER frontend = server ttft - engine(queue+prefill): immune to client event-loop.
    server_frontend = None if None in (sttft, q, p) else sttft - q - p
    # CLIENT frontend = client wall - engine: includes any client-side serialization.
    client_frontend = None if None in (cm, q, p) else cm - q - p
    return {"new": new, "cached": cached, "conc": conc, "trials": trials,
            "ttft_client_med_ms": cm, "ttft_client_max_ms": med(c_max),
            "ttft_server_ms": sttft, "mean_queue_ms": q, "mean_prefill_ms": p,
            "server_frontend_ms": server_frontend, "client_frontend_ms": client_frontend}


def wait_health(port, timeout=420):
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


def launch_server(model, port, gpu_mem, max_model_len, api_key, log_path):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--served-model-name", "llama",
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--gpu-memory-utilization", str(gpu_mem),
           "--max-model-len", str(max_model_len), "--tensor-parallel-size", "1",
           "--enable-prefix-caching", "--enable-chunked-prefill",
           "--api-key", api_key, "--prefix-caching-hash-algo", "sha256",
           "--no-enable-log-requests"]
    print("launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


_CSV_FIELDS = ["new", "cached", "conc", "trials", "ttft_client_med_ms", "ttft_client_max_ms",
               "ttft_server_ms", "mean_queue_ms", "mean_prefill_ms",
               "server_frontend_ms", "client_frontend_ms"]


async def run(base_url, model, news, cacheds, concs, trials, out_path):
    rows = []
    conn = aiohttp.TCPConnector(limit=0)  # no client-side cap -> true concurrency
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=conn) as s:
        present = [k for k, v in (await scrape(s, base_url)).items() if v is not None]
        print(f"/metrics histograms present: {present}", flush=True)
        for cached in cacheds:
            for new in news:
                for conc in concs:
                    r = await burst(s, base_url, model, cached, new, conc, trials)
                    rows.append(r)
                    def f(x):
                        return "n/a" if x is None else f"{x:.1f}"
                    print(f"  new={new:>4} cached={cached:>5} conc={conc:>3} | "
                          f"cli_med={f(r['ttft_client_med_ms']):>7} srv_ttft={f(r['ttft_server_ms']):>7} "
                          f"queue={f(r['mean_queue_ms']):>6} prefill={f(r['mean_prefill_ms']):>6} "
                          f"| SRV_frontend={f(r['server_frontend_ms']):>7} "
                          f"cli_frontend={f(r['client_frontend_ms']):>7}", flush=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})
    print(f"\nwrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--served-model-name", default="llama")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--news", default="128,2048", help="fresh-token counts (isolate new dependence)")
    ap.add_argument("--cacheds", default="0,8000", help="shared primed-prefix token counts (cached dependence)")
    ap.add_argument("--concs", default="1,5,10,20", help="burst concurrencies (overlap curve)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="profile_data/results/serving_herd_scaling_H100.csv")
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--server-log", default="vllm_server_herd_scaling.log")
    a = ap.parse_args()

    base_url = f"http://127.0.0.1:{a.port}"
    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]
    concs = [int(x) for x in a.concs.split(",")]
    proc = None
    if not a.no_launch:
        proc = launch_server(a.model, a.port, a.gpu_mem, a.max_model_len, a.api_key, a.server_log)
    try:
        if not wait_health(a.port):
            print(f"SERVER DID NOT BECOME HEALTHY -- see {a.server_log}", flush=True)
            sys.exit(1)
        print("server healthy; starting herd-scaling sweep", flush=True)
        asyncio.run(run(base_url, a.served_model_name, news, cacheds, concs, a.trials, a.out))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
