#!/usr/bin/env python3
"""Prefill stage-split microbench — the one measurement that closes the TTFT prefill-law de-fit.

Measures c1 TTFT vs new/cached tokens, in-process. See profiling/docs/prefill_law_defit_trace.md
and the 2026-06-03 results in profiling/docs/prefill_stage_split_results.md.

RUN NOTES (learned the hard way — see results doc):
  * Run from a CLEAN cwd — a stray flash_attn.py shadows the package ('flash_attn' is not a package').
  * For device_ms: set VLLM_ENABLE_V1_MULTIPROCESSING=0 (vLLM v1's engine is a separate process, so an
    in-main-process torch.profiler sees 0 CUDA kernels). NOTE this is incompatible with --tp > 1.
  * CAVEAT: even then the host/device split is UNRELIABLE here — device_ms (Σ CUDA self-time) over-counts
    wall in eager and is hidden under CUDA graphs. Only end-to-end ttft_ms + tokenize_ms are trustworthy.
  * The `new` tail is freshly random PER TRIAL so it is a real cache MISS (else the warmup primes it and
    every measured trial is a hit — the original bug).
  Example:  CUDA_VISIBLE_DEVICES=7 VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 prefill_stage_split.py \\
              --model /data48/kevinlau/models/Llama-3.1-8B-Instruct --tp 1   (then --tp 2 without mp=0)
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import time
from pathlib import Path

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL_RNG = random.Random(777)  # fresh new-tail per trial -> the `new` tokens are a real cache MISS


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[:int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    """A UNIQUE fresh tail each call (random over the vocab) so `new` is never prefix-cached."""
    return " ".join(_TAIL_RNG.choice(_VOCAB) for _ in range(int(new * 0.96)))


def _cuda_self_ms(prof) -> float:
    total_us = 0.0
    for k in prof.key_averages():
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(k, attr, None)
            if v:
                total_us += float(v)
                break
    return total_us / 1000.0


def measure(llm, tok, cached: int, new: int, trials: int) -> dict:
    from vllm import SamplingParams
    from torch.profiler import profile, ProfilerActivity

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    prefix = cached_prefix(cached)

    # Prime the cached prefix once so it is a HIT every trial (we only want to prefill `new`).
    if cached > 0:
        llm.generate([prefix], sp, use_tqdm=False)

    tok_ms, dev_ms, host_ms, ttft_ms = [], [], [], []
    for _ in range(trials + 1):  # +1 warmup (discarded)
        # Fresh `new` tail EVERY trial -> `new` is a genuine cache miss (real prefill); cached part hits.
        prompt = (prefix + " " + fresh_tail(new)) if cached > 0 else fresh_tail(new)
        t0 = time.perf_counter()
        _ = tok.encode(prompt)
        t_tok = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt], sp, use_tqdm=False)
        wall = (time.perf_counter() - t1) * 1000.0
        dev = _cuda_self_ms(prof)
        tok_ms.append(t_tok); dev_ms.append(dev)
        host_ms.append(max(0.0, wall - dev)); ttft_ms.append(wall)
    med = lambda xs: st.median(xs[1:])
    return dict(new=new, cached=cached, n=trials,
                tokenize_ms=med(tok_ms), device_ms=med(dev_ms),
                host_ms=med(host_ms), ttft_ms=med(ttft_ms))


def regress(rows: list[dict]) -> None:
    try:
        import numpy as np
    except ImportError:
        print("(numpy unavailable — skipping regression; CSV has the raw stage rows)")
        return
    X = np.array([[1.0, r["new"], r["cached"]] for r in rows])
    print("\n=== stage regressions: stage ~ FLOOR + a*new + b*cached ===")
    for stage in ("ttft_ms", "device_ms", "host_ms", "tokenize_ms"):
        y = np.array([r[stage] for r in rows])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        print(f"  {stage:<12} FLOOR={beta[0]:7.2f} ms | new={beta[1]*1000:7.3f} ms/1k | cached={beta[2]*1000:7.3f} ms/1k")
    print("\nInterpret: device.new ~ GEMM roofline (~25 ms/1k tp1); host.new = NEW dispatch residual; "
          "device.cached ~ GPU paged-attn (~1.5 ms/1k); host.cached = CACHED host residual. "
          "Run --hash builtin vs sha256 -> host.cached delta = block-hash cost.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (needs that many visible GPUs)")
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--news", default="8,128,512,1024,2048", help="fresh-token counts to sweep")
    ap.add_argument("--cacheds", default="0,2000,8000,16000", help="cached-prefix token counts to sweep")
    ap.add_argument("--hash", default="sha256", choices=["sha256", "builtin"], help="prefix-cache hash algo")
    ap.add_argument("--eager", action="store_true", help="enforce_eager (no CUDA graphs) — isolates launch overhead")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="prefill_stage_split_H100.csv")
    a = ap.parse_args()

    from vllm import LLM
    kw = dict(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              enable_chunked_prefill=True, enforce_eager=a.eager,
              tensor_parallel_size=a.tp)
    try:
        llm = LLM(prefix_caching_hash_algo=a.hash, **kw)
    except TypeError:
        print(f"[warn] LLM() rejected prefix_caching_hash_algo; launching without it "
              f"({a.hash}).", flush=True)
        llm = LLM(**kw)
    tok = llm.get_tokenizer()

    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]
    rows = []
    for cached in cacheds:
        for new in news:
            r = measure(llm, tok, cached, new, a.trials)
            r["hash_algo"] = a.hash
            r["eager"] = int(a.eager)
            rows.append(r)
            print(f"  new={new:>5} cached={cached:>6}  tok={r['tokenize_ms']:6.2f}  dev={r['device_ms']:7.2f}  "
                  f"host={r['host_ms']:7.2f}  ttft={r['ttft_ms']:7.2f} ms", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new", "cached", "hash_algo", "eager", "n",
                                          "tokenize_ms", "device_ms", "host_ms", "ttft_ms"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\nwrote {out}")
    regress(rows)


if __name__ == "__main__":
    main()
