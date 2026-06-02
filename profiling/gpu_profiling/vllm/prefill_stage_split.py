#!/usr/bin/env python3
"""Prefill stage-split microbench — the one measurement that closes the TTFT prefill-law de-fit.

The serving prefill law `ttft = FLOOR + NEW·new + HOST·cached` was fit to end-to-end c1 TTFT, so
its coefficients bundle GPU and off-GPU work that the kernel grids (GPU-forward only) can't separate.
The de-fit audit (profiling/docs/prefill_law_defit_trace.md) reduced everything to a single missing
fact: **the host-vs-device split of c1 TTFT vs new/cached tokens.** This script measures exactly that,
in-process, so each fitted coefficient becomes a measured/derived quantity:

  * FLOOR (22.5)        -> intercept = TTFT at new->0, cached=0 (smallest pure-prefill, here new~8)
  * NEW dispatch tail   -> the HOST (CPU) slope vs `new` ABOVE the device GEMM slope
                           (we already derive the GEMM part from the roofline = 0.025 ms/tok tp1)
  * CACHED host (~2.5/1k) -> the HOST slope vs `cached` ABOVE the measured GPU paged-attn (~1.5/1k)
  * SHA-256 hypothesis  -> rerun with --hash builtin vs sha256; the CACHED host-slope DELTA is the
                           prefix-cache block-hash cost (vLLM default sha256 >= v0.11)
  * CUDA-graph launch   -> rerun with --eager; the host-slope delta vs graph mode is launch overhead

Method (offline vLLM `LLM`, B=1, isolates one request's stages):
  * For each (new, cached): build a `cached`-token shared prefix + `new` fresh tokens. PRIME the prefix
    once (so the measured request is a cache HIT on `cached`, prefilling only `new`). Then measure the
    request with max_tokens=1 (= TTFT), capturing per stage:
       tokenize_ms = wall of tokenizer.encode(prompt)            [pure host]
       device_ms   = sum of CUDA kernel self-time (torch.profiler) [GPU forward]
       host_ms     = wall(generate) - device_ms                   [non-overlapped host: dispatch/sched/sample/return]
  * Median over --trials. Sweep new x cached. Regress each stage on (new, cached).

Output CSV: profile_data/results/prefill_stage_split_H100.csv
  columns: new, cached, hash_algo, eager, n, tokenize_ms, device_ms, host_ms, ttft_ms
Plus a printed regression: device/host slopes vs new and vs cached, the SHA-256 delta, and the FLOOR.

Run on the H100 (see profiling/docs/h100_setup.md) from a clean CWD (avoid a local flash_attn.py shadow),
e.g.:  VLLM_WORKER_MULTIPROC_METHOD=spawn python3 prefill_stage_split.py \
         --model meta-llama/Llama-3.1-8B-Instruct --hash sha256
       (then again with --hash builtin, and once with --eager, to fill the deltas)

NOTE: vLLM's offline kwarg for the hash algo and torch.profiler's CUDA-time attribute name vary by
version — both are handled with fallbacks below; if your vLLM rejects `prefix_caching_hash_algo`, set it
via the documented flag for your version (the two TODO spots are marked).
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import time
from pathlib import Path

# 1-token-per-word vocab so token count ~= word count (server reports true prompt_tokens; here we
# approximate, which is fine — we regress on the REQUESTED new/cached, the controlled variable).
_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most "
          "made after also did many before must through back years where much your way well down").split()
import random
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]


def build_prompt(cached: int, new: int) -> str:
    """`cached` shared-prefix tokens (FIXED across calls → cache HIT) + `new` fresh-ish tokens."""
    nc = int(cached * 0.96)
    prefix = _WORDS[:nc]
    # fresh tail: shift into a disjoint region of the word stream so it isn't already cached
    tail = _WORDS[20000:20000 + int(new * 0.96)]
    return " ".join(prefix + tail)


def _cuda_self_ms(prof) -> float:
    """Sum CUDA kernel self-time (ms) across the profiled region, across torch versions."""
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
    prompt = build_prompt(cached, new)

    # PRIME: send the cached prefix alone so its blocks are prefix-cached (HIT on the next call).
    if cached > 0:
        llm.generate([" ".join(_WORDS[:int(cached * 0.96)])], sp, use_tqdm=False)

    tok_ms, dev_ms, host_ms, ttft_ms = [], [], [], []
    for _ in range(trials + 1):  # +1 warmup (discarded)
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
    # drop warmup (first)
    med = lambda xs: st.median(xs[1:])
    return dict(new=new, cached=cached, n=trials,
                tokenize_ms=med(tok_ms), device_ms=med(dev_ms),
                host_ms=med(host_ms), ttft_ms=med(ttft_ms))


def regress(rows: list[dict]) -> None:
    """OLS each stage on [1, new, cached]; print the de-fit-relevant slopes."""
    try:
        import numpy as np
    except ImportError:
        print("(numpy unavailable — skipping regression; CSV has the raw stage rows)")
        return
    X = np.array([[1.0, r["new"], r["cached"]] for r in rows])
    print("\n=== stage regressions: stage ~ FLOOR + a·new + b·cached ===")
    for stage in ("ttft_ms", "device_ms", "host_ms", "tokenize_ms"):
        y = np.array([r[stage] for r in rows])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        print(f"  {stage:<12} FLOOR={beta[0]:7.2f} ms | new={beta[1]*1000:7.3f} ms/1k | cached={beta[2]*1000:7.3f} ms/1k")
    print("\nInterpret: device.new ≈ GEMM roofline (~25 ms/1k tp1); host.new = the NEW dispatch residual; "
          "device.cached ≈ GPU paged-attn (~1.5 ms/1k); host.cached = the CACHED host residual. "
          "Run --hash builtin vs sha256 → host.cached delta = block-hash cost.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--news", default="8,128,512,1024,2048", help="fresh-token counts to sweep")
    ap.add_argument("--cacheds", default="0,2000,8000,16000", help="cached-prefix token counts to sweep")
    ap.add_argument("--hash", default="sha256", choices=["sha256", "builtin"], help="prefix-cache hash algo")
    ap.add_argument("--eager", action="store_true", help="enforce_eager (no CUDA graphs) — isolates launch overhead")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="profile_data/results/prefill_stage_split_H100.csv")
    a = ap.parse_args()

    from vllm import LLM
    kw = dict(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              enable_chunked_prefill=True, enforce_eager=a.eager)
    # TODO(version): kwarg name for the prefix-cache hash algo varies; try the common one, fall back.
    try:
        llm = LLM(prefix_caching_hash_algo=a.hash, **kw)
    except TypeError:
        print(f"[warn] LLM() rejected prefix_caching_hash_algo; launching without it "
              f"(set --prefix-caching-hash-algo {a.hash} via your vLLM version's flag).", flush=True)
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
