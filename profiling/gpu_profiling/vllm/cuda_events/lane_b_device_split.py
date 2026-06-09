#!/usr/bin/env python3
"""LANE B — device-vs-dispatch split of the prefill forward window (the 22.7 ms/1k `prefill_span`).

Measures, vs Lane A's `prefill_span` (= scheduled->first-token engine wall, ~22.7 ms/1k-new):
  * device_ms     : pure GPU kernel time, measured with torch.cuda.Event (reliable UNDER CUDA graphs --
                    unlike torch.profiler self-time, which over-counts in eager / is hidden under graphs;
                    that is why the original offline split in prefill_stage_split.py was unusable). VALID.
  * dispatch_ms / forward_wall_ms : INVALID -- DO NOT USE. CUDA is async: execute_model returns at
                    kernel-LAUNCH (host_wall ~4-6 ms), before the GPU finishes (device ~24 ms), so
                    `host_wall - device` goes negative and clamps to 0. The host wall here measures enqueue,
                    not execution. The device-vs-overhead split comes from `device_ms` vs Lane A's
                    `prefill_span`, NOT from these columns. (Kept in the CSV only for transparency.)

RESULT (2026-06-05, h100 GPU 7, 5x4 sweep -> lane_b_device_H100.csv): device.new = 23.85 ms/1k
(util ~0.68, roofline-consistent) ~= Lane A prefill_span.new 22.7 -> the engine prefill window is
~100% GPU kernel; there is NO GPU-side in-engine dispatch residual. device.cached = 0.59 ms/1k
(paged-attn KV-gather). Corroborates the de-fit: the NEW "dispatch residual" is FRONTEND host
serving-stack (Lane A frontend.new 5.7), entirely outside the GPU forward.

Mechanism (no nsys, no --worker-extension-cls -- both the worker-extension arg AND a clean nsys NVTX join
were unavailable on vLLM 0.19.0 here): run the engine IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0, tp1 only)
so GPUModelRunner lives in this process, and monkeypatch its execute_model BEFORE LLM() so every prefill
step is bracketed by CUDA events (+ an NVTX range, so the SAME script can ALSO be wrapped in
`nsys profile -t cuda,nvtx --cuda-graph-trace=node` to further break device_ms into GEMM/attn/elementwise).

At max_tokens=1 a request's execute_model calls ARE its prefill steps (the first token is emitted at the
end of prefill; no separate decode), so summing the ring over one generate() = that request's prefill.

RUN (h100, GPU 7, clean cwd):
  CUDA_VISIBLE_DEVICES=7 VLLM_ENABLE_V1_MULTIPROCESSING=0 TMPDIR=/data48/kevinlau/tmp \
  ~/miniconda3/envs/vllm/bin/python lane_b_device_split.py \
    --model /data48/kevinlau/models/Llama-3.1-8B-Instruct \
    --news 8,128,512,1024,2048 --cacheds 0,2000,8000,16000 --trials 5 \
    --out /home/kevinlau/serving_split_run/lane_b_device_H100.csv
  # OPTIONAL per-kernel layer: prefix the python with
  #   /usr/local/cuda/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node -o lane_b_nsys -f true --
  # then export lane_b_nsys.nsys-rep -> .sqlite and SUM kernels inside the 'PF|...' NVTX ranges.
  [VERIFY ON H100] GPUModelRunner.execute_model is the v1 0.19.0 prefill entrypoint and the monkeypatch
  (patched on the CLASS before LLM(), mp=0 in-process) is actually invoked -- the script asserts the ring
  is non-empty and aborts loudly if not.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:   # don't let a stray flash_attn.py shadow the pkg
    sys.path.remove(_p)

import torch  # noqa: E402

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most").split()
_RNG = random.Random(0)
_WORDS = [_RNG.choice(_VOCAB) for _ in range(40000)]
_TAIL = random.Random(777)

# --- the per-step ring the monkeypatch fills; cleared per measured generate() ---
_RING: list[tuple[torch.cuda.Event, torch.cuda.Event, float]] = []
_CELL = {"label": "none"}


def _install_hook() -> None:
    """Monkeypatch GPUModelRunner.execute_model BEFORE LLM() (mp=0 -> in-process -> reachable)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # [VERIFY] v1 0.19.0 path
    _orig = GPUModelRunner.execute_model

    def timed(self, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(f"PF|{_CELL['label']}")
        t0 = time.perf_counter()
        s.record()
        out = _orig(self, *a, **k)
        e.record()
        host = (time.perf_counter() - t0) * 1e3
        torch.cuda.nvtx.range_pop()
        _RING.append((s, e, host))
        return out

    GPUModelRunner.execute_model = timed


def cached_prefix(cached: int) -> str:
    return " ".join(_WORDS[: int(cached * 0.96)])


def fresh_tail(new: int) -> str:
    return " ".join(_TAIL.choice(_VOCAB) for _ in range(int(new * 0.96)))


def measure(llm, cached: int, new: int, trials: int) -> dict:
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    prefix = cached_prefix(cached)
    if cached > 0:
        llm.generate([prefix], sp, use_tqdm=False)  # prime the cached prefix -> HIT every trial
    dev_ms, host_ms, calls = [], [], []
    for t in range(trials + 1):  # +1 warmup discarded
        prompt = (prefix + " " + fresh_tail(new)) if cached > 0 else fresh_tail(new)
        _CELL["label"] = f"n{new}_c{cached}"
        _RING.clear()
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()                       # one sync, then read the events off the hot path
        if not _RING:
            sys.exit("FATAL: execute_model hook never fired -- the monkeypatch did not take (check mp=0 / "
                     "the v1 GPUModelRunner path). Aborting rather than emitting bogus zeros.")
        dev = sum(s.elapsed_time(e) for s, e, _ in _RING)   # GPU kernel ms (events)
        wall = sum(h for _, _, h in _RING)                  # host wall of the forward calls
        dev_ms.append(dev); host_ms.append(wall); calls.append(len(_RING))
    med = lambda xs: st.median(xs[1:])  # drop warmup
    d, w = med(dev_ms), med(host_ms)
    return dict(new=new, cached=cached, n=trials, n_calls=med(calls),
                device_ms=round(d, 4), forward_wall_ms=round(w, 4),
                dispatch_ms=round(max(0.0, w - d), 4))


def regress(rows: list[dict]) -> None:
    try:
        import numpy as np
    except ImportError:
        print("(numpy unavailable; CSV has the raw rows)"); return
    X = np.array([[1.0, r["new"], r["cached"]] for r in rows])
    print("\n=== LANE B device-vs-dispatch: stage ~ FLOOR + new*ms/1k + cached*ms/1k ===")
    for stage in ("device_ms", "dispatch_ms", "forward_wall_ms"):
        y = np.array([r[stage] for r in rows])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        print(f"  {stage:<16} FLOOR={b[0]:7.2f} | new={b[1]*1000:7.3f} ms/1k | cached={b[2]*1000:7.3f} ms/1k")
    print("Interpret: device.new (CUDA events) = GPU prefill GEMM kernel, the ONLY valid number here "
          "(~roofline 16-25 ms/1k at the c1 util). dispatch_ms / forward_wall_ms are INVALID (CUDA async, "
          "see header) -- ignore. The split is device.new vs Lane A prefill_span.new (22.7): if they match, "
          "the engine prefill window is pure GPU kernel (no in-engine dispatch residual).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--news", default="8,128,512,1024,2048")
    ap.add_argument("--cacheds", default="0,2000,8000,16000")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="lane_b_device_H100.csv")
    a = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print("[warn] set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine is in-process and the hook is "
              "reachable (and use tp1 only).", flush=True)

    _install_hook()  # BEFORE LLM()
    from vllm import LLM
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              enable_chunked_prefill=True, tensor_parallel_size=1)

    news = [int(x) for x in a.news.split(",")]
    cacheds = [int(x) for x in a.cacheds.split(",")]
    rows = []
    for cached in cacheds:
        for new in news:
            r = measure(llm, cached, new, a.trials)
            rows.append(r)
            print(f"  new={new:>5} cached={cached:>6}  device={r['device_ms']:7.2f}  "
                  f"forward_wall={r['forward_wall_ms']:7.2f}  dispatch={r['dispatch_ms']:6.2f}  "
                  f"calls={r['n_calls']}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new", "cached", "n", "n_calls",
                                          "device_ms", "forward_wall_ms", "dispatch_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")
    regress(rows)


if __name__ == "__main__":
    main()
