#!/usr/bin/env python3
"""Per-STEP prefill GEMM utilization vs chunk size — the Phase-B measurement for audit-v2 R1/S6
(`ttft_pricing_defit_plan.md` Item 2).

What it pins down: ``util(m)`` — the fraction of the util-1 compute roofline a prefill engine
step achieves when it processes ``m`` batched tokens. The production pricing
(`ttft_queue_sim._prefill_gemm_per_tok_loaded`) currently RAMPS util from util_flops (0.65) to
``PREFILL_GEMM_UTIL_SAT = 1.0`` as the step fills the chunked-prefill budget — a
validation-anchored cap (the "15.5 ms/1k GT cohort" anchor was debunked: cohort wall over a
shared-prefix double-counted token denominator; correctly-accounted GT gives a ~0.62 gross
floor, and three offline artifacts cohere at ~0.68–0.77 with NO ramp).

Mechanism (the proven lane-B technique — see lane_b_device_split.py): run the engine IN-PROCESS
(VLLM_ENABLE_V1_MULTIPROCESSING=0, tp1), monkeypatch ``GPUModelRunner.execute_model`` with CUDA
events BEFORE ``LLM()``. Per-step chunk size is controlled by ``max_num_batched_tokens = B``:
a single long prompt of L tokens runs ceil(L/B) prefill steps, all but the last processing
exactly B tokens — so each full step is one (m=B, device_ms) sample. One LLM per budget
(sequential model loads). ``enable_prefix_caching=False`` so every step does its full compute
(we measure the compute rate, not cache behavior); ``max_tokens=1`` so execute_model calls ==
prefill steps (lane-B precedent).

util accounting (BOTH conventions, the Phase-A confusion made explicit):
  * util_sim    = roofline_ms_sim(m) / device_ms, roofline_sim = 2·N_PARAMS_SIM·m/PEAK — the
    convention `_prefill_gemm_per_tok` prices with (N_PARAMS_SIM = 8.03e9). THIS is the column
    Phase C wires.
  * util_gemm   = same with N_GEMM = 6.979e9 (executed fused-linear FLOPs only) — comparable to
    the offline microbench's 0.655–0.672.

ONE BUDGET PER PROCESS (vLLM does not reliably release GPU memory between LLM() instances in
one process, and re-installing the hook would double-wrap execute_model). The launcher loops:

RUN (h100 — env per profiling/docs/h100_setup.md: GPU 6, tmp+cache on the data48 mount, run dir
on data48; output filename must be FRESH — the script appends):
  for B in 512 1310 2048 4096 8192; do
    CUDA_VISIBLE_DEVICES=6 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    TMPDIR=/data48/kevinlau/tmp XDG_CACHE_HOME=/data48/kevinlau/tmp/.cache \
    ~/miniconda3/envs/vllm/bin/python prefill_util_sweep.py \
      --model /data48/kevinlau/models/Llama-3.1-8B-Instruct \
      --budget $B --prompt-tokens 28000 --trials 3 \
      --out /data48/kevinlau/serving_split_run/prefill_util_sweep_H100.csv  # appends; header once
  done
  [VERIFY ON H100] per budget B: ring length per generate == ceil(prompt_kv_tokens/B) — the
  script asserts it and aborts loudly on mismatch (chunking semantics changed -> bad samples).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
for _p in [e for e in sys.path if e == _SCRIPT_DIR]:   # don't let a stray module shadow the pkg
    sys.path.remove(_p)

import torch  # noqa: E402

N_PARAMS_SIM = 8.03e9    # the sim's pricing convention (total params)
N_GEMM = 6.979e9         # executed fused-linear FLOPs convention (microbench comparison)
PEAK_FLOPS = 989e12      # H100 bf16 dense

_VOCAB = ("the of and to in a is that for it as was with on be at by this had not are but from or "
          "have an they which one you were her all she there would their we him been has when who "
          "will more no if out so up said what its about into than them can only other new some "
          "could time these two may then do first any my now such like our over man me even most").split()

_RING: list[tuple[torch.cuda.Event, torch.cuda.Event, int]] = []


def _install_hook() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _orig = GPUModelRunner.execute_model

    def timed(self, *a, **k):
        # a[0] is the v1 SchedulerOutput: read the step's ACTUAL batched token count from the
        # engine rather than assuming it (the first generate can interleave a lazy compile /
        # capture step, so call counts are not exactly ceil(prompt/budget)).
        sched = a[0] if a else k.get("scheduler_output")
        m = int(getattr(sched, "total_num_scheduled_tokens", -1) or -1)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = _orig(self, *a, **k)
        e.record()
        _RING.append((s, e, m))
        return out

    GPUModelRunner.execute_model = timed


def fresh_prompt(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(words))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data48/kevinlau/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--budget", type=int, required=True, help="max_num_batched_tokens = per-step m")
    ap.add_argument("--prompt-tokens", type=int, default=28000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--out", default="prefill_util_sweep_H100.csv")
    a = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print("[warn] set VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process hook) and tp1.", flush=True)

    budget = a.budget
    _install_hook()  # BEFORE LLM(); one budget per process -> single wrap, clean memory
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_num_batched_tokens=budget,
              tensor_parallel_size=1)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    tok = llm.get_tokenizer()
    rows: list[dict] = []
    per_step: list[float] = []
    for t in range(a.trials + 1):                     # +1 warmup, discarded
        rng = random.Random(1000 * budget + t)
        prompt = fresh_prompt(rng, int(a.prompt_tokens * 0.96))
        n_kv = len(tok(prompt).input_ids)             # actual prompt tokens the engine chunks
        expect = math.ceil(n_kv / budget)
        _RING.clear()
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
        full = [(s, e) for (s, e, m) in _RING if m == budget]   # exact full-budget steps only
        if abs(len(_RING) - expect) > 2 or not full:
            sys.exit(f"FATAL: budget={budget} trial={t}: {len(_RING)} execute_model calls "
                     f"(expected ~{expect}), {len(full)} full-budget steps — chunking semantics "
                     f"changed; aborting rather than emitting mislabeled samples.")
        if t == 0:
            continue                                  # warmup trial (lazy compile/capture steps)
        for i, (s, e) in enumerate(full):
            per_step.append(s.elapsed_time(e))
            rows.append(dict(budget=budget, trial=t, step=i,
                             tokens=budget, device_ms=round(s.elapsed_time(e), 4)))
    med = st.median(per_step)
    roof_sim = 2.0 * N_PARAMS_SIM * budget / PEAK_FLOPS * 1e3
    roof_gemm = 2.0 * N_GEMM * budget / PEAK_FLOPS * 1e3
    print(f"budget={budget:>5}  steps={len(per_step):>3}  median {med:8.3f} ms  "
          f"util_sim={roof_sim / med:.4f}  util_gemm={roof_gemm / med:.4f}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["budget", "trial", "step", "tokens", "device_ms"])
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} full-step samples -> {out}")
    print("Interpret: util_sim is the column the sim's pricing convention wires (n_params=8.03e9);"
          " util_gemm compares to the offline microbench (0.655-0.672). Expect a flat ~0.62-0.77"
          " plateau if Phase A holds; a genuine ramp to 1.0 would rehabilitate UTIL_SAT.")


if __name__ == "__main__":
    main()
