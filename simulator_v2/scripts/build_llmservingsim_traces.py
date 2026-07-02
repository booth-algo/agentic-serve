# simulator_v2/scripts/build_llmservingsim_traces.py

"""Build LLMServingSim agentic JSONL traces from the SAME synthetic workload the
inference-benchmark fed to vLLM (so both engines get identical input and each
independently decides cache/eviction/latency).

Reuses inference-benchmark's own generator: `make_dataset(...)` -> DistributionalSampler
(seed 42) with the exact env config from scripts/bench_jobs.json. Each turn's chat-templated
prompt is tokenized with the Llama tokenizer -- byte-for-byte what vLLM tokenizes -- so the
engineered block-aligned shared prefix (cross-session cache hits) is preserved.

Output is HARDWARE-AGNOSTIC (make_dataset has no GPU/TP/engine input): one set per
(profile, conc), fed to any hardware config in LLMServingSim.

  input_toks    = len(chat-templated tokenized prompt)  (== GT total_context_tokens, verified)
  output_toks   = sampler-prescribed max_tokens          (agnostic; see handoff note (a))
  input_tok_ids = the real token ids (shared prefix + growing context)
  output_tok_ids= []  (responses re-prefilled next turn, matching GT previous_prompt accounting)
  arrival_time_ns=0, tool_duration_ns=0  (closed loop: semaphore == concurrency == num_sessions)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

IB_ROOT = "/root/agentic-serve/inference-benchmark"
TOKENIZER = "/root/agentic-serve/.llama_tokenizer"
PROFILES = ["chat", "osworld", "swebench", "terminalbench"]
CONCS = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]

# Exact synthetic-workload env from inference-benchmark/scripts/bench_jobs.json
os.environ.setdefault("DISTRIBUTIONAL_SYNTHETIC_STYLE", "code")
os.environ.setdefault("DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN", "3.8")
os.environ.setdefault("DISTRIBUTIONAL_PREFIX_AWARE", "1")
os.environ.setdefault("DISTRIBUTIONAL_SHARED_PREFIX_TOKENS", "1024")

sys.path.insert(0, IB_ROOT)


def _encode_turn(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def build_cell(profile_name, conc, tok, tokenizer_path, make_dataset, get_profile, out_path):
    prof = get_profile(f"{profile_name}-multiturn-synth")
    num_sessions = max(int(getattr(prof, "num_sessions", 1) or 1), conc)
    ds = make_dataset(prof, max_context_tokens=32768, random_seed=42,
                      context_safety_margin_tokens=256, num_sessions=num_sessions,
                      tokenizer_name=tokenizer_path, source_session_ids=None)
    sessions = ds.sessions
    n_turns, n_ids = 0, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for s in sessions:
            subs = []
            for t in s.turns:
                ids = _encode_turn(tok, t.messages)
                subs.append({"input_toks": len(ids), "output_toks": int(t.max_tokens),
                             "tool_duration_ns": 0, "input_tok_ids": ids, "output_tok_ids": []})
            if not subs:
                continue
            n_turns += len(subs)
            n_ids += sum(x["input_toks"] for x in subs)
            fh.write(json.dumps({"session_id": str(s.session_id),
                                 "arrival_time_ns": 0, "sub_requests": subs}) + "\n")
    return {"sessions": len(sessions), "turns": n_turns, "id_ints": n_ids,
            "size_mb": round(Path(out_path).stat().st_size / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="profile_data/results/llmservingsim_traces")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="path to the model tokenizer dir")
    ap.add_argument("--label", default="llama-3.1", help="tokenizer dir name (output goes to <out>/<label>/)")
    ap.add_argument("--only", default="", help="substring filter on '<profile>_conc<N>'")
    ap.add_argument("--concs", default="", help="comma list override")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from src.workloads.dataset import make_dataset
    from src.workloads.profiles import get_profile
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    concs = [int(x) for x in a.concs.split(",")] if a.concs else CONCS
    out_root = Path(a.out) / a.label
    manifest = []
    for pf in PROFILES:
        for c in concs:
            cell = f"{pf}_conc{c}"
            if a.only and a.only not in cell:
                continue
            out = out_root / f"{cell}.jsonl"
            info = build_cell(pf, c, tok, a.tokenizer, make_dataset, get_profile, out)
            row = {"profile": pf, "conc": c, **info, "path": str(out)}
            manifest.append(row)
            print(f"{cell:26} {info['sessions']:>4} sess {info['turns']:>6} turns {info['size_mb']:>8} MB", flush=True)
    mpath = out_root / "MANIFEST.csv"
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["profile", "conc", "sessions", "turns", "id_ints", "size_mb", "path"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\nwrote manifest: {mpath}  ({len(manifest)} cells)")


if __name__ == "__main__":
    main()
