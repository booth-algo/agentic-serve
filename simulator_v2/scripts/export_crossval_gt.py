# simulator_v2/scripts/export_crossval_gt.py

"""Export the LLMServingSim cross-validation Google-Sheet rows (docs/
llmservingsim_crossval.md §6): one row per (config, profile, conc, metric) with
GT (per-turn median -> mean over turns, same reduction as score_llmservingsim),
our sim v2 prediction + cell MAPE (h100 8B tp1 only), and KV pressure
(max-turn herd demand / pool; analytic pool except the measured h100-8B-tp1).
George's columns (LLMServingSim2.0, MAPE_llmsim) are left blank.

  python3 simulator_v2/scripts/export_crossval_gt.py \
      --out profile_data/results/llmservingsim_crossval_gt.csv
"""

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.kv_pool import available_kv_blocks
from simulator_v2.configs.config_loader import load_gpu_config, load_model_config
from simulator_v2.core.mode import Mode, mode

BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
V2_JSON = Path("inference-benchmark/dashboard/public/simulator-v2sim-predictions.json")
CFG = Path("simulator_v2/configs")

# docs/llmservingsim_crossval.md §1 matrix (vllm GT only; ragged cells welcome)
MODELS = {  # bench name -> (model yaml, gpu yamls it appears on)
    "Llama-3.1-8B": "llama3.1-8b.yaml",
    "Llama-3.1-70B": "llama-3.1-70b.yaml",
    "Mixtral-8x7B": "mixtral-8x7b.yaml",
    "gpt-oss-20b": "gpt-oss-20b.yaml",
    "gpt-oss-120b": "gpt-oss-120b.yaml",
}
GPUS = {"h100": "h100.yaml", "a100": "a100.yaml", "3090": "rtx3090.yaml"}
PROFILES = ["chat", "osworld", "swebench", "terminalbench"]
CONCS = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320]
MEASURED_POOL = {("h100", "Llama-3.1-8B", 1): 27250}  # blocks; everything else analytic

FIELDS = ["model", "gpu", "tp", "backend", "profile", "conc", "metric", "GT",
          "LLMServingSim2.0", "Ours (roofline)", "Ours (sim v2)", "MAPE_llmsim",
          "MAPE_ours", "pressure"]


def pool_tokens(gpu_key, model_name, tp):
    blocks = MEASURED_POOL.get((gpu_key, model_name, tp))
    model = load_model_config(CFG / "model_configs" / MODELS[model_name])
    if blocks is None:
        gpu = load_gpu_config(CFG / "gpu_configs" / GPUS[gpu_key])
        blocks = available_kv_blocks(
            total_memory_bytes=gpu.total_bytes, gpu_mem_util=gpu.gpu_mem_util,
            weight_bytes=model.n_params * model.bytes_per_param, tp=tp,
            kv_bytes_per_token=model.kv_bytes_per_token, kv_heads=model.kv_heads,
            block_size=model.cache_block_size,
        )
    return max(1, blocks) * model.cache_block_size


def cell_stats(path):
    """GT per metric (per-turn median -> mean over turns) + max-turn demand tokens."""
    d = json.load(path.open())
    by_turn = defaultdict(list)
    for r in d.get("per_request", []):
        if r.get("success"):
            by_turn[r["turn_index"]].append(r)
    if not by_turn:
        return None
    gt, demand = {}, 0.0
    for key, field in [("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"), ("e2el_ms", "e2el_ms")]:
        vals = []
        for t in sorted(by_turn):
            xs = [r[field] for r in by_turn[t] if r.get(field) is not None]
            if xs:
                vals.append(st.median(xs))
        gt[key] = st.mean(vals) if vals else None
    for t, rs in by_turn.items():
        demand = max(demand, sum(r["total_context_tokens"] + r["output_tokens"] for r in rs))
    return gt, demand


def v2_cells():
    """(profile, conc) -> {metric: (pred_scalar, cell_mape)} from the dashboard JSON."""
    out = {}
    if not V2_JSON.exists():
        return out
    for row in json.load(V2_JSON.open())["H100"]:
        tps = row["multiturn_turn_predictions"]
        prof = row["profile"].split("-")[0]
        entry = {}
        for m in ["ttft", "tpot", "e2el"]:
            preds = [tp[f"{m}_pred"] for tp in tps]
            errs = [tp[f"{m}_err"] for tp in tps if tp.get(f"{m}_err") is not None]
            entry[f"{m}_ms"] = (st.mean(preds), st.mean(errs) if errs else None)
        out[(prof, row["concurrency"])] = entry
    return out


@mode(Mode.BACKTEST)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profile_data/results/llmservingsim_crossval_gt.csv")
    a = ap.parse_args()

    v2 = v2_cells()
    rows, skipped = [], []
    for gpu_key in GPUS:
        for model_name in MODELS:
            for tp in (1, 2, 4):
                cfg = f"{gpu_key}_{model_name}_tp{tp}_vllm"
                bench = BASE / cfg
                if not bench.is_dir():
                    continue
                pool = pool_tokens(gpu_key, model_name, tp)
                for prof in PROFILES:
                    for conc in CONCS:
                        path = bench / f"{prof}-multiturn-synth_conc{conc}.json"
                        if not path.exists():
                            continue
                        try:
                            stats = cell_stats(path)
                        except (json.JSONDecodeError, KeyError) as e:
                            skipped.append(f"{cfg}/{prof}/c{conc}: {e}")
                            continue
                        if stats is None:
                            skipped.append(f"{cfg}/{prof}/c{conc}: no successful requests")
                            continue
                        gt, demand = stats
                        pressure = demand / pool
                        is_v2 = (cfg == "h100_Llama-3.1-8B_tp1_vllm")
                        for metric in ["ttft_ms", "tpot_ms", "e2el_ms"]:
                            pred, mape = ("", "")
                            if is_v2 and (prof, conc) in v2:
                                p, m = v2[(prof, conc)][metric]
                                pred = round(p, 2)
                                mape = round(m, 2) if m is not None else ""
                            rows.append({
                                "model": model_name, "gpu": gpu_key, "tp": tp,
                                "backend": "vllm", "profile": prof, "conc": conc,
                                "metric": metric,
                                "GT": round(gt[metric], 2) if gt[metric] is not None else "",
                                "LLMServingSim2.0": "", "Ours (roofline)": "",
                                "Ours (sim v2)": pred, "MAPE_llmsim": "",
                                "MAPE_ours": mape, "pressure": round(pressure, 3),
                            })
                n_cells = sum(1 for r in rows
                              if (r["gpu"], r["model"], r["tp"]) == (gpu_key, model_name, tp)) // 3
                print(f"{cfg}: {n_cells} cells", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    if skipped:
        print(f"skipped {len(skipped)} cells:")
        for s in skipped[:20]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
