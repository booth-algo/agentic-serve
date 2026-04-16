#!/usr/bin/env python3
"""
Validate LLMCompass dense model predictions on H100 against real isolated-layer measurements.

Usage:
    python experiment/validate_dense_h100.py --model llama-8b --device cuda:0
    python experiment/validate_dense_h100.py --model all --device cuda:0
"""
import argparse, json, os, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch

MODELS = {
    "llama-8b": {
        "path": "/workspace/models/Llama-3.1-8B-Instruct",
        "d": 4096, "h": 32, "kv": 8, "ffn": 14336, "layers": 32,
    },
    "qwen3.5-27b": {
        "path": "/workspace/models/Qwen3.5-27B",
        "d": 3584, "h": 28, "kv": 4, "ffn": 18944, "layers": 36,
    },
    "qwen2.5-72b": {
        "path": "/workspace/models/Qwen2.5-72B-Instruct",
        "d": 8192, "h": 64, "kv": 8, "ffn": 29568, "layers": 80,
    },
}

BATCH_SIZES = [1, 4, 8]
PREFILL_SEQ_LENS = [64, 128, 256, 512, 1024]
DECODE_KV_LENS = [256, 512, 1024]
WARMUP = 5
REPEAT = 20


def load_h100_predictor():
    import llm_predict.models.software.transformer as tmod
    from llm_predict.predictors.per_category.predictor import CategoryPredictor
    tmod._kernel_predictor = None
    p = CategoryPredictor("llm_predict/profiling/data/H100")
    p.train_all(force_retrain=False)
    tmod._kernel_predictor = p
    return p


def run_predictions(cfg):
    from llm_predict.models.software.transformer import TransformerBlockInitComputationTP
    from llm_predict.models.software.utils import data_type_dict, Tensor
    from llm_predict.dse.dse import template_to_system, read_architecture_template
    import llm_predict.models.software.transformer as tmod

    predictor = load_h100_predictor()
    arch = read_architecture_template(os.path.join(str(REPO_ROOT), "device_configs", "GH100.json"))
    system = template_to_system(arch)

    results = {}
    for bs in BATCH_SIZES:
        for seq in PREFILL_SEQ_LENS:
            try:
                block = TransformerBlockInitComputationTP(
                    d_model=cfg["d"], n_heads=cfg["h"], device_count=1,
                    data_type=data_type_dict["fp16"],
                    intermediate_size=cfg["ffn"], n_kv_heads=cfg["kv"],
                    use_flash_attention=True, use_ml_predictor=True,
                )
                X = Tensor([bs, seq, cfg["d"]], data_type_dict["fp16"])
                _ = block(X)
                tmod._kernel_predictor = predictor
                lat = block.compile_and_simulate(system, "heuristic-GPU") * 1e3
                results[("prefill", bs, seq)] = lat
            except Exception as e:
                print(f"  pred prefill bs={bs} seq={seq}: ERROR {e}")

        for kv in DECODE_KV_LENS:
            try:
                block = TransformerBlockInitComputationTP(
                    d_model=cfg["d"], n_heads=cfg["h"], device_count=1,
                    data_type=data_type_dict["fp16"],
                    intermediate_size=cfg["ffn"], n_kv_heads=cfg["kv"],
                    use_flash_attention=True, use_ml_predictor=True,
                )
                X = Tensor([bs, 1, cfg["d"]], data_type_dict["fp16"])
                _ = block(X)
                tmod._kernel_predictor = predictor
                lat = block.compile_and_simulate(system, "heuristic-GPU") * 1e3
                results[("decode", bs, kv)] = lat
            except Exception as e:
                print(f"  pred decode bs={bs} kv={kv}: ERROR {e}")
    return results


def load_2layers(path, device):
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16, device_map=device,
        ignore_mismatched_sizes=True, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def measure(model, device, bs, seq, past_kv=None):
    layer = model.model.layers[0]
    if past_kv is not None:
        ids = torch.randint(0, 1000, (bs, 1), device=device)
        cap = {}
        def hfn(m, a, kw): cap["a"], cap["kw"] = tuple(x.detach() if hasattr(x,"detach") else x for x in a), {k:(v.detach() if hasattr(v,"detach") else v) for k,v in kw.items()}
        h = layer.register_forward_pre_hook(hfn, with_kwargs=True)
        with torch.no_grad(): model(input_ids=ids, past_key_values=past_kv, use_cache=True)
        h.remove()
        args, kwargs = cap["a"], dict(cap["kw"])
    else:
        ids = torch.randint(0, 1000, (bs, seq), device=device)
        cap = {}
        def hfn(m, a, kw): cap["a"], cap["kw"] = tuple(x.detach() if hasattr(x,"detach") else x for x in a), {k:(v.detach() if hasattr(v,"detach") else v) for k,v in kw.items()}
        h = layer.register_forward_pre_hook(hfn, with_kwargs=True)
        with torch.no_grad(): model(input_ids=ids)
        h.remove()
        args = cap["a"]
        kwargs = {k:v for k,v in cap["kw"].items() if k != "past_key_values"}
        kwargs["use_cache"] = False

    dev = torch.device(device)
    with torch.no_grad():
        for _ in range(WARMUP): layer(*args, **kwargs)
        torch.cuda.synchronize(dev)
    lats = []
    with torch.no_grad():
        for _ in range(REPEAT):
            s,e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(dev); s.record()
            layer(*args, **kwargs)
            e.record(); torch.cuda.synchronize(dev)
            lats.append(s.elapsed_time(e))
    lats.sort()
    return lats[len(lats)//2]


def run_measurements(path, device):
    model = load_2layers(path, device)
    results = {}
    for bs in BATCH_SIZES:
        for seq in PREFILL_SEQ_LENS:
            try:
                lat = measure(model, device, bs, seq)
                results[("prefill", bs, seq)] = lat
                print(f"  prefill bs={bs} seq={seq:4d}: {lat:.3f}ms")
            except Exception as e:
                print(f"  prefill bs={bs} seq={seq:4d}: {e}")
                torch.cuda.empty_cache()

        for kv in DECODE_KV_LENS:
            try:
                pf = torch.randint(0,1000,(bs,kv),device=device)
                with torch.no_grad(): out = model(input_ids=pf, use_cache=True)
                pkv = out.past_key_values
                lat = measure(model, device, bs, 1, past_kv=pkv)
                results[("decode", bs, kv)] = lat
                print(f"  decode  bs={bs} kv={kv:4d}: {lat:.3f}ms")
                del pkv, out; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  decode  bs={bs} kv={kv:4d}: {e}")
                torch.cuda.empty_cache()
    del model; torch.cuda.empty_cache()
    return results


def rate(r):
    d = abs(r-1.0)
    return "GOOD" if d<=0.2 else "OK" if d<=0.3 else "BAD"


def compare(name, cfg, preds, meas):
    print(f"\n{'='*80}")
    print(f"  {name} — d={cfg['d']} h={cfg['h']} kv={cfg['kv']} ffn={cfg['ffn']}")
    print(f"  H100-SXM5-80GB | TP=1 | H100 ML predictor")
    print(f"{'='*80}")
    print(f"{'Phase':<10}{'BS':>4}{'Seq/KV':>8}{'Predicted':>12}{'Measured':>12}{'Ratio':>8}{'Rating':>7}")
    print("-"*80)

    rows = []; g=o=b=s=0
    for phase in ["prefill","decode"]:
        lens = PREFILL_SEQ_LENS if phase=="prefill" else DECODE_KV_LENS
        for bs in BATCH_SIZES:
            for sl in lens:
                p,m = preds.get((phase,bs,sl)), meas.get((phase,bs,sl))
                if p and m and m>0:
                    r = p/m; rt = rate(r)
                    if rt=="GOOD": g+=1
                    elif rt=="OK": o+=1
                    else: b+=1
                    print(f"{phase:<10}{bs:>4}{sl:>8}{p:>11.3f}ms{m:>11.3f}ms{r:>7.3f}x{rt:>7}")
                    rows.append({"phase":phase,"bs":bs,"seq_kv":sl,"pred_ms":round(float(p),3),"meas_ms":round(float(m),3),"ratio":round(float(r),3),"rating":rt})
                else: s+=1
    tot = g+o+b
    print(f"{'='*80}")
    print(f"  GOOD: {g}/{tot}  OK: {o}/{tot}  BAD: {b}/{tot}  (skip: {s})")

    out = Path(f"experiment/dense_validation_{name.replace('-','_')}_h100.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w") as f:
        json.dump({"model":name,"hardware":"H100-SXM5-80GB","arch":cfg,"results":rows,"summary":{"good":g,"ok":o,"bad":b,"total":tot}},f,indent=2)
    print(f"  Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-8b", choices=list(MODELS.keys())+["all"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    targets = list(MODELS.items()) if args.model=="all" else [(args.model, MODELS[args.model])]
    for name, cfg in targets:
        print(f"\n### {name} ###")
        preds = run_predictions(cfg)
        meas = run_measurements(cfg["path"], args.device)
        compare(name, cfg, preds, meas)

if __name__ == "__main__":
    main()
