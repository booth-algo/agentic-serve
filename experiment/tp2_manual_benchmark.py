"""Manual TP=2 benchmark using accelerate device_map=auto across 2 GPUs."""
import os, sys, json, time, torch, statistics
sys.path.insert(0, "/data/kevinlau/python-packages")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "/data/models/Llama-3.1-8B-Instruct"
print("Loading %s across 2 GPUs (pipeline parallel)..." % os.path.basename(MODEL))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL)
print("Loaded. Layers: %d" % config.num_hidden_layers)

prompt = "Explain tensor parallelism in LLM inference."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()

results = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000

    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    decode_times = []
    for _ in range(31):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t1) * 1000
        decode_times.append(dt)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    tpot = statistics.median(decode_times)
    results.append({"ttft_ms": ttft, "tpot_ms": tpot})
    print("  Run %d: TTFT=%.1fms TPOT=%.1fms" % (i+1, ttft, tpot))

med_ttft = statistics.median([r["ttft_ms"] for r in results])
med_tpot = statistics.median([r["tpot_ms"] for r in results])
print("\n=== 2-GPU Pipeline Parallel Results ===")
print("TTFT: %.1f ms" % med_ttft)
print("TPOT: %.1f ms" % med_tpot)

# Now TP=1 comparison
del model
torch.cuda.empty_cache()
print("\nLoading on single GPU for comparison...")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model1 = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True)
model1.eval()

input_ids1 = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
for _ in range(3):
    with torch.no_grad():
        _ = model1(input_ids=input_ids1, use_cache=True)
    torch.cuda.synchronize()

results1 = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model1(input_ids=input_ids1, use_cache=True)
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000

    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    decode_times = []
    for _ in range(31):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model1(input_ids=next_tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t1) * 1000
        decode_times.append(dt)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    tpot = statistics.median(decode_times)
    results1.append({"ttft_ms": ttft, "tpot_ms": tpot})

med_ttft1 = statistics.median([r["ttft_ms"] for r in results1])
med_tpot1 = statistics.median([r["tpot_ms"] for r in results1])
print("=== 1-GPU Results ===")
print("TTFT: %.1f ms" % med_ttft1)
print("TPOT: %.1f ms" % med_tpot1)
print("\nSpeedup PP2/TP1: TTFT=%.2fx TPOT=%.2fx" % (med_ttft1/med_ttft, med_tpot1/med_tpot))

out_data = {
    "model": "Llama-3.1-8B-Instruct",
    "tp1": {"ttft_ms": med_ttft1, "tpot_ms": med_tpot1},
    "pp2": {"ttft_ms": med_ttft, "tpot_ms": med_tpot},
    "note": "pp2 = pipeline parallel (device_map=auto), not true tensor parallel"
}
with open("/home/kevinlau/llmserve/experiment/tp2_benchmark_results.json", "w") as f:
    json.dump(out_data, f, indent=2)
print("\nSaved results")
