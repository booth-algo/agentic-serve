"""End-to-end TTFT/TPOT benchmark using transformers generate (no serving framework)."""
import torch, time, json, os, sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/data/models/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"
NUM_LAYERS = None  # full model
OUTPUT_TOKENS = 32
WARMUP = 2
REPEAT = 5

print(f"Loading {os.path.basename(MODEL_PATH)} (full model)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

prompt = "Explain the concept of tensor parallelism in large language model inference."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
input_len = input_ids.shape[1]
print(f"Input tokens: {input_len}, Output tokens: {OUTPUT_TOKENS}")

results = []
for i in range(WARMUP + REPEAT):
    torch.cuda.synchronize()
    
    # Measure TTFT: time to first token (prefill)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    
    # Measure generation (decode tokens one by one)
    past = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    
    decode_times = []
    for j in range(OUTPUT_TOKENS - 1):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        decode_times.append(dt)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    
    tpot = sum(decode_times) / len(decode_times) if decode_times else 0
    e2e = ttft + sum(decode_times)
    
    if i >= WARMUP:
        results.append({"ttft_ms": ttft*1000, "tpot_ms": tpot*1000, "e2e_ms": e2e*1000})
        print(f"  Run {i-WARMUP+1}: TTFT={ttft*1000:.1f}ms TPOT={tpot*1000:.1f}ms E2E={e2e*1000:.1f}ms")

# Median results
import statistics
ttfts = [r["ttft_ms"] for r in results]
tpots = [r["tpot_ms"] for r in results]
e2es = [r["e2e_ms"] for r in results]

config = AutoConfig.from_pretrained(MODEL_PATH)
n_layers = config.num_hidden_layers

print(f"\n=== {os.path.basename(MODEL_PATH)} E2E Results (TP=1, {n_layers} layers) ===")
print(f"  TTFT:  {statistics.median(ttfts):.1f} ms (prefill {input_len} tokens)")
print(f"  TPOT:  {statistics.median(tpots):.1f} ms (decode, median per token)")
print(f"  E2E:   {statistics.median(e2es):.1f} ms ({input_len} in, {OUTPUT_TOKENS} out)")
print(f"  Per-layer prefill: {statistics.median(ttfts)/n_layers:.3f} ms")
print(f"  Per-layer decode:  {statistics.median(tpots)/n_layers:.3f} ms")

# Compare with XGBoost per-layer prediction
sys.path.insert(0, "/home/kevinlau/llmserve"); os.chdir("/home/kevinlau/llmserve")
from llm_predict.models.software.utils import data_type_dict, Tensor
from llm_predict.models.software.transformer import TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP
from llm_predict.dse.dse import template_to_system, read_architecture_template
arch = read_architecture_template("device_configs/GA100.json"); system = template_to_system(arch)

d = config.hidden_size; h = config.num_attention_heads
kv = getattr(config, "num_key_value_heads", h); ffn = config.intermediate_size

pf_block = TransformerBlockInitComputationTP(d_model=d, n_heads=h, device_count=1,
    data_type=data_type_dict["fp16"], intermediate_size=ffn, n_kv_heads=kv,
    use_flash_attention=True, activation_type="silu", use_ml_predictor=True, use_cuda_graph=False)
X = Tensor([1, input_len, d], data_type_dict["fp16"]); _ = pf_block(X)
pf_pred = pf_block.compile_and_simulate(system, "heuristic-GPU") * 1e3

dc_block = TransformerBlockAutoRegressionTP(d_model=d, n_heads=h, device_count=1,
    data_type=data_type_dict["fp16"], intermediate_size=ffn, n_kv_heads=kv,
    use_flash_attention=True, activation_type="silu", use_ml_predictor=True, use_cuda_graph=False)
x = Tensor([1, 1, d], data_type_dict["fp16"]); _ = dc_block(x, input_len)
dc_pred = dc_block.compile_and_simulate(system, "heuristic-GPU") * 1e3

ttft_pred = pf_pred * n_layers
tpot_pred = dc_pred * n_layers
e2e_pred = ttft_pred + tpot_pred * (OUTPUT_TOKENS - 1)

med_ttft = statistics.median(ttfts)
med_tpot = statistics.median(tpots)
med_e2e = statistics.median(e2es)

print(f"\n=== LLMCompass vs Measured ===")
print(f"{'Metric':>8} {'Pred(ms)':>10} {'Meas(ms)':>10} {'Ratio':>8} {'Rating':>6}")
print("-" * 48)
for label, pred, meas in [("TTFT", ttft_pred, med_ttft), ("TPOT", tpot_pred, med_tpot), ("E2E", e2e_pred, med_e2e)]:
    ratio = pred / meas
    rating = "GOOD" if abs(ratio-1) <= 0.2 else ("OK" if abs(ratio-1) <= 0.3 else "BAD")
    print(f"{label:>8} {pred:>10.1f} {meas:>10.1f} {ratio:>7.2fx} {rating:>6}")

# Save results
with open(f"/tmp/e2e_{os.path.basename(MODEL_PATH)}.json", "w") as f:
    json.dump({"model": os.path.basename(MODEL_PATH), "n_layers": n_layers,
        "input_len": input_len, "output_tokens": OUTPUT_TOKENS,
        "measured": {"ttft_ms": med_ttft, "tpot_ms": med_tpot, "e2e_ms": med_e2e},
        "predicted": {"ttft_ms": ttft_pred, "tpot_ms": tpot_pred, "e2e_ms": e2e_pred}}, f, indent=2)
print("\nDONE")
