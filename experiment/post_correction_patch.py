"""Patch transformer.py: add post-prediction correction for MoE decode and small-seq."""
filepath = "/home/kevinlau/llmserve/llmcompass/software_model/transformer.py"
with open(filepath, "r") as f:
    content = f.read()

correction_fn = '''
def _post_prediction_correction(pred_us, n_tokens, d_model, num_experts, batch_size, seq_len):
    """Apply targeted post-prediction corrections for known weak regimes.

    1. MoE decode: expert weight loading causes ~2.5x more HBM traffic than
       analytical model predicts (cache thrashing from separate weight tensors).
    2. Small-seq overhead: kernel launch + MoE routing has a minimum floor.
    """
    corrected = pred_us

    # MoE decode correction: when tok is small and model has many experts,
    # FFN weight loading is much slower than bandwidth model predicts
    is_decode = (seq_len <= 1 and n_tokens <= 8)
    if is_decode and num_experts >= 8:
        moe_decode_scale = 1.0 + 0.15 * num_experts
        corrected = pred_us * min(moe_decode_scale, 4.0)

    # Small-seq minimum floor: kernel launch overhead
    if n_tokens * seq_len <= 64:
        min_floor_us = 150.0 + d_model * 0.1
        if num_experts > 0:
            min_floor_us += 300.0 * min(num_experts, 8)
        corrected = max(corrected, min_floor_us)

    return corrected

'''

marker = "\n_allreduce_calibration_cache = None"
if "_post_prediction_correction" not in content:
    content = content.replace(marker, correction_fn + marker)
    print("Added _post_prediction_correction function")
else:
    print("Already present, skipping")

# Wire it into prediction blocks
old_ret = "            self.latency = lat_sum / 1e6  # us -> seconds\n            return self.latency"
new_ret = "            lat_sum = _post_prediction_correction(lat_sum, n_tokens, d, getattr(self, 'num_experts', 0), b, s)\n            self.latency = lat_sum / 1e6  # us -> seconds\n            return self.latency"

count = content.count(old_ret)
content = content.replace(old_ret, new_ret)
print("Wired into %d prediction blocks" % count)

with open(filepath, "w") as f:
    f.write(content)
print("Done")
