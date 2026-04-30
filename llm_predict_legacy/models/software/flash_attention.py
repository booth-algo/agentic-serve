"""Analytical FlashAttention2 model.

FlashAttention fuses Q×K^T + softmax + A×V into a single kernel that
tiles over sequence length, keeping intermediate attention scores in SRAM.
This avoids materializing the O(s^2) attention matrix to HBM.

For prefill (all tokens attend to all): compute-bound for long sequences.
For decode (1 new token attends to full KV cache): memory-bound.
"""
