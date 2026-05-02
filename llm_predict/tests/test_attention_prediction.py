from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm_predict.composer import Composer
from llm_predict.kernels.flash_attn import _flash_attn_bytes, _flash_attn_flops
from llm_predict.sweep.generate_shapes import generate_attention_shapes


class RecordingGemm:
    def __init__(self):
        self.calls = []

    def predict(self, M, N, K):
        self.calls.append((M, N, K))
        return 1.0


class RecordingFlash:
    def __init__(self):
        self.calls = []

    def predict(self, seq_len, n_heads, head_dim, **kwargs):
        self.calls.append({
            "q_len": seq_len,
            "n_heads": n_heads,
            "head_dim": head_dim,
            **kwargs,
        })
        return 10.0


class ZeroElementwise:
    def predict_from_shape(self, *args, **kwargs):
        return 0.0


def fake_composer() -> Composer:
    composer = object.__new__(Composer)
    composer.gpu = "test"
    composer.gemm = RecordingGemm()
    composer.flash = RecordingFlash()
    composer.rmsnorm = ZeroElementwise()
    composer.silu_mul = ZeroElementwise()
    composer.rotary = ZeroElementwise()
    composer.residual = ZeroElementwise()
    return composer


def fake_cfg():
    return SimpleNamespace(
        hidden_dim=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        top_k=1,
    )


class AttentionPredictionTests(unittest.TestCase):
    def test_flash_flops_include_qk_and_av(self):
        q_len = 128
        kv_len = 4096
        n_heads = 32
        head_dim = 128
        batch = 2

        self.assertEqual(
            _flash_attn_flops(q_len, kv_len, n_heads, head_dim, batch),
            4.0 * batch * q_len * kv_len * n_heads * head_dim,
        )

    def test_flash_bytes_use_kv_heads_for_cache_traffic(self):
        bytes_moved = _flash_attn_bytes(
            q_len=1,
            kv_len=2048,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            batch=16,
            dtype_bytes=2,
        )

        q_bytes = 16 * 1 * 32 * 128 * 2
        kv_bytes = 2 * 16 * 2048 * 8 * 128 * 2
        o_bytes = 16 * 1 * 32 * 128 * 2
        self.assertEqual(bytes_moved, q_bytes + kv_bytes + o_bytes)

    def test_prefill_attention_shape_is_full_context(self):
        composer = fake_composer()
        composer.predict_layer(fake_cfg(), seq_len=512, bs=1, phase="prefill")

        self.assertEqual(composer.flash.calls[0]["q_len"], 512)
        self.assertEqual(composer.flash.calls[0]["kv_len"], 512)
        self.assertEqual(composer.flash.calls[0]["batch"], 1)
        self.assertEqual(composer.flash.calls[0]["n_kv_heads"], 8)
        self.assertEqual(composer.flash.calls[0]["phase"], "prefill")

    def test_cached_prefill_attention_shape_uses_suffix_and_full_kv(self):
        composer = fake_composer()
        composer.predict_layer(fake_cfg(), seq_len=128, bs=2, kv_len=4096, phase="prefill")

        self.assertEqual(composer.flash.calls[0]["q_len"], 128)
        self.assertEqual(composer.flash.calls[0]["kv_len"], 4096)
        self.assertEqual(composer.flash.calls[0]["batch"], 2)
        self.assertEqual(composer.flash.calls[0]["phase"], "cached_prefill")

    def test_decode_attention_shape_uses_one_query_and_effective_batch(self):
        composer = fake_composer()
        composer.predict_layer(fake_cfg(), seq_len=1, bs=16, kv_len=2048, phase="decode")

        self.assertEqual(composer.flash.calls[0]["q_len"], 1)
        self.assertEqual(composer.flash.calls[0]["kv_len"], 2048)
        self.assertEqual(composer.flash.calls[0]["batch"], 16)
        self.assertEqual(composer.flash.calls[0]["phase"], "decode")

    def test_fused_attention_does_not_add_qk_to_generic_gemm_path(self):
        composer = fake_composer()
        composer.predict_layer(fake_cfg(), seq_len=128, bs=2, kv_len=4096, phase="prefill")

        self.assertEqual(len(composer.gemm.calls), 7)
        self.assertNotIn((2, 128, 4096), composer.gemm.calls)
        self.assertNotIn((256, 128, 4096), composer.gemm.calls)

    def test_tensor_parallel_shards_gemm_and_attention_widths(self):
        composer = fake_composer()
        composer.predict_layer(
            fake_cfg(),
            seq_len=128,
            bs=2,
            kv_len=4096,
            phase="prefill",
            tensor_parallel_size=4,
        )

        self.assertIn((256, 1024, 4096), composer.gemm.calls)  # Q
        self.assertIn((256, 256, 4096), composer.gemm.calls)   # K/V
        self.assertIn((256, 4096, 1024), composer.gemm.calls)  # O
        self.assertIn((256, 3584, 4096), composer.gemm.calls)  # gate/up
        self.assertIn((256, 4096, 3584), composer.gemm.calls)  # down
        self.assertEqual(composer.flash.calls[0]["n_heads"], 8)
        self.assertEqual(composer.flash.calls[0]["n_kv_heads"], 2)

    def test_attention_shape_inventory_has_all_phases(self):
        phases = {row["phase"] for row in generate_attention_shapes()}

        self.assertEqual(phases, {"prefill", "cached_prefill", "decode"})


if __name__ == "__main__":
    unittest.main()
