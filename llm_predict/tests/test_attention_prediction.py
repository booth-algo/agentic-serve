from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm_predict.composer import Composer
from llm_predict.configs.model_configs import MODEL_CONFIGS
from llm_predict.kernels.flash_attn import _flash_attn_bytes, _flash_attn_flops
from llm_predict.sweep.generate_shapes import generate_attention_shapes


class RecordingGemm:
    def __init__(self):
        self.calls = []
        self.dtype_bytes = []

    def predict(self, M, N, K, dtype_bytes=2):
        self.calls.append((M, N, K))
        self.dtype_bytes.append(dtype_bytes)
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
        expert_intermediate_size=None,
        is_moe=False,
        n_experts=1,
        top_k=1,
    )


class AttentionPredictionTests(unittest.TestCase):
    def test_gpt_oss_configs_match_openai_architecture(self):
        gpt_oss_20b = MODEL_CONFIGS["gpt-oss-20b"]
        self.assertEqual(gpt_oss_20b.hidden_dim, 2880)
        self.assertEqual(gpt_oss_20b.n_layers, 24)
        self.assertEqual(gpt_oss_20b.n_heads, 64)
        self.assertEqual(gpt_oss_20b.n_kv_heads, 8)
        self.assertEqual(gpt_oss_20b.head_dim, 64)
        self.assertEqual(gpt_oss_20b.intermediate_size, 2880)
        self.assertEqual(gpt_oss_20b.n_experts, 32)
        self.assertEqual(gpt_oss_20b.top_k, 4)
        self.assertEqual(gpt_oss_20b.expert_weight_bits, 4)
        self.assertEqual(gpt_oss_20b.sliding_window, 128)
        self.assertEqual(gpt_oss_20b.full_attention_layers, 12)
        self.assertAlmostEqual(gpt_oss_20b.active_params_b or 0.0, 3.6)

        gpt_oss_120b = MODEL_CONFIGS["gpt-oss-120b"]
        self.assertEqual(gpt_oss_120b.hidden_dim, 2880)
        self.assertEqual(gpt_oss_120b.n_layers, 36)
        self.assertEqual(gpt_oss_120b.n_experts, 128)
        self.assertEqual(gpt_oss_120b.top_k, 4)
        self.assertEqual(gpt_oss_120b.expert_weight_bits, 4)
        self.assertEqual(gpt_oss_120b.full_attention_layers, 18)
        self.assertAlmostEqual(gpt_oss_120b.active_params_b or 0.0, 5.1)

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

    def test_moe_uses_topk_scale_on_ffn_gemms(self):
        composer = fake_composer()
        cfg = SimpleNamespace(
            hidden_dim=8,
            n_heads=2,
            n_kv_heads=1,
            head_dim=4,
            intermediate_size=16,
            is_moe=True,
            n_experts=4,
            top_k=2,
        )

        # M = seq_len(3) * bs(2) = 6 for prefill
        layer = composer.predict_layer(cfg, seq_len=3, bs=2, phase="prefill")

        # Router: predict(M=6, N=n_experts=4, K=h=8)
        self.assertIn((6, 4, 8), composer.gemm.calls)  # router
        # Gate/up: predict(M=6, N=ffn=16, K=h=8) — each called once, scaled by top_k internally
        self.assertEqual(composer.gemm.calls.count((6, 16, 8)), 2)  # gate + up
        # Down: predict(M=6, N=h=8, K=ffn=16), scaled by top_k
        self.assertIn((6, 8, 16), composer.gemm.calls)  # down
        # Router output: raw GEMM value (fake composer returns 1.0 per call)
        self.assertEqual(layer.router_proj_us, 1.0)
        # FFN outputs: each GEMM returns 1.0, scaled by top_k=2 → 2.0 each
        self.assertEqual(layer.gate_proj_us, 2.0)
        self.assertEqual(layer.up_proj_us, 2.0)
        self.assertEqual(layer.down_proj_us, 2.0)
        # Q/K/V/O (4 calls) + router + gate + up + down (4 MoE calls) = 8 total, all default fp16
        self.assertEqual(composer.gemm.dtype_bytes, [2, 2, 2, 2, 2, 2, 2, 2])

    def test_sliding_attention_uses_window_for_sliding_layers(self):
        composer = fake_composer()
        cfg = SimpleNamespace(
            hidden_dim=4096,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            intermediate_size=14336,
            expert_intermediate_size=None,
            expert_weight_bits=None,
            is_moe=False,
            n_experts=1,
            top_k=1,
            n_layers=4,
            sliding_window=128,
            full_attention_layers=1,
        )

        composer._predict_model_us(
            cfg, seq_len=1, bs=1, kv_len=2048, phase="decode"
        )

        self.assertEqual(composer.flash.calls[0]["kv_len"], 2048)
        self.assertEqual(composer.flash.calls[1]["kv_len"], 128)

    def test_attention_shape_inventory_has_all_phases(self):
        phases = {row["phase"] for row in generate_attention_shapes()}

        self.assertEqual(phases, {"prefill", "cached_prefill", "decode"})


if __name__ == "__main__":
    unittest.main()
