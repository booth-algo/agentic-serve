from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

from llm_predict_2.cache_aware import (
    aggregate_turn_cache_feature,
    derive_turn_cache_features,
    weighted_median,
)
from llm_predict_2.framework_corrections import (
    prefix_cache_contention_factors,
    prefix_cache_prior,
)
from llm_predict_2.kernels.gemm import GemmPredictor, _MAX_PREDICT_CACHE
from llm_predict_2.serving import predict_serving


class FakeComposer:
    def __init__(self):
        self.ttft_calls = []
        self.decode_kv_lens = []

    def predict_ttft_ms(self, cfg, isl, bs=1, kv_len=None):
        self.ttft_calls.append((isl, bs, kv_len))
        return float(isl) + float(kv_len or isl) / 1000.0

    def predict_decode_step_us(self, cfg, kv_len, bs=1):
        self.decode_kv_lens.append(kv_len)
        return float(kv_len)


class CacheAwareTests(unittest.TestCase):
    def test_weighted_median_interpolates_even_weight_boundary(self):
        self.assertEqual(weighted_median([(10, 1), (20, 1)]), 15)
        self.assertAlmostEqual(weighted_median([(0.2, 3), (0.8, 3)]), 0.5)

    def test_skipped_turn_does_not_advance_cached_context(self):
        features = derive_turn_cache_features([
            {
                "turn_index": 0,
                "successful": 10,
                "avg_input_tokens": 100,
                "avg_output_tokens": 20,
            },
            {
                "turn_index": 1,
                "successful": 0,
                "avg_input_tokens": 200,
                "avg_output_tokens": 0,
            },
            {
                "turn_index": 2,
                "successful": 10,
                "avg_input_tokens": 250,
                "avg_output_tokens": 25,
            },
        ])

        self.assertEqual(len(features), 2)
        self.assertEqual(features[1].total_context_tokens, 250)
        self.assertEqual(features[1].new_prefill_tokens, 150)
        self.assertEqual(features[1].cached_context_tokens, 100)

    def test_aggregate_turn_uses_unbiased_weighted_median(self):
        feature = aggregate_turn_cache_feature([
            {
                "turn_index": 0,
                "successful": 10,
                "avg_input_tokens": 100,
                "avg_output_tokens": 20,
            },
            {
                "turn_index": 1,
                "successful": 10,
                "avg_input_tokens": 200,
                "avg_output_tokens": 40,
            },
        ])

        self.assertIsNotNone(feature)
        assert feature is not None
        self.assertEqual(feature.total_context_tokens, 150)
        self.assertEqual(feature.output_tokens, 30)
        self.assertEqual(feature.new_prefill_tokens, 100)
        self.assertAlmostEqual(feature.cache_hit_rate, 50 / 150)

    def test_predict_serving_prefills_new_tokens_but_decodes_full_context(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False)

        pred = predict_serving(
            composer, cfg, "H100",
            isl=1000, osl=8, concurrency=1,
            backend=None,
            total_context_tokens=1000,
            new_prefill_tokens=100,
        )

        self.assertEqual(composer.ttft_calls, [(100, 1, 1000)])
        self.assertGreaterEqual(min(composer.decode_kv_lens), 1000)
        self.assertEqual(pred.total_context_tokens, 1000)
        self.assertEqual(pred.new_prefill_tokens, 100)
        self.assertEqual(pred.cached_context_tokens, 900)
        self.assertAlmostEqual(pred.cache_hit_rate, 0.9)
        self.assertTrue(pred.cache_aware_applied)

    def test_prefix_cache_contention_requires_applicable_calibration(self):
        artifact = {
            "calibration_status": "high_confidence",
            "prefix_cache_factors_by_profile": {
                "swebench-multiturn-short": {
                    "10": {"ttft_factor": 2.0, "decode_factor": 3.0},
                    "20": {"ttft_factor": 4.0, "decode_factor": 5.0},
                }
            },
        }
        with patch(
            "llm_predict_2.framework_corrections._artifact_for",
            lambda *args, **kwargs: artifact,
        ):
            ttft, decode, applied = prefix_cache_contention_factors(
                "H100", "vllm", "0.19.0", "Llama-3.1-8B",
                15, "swebench-multiturn-short",
            )

            self.assertTrue(applied)
            self.assertAlmostEqual(ttft, 3.0)
            self.assertAlmostEqual(decode, 4.0)

            artifact["calibration_status"] = "low_confidence"
            self.assertEqual(
                prefix_cache_contention_factors(
                    "H100", "vllm", "0.19.0", "Llama-3.1-8B",
                    15, "swebench-multiturn-short",
                ),
                (1.0, 1.0, False),
            )

    def test_prefix_cache_prior_requires_applicable_calibration(self):
        artifact = {
            "calibration_status": "high_confidence",
            "prefix_cache_priors_by_profile": {
                "coding-agent": {
                    "new_prefill_tokens": 512,
                    "median_cache_hit_rate": 0.9,
                }
            },
        }
        with patch(
            "llm_predict_2.framework_corrections._artifact_for",
            lambda *args, **kwargs: artifact,
        ):
            new_tokens, hit_rate, applied = prefix_cache_prior(
                "H100", "vllm", "0.19.0", "Llama-3.1-8B",
                "coding-agent", 4096,
            )
            self.assertTrue(applied)
            self.assertEqual(new_tokens, 512)
            self.assertAlmostEqual(hit_rate, (4096 - 512) / 4096)

            artifact["calibration_status"] = "low_confidence"
            self.assertEqual(
                prefix_cache_prior(
                    "H100", "vllm", "0.19.0", "Llama-3.1-8B",
                    "coding-agent", 4096,
                ),
                (4096, 0.0, False),
            )

    def test_gemm_prediction_cache_is_bounded(self):
        predictor = object.__new__(GemmPredictor)
        predictor._predict_cache = OrderedDict()

        for i in range(_MAX_PREDICT_CACHE + 3):
            predictor._remember_prediction((i, i + 1, i + 2, 2), float(i))

        self.assertEqual(len(predictor._predict_cache), _MAX_PREDICT_CACHE)
        self.assertNotIn((0, 1, 2, 2), predictor._predict_cache)
        self.assertIn(
            (_MAX_PREDICT_CACHE + 2, _MAX_PREDICT_CACHE + 3,
             _MAX_PREDICT_CACHE + 4, 2),
            predictor._predict_cache,
        )


if __name__ == "__main__":
    unittest.main()
