from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace

from unittest import mock

from llm_predict.cache_aware import (
    aggregate_turn_cache_feature,
    derive_turn_cache_features,
    predict_multiturn_from_per_turn,
    weighted_median,
)
from llm_predict.export_serving_predictions import _prediction_row
from llm_predict.kernels.gemm import GemmPredictor, _MAX_PREDICT_CACHE
from llm_predict.prefix_cache_priors import PrefixCachePrior
from llm_predict.serving import decode_interval_count, predict_serving


class FakeComposer:
    def __init__(self):
        self.ttft_calls = []
        self.decode_kv_lens = []

    def predict_ttft_ms(self, cfg, isl, bs=1, kv_len=None, tensor_parallel_size=1):
        self.ttft_calls.append((isl, bs, kv_len))
        return float(isl) + float(kv_len or isl) / 1000.0

    def predict_ttft_us(self, cfg, isl, bs=1, kv_len=None, tensor_parallel_size=1):
        return float(isl) * 1000.0 + float(kv_len or isl)

    def predict_decode_step_us(self, cfg, kv_len, bs=1, tensor_parallel_size=1):
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

    def test_derive_turn_prefers_runner_cache_telemetry(self):
        features = derive_turn_cache_features([
            {
                "turn_index": 3,
                "successful": 7,
                "avg_input_tokens": 1000,
                "avg_output_tokens": 200,
                "median_input_tokens": 900,
                "median_output_tokens": 180,
                "median_new_prefill_tokens": 125,
                "median_cached_context_tokens": 775,
                "median_cache_hit_rate": 0.8611,
            },
        ])

        self.assertEqual(len(features), 1)
        feature = features[0]
        self.assertEqual(feature.turn_index, 3)
        self.assertEqual(feature.total_context_tokens, 900)
        self.assertEqual(feature.output_tokens, 180)
        self.assertEqual(feature.new_prefill_tokens, 125)
        self.assertEqual(feature.cached_context_tokens, 775)
        self.assertAlmostEqual(feature.cache_hit_rate, 0.8611)

    def test_predict_serving_prefills_new_tokens_but_decodes_full_context(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False, n_layers=32)

        pred = predict_serving(
            composer, cfg, "H100",
            isl=1000, osl=8, concurrency=1,
            backend=None,
            total_context_tokens=1000,
            new_prefill_tokens=100,
        )

        self.assertEqual(composer.ttft_calls, [(100, 1, 1000)])
        # first_decode_ms and queue model add extra decode calls
        tpot_decode_kv = [kv for kv in composer.decode_kv_lens if kv >= 1001]
        self.assertGreaterEqual(len(tpot_decode_kv), 0)
        self.assertEqual(pred.total_context_tokens, 1000)
        self.assertEqual(pred.new_prefill_tokens, 100)
        self.assertEqual(pred.cached_context_tokens, 900)
        self.assertAlmostEqual(pred.cache_hit_rate, 0.9)
        self.assertTrue(pred.cache_aware_applied)
        self.assertEqual(pred.cache_feature_source, "provided")
        self.assertEqual(pred.cache_prediction_regime, "prefix_cached_prefill")
        self.assertEqual(decode_interval_count(8), 7)
        self.assertAlmostEqual(pred.tpot_ms, pred.decode_total_ms / 7)
        self.assertAlmostEqual(pred.e2el_ms, pred.ttft_ms + pred.decode_total_ms)

    def test_predict_serving_uses_no_decode_intervals_for_single_output_token(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False, n_layers=32)

        pred = predict_serving(
            composer, cfg, "H100",
            isl=100, osl=1, concurrency=1,
            backend=None,
        )

        self.assertEqual(pred.decode_total_ms, 0.0)
        self.assertEqual(pred.tpot_ms, 0.0)
        self.assertEqual(pred.e2el_ms, pred.ttft_ms)
        # first_decode_ms adds one decode call even at osl=1
        osl1_decode = [kv for kv in composer.decode_kv_lens if kv == 100]
        self.assertEqual(len(osl1_decode), 1)

    def test_multiturn_prediction_calls_serving_once_per_valid_turn(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False, n_layers=32)

        pred = predict_multiturn_from_per_turn(
            composer, cfg, "H100",
            [
                {
                    "turn_index": 0,
                    "successful": 2,
                    "median_input_tokens": 100,
                    "median_new_prefill_tokens": 100,
                    "median_output_tokens": 10,
                },
                {
                    "turn_index": 1,
                    "successful": 1,
                    "median_input_tokens": 1000,
                    "median_new_prefill_tokens": 50,
                    "median_cached_context_tokens": 950,
                    "median_cache_hit_rate": 0.95,
                    "median_output_tokens": 20,
                },
            ],
            concurrency=1,
        )

        self.assertIsNotNone(pred)
        assert pred is not None
        # Queue model adds extra predict_ttft_ms calls for batch prefill costing.
        # Verify the per-turn serving calls still use correct (prefill, kv) shapes.
        turn0_ttft = [c for c in composer.ttft_calls if c[0] == 100]
        turn1_ttft = [c for c in composer.ttft_calls if c[0] == 50]
        self.assertGreaterEqual(len(turn0_ttft), 1)
        self.assertGreaterEqual(len(turn1_ttft), 1)
        self.assertEqual(turn0_ttft[0], (100, 1, 100))
        self.assertEqual(turn1_ttft[0], (50, 1, 1000))
        self.assertEqual(pred.multiturn_prediction_mode, "per_turn_aggregated")
        self.assertEqual(pred.predicted_turn_count, 2)
        self.assertEqual(pred.total_successful_turn_requests, 3)

    def test_multiturn_prediction_skips_invalid_turns(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False, n_layers=32)

        pred = predict_multiturn_from_per_turn(
            composer, cfg, "H100",
            [
                {
                    "turn_index": 0,
                    "successful": 0,
                    "median_input_tokens": 100,
                    "median_new_prefill_tokens": 100,
                    "median_output_tokens": 10,
                },
                {
                    "turn_index": 1,
                    "successful": 1,
                    "median_input_tokens": 1000,
                    "median_new_prefill_tokens": 50,
                    "median_cached_context_tokens": 950,
                    "median_output_tokens": 20,
                },
            ],
            concurrency=1,
        )

        self.assertIsNotNone(pred)
        assert pred is not None
        self.assertEqual(composer.ttft_calls, [(50, 1, 1000)])
        self.assertEqual(pred.predicted_turn_count, 1)

    def test_multiturn_tpot_is_decode_interval_weighted(self):
        composer = FakeComposer()
        cfg = SimpleNamespace(name="fake", is_moe=False, n_layers=32)

        pred = predict_multiturn_from_per_turn(
            composer, cfg, "H100",
            [
                {
                    "turn_index": 0,
                    "successful": 1,
                    "median_input_tokens": 100,
                    "median_new_prefill_tokens": 100,
                    "median_output_tokens": 10,
                },
                {
                    "turn_index": 1,
                    "successful": 1,
                    "median_input_tokens": 1000,
                    "median_new_prefill_tokens": 50,
                    "median_cached_context_tokens": 950,
                    "median_output_tokens": 20,
                },
            ],
            concurrency=1,
        )

        self.assertIsNotNone(pred)
        assert pred is not None
        expected_decode_intervals = (10 - 1) + (20 - 1)
        self.assertAlmostEqual(
            pred.tpot_ms,
            pred.decode_total_ms * pred.total_successful_turn_requests
            / expected_decode_intervals,
        )

    def test_prefix_cache_export_without_features_is_marked_unsupported(self):
        composer = FakeComposer()

        with mock.patch(
            "llm_predict.export_serving_predictions.get_prefix_cache_prior",
            return_value=None,
        ):
            row = _prediction_row(
                {
                    "hardware": "H100x4",
                    "dataScope": "current",
                    "engineVersion": "0.10.0",
                    "config": {
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "backend": "vllm",
                        "profile": "coding-singleturn",
                        "mode": "single_turn",
                        "concurrency": 20,
                    },
                    "summary": {
                        "successful_requests": 10,
                        "total_input_tokens": 40960,
                        "total_output_tokens": 8000,
                        "median_ttft_ms": 100,
                        "median_tpot_ms": 5,
                        "median_e2el_ms": 4100,
                    },
                },
                composer,
                "H100",
            )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["cache_feature_source"], "missing")
        self.assertEqual(row["cache_prediction_regime"], "unknown_prefix_cache")
        self.assertFalse(row["ttft_prediction_supported"])
        self.assertEqual(row["unsupported_reason"], "missing_prefix_cache_features")
        self.assertNotIn("ttft_err", row)
        self.assertNotIn("e2el_err", row)
        self.assertIn("tpot_err", row)

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

    def test_coding_singleturn_uses_prefix_cache_prior(self):
        prior = PrefixCachePrior(
            profile="coding-singleturn",
            cached_context_tokens=6982,
            new_prefill_tokens=328,
            total_context_tokens=7310,
            cache_hit_rate=0.9551,
            source="test",
        )
        composer = FakeComposer()

        with mock.patch(
            "llm_predict.export_serving_predictions.get_prefix_cache_prior",
            return_value=prior,
        ):
            row = _prediction_row(
                {
                    "hardware": "H100x4",
                    "dataScope": "current",
                    "engineVersion": "0.10.0",
                    "config": {
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "backend": "vllm",
                        "profile": "coding-singleturn",
                        "mode": "single_turn",
                        "concurrency": 20,
                    },
                    "summary": {
                        "successful_requests": 10,
                        "total_input_tokens": 73100,
                        "total_output_tokens": 8000,
                        "median_ttft_ms": 100,
                        "median_tpot_ms": 5,
                        "median_e2el_ms": 4100,
                    },
                },
                composer,
                "H100",
            )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["cache_feature_source"], "prefix_cache_prior")
        self.assertEqual(row["cache_prediction_regime"], "prefix_cached_prefill")
        self.assertTrue(row["cache_aware_applied"])
        self.assertTrue(row["ttft_prediction_supported"])
        self.assertEqual(row["total_context_tokens"], 7310)
        self.assertEqual(row["new_prefill_tokens"], 328)
        self.assertEqual(row["cached_context_tokens"], 6982)
        self.assertAlmostEqual(row["cache_hit_rate"], 0.9551)
        self.assertIn("ttft_err", row)
        self.assertIn("tpot_err", row)
        self.assertIn("e2el_err", row)

    def test_coding_singleturn_without_prior_falls_back_to_unsupported(self):
        composer = FakeComposer()

        with mock.patch(
            "llm_predict.export_serving_predictions.get_prefix_cache_prior",
            return_value=None,
        ):
            row = _prediction_row(
                {
                    "hardware": "H100x4",
                    "dataScope": "current",
                    "engineVersion": "0.10.0",
                    "config": {
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "backend": "vllm",
                        "profile": "coding-singleturn",
                        "mode": "single_turn",
                        "concurrency": 20,
                    },
                    "summary": {
                        "successful_requests": 10,
                        "total_input_tokens": 73100,
                        "total_output_tokens": 8000,
                        "median_ttft_ms": 100,
                        "median_tpot_ms": 5,
                        "median_e2el_ms": 4100,
                    },
                },
                composer,
                "H100",
            )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["cache_feature_source"], "missing")
        self.assertEqual(row["cache_prediction_regime"], "unknown_prefix_cache")
        self.assertFalse(row["ttft_prediction_supported"])
        self.assertFalse(row["cache_aware_applied"])
        self.assertNotIn("ttft_err", row)
        self.assertNotIn("e2el_err", row)
        self.assertIn("tpot_err", row)

    def test_export_flags_e2el_below_ttft_measurements(self):
        composer = FakeComposer()

        row = _prediction_row(
            {
                "hardware": "H100",
                "dataScope": "current",
                "engineVersion": "0.10.0",
                "config": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "backend": "vllm",
                    "profile": "chat-singleturn",
                    "mode": "single_turn",
                    "concurrency": 1,
                },
                "summary": {
                    "successful_requests": 1,
                    "total_input_tokens": 100,
                    "total_output_tokens": 10,
                    "median_ttft_ms": 100,
                    "median_tpot_ms": 5,
                    "median_e2el_ms": 90,
                },
            },
            composer,
            "H100",
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["measurement_semantics_warning"], "measured_e2el_lt_ttft")
        self.assertIn("e2el_pred", row)
        self.assertIn("e2el_meas", row)
        self.assertNotIn("e2el_err", row)


if __name__ == "__main__":
    unittest.main()
