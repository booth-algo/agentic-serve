import unittest

from src.benchmark.metrics import (
    RequestResult,
    aggregate_per_turn,
    annotate_multi_turn_cache_estimate,
)


class RunnerCacheMetadataTests(unittest.TestCase):
    def test_annotates_successful_multi_turn_request(self):
        result = RequestResult(success=True, input_tokens=250, output_tokens=40)

        annotate_multi_turn_cache_estimate(
            result,
            session_id=12,
            turn_index=2,
            previous_context_tokens=175,
        )

        self.assertEqual(result.session_id, 12)
        self.assertEqual(result.turn_index, 2)
        self.assertEqual(result.previous_context_tokens, 175)
        self.assertEqual(result.total_context_tokens, 250)
        self.assertEqual(result.cached_context_tokens, 175)
        self.assertEqual(result.new_prefill_tokens, 75)
        self.assertAlmostEqual(result.cache_hit_rate, 0.7)
        self.assertEqual(result.cache_estimate_source, "previous_prompt_tokens")

    def test_per_turn_summary_includes_cache_estimate_medians(self):
        first = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=100, output_tokens=20),
            session_id=0,
            turn_index=0,
            previous_context_tokens=0,
        )
        second = annotate_multi_turn_cache_estimate(
            RequestResult(success=True, input_tokens=260, output_tokens=30),
            session_id=1,
            turn_index=0,
            previous_context_tokens=200,
        )

        summaries = aggregate_per_turn({0: [first, second]})

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.median_input_tokens, 180)
        self.assertEqual(summary.median_output_tokens, 25)
        self.assertEqual(summary.median_new_prefill_tokens, 80)
        self.assertEqual(summary.median_cached_context_tokens, 100)
        self.assertAlmostEqual(summary.median_cache_hit_rate, (0.0 + 200 / 260) / 2)


if __name__ == "__main__":
    unittest.main()
