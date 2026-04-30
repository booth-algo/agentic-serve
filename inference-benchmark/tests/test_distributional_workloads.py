import unittest
import json
import tempfile
from pathlib import Path

from src.workloads.distributional import DistributionalSampler
from src.workloads.dataset import DistributionalMultiTurnDataset, make_dataset
from src.workloads.profiles import WorkloadProfile
from src.workloads.trace_distributions import (
    TraceDistributionError,
    parse_trace_distribution,
)


def fixture_distribution():
    payload = {
        "schema_version": 1,
        "name": "fixture_multiturn",
        "source": {"kind": "unit-test"},
        "summary": {},
        "samples": {
            "turn_count": [3],
            "turns": [
                {
                    "turn_index": 0,
                    "total_context_tokens": 100,
                    "new_prefill_tokens": 100,
                    "output_tokens": 20,
                    "cache_hit_rate": 0.0,
                },
                {
                    "turn_index": 1,
                    "total_context_tokens": 150,
                    "new_prefill_tokens": 50,
                    "output_tokens": 30,
                    "cache_hit_rate": 100 / 150,
                },
                {
                    "turn_index": 2,
                    "total_context_tokens": 190,
                    "new_prefill_tokens": 40,
                    "output_tokens": 10,
                    "cache_hit_rate": 150 / 190,
                },
            ],
        },
    }
    return parse_trace_distribution(payload, path=Path("fixture.json"))


class TraceDistributionLoaderTests(unittest.TestCase):
    def test_parses_fixture_distribution(self):
        dist = fixture_distribution()

        self.assertEqual(dist.name, "fixture_multiturn")
        self.assertEqual(dist.turn_counts, (3,))
        self.assertEqual(len(dist.turns), 3)
        self.assertEqual(dist.turns_by_index[1][0].new_prefill_tokens, 50)

    def test_rejects_invalid_schema(self):
        with self.assertRaises(TraceDistributionError):
            parse_trace_distribution(
                {"schema_version": 999, "name": "bad", "samples": {"turn_count": [1], "turns": []}},
                path=Path("bad.json"),
            )


class DistributionalSamplerTests(unittest.TestCase):
    def test_builds_growing_context_from_prefill_deltas(self):
        sampler = DistributionalSampler(fixture_distribution(), seed=7)
        session = sampler.sample_session(session_id=3)

        self.assertEqual(len(session.turns), 3)
        specs = session.specs

        self.assertEqual([s.total_context_tokens for s in specs], [100, 150, 190])
        self.assertEqual([s.actual_new_prefill_tokens for s in specs], [100, 50, 40])
        self.assertEqual([s.cached_context_tokens for s in specs], [0, 100, 150])
        self.assertEqual([s.new_user_tokens for s in specs], [100, 30, 10])
        self.assertEqual([r.max_tokens for r in session.turns], [20, 30, 10])
        self.assertAlmostEqual(specs[1].cache_hit_rate, 100 / 150)
        self.assertAlmostEqual(specs[2].cache_hit_rate, 150 / 190)

    def test_stops_before_context_overflow(self):
        sampler = DistributionalSampler(
            fixture_distribution(),
            seed=7,
            max_context_tokens=160,
        )
        session = sampler.sample_session(session_id=0)

        self.assertEqual(len(session.turns), 2)
        self.assertEqual([s.total_context_tokens for s in session.specs], [100, 150])
        self.assertTrue(all(s.total_context_tokens <= 160 for s in session.specs))

    def test_clips_turn_that_would_cross_context_limit(self):
        sampler = DistributionalSampler(
            fixture_distribution(),
            seed=7,
            max_context_tokens=140,
        )
        session = sampler.sample_session(session_id=0)

        self.assertEqual(len(session.turns), 2)
        self.assertEqual(session.specs[-1].total_context_tokens, 140)
        self.assertEqual(session.specs[-1].new_user_tokens, 20)
        self.assertTrue(session.specs[-1].truncated_by_context_limit)


class DistributionalMultiTurnDatasetTests(unittest.TestCase):
    def test_make_dataset_loads_distributional_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture_multiturn.json"
            payload = {
                "schema_version": 1,
                "name": "fixture_multiturn",
                "source": {"kind": "unit-test"},
                "summary": {},
                "samples": {
                    "turn_count": [3],
                    "turns": [
                        {
                            "turn_index": 0,
                            "total_context_tokens": 100,
                            "new_prefill_tokens": 100,
                            "output_tokens": 20,
                            "cache_hit_rate": 0.0,
                        },
                        {
                            "turn_index": 1,
                            "total_context_tokens": 150,
                            "new_prefill_tokens": 50,
                            "output_tokens": 30,
                            "cache_hit_rate": 100 / 150,
                        },
                        {
                            "turn_index": 2,
                            "total_context_tokens": 190,
                            "new_prefill_tokens": 40,
                            "output_tokens": 10,
                            "cache_hit_rate": 150 / 190,
                        },
                    ],
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            profile = WorkloadProfile(
                name="fixture-multiturn",
                isl_tokens=512,
                osl_tokens=64,
                isl_stddev=0.0,
                description="fixture",
                dataset="distributional-multi-turn",
                file_path=str(path),
                mode="multi-turn",
                num_sessions=2,
                agent_type="coding",
                turn_style="multi-turn",
                data_source="distributional",
            )

            dataset = make_dataset(profile, max_context_tokens=140)

            self.assertIsInstance(dataset, DistributionalMultiTurnDataset)
            self.assertEqual(len(dataset.sessions), 2)
            self.assertEqual([len(s.turns) for s in dataset.sessions], [2, 2])
            self.assertEqual(dataset.sessions[0].turns[0].max_tokens, 20)
            self.assertLessEqual(dataset.session_specs[0][-1].total_context_tokens, 140)


if __name__ == "__main__":
    unittest.main()
