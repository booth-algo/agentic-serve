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

    def test_request_metadata_matches_synthetic_turn_specs(self):
        sampler = DistributionalSampler(fixture_distribution(), seed=7)
        session = sampler.sample_session(session_id=11)

        for request, spec in zip(session.turns, session.specs):
            meta = request.metadata
            self.assertEqual(meta["synthetic_session_id"], 11)
            self.assertEqual(meta["synthetic_turn_index"], spec.turn_index)
            self.assertEqual(meta["sampled_new_prefill_tokens"], spec.sampled_new_prefill_tokens)
            self.assertEqual(meta["planned_new_prefill_tokens"], spec.actual_new_prefill_tokens)
            self.assertEqual(meta["planned_cached_context_tokens"], spec.cached_context_tokens)
            self.assertEqual(meta["planned_total_context_tokens"], spec.total_context_tokens)
            self.assertEqual(meta["planned_new_user_tokens"], spec.new_user_tokens)
            self.assertEqual(meta["planned_output_tokens"], spec.output_tokens)
            self.assertEqual(
                meta["planned_total_with_output_tokens"],
                spec.total_context_tokens + spec.output_tokens,
            )
            self.assertEqual(meta["context_window_tokens"], spec.context_window_tokens)
            self.assertEqual(
                meta["context_safety_margin_tokens"],
                spec.context_safety_margin_tokens,
            )
            self.assertEqual(meta["prompt_token_budget"], spec.prompt_token_budget)
            self.assertAlmostEqual(meta["planned_cache_hit_rate"], spec.cache_hit_rate, places=6)
            self.assertEqual(meta["truncated_by_context_limit"], spec.truncated_by_context_limit)

    def test_stops_before_context_overflow(self):
        sampler = DistributionalSampler(
            fixture_distribution(),
            seed=7,
            max_context_tokens=160,
            context_safety_margin_tokens=0,
        )
        session = sampler.sample_session(session_id=0)

        self.assertEqual(len(session.turns), 2)
        self.assertEqual([s.total_context_tokens for s in session.specs], [100, 130])
        self.assertTrue(
            all(s.total_context_tokens + s.output_tokens <= 160 for s in session.specs)
        )

    def test_clips_turn_that_would_cross_context_limit(self):
        sampler = DistributionalSampler(
            fixture_distribution(),
            seed=7,
            max_context_tokens=170,
            context_safety_margin_tokens=0,
        )
        session = sampler.sample_session(session_id=0)

        self.assertEqual(len(session.turns), 2)
        self.assertEqual(session.specs[-1].total_context_tokens, 140)
        self.assertEqual(session.specs[-1].new_user_tokens, 20)
        self.assertTrue(session.specs[-1].truncated_by_context_limit)

    def test_reserves_output_and_safety_margin_under_context_limit(self):
        sampler = DistributionalSampler(
            fixture_distribution(),
            seed=7,
            max_context_tokens=160,
            context_safety_margin_tokens=8,
        )
        session = sampler.sample_session(session_id=0)

        self.assertEqual(len(session.turns), 2)
        self.assertEqual([s.total_context_tokens for s in session.specs], [100, 122])
        self.assertEqual(session.specs[-1].prompt_token_budget, 122)
        self.assertTrue(session.specs[-1].truncated_by_context_limit)
        for request, spec in zip(session.turns, session.specs):
            self.assertLessEqual(
                spec.total_context_tokens + request.max_tokens,
                160 - 8,
            )
            self.assertEqual(
                request.metadata["planned_total_with_output_tokens"],
                spec.total_context_tokens + request.max_tokens,
            )

    def test_same_seed_produces_reproducible_sessions(self):
        first = DistributionalSampler(fixture_distribution(), seed=123).sample_sessions(3)
        second = DistributionalSampler(fixture_distribution(), seed=123).sample_sessions(3)

        first_specs = [
            [(s.total_context_tokens, s.actual_new_prefill_tokens, s.output_tokens) for s in session.specs]
            for session in first
        ]
        second_specs = [
            [(s.total_context_tokens, s.actual_new_prefill_tokens, s.output_tokens) for s in session.specs]
            for session in second
        ]

        self.assertEqual(first_specs, second_specs)


class DistributionalMultiTurnDatasetTests(unittest.TestCase):
    def test_benchmark_request_metadata_uses_independent_default_dicts(self):
        first = DistributionalSampler(fixture_distribution(), seed=7).sample_session(session_id=0).turns[0]
        second = DistributionalSampler(fixture_distribution(), seed=7).sample_session(session_id=1).turns[0]

        first.metadata["mutated"] = True

        self.assertNotIn("mutated", second.metadata)

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

            dataset = make_dataset(
                profile,
                max_context_tokens=170,
                context_safety_margin_tokens=0,
            )

            self.assertIsInstance(dataset, DistributionalMultiTurnDataset)
            self.assertEqual(len(dataset.sessions), 2)
            self.assertEqual([len(s.turns) for s in dataset.sessions], [2, 2])
            self.assertEqual(dataset.sessions[0].turns[0].max_tokens, 20)
            self.assertEqual(
                dataset.sessions[0].turns[0].metadata["planned_new_prefill_tokens"],
                100,
            )
            self.assertEqual(
                dataset.sessions[0].turns[1].metadata["planned_total_context_tokens"],
                140,
            )
            self.assertLessEqual(dataset.session_specs[0][-1].total_context_tokens, 140)

    def test_make_dataset_num_sessions_override_controls_distributional_count(self):
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

            dataset = make_dataset(
                profile,
                max_context_tokens=170,
                context_safety_margin_tokens=0,
                num_sessions=4,
            )

            self.assertEqual(len(dataset.sessions), 4)

    def test_make_dataset_random_seed_controls_distributional_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture_multiturn.json"
            payload = {
                "schema_version": 1,
                "name": "fixture_multiturn",
                "source": {"kind": "unit-test"},
                "summary": {},
                "samples": {
                    "turn_count": [1, 2, 3],
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
                num_sessions=5,
                agent_type="coding",
                turn_style="multi-turn",
                data_source="distributional",
            )

            first = make_dataset(profile, random_seed=99)
            second = make_dataset(profile, random_seed=99)

            self.assertEqual(
                [[r.max_tokens for r in s.turns] for s in first.sessions],
                [[r.max_tokens for r in s.turns] for s in second.sessions],
            )


if __name__ == "__main__":
    unittest.main()
