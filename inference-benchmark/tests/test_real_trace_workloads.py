import json
import tempfile
import unittest
from pathlib import Path

from src.workloads.dataset import TrajectoryMultiTurnDataset, make_dataset
from src.workloads.profiles import WorkloadProfile


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class RealTraceWorkloadTests(unittest.TestCase):
    def test_trajectory_dataset_reserves_output_and_safety_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            session = {
                "session_id": "trace-1",
                "source": "unit-trace",
                "turns": [
                    {
                        "turn_idx": 0,
                        "messages": [{"role": "user", "content": words(50)}],
                        "osl_tokens": 20,
                    },
                    {
                        "turn_idx": 1,
                        "messages": [
                            {"role": "user", "content": words(50)},
                            {"role": "assistant", "content": words(20)},
                            {"role": "user", "content": words(20)},
                        ],
                        "osl_tokens": 20,
                    },
                ],
            }
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")

            dataset = TrajectoryMultiTurnDataset(
                filepath=str(path),
                min_turns=1,
                max_turns=2,
                num_sessions=1,
                max_isl_tokens=120,
                max_osl_tokens=50,
                context_safety_margin_tokens=10,
            )

            self.assertEqual(len(dataset.sessions), 1)
            turns = dataset.sessions[0].turns
            self.assertEqual(len(turns), 1)
            meta = turns[0].metadata
            self.assertEqual(meta["source_session_id"], "trace-1")
            self.assertEqual(meta["source_turn_index"], 0)
            self.assertEqual(meta["trace_content_source"], "unit-trace")
            self.assertLessEqual(
                meta["planned_total_with_output_tokens"],
                meta["context_window_tokens"] - meta["context_safety_margin_tokens"],
            )

    def test_make_dataset_applies_context_cap_to_real_trace_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                json.dumps({
                    "session_id": "trace-2",
                    "source": "unit-trace",
                    "turns": [
                        {
                            "turn_idx": 0,
                            "messages": [{"role": "user", "content": words(50)}],
                            "osl_tokens": 20,
                        }
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            profile = WorkloadProfile(
                name="fixture-real-trace",
                isl_tokens=4096,
                osl_tokens=50,
                isl_stddev=0.0,
                description="fixture",
                dataset="swebench-multi-turn",
                file_path=str(path),
                mode="multi-turn",
                min_turns=1,
                max_turns=2,
                num_sessions=1,
                agent_type="coding",
                turn_style="multi-turn",
                data_source="swebench",
            )

            dataset = make_dataset(
                profile,
                max_context_tokens=120,
                context_safety_margin_tokens=10,
            )

            meta = dataset.sessions[0].turns[0].metadata
            self.assertEqual(meta["context_window_tokens"], 120)
            self.assertEqual(meta["context_safety_margin_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
