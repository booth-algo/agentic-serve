from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


class AggregateJsonlTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        with path.open("w") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")

    def test_rolls_up_records_by_profile_concurrency_turn(self) -> None:
        from profiling.process.extractors.aggregate_engine_step_trace_jsonl import (
            aggregate,
        )

        records = [
            # Turn 0: 3 steps: 1 prefill_only, 1 mixed, 1 decode_only
            {"profile": "smoke", "concurrency": 2, "target_turn_index": 0,
             "decode_batch": 0, "prefill_tokens": 10, "total_scheduled_tokens": 10,
             "free_kv_blocks": 100, "waiting_queue": 1, "running_queue": 1,
             "preemptions": 0, "scheduled_request_count": 2, "batch_size": 2,
             "primary_eval": "true"},
            {"profile": "smoke", "concurrency": 2, "target_turn_index": 0,
             "decode_batch": 1, "prefill_tokens": 8, "total_scheduled_tokens": 9,
             "free_kv_blocks": 92, "waiting_queue": 0, "running_queue": 2,
             "preemptions": 0, "scheduled_request_count": 2},
            {"profile": "smoke", "concurrency": 2, "target_turn_index": 0,
             "decode_batch": 2, "prefill_tokens": 0, "total_scheduled_tokens": 2,
             "free_kv_blocks": 90, "waiting_queue": 0, "running_queue": 2,
             "preemptions": 0, "scheduled_request_count": 2},
        ]
        rows = aggregate(records, profile_alias={})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["profile"], "smoke")
        self.assertEqual(row["concurrency"], "2")
        self.assertEqual(row["turn_index"], "0")
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["decode_only_steps"], 1)
        self.assertEqual(row["prefill_only_steps"], 1)
        self.assertEqual(row["mixed_decode_prefill_steps"], 1)
        self.assertEqual(row["total_decode_slots"], 3)
        self.assertEqual(row["total_prefill_tokens"], 18)
        self.assertEqual(row["total_scheduled_tokens"], 21)
        self.assertEqual(row["max_decode_batch"], 2)
        self.assertEqual(row["min_free_kv_blocks"], 90)
        self.assertEqual(row["prefill_intrusion_candidate"], "true")
        self.assertEqual(row["scheduled_request_count"], 2)

    def test_profile_alias_remaps_name(self) -> None:
        from profiling.process.extractors.aggregate_engine_step_trace_jsonl import (
            aggregate,
        )

        records = [
            {"profile": "swebench-multiturn", "concurrency": 320,
             "turn_index": 1, "decode_batch": 1, "prefill_tokens": 1,
             "total_scheduled_tokens": 2, "free_kv_blocks": 100,
             "waiting_queue": 0, "running_queue": 1, "preemptions": 0},
        ]
        rows = aggregate(
            records,
            profile_alias={"swebench-multiturn": "swebench-multiturn-synth"},
        )
        self.assertEqual(rows[0]["profile"], "swebench-multiturn-synth")

    def test_main_end_to_end(self) -> None:
        from profiling.process.extractors.aggregate_engine_step_trace_jsonl import (
            main,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            inp = tmp / "trace.jsonl"
            self._write_jsonl(inp, [
                {"profile": "foo", "concurrency": 4, "turn_index": 0,
                 "decode_batch": 0, "prefill_tokens": 16,
                 "total_scheduled_tokens": 16, "free_kv_blocks": 50,
                 "waiting_queue": 3, "running_queue": 1, "preemptions": 0,
                 "scheduled_request_count": 4},
            ])
            out = tmp / "summary.csv"

            argv = sys.argv
            sys.argv = ["aggregate", "--input", str(inp), "--output", str(out)]
            try:
                main()
            finally:
                sys.argv = argv

            self.assertTrue(out.exists())
            with out.open() as h:
                rows = list(csv.DictReader(h))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["profile"], "foo")
            self.assertEqual(int(rows[0]["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
