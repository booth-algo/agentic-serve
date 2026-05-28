from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


class CompareSimToEngineTraceTest(unittest.TestCase):
    def test_diff_csv_joins_on_profile_concurrency_turn(self) -> None:
        from profiling.process.comparators.compare_sim_to_engine_trace import (
            compute_diffs,
            load_sim_rows,
            load_truth_rows,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace = tmp / "trace_summary.csv"
            trace.write_text(
                "profile,concurrency,turn_index,steps,decode_only_steps,"
                "prefill_only_steps,mixed_decode_prefill_steps,total_decode_slots,"
                "total_prefill_tokens,max_decode_batch,mean_decode_batch,"
                "min_free_kv_blocks,total_preemptions\n"
                "swebench-multiturn-synth,40,12,39,21,1,15,840,247624,40,21.5385,12119,0\n"
            )
            sim_csv = tmp / "sim.csv"
            sim_csv.write_text(
                "profile,concurrency,turn_index,sim_steps,sim_decode_only_steps,"
                "sim_prefill_only_steps,sim_mixed_decode_prefill_steps,"
                "sim_total_decode_slots,sim_total_prefill_tokens,sim_max_decode_batch,"
                "sim_mean_decode_batch,sim_min_free_kv_blocks,sim_total_preemptions\n"
                "swebench-multiturn-synth,40,12,23,22,1,0,880,5760,40,38.26,10811,0\n"
            )

            truth = load_truth_rows(str(trace))
            sim = load_sim_rows(sim_csv)
            diffs = compute_diffs(truth, sim)

        diffs_by_metric = {d.metric: d for d in diffs}
        mixed = diffs_by_metric["mixed_decode_prefill_steps"]
        self.assertEqual(mixed.real, 15.0)
        self.assertEqual(mixed.sim, 0.0)
        self.assertEqual(mixed.abs_delta, -15.0)
        self.assertAlmostEqual(mixed.pct_delta, -100.0, places=2)

        steps = diffs_by_metric["steps"]
        self.assertEqual(steps.real, 39.0)
        self.assertEqual(steps.sim, 23.0)
        self.assertEqual(steps.abs_delta, -16.0)

    def test_main_writes_csv_and_report(self) -> None:
        from profiling.process.comparators.compare_sim_to_engine_trace import main

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace = tmp / "vllm_engine_step_trace_smoke_summary.csv"
            trace.write_text(
                "profile,concurrency,turn_index,steps,decode_only_steps,"
                "prefill_only_steps,mixed_decode_prefill_steps,total_decode_slots,"
                "total_prefill_tokens,max_decode_batch,mean_decode_batch,"
                "min_free_kv_blocks,total_preemptions\n"
                "smoke,2,0,5,3,1,1,6,10,2,1.5,100,0\n"
            )
            sim_csv = tmp / "sim.csv"
            sim_csv.write_text(
                "profile,concurrency,turn_index,sim_steps,sim_decode_only_steps,"
                "sim_prefill_only_steps,sim_mixed_decode_prefill_steps,"
                "sim_total_decode_slots,sim_total_prefill_tokens,sim_max_decode_batch,"
                "sim_mean_decode_batch,sim_min_free_kv_blocks,sim_total_preemptions\n"
                "smoke,2,0,4,3,1,0,6,10,2,2.0,90,0\n"
            )
            output = tmp / "diff.csv"
            report = tmp / "diff.md"

            argv = sys.argv
            sys.argv = [
                "compare",
                "--trace-glob", str(trace),
                "--predictions", str(sim_csv),
                "--output", str(output),
                "--report-output", str(report),
            ]
            try:
                main()
            finally:
                sys.argv = argv

            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            metrics = {r["metric"] for r in rows}
            self.assertIn("mixed_decode_prefill_steps", metrics)
            self.assertIn("steps", metrics)
            report_text = report.read_text()
            self.assertIn("smoke c=2 turn=0", report_text)


if __name__ == "__main__":
    unittest.main()
