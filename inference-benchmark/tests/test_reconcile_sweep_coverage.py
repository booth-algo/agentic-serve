import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reconcile_sweep_coverage as reconcile  # noqa: E402


def coverage(status="skipped"):
    expected = {
        ("A100-40GB", "Tiny", "vllm", "single-turn", "chat-singleturn-synth", 1),
        ("A100-40GB", "Tiny", "vllm", "single-turn", "chat-singleturn-synth", 2),
    }
    present = {
        ("A100-40GB", "Tiny", "vllm", "single-turn", "chat-singleturn-synth", 1),
    }
    return reconcile.JobCoverage(
        job_id="a100_Tiny_tp1_single",
        data_scope="synthetic_distributional",
        host="a100",
        hw_label="A100-40GB",
        model="Tiny",
        tp=1,
        mode="single",
        backend="vllm",
        status=status,
        reason="old terminal state",
        attempt=2,
        failure_metadata={
            "kind": "incomplete_outputs",
            "status": status,
            "reason": "retry limit reached after 2/2 incomplete attempts: ABORT: Success rate 50.0% below minimum 75%",
            "attempt": 2,
            "max_attempts": 2,
            "expected_outputs_present": 1,
            "expected_outputs_total": 2,
            "missing_outputs": ["Tiny_tp1_vllm_chat-singleturn-synth_conc2.json"],
            "remote_log": "/tmp/bench_Tiny.log",
        },
        expected=expected,
        present=present,
    )


class ReconcileSweepCoverageTests(unittest.TestCase):
    def test_reset_stale_job_requeues_once_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            outcome = reconcile.reset_stale_jobs(
                [coverage()],
                state_root,
                {"skipped"},
                "synthetic_distributional",
                write_reason=True,
                max_requeues=1,
            )

            scope_dir = state_root / "synthetic_distributional"
            jid = "a100_Tiny_tp1_single"
            blocker = json.loads((scope_dir / f"{jid}.coverage_blocker.json").read_text())

            self.assertEqual([cov.job_id for cov in outcome.reset], ["a100_Tiny_tp1_single"])
            self.assertEqual(outcome.exhausted, [])
            self.assertEqual((scope_dir / f"{jid}.status").read_text(), "pending\n")
            self.assertEqual((scope_dir / f"{jid}.coverage_requeue_count").read_text(), "1\n")
            self.assertIn("coverage requeue 1/1", (scope_dir / f"{jid}.reason").read_text())
            self.assertEqual(blocker["status"], "requeued")
            self.assertEqual(blocker["missing_count"], 1)
            self.assertEqual(blocker["missing_points"], [{
                "hardware": "A100-40GB",
                "model": "Tiny",
                "backend": "vllm",
                "mode": "single-turn",
                "profile": "chat-singleturn-synth",
                "concurrency": 2,
            }])
            self.assertEqual(blocker["failure"]["category"], "success_rate_below_min")
            self.assertEqual(blocker["failure"]["attempt"], 2)
            self.assertEqual(blocker["requeue_count"], 1)
            self.assertEqual(blocker["max_requeues"], 1)

    def test_reset_stale_job_stops_at_requeue_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            scope_dir = state_root / "synthetic_distributional"
            scope_dir.mkdir(parents=True)
            jid = "a100_Tiny_tp1_single"
            (scope_dir / f"{jid}.status").write_text("skipped\n")
            (scope_dir / f"{jid}.coverage_requeue_count").write_text("1\n")

            outcome = reconcile.reset_stale_jobs(
                [coverage()],
                state_root,
                {"skipped"},
                "synthetic_distributional",
                write_reason=True,
                max_requeues=1,
            )
            blocker = json.loads((scope_dir / f"{jid}.coverage_blocker.json").read_text())

            self.assertEqual(outcome.reset, [])
            self.assertEqual([cov.job_id for cov in outcome.exhausted], ["a100_Tiny_tp1_single"])
            self.assertEqual((scope_dir / f"{jid}.status").read_text(), "skipped\n")
            self.assertIn("coverage requeue limit reached 1/1", (scope_dir / f"{jid}.reason").read_text())
            self.assertIn("last failure: success rate below threshold after 2/2 attempts", (scope_dir / f"{jid}.reason").read_text())
            self.assertEqual(blocker["status"], "requeue_exhausted")
            self.assertEqual(blocker["requeue_count"], 1)
            self.assertEqual(blocker["max_requeues"], 1)
            self.assertEqual(blocker["failure"]["category"], "success_rate_below_min")


if __name__ == "__main__":
    unittest.main()
