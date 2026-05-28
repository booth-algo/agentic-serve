import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import publish_sweep_state as publisher  # noqa: E402


class PublishSweepStateTests(unittest.TestCase):
    def test_active_run_lease_overrides_stale_pending_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_state_dir = publisher.STATE_DIR
            old_legacy_state_dir = publisher.LEGACY_STATE_DIR
            try:
                publisher.STATE_DIR = Path(tmp) / "state"
                publisher.LEGACY_STATE_DIR = Path(tmp) / "legacy"
                scope_dir = publisher.STATE_DIR / "synthetic_distributional"
                runs_dir = scope_dir / "runs"
                runs_dir.mkdir(parents=True)
                jid = "a100_Tiny_tp1_single"
                run_id = "run_20260512T120000Z_test"
                (scope_dir / f"{jid}.status").write_text("pending\n")
                (scope_dir / f"{jid}.run_id").write_text(f"{run_id}\n")
                (runs_dir / f"{run_id}.json").write_text(
                    """{
  "run_id": "run_20260512T120000Z_test",
  "job_id": "a100_Tiny_tp1_single",
  "status": "running",
  "updated_at": "2026-05-12T12:00:30+00:00"
}
"""
                )

                state = publisher.read_state(jid, "synthetic_distributional")
            finally:
                publisher.STATE_DIR = old_state_dir
                publisher.LEGACY_STATE_DIR = old_legacy_state_dir

        self.assertEqual(state["status"], "running")
        self.assertEqual(state["run_id"], "run_20260512T120000Z_test")
        self.assertEqual(state["updated_at"], "2026-05-12T12:00:30+00:00")

    def test_failure_metadata_is_published_from_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_state_dir = publisher.STATE_DIR
            old_legacy_state_dir = publisher.LEGACY_STATE_DIR
            try:
                publisher.STATE_DIR = Path(tmp) / "state"
                publisher.LEGACY_STATE_DIR = Path(tmp) / "legacy"
                scope_dir = publisher.STATE_DIR / "synthetic_distributional"
                scope_dir.mkdir(parents=True)
                jid = "a100_Tiny_tp1_single"
                (scope_dir / f"{jid}.status").write_text("skipped\n")
                (scope_dir / f"{jid}.attempt").write_text("2\n")
                (scope_dir / f"{jid}.reason").write_text("retry limit reached\n")
                (scope_dir / f"{jid}.failure.json").write_text(
                    """{
  "kind": "incomplete_outputs",
  "attempt": 2,
  "max_attempts": 2,
  "missing_outputs": ["Tiny_tp1_vllm_chat-singleturn-synth_conc500.json"],
  "reason": "retry limit reached"
}
"""
                )

                state = publisher.read_state(jid, "synthetic_distributional")
            finally:
                publisher.STATE_DIR = old_state_dir
                publisher.LEGACY_STATE_DIR = old_legacy_state_dir

        self.assertEqual(state["status"], "skipped")
        self.assertEqual(state["attempt"], 2)
        self.assertEqual(state["reason"], "retry limit reached")
        self.assertEqual(state["failure_metadata"]["max_attempts"], 2)
        self.assertEqual(
            state["failure_metadata"]["missing_outputs"],
            ["Tiny_tp1_vllm_chat-singleturn-synth_conc500.json"],
        )


if __name__ == "__main__":
    unittest.main()
