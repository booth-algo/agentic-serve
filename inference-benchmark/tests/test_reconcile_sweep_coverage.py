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
            self.assertEqual(blocker["failure"]["category"], "low_success_rate")
            self.assertEqual(blocker["failure"]["failure_class"], "low_success_rate")
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
            self.assertEqual(blocker["failure"]["category"], "low_success_rate")
            self.assertEqual(blocker["failure"]["failure_class"], "low_success_rate")


class FailureClassificationTests(unittest.TestCase):
    """Golden mapping: structured/legacy failure record -> (failure_class, disposition).

    See docs/coverage-classification-rfc.md. Policy changes = edit these
    expectations + disposition_for_class(); no regex hunt across two languages.
    """

    def _cov(self, *, status="skipped", reason=None, **meta):
        meta.setdefault("status", status)
        if reason is not None:
            meta["reason"] = reason
        return reconcile.JobCoverage(
            job_id="h_M_tp2_single", data_scope="synthetic_distributional",
            host="h", hw_label="H100x2", model="M", tp=2, mode="single",
            backend="vllm", status=status, reason=reason, attempt=meta.get("attempt"),
            failure_metadata=meta, expected={("H100x2", "M", "vllm", "single-turn", "p", 1)},
            present=set(),
        )

    def assert_maps(self, cov, failure_class, disposition):
        fp = reconcile.failure_payload(cov)
        self.assertEqual(fp["failure_class"], failure_class)
        self.assertEqual(reconcile.coverage_disposition(cov), disposition)

    # --- the RFC invariant: never N/A without positive evidence -------------
    def test_unknown_zero_results_is_todo_not_na(self):
        # The bug that started this: "zero results, no retryable OOM" has no
        # captured cause -> must be fillable TODO, never N/A.
        self.assert_maps(self._cov(
            reason="zero results and retry limit exhausted or no retryable OOM; attempt=0 oom_log="
        ), "unknown", "todo")

    # --- legacy reason-string bridge (single place that parses prose) -------
    def test_legacy_config_error_is_model_missing_failed(self):
        self.assert_maps(self._cov(
            reason="OSError: Can't load the configuration of '/models/gpt-oss-120b'"
        ), "model_missing", "failed")

    def test_legacy_kv_cache_reason_is_oom(self):
        self.assert_maps(self._cov(
            reason="ValueError: No available memory for the cache blocks"
        ), "oom_kv_cache", "na")  # no util captured -> treated as the limit

    def test_legacy_low_success_rate(self):
        self.assert_maps(self._cov(
            reason="ABORT: Success rate 38% below minimum 75%"
        ), "low_success_rate", "na")

    # --- structured launcher path (preferred) ------------------------------
    def test_structured_model_missing_is_failed(self):
        self.assert_maps(self._cov(failure_class="model_missing"), "model_missing", "failed")

    def test_structured_engine_crash_is_failed(self):
        self.assert_maps(self._cov(failure_class="engine_crash"), "engine_crash", "failed")

    def test_oom_below_max_util_is_fixable_todo(self):
        # 3090 vllm gpt-oss-120b: real OOM at gpu_mem=0.85 -> raise gpu_mem, not N/A.
        self.assert_maps(self._cov(
            failure_class="oom_kv_cache", evidence={"gpu_mem_util": 0.85},
        ), "oom_kv_cache", "todo")

    def test_oom_at_max_util_is_irreducible_na(self):
        self.assert_maps(self._cov(
            failure_class="oom_kv_cache", evidence={"gpu_mem_util": 0.95},
        ), "oom_kv_cache", "na")

    def test_hw_infeasible_is_na(self):
        self.assert_maps(self._cov(failure_class="hw_infeasible"), "hw_infeasible", "na")

    def test_known_oom_status_is_na(self):
        self.assert_maps(self._cov(status="known_oom", reason="hard OOM"),
                         "oom_kv_cache", "na")

    def test_explicit_failure_class_overrides_reason_text(self):
        # structured field wins over a misleading legacy reason string
        self.assert_maps(self._cov(
            failure_class="engine_crash", reason="zero results no retryable OOM",
        ), "engine_crash", "failed")

    def test_labels_are_class_specific(self):
        cov = self._cov(failure_class="oom_kv_cache", evidence={"gpu_mem_util": 0.85})
        self.assertEqual(reconcile.coverage_disposition_label(cov, "todo"),
                         "TODO — raise gpu_mem and retry")


if __name__ == "__main__":
    unittest.main()
