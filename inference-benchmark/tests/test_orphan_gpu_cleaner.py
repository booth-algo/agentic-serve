import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import clean_orphan_gpus as cleaner  # noqa: E402


def base_state(
    process,
    *,
    status="same-user-orphan",
    ports=None,
    assignments=None,
    remote_user="kevin",
    drained=False,
    util_pct=0,
):
    return {
        "generated_at": "2026-05-12T12:00:00+00:00",
        "hosts": [
            {
                "host": "2080ti",
                "ok": True,
                "remote_user": remote_user,
                "drained": drained,
                "ports": ports or [],
                "gpus": [
                    {
                        "index": "1",
                        "status": status,
                        "util_pct": util_pct,
                        "assignments": assignments or [],
                        "processes": [] if process is None else [process],
                    }
                ],
            }
        ],
    }


def vllm_orphan_process():
    return {
        "pid": "3911703",
        "ppid": "3911581",
        "user": "kevin",
        "kind": "same-user-orphan",
        "age_seconds": 3600,
        "command": "VLLM::Worker",
        "parent_user": "kevin",
        "parent_ppid": "1",
        "parent_command": "VLLM::EngineCore",
        "orphan_reason": "vLLM engine parent is orphaned under init",
    }


def vllm_nonsweep_process():
    return {
        "pid": "200",
        "ppid": "100",
        "user": "kevin",
        "kind": "same-user-nonsweep",
        "age_seconds": 7200,
        "process_name": "VLLM::Worker",
        "command": "VLLM::Worker",
        "parent_user": "kevin",
        "parent_ppid": "50",
        "parent_command": "VLLM::EngineCore",
        "grandparent_ppid": "1",
        "grandparent_command": "python -m vllm.entrypoints.openai.api_server --port 8089",
    }


def sglang_stale_sweep_process():
    return {
        "pid": "200",
        "ppid": "100",
        "user": "kevin",
        "kind": "sweep",
        "age_seconds": 7200,
        "process_name": "sglang::scheduler_TP0",
        "command": "sglang::scheduler_TP0",
        "parent_user": "kevin",
        "parent_ppid": "50",
        "parent_command": "python -m sglang.launch_server --model-path model --port 8095",
        "grandparent_ppid": "1",
        "grandparent_command": "bash sweep_multiturn_profiles_sglang.sh",
    }


def sglang_stale_listener_port():
    return {
        "port": "8095",
        "detail": "LISTEN 0 2048 0.0.0.0:8095 0.0.0.0:* users:((\"python\",pid=100,fd=72))",
        "pid": "100",
        "user": "kevin",
        "ppid": "50",
        "age_seconds": 7200,
        "command": "python -m sglang.launch_server --model-path model --port 8095",
    }


def reclaim_config():
    config = dict(cleaner.DEFAULT_CONFIG)
    config["reclaim_same_user_nonsweep"] = dict(cleaner.DEFAULT_CONFIG["reclaim_same_user_nonsweep"])
    config["reclaim_stale_sweep_servers"] = dict(cleaner.DEFAULT_CONFIG["reclaim_stale_sweep_servers"])
    config["reclaim_drained_sweep_servers"] = dict(cleaner.DEFAULT_CONFIG["reclaim_drained_sweep_servers"])
    return config


class OrphanGpuCleanerTests(unittest.TestCase):
    def test_vllm_orphan_targets_engine_parent_and_worker(self):
        config = dict(cleaner.DEFAULT_CONFIG)
        process = vllm_orphan_process()

        candidates = cleaner.find_candidates(base_state(process), config)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kill_pids, ["3911581", "3911703"])

    def test_observation_gate_requires_repeated_live_sightings(self):
        config = dict(cleaner.DEFAULT_CONFIG)
        config["min_age_seconds"] = 0
        config["required_observations"] = 2
        process = vllm_orphan_process()

        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.json"
            audit_path = Path(tmp) / "events.jsonl"

            first = cleaner.cleanup_from_state(
                base_state(process),
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )
            second = cleaner.cleanup_from_state(
                base_state(process),
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:05:00+00:00",
            )

        self.assertEqual(first["events"], {"skip": 1})
        self.assertEqual(second["events"], {"dry-run": 1})
        self.assertEqual(second["eligible"], 1)

    def test_direct_init_orphan_targets_only_process(self):
        config = dict(cleaner.DEFAULT_CONFIG)
        process = vllm_orphan_process()
        process.update({
            "pid": "42",
            "ppid": "1",
            "parent_ppid": "",
            "parent_command": "",
            "orphan_reason": "process parent is init",
        })

        candidates = cleaner.find_candidates(base_state(process), config)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kill_pids, ["42"])

    def test_unapproved_parent_command_is_not_eligible(self):
        config = dict(cleaner.DEFAULT_CONFIG)
        config["min_age_seconds"] = 0
        config["required_observations"] = 1
        process = vllm_orphan_process()
        process["parent_command"] = "python manual_server.py"

        candidates = cleaner.find_candidates(base_state(process), config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(
            candidates,
            observations,
            "2026-05-12T12:00:01+00:00",
        )

        self.assertEqual(candidates[0].kill_pids, [])
        self.assertEqual(
            cleaner.skip_reason(candidates[0], config),
            "no configured kill target for orphan shape",
        )

    def test_same_user_nonsweep_managed_server_becomes_reclaimable(self):
        config = reclaim_config()
        config["reclaim_same_user_nonsweep"]["min_age_seconds"] = 0
        config["reclaim_same_user_nonsweep"]["required_observations"] = 2
        process = vllm_nonsweep_process()
        state = base_state(process, status="same-user-nonsweep")

        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.json"
            audit_path = Path(tmp) / "events.jsonl"

            first = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )
            second = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:05:00+00:00",
            )

        self.assertEqual(first["events"], {"skip": 1})
        self.assertEqual(second["events"], {"dry-run": 1})
        candidates = cleaner.find_candidates(state, config)
        self.assertEqual(candidates[0].policy, "same-user-nonsweep")
        self.assertEqual(candidates[0].port, "8089")
        self.assertEqual(candidates[0].kill_pids, ["50", "100", "200"])

    def test_same_user_nonsweep_with_run_lease_is_protected(self):
        config = reclaim_config()
        config["reclaim_same_user_nonsweep"]["min_age_seconds"] = 0
        config["reclaim_same_user_nonsweep"]["required_observations"] = 1
        process = vllm_nonsweep_process()
        process["bench_run_id"] = "run_20260512T120000Z_test"

        candidates = cleaner.find_candidates(base_state(process, status="same-user-nonsweep"), config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(candidates, observations, "2026-05-12T12:01:00+00:00")

        self.assertEqual(
            cleaner.skip_reason(candidates[0], config),
            "process has active BENCH_RUN_ID lease",
        )

    def test_same_user_nonsweep_unmanaged_port_is_not_reclaimable(self):
        config = reclaim_config()
        config["reclaim_same_user_nonsweep"]["min_age_seconds"] = 0
        config["reclaim_same_user_nonsweep"]["required_observations"] = 1
        process = vllm_nonsweep_process()
        process["grandparent_command"] = "python -m vllm.entrypoints.openai.api_server --port 9000"

        candidates = cleaner.find_candidates(base_state(process, status="same-user-nonsweep"), config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(candidates, observations, "2026-05-12T12:01:00+00:00")

        self.assertEqual(cleaner.skip_reason(candidates[0], config), "no managed scheduler port found")

    def test_other_user_nonsweep_is_never_a_candidate(self):
        config = reclaim_config()
        process = vllm_nonsweep_process()
        process["user"] = "teammate"

        candidates = cleaner.find_candidates(
            base_state(process, status="same-user-nonsweep", remote_user="kevin"),
            config,
        )

        self.assertEqual(candidates, [])

    def test_stale_sweep_server_without_assignment_becomes_reclaimable(self):
        config = reclaim_config()
        config["reclaim_stale_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_stale_sweep_servers"]["required_observations"] = 2
        process = sglang_stale_sweep_process()
        state = base_state(process, status="sweep")

        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.json"
            audit_path = Path(tmp) / "events.jsonl"

            first = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )
            second = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:05:00+00:00",
            )

        self.assertEqual(first["events"], {"skip": 1})
        self.assertEqual(second["events"], {"dry-run": 1})
        candidates = cleaner.find_candidates(state, config)
        self.assertEqual(candidates[0].policy, "stale-sweep-server")
        self.assertEqual(candidates[0].port, "8095")
        self.assertEqual(candidates[0].kill_pids, ["50", "100", "200"])

    def test_stale_sweep_server_with_assignment_is_not_a_candidate(self):
        config = reclaim_config()
        process = sglang_stale_sweep_process()
        state = base_state(process, status="sweep", assignments=[{"job_id": "active"}])

        candidates = cleaner.find_candidates(state, config)

        self.assertEqual(candidates, [])

    def test_drained_sweep_assignment_can_reclaim_old_idle_server(self):
        config = reclaim_config()
        config["reclaim_drained_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_drained_sweep_servers"]["required_observations"] = 2
        process = sglang_stale_sweep_process()
        process["bench_run_id"] = "run_20260512T120000Z_test"
        state = base_state(
            process,
            status="sweep",
            assignments=[
                {
                    "id": "2080ti_Tiny_tp1_multi_sglang",
                    "port": "8095",
                    "run_id": "run_20260512T120000Z_test",
                }
            ],
            drained=True,
            util_pct=0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.json"
            audit_path = Path(tmp) / "events.jsonl"

            first = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )
            second = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:05:00+00:00",
            )

        self.assertEqual(first["events"], {"skip": 1})
        self.assertEqual(second["events"], {"dry-run": 1})
        candidates = cleaner.find_candidates(state, config)
        self.assertEqual(candidates[0].policy, "drained-stale-sweep-server")
        self.assertEqual(candidates[0].port, "8095")
        self.assertEqual(candidates[0].run_id, "run_20260512T120000Z_test")
        self.assertEqual(candidates[0].gpu_util_pct, 0)
        self.assertEqual(candidates[0].kill_pids, ["50", "100", "200"])

    def test_drained_sweep_assignment_keeps_busy_gpu(self):
        config = reclaim_config()
        config["reclaim_drained_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_drained_sweep_servers"]["required_observations"] = 1
        process = sglang_stale_sweep_process()
        state = base_state(
            process,
            status="sweep",
            assignments=[{"id": "active", "port": "8095"}],
            drained=True,
            util_pct=35,
        )

        candidates = cleaner.find_candidates(state, config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(candidates, observations, "2026-05-12T12:01:00+00:00")

        self.assertEqual(
            cleaner.skip_reason(candidates[0], config),
            "gpu utilization above max_gpu_util_pct=0",
        )

    def test_stale_sweep_server_with_run_lease_is_protected(self):
        config = reclaim_config()
        config["reclaim_stale_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_stale_sweep_servers"]["required_observations"] = 1
        config["_active_run_ids"] = {"run_20260512T120000Z_test"}
        process = sglang_stale_sweep_process()
        process["bench_run_id"] = "run_20260512T120000Z_test"

        candidates = cleaner.find_candidates(base_state(process, status="sweep"), config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(candidates, observations, "2026-05-12T12:01:00+00:00")

        self.assertEqual(
            cleaner.skip_reason(candidates[0], config),
            "process has active BENCH_RUN_ID lease",
        )

    def test_stale_sweep_server_with_inactive_run_lease_is_reclaimable(self):
        config = reclaim_config()
        config["reclaim_stale_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_stale_sweep_servers"]["required_observations"] = 1
        process = sglang_stale_sweep_process()
        process["bench_run_id"] = "run_20260512T120000Z_test"
        state = base_state(process, status="sweep")

        with tempfile.TemporaryDirectory() as tmp:
            result = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=Path(tmp) / "observations.json",
                audit_log=Path(tmp) / "events.jsonl",
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )

        self.assertEqual(result["events"], {"dry-run": 1})

    def test_stale_listener_port_without_assignment_becomes_reclaimable(self):
        config = reclaim_config()
        config["reclaim_stale_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_stale_sweep_servers"]["required_observations"] = 2
        state = base_state(None, status="free", ports=[sglang_stale_listener_port()])

        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.json"
            audit_path = Path(tmp) / "events.jsonl"

            first = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:00:00+00:00",
            )
            second = cleaner.cleanup_from_state(
                state,
                config,
                observations_path=observation_path,
                audit_log=audit_path,
                dry_run=True,
                timestamp="2026-05-12T12:05:00+00:00",
            )

        self.assertEqual(first["events"], {"skip": 1})
        self.assertEqual(second["events"], {"dry-run": 1})
        candidates = cleaner.find_candidates(state, config)
        self.assertEqual(candidates[0].policy, "stale-sweep-listener")
        self.assertEqual(candidates[0].port, "8095")
        self.assertEqual(candidates[0].kill_pids, ["100"])

    def test_stale_listener_port_with_live_assignment_is_not_a_candidate(self):
        config = reclaim_config()
        state = base_state(
            None,
            status="free",
            ports=[sglang_stale_listener_port()],
            assignments=[{"port": "8095"}],
        )

        candidates = cleaner.find_candidates(state, config)

        self.assertEqual(candidates, [])

    def test_stale_listener_port_with_run_lease_is_protected(self):
        config = reclaim_config()
        config["reclaim_stale_sweep_servers"]["min_age_seconds"] = 0
        config["reclaim_stale_sweep_servers"]["required_observations"] = 1
        config["_active_run_ids"] = {"run_20260512T120000Z_test"}
        port = sglang_stale_listener_port()
        port["bench_run_id"] = "run_20260512T120000Z_test"
        candidates = cleaner.find_candidates(base_state(None, status="free", ports=[port]), config)
        observations = cleaner.update_observations(
            candidates,
            {"candidates": {}},
            "2026-05-12T12:00:00+00:00",
        )
        cleaner.update_observations(candidates, observations, "2026-05-12T12:01:00+00:00")

        self.assertEqual(
            cleaner.skip_reason(candidates[0], config),
            "process has active BENCH_RUN_ID lease",
        )


if __name__ == "__main__":
    unittest.main()
